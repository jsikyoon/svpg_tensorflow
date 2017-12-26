import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

import sklearn.pipeline
import sklearn.preprocessing

from sklearn.kernel_approximation import RBFSampler

from svpg import SVPG

###################
# parameters
###################
ENV_NAME="MountainCarContinuous-v0";
#ENV_NAME="Pendulum-v0";
#ENV_NAME="Reacher-v1";
NUM_EPISODES=50;
n_particles=2;
policy_lr=0.001;
value_lr=0.1;
independent_flag_svpg=1;
NUM_VARS=4;
###################

# gym env
env=np.zeros(n_particles,dtype=object);
for i in range(n_particles):
  env[i] = gym.envs.make(ENV_NAME)

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
obsp_high=env[0].observation_space.high;
obsp_low=env[0].observation_space.low;
for i in range(len(obsp_high)):
  if(obsp_high[i]==float('Inf')):
    obsp_high[i]=1e+10;
for i in range(len(obsp_low)):
  if(obsp_low[i]==-float('Inf')):
    obsp_low[i]=-1e+10;
observation_examples = np.array([np.random.uniform(low=obsp_low, high=obsp_high,size=env[0].observation_space.low.shape) for x in range(10000)])

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    def __init__(self, learning_rate=0.01, par_idx=0,scope="policy_estimator"):
        with tf.variable_scope(scope+"_"+str(par_idx)):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.mu = tf.squeeze(self.mu)
            
            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            
            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            self.action = tf.clip_by_value(self.action, env[0].action_space.low[0], env[0].action_space.high[0])

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads=[];
            self.vars=[];
            for i in range(len(self.grads_and_vars)):
              self.grads.append(self.grads_and_vars[i][0]);
              self.vars.append(self.grads_and_vars[i][1]);
            self.grads=self.grads[-1*NUM_VARS:];
            self.vars=self.vars[-1*NUM_VARS:];
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars, global_step=tf.contrib.framework.get_global_step())
             
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.action, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.1, par_idx=0,scope="value_estimator"):
        with tf.variable_scope(scope+"_"+str(par_idx)):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def actor_critic(env, estimator_policy, estimator_value, svpg, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environments List.
        estimator_policy: Policy Function List to be optimized 
        estimator_value: Value function approximator List, used as a baseline
        svpg: svpg module connected with estimator_policy
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = {};
    stats["episode_lengths"]=np.zeros((n_particles,num_episodes));
    stats["episode_rewards"]=np.zeros((n_particles,num_episodes));
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
   
    # state list
    state=np.zeros(n_particles,dtype=object);
    episode=np.zeros(n_particles,dtype=object);
    action=np.zeros(n_particles,dtype=object);
    next_state=np.zeros(n_particles,dtype=object);
    reward=np.zeros(n_particles,dtype=object);
    done=np.zeros(n_particles,dtype=object);
    policy_grads=np.zeros(n_particles,dtype=object);

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        for i in range(n_particles):
          state[i] = env[i].reset()
          episode[i] = []
        
        # One step in the environment
        for t in itertools.count():
            feed_dict={}
            for i in range(n_particles):
              # Take a step
              action[i] = estimator_policy[i].predict(state[i])
              next_state[i], reward[i], done[i], _ = env[i].step(action[i])
            
              # Keep track of the transition
              episode[i].append(Transition(
                state=state[i], action=action[i], reward=reward[i], next_state=next_state[i], done=done[i]))
            
              # Update statistics
              stats["episode_rewards"][i][i_episode] += reward[i]
              stats["episode_lengths"][i][i_episode] = t
            
              # Calculate TD Target
              value_next = estimator_value[i].predict(next_state[i])
              td_target = reward[i] + discount_factor * value_next
              td_error = td_target - estimator_value[i].predict(state[i])
            
              # Update the value estimator
              estimator_value[i].update(state[i], td_target)
            
              # Update the policy estimator
              # using the td error as our advantage estimate
              # estimator_policy[i].update(state[i], td_error, action[i])
              feed_dict.update({estimator_policy[i].state:featurize_state(state[i])});
              feed_dict.update({estimator_policy[i].target:td_error});
              feed_dict.update({estimator_policy[i].action:action[i]});
              state[i] = next_state[i]
            svpg.run(feed_dict);

            # checking one of them is done
            Done=False;
            for i in range(n_particles):
              if done[i]:
                Done=True;
            
            if Done:
                break
            
        # Print out which step we're on, useful for debugging.
        print("Episode {}/{} ({})".format(i_episode + 1, num_episodes, np.max(stats["episode_rewards"][:,i_episode - 1])))
    
    return stats

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = np.zeros(n_particles,dtype=object);
value_estimator = np.zeros(n_particles,dtype=object);
for i in range(n_particles):
  policy_estimator[i] = PolicyEstimator(learning_rate=policy_lr,par_idx=i)
  value_estimator[i] = ValueEstimator(learning_rate=value_lr,par_idx=i)

svpg=SVPG(policy_estimator,independent_flag_svpg,learning_rate=policy_lr);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need varies
    # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
    stats = actor_critic(env, policy_estimator, value_estimator, svpg, NUM_EPISODES, discount_factor=0.95)



