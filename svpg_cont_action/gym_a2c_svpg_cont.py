import roboschool, gym
import itertools
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
#ENV_NAME="MountainCarContinuous-v0";
#ENV_NAME="Pendulum-v0";
ENV_NAME="RoboschoolAnt-v1";
if(ENV_NAME=="Pendulum-v0"):
  NUM_EPISODES=2000;
  ENTROPY_BETA=0.01;
  POLICY_LR=0.0001;
  VALUE_LR=0.001;
  NUM_VARS=6;
  UPDATE_ITER=10;
  MAX_EPI_STEP=200;
  DISCOUNT_FACTOR=0.9;
if(ENV_NAME=="MountainCarContinuous-v0"):
  NUM_EPISODES=100;
  ENTROPY_BETA=0.1;
  POLICY_LR=0.001;
  VALUE_LR=0.1;
  NUM_VARS=4;
  UPDATE_ITER=20;
  MAX_EPI_STEP=1000;
  DISCOUNT_FACTOR=0.95;
if(ENV_NAME=="RoboschoolAnt-v1"):
  NUM_EPISODES=50000;
  ENTROPY_BETA=0.01;
  POLICY_LR=0.0001;
  VALUE_LR=0.001;
  NUM_VARS=6;
  UPDATE_ITER=100;
  MAX_EPI_STEP=200;
  DISCOUNT_FACTOR=0.9;

# for SVPG
n_particles=1;
independent_flag_svpg=1;
###################

# gym env
env=np.zeros(n_particles,dtype=object);
for i in range(n_particles):
  env[i] = gym.envs.make(ENV_NAME)
num_state=env[0].observation_space.shape[0]
num_action=env[0].action_space.shape[0]
action_bound=[env[0].action_space.low, env[0].action_space.high]
# MAX EPI STEP is setted in gym 
MAX_EPI_STEP=env[0].spec.timestep_limit;

"""
For MountainCarContinuous-v0
"""
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
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]

class PolicyEstimator_MountainCarContinuous():
    def __init__(self, entropy_beta=0.1, learning_rate=0.001, par_idx=0,scope="policy_estimator"):
        w_init = tf.random_normal_initializer(0.,.1);
        with tf.variable_scope(scope+"_"+str(par_idx)):

            # state, target and action
            self.state = tf.placeholder(tf.float32, [None,400], name="state")
            self.target = tf.placeholder(tf.float32,[None,1], name="target")
            self.a_his = tf.placeholder(tf.float32, [None, num_action], name="action_hist")

            # layers
            self.mu = tf.layers.dense(self.state, num_action, tf.nn.tanh, kernel_initializer=w_init, name='mu') # estimated action value
            self.sigma = tf.layers.dense(self.state, num_action, tf.nn.softplus, kernel_initializer=w_init, name='sigma') # estimated variance

            # wrap output
            self.mu = self.mu * action_bound[1];
            self.sigma = self.sigma + 1e-5
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = tf.squeeze(self.normal_dist.sample(1),axis=0);
            self.action = tf.clip_by_value(self.action, action_bound[0], action_bound[1])

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.a_his) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= entropy_beta * self.normal_dist.entropy()
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
        state=featurize_state(state);
        return sess.run(self.action, { self.state: [state] })[0]

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state2 = np.zeros(len(state),dtype=object);
        for st_idx in range(len(state)):
            state2[st_idx]=featurize_state(state[st_idx]);
        feed_dict = { self.state: state2.tolist(), self.target: target, self.a_his: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator_MountainCarContinuous():
    def __init__(self, learning_rate=0.1, par_idx=0,scope="value_estimator"):
        w_init = tf.random_normal_initializer(0.,.1);
        with tf.variable_scope(scope+"_"+str(par_idx)):
            # state and target
            self.state = tf.placeholder(tf.float32, [None,400], "state")
            self.target = tf.placeholder(tf.float32, [None,1], name="target")

            # layers
            self.value_estimate = tf.layers.dense(self.state, 1, kernel_initializer=w_init, name='v')  # estimated value for state

            # loss and optimizer
            self.loss = tf.squared_difference(self.value_estimate, self.target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                                    self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state=featurize_state(state);
        return sess.run(self.value_estimate, { self.state: [state] })[0][0]

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state2 = np.zeros(len(state),dtype=object);
        for st_idx in range(len(state)):
            state2[st_idx]=featurize_state(state[st_idx]);
        feed_dict = { self.state: state2.tolist(), self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

"""
For Pendulum-v0
"""
class PolicyEstimator_Pendulum():
    def __init__(self, entropy_beta=0.01, learning_rate=0.01, par_idx=0,scope="policy_estimator"):
        w_init = tf.random_normal_initializer(0.,.1);
        with tf.variable_scope(scope+"_"+str(par_idx)):
            
            # state, target and action
            self.state = tf.placeholder(tf.float32, [None,num_state], name="state")
            self.target = tf.placeholder(tf.float32,[None,1], name="target")
            self.a_his = tf.placeholder(tf.float32, [None, num_action], name="action_hist")        
            
            # layers
            l_a = tf.layers.dense(self.state, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            self.mu = tf.layers.dense(l_a, num_action, tf.nn.tanh, kernel_initializer=w_init, name='mu') # estimated action value
            self.sigma = tf.layers.dense(l_a, num_action, tf.nn.softplus, kernel_initializer=w_init, name='sigma') # estimated variance
            
            # wrap output
            self.mu = self.mu * action_bound[1];
            self.sigma = self.sigma + 1e-4

            # get action from distribution
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = tf.squeeze(self.normal_dist.sample(1),axis=0);
            self.action = tf.clip_by_value(self.action, action_bound[0], action_bound[1])
            
            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.a_his) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= entropy_beta * self.normal_dist.entropy()
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
        return sess.run(self.action, { self.state: [state] })[0]

    def update(self, state, target, a_his, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.a_his: a_his  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator_Pendulum():
    def __init__(self, learning_rate=0.1, par_idx=0,scope="value_estimator"):
        w_init = tf.random_normal_initializer(0.,.1);
        with tf.variable_scope(scope+"_"+str(par_idx)):
            # state and target
            self.state = tf.placeholder(tf.float32, [None,num_state], "state")
            self.target = tf.placeholder(tf.float32, [None,1], name="target")

            # layers
            l_c = tf.layers.dense(self.state, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            self.value_estimate = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # estimated value for state
        
            # loss and optimizer
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.value_estimate, self.target)))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: [state] })[0][0]

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

"""
For RoboschoolAnt-v1
"""
class PolicyEstimator_Ant():
    def __init__(self, entropy_beta=0.01, learning_rate=0.01, par_idx=0,scope="policy_estimator"):
        w_init = tf.random_normal_initializer(0.,.1);
        with tf.variable_scope(scope+"_"+str(par_idx)):
            
            # state, target and action
            self.state = tf.placeholder(tf.float32, [None,num_state], name="state")
            self.target = tf.placeholder(tf.float32,[None,1], name="target")
            self.a_his = tf.placeholder(tf.float32, [None, num_action], name="action_hist")        
            
            # layers
            l_a = tf.layers.dense(self.state, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            self.mu = tf.layers.dense(l_a, num_action, tf.nn.tanh, kernel_initializer=w_init, name='mu') # estimated action value
            self.sigma = tf.layers.dense(l_a, num_action, tf.nn.softplus, kernel_initializer=w_init, name='sigma') # estimated variance
            
            # wrap output
            self.mu = self.mu * action_bound[1];
            self.sigma = self.sigma + 1e-4

            # get action from distribution
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = tf.squeeze(self.normal_dist.sample(1),axis=0);
            self.action = tf.clip_by_value(self.action, action_bound[0], action_bound[1])
            
            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.a_his) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= entropy_beta * self.normal_dist.entropy()
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
        return sess.run(self.action, { self.state: [state] })[0]

    def update(self, state, target, a_his, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.a_his: a_his  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator_Ant():
    def __init__(self, learning_rate=0.1, par_idx=0,scope="value_estimator"):
        w_init = tf.random_normal_initializer(0.,.1);
        with tf.variable_scope(scope+"_"+str(par_idx)):
            # state and target
            self.state = tf.placeholder(tf.float32, [None,num_state], "state")
            self.target = tf.placeholder(tf.float32, [None,1], name="target")

            # layers
            l_c = tf.layers.dense(self.state, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            self.value_estimate = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # estimated value for state
        
            # loss and optimizer
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.value_estimate, self.target)))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: [state] })[0][0]

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def advantage_actor_critic(env, estimator_policy, estimator_value, svpg, num_episodes,max_epi_step, discount_factor=1.0):
    # Keeps track of useful statistics
    stats = {};
    stats["episode_lengths"]=np.zeros((n_particles,num_episodes));
    stats["episode_rewards"]=np.zeros((n_particles,num_episodes));
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
   
    # state list
    state=np.zeros(n_particles,dtype=object);
    action=np.zeros(n_particles,dtype=object);
    next_state=np.zeros(n_particles,dtype=object);
    episode=np.zeros(n_particles,dtype=object);
    reward=np.zeros(n_particles,dtype=object);
    done=np.zeros(n_particles,dtype=object);
    policy_grads=np.zeros(n_particles,dtype=object);

    # total step
    total_step=1;
    # trasition backup initialization
    for i in range(n_particles):
      episode[i]=[];

    for i_episode in range(num_episodes):
        # Reset the environment
        for i in range(n_particles):
          state[i] = env[i].reset()
        # run
        for t in range(MAX_EPI_STEP):
            for i in range(n_particles):
              # Take a step
              action[i] = estimator_policy[i].predict(state[i])
              next_state[i], reward[i], done[i], _ = env[i].step(action[i])
              # Pendulum case maximum running is just done (there are no reward threshold)
              if(ENV_NAME=="Pendulum-v0"):
                done[i] = True if t == max_epi_step -1 else False

              # Keep track of the transition
              episode[i].append(Transition(
                 state=state[i], action=action[i], reward=reward[i], next_state=next_state[i], done=done[i]))

              # Update statistics
              stats["episode_rewards"][i][i_episode] += reward[i]
              stats["episode_lengths"][i][i_episode] = t
              
              state[i] = next_state[i]
            
            # checking one of them is done
            Done=False;
            for i in range(n_particles):
              if done[i]:
                Done=True;
            
            if((total_step%UPDATE_ITER==0)or(Done)):
              feed_dict={};
              # Buffer for each particle
              buffer_s=np.zeros(n_particles,dtype=object);
              buffer_a=np.zeros(n_particles,dtype=object);
              buffer_v=np.zeros(n_particles,dtype=object);
              buffer_td_target=np.zeros(n_particles,dtype=object);
              buffer_td_error=np.zeros(n_particles,dtype=object);
              for i in range(n_particles):
                buffer_s[i]=[];
                buffer_a[i]=[];
                buffer_v[i]=[];
                buffer_td_target[i]=[];
                buffer_td_error[i]=[];
                for t in range(len(episode[i])):
                  transition=episode[i][t];
                  buffer_s[i].append(transition.state);
                  buffer_a[i].append(transition.action);
                  # normalize reward for Pendulum case
                  if(ENV_NAME=="Pendulum-v0"):
                    buffer_v[i].append((transition.reward+8)/8)
                  else:
                    buffer_v[i].append(transition.reward)
                if done[i]:
                  v_s_=0;
                else:
                  v_s_=estimator_value[i].predict(episode[i][-1].next_state);
                buffer_v[i].reverse();
                for r in buffer_v[i]:
                  v_s_=r+discount_factor*v_s_
                  buffer_td_target[i].append(v_s_);
                buffer_td_target[i].reverse();
                for t in range(len(buffer_s[i])):
                  buffer_td_error[i].append(buffer_td_target[i][t]-estimator_value[i].predict(buffer_s[i][t]));
                estimator_value[i].update(buffer_s[i],np.reshape(buffer_td_target[i],[-1,1]));
                # For MountainCarContinuous case, we uses RBF for pre-processing
                if(ENV_NAME=="MountainCarContinuous-v0"):
                  for st_idx in range(len(buffer_s[i])):
                    buffer_s[i][st_idx]=featurize_state(buffer_s[i][st_idx]);
                feed_dict.update({estimator_policy[i].state:buffer_s[i]});
                feed_dict.update({estimator_policy[i].target:np.reshape(buffer_td_error[i],[-1,1])});
                feed_dict.update({estimator_policy[i].a_his:np.reshape(buffer_a[i],[-1,num_action])});
              svpg.run(feed_dict);
              # trasition backup re-set
              for i in range(n_particles):
                episode[i]=[];
            total_step+=1;
            if Done:
                break
            
        # Print out which step we're on, useful for debugging. (average recent 10 episode scores)
        if(i_episode>=10):
          print("Episode {}/{} ({})".format(i_episode + 1, num_episodes, np.max(np.mean(stats["episode_rewards"][:,i_episode-10:i_episode - 1],axis=1))))
    
    return stats

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = np.zeros(n_particles,dtype=object);
value_estimator = np.zeros(n_particles,dtype=object);

# call proper policy and value estimators for each envs
for i in range(n_particles):
  if(ENV_NAME=="Pendulum-v0"):
    policy_estimator[i] = PolicyEstimator_Pendulum(entropy_beta=ENTROPY_BETA,learning_rate=POLICY_LR,par_idx=i)
    value_estimator[i] = ValueEstimator_Pendulum(learning_rate=VALUE_LR,par_idx=i)
  if(ENV_NAME=="MountainCarContinuous-v0"):
    policy_estimator[i] = PolicyEstimator_MountainCarContinuous(entropy_beta=ENTROPY_BETA,learning_rate=POLICY_LR,par_idx=i)
    value_estimator[i] = ValueEstimator_MountainCarContinuous(learning_rate=VALUE_LR,par_idx=i)
  if(ENV_NAME=="RoboschoolAnt-v1"):
    policy_estimator[i] = PolicyEstimator_Ant(entropy_beta=ENTROPY_BETA,learning_rate=POLICY_LR,par_idx=i)
    value_estimator[i] = ValueEstimator_Ant(learning_rate=VALUE_LR,par_idx=i)

svpg=SVPG(policy_estimator,independent_flag_svpg,learning_rate=POLICY_LR);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need varies
    # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
    stats = advantage_actor_critic(env, policy_estimator, value_estimator, svpg, NUM_EPISODES, MAX_EPI_STEP,discount_factor=DISCOUNT_FACTOR)

