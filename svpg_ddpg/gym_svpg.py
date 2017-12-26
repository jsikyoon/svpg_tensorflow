import filter_env
from ddpg import *
from svpg import SVPG
import gc
gc.enable()

#ENV_NAME = 'InvertedPendulum-v1'
ENV_NAME = 'MountainCarContinuous-v0'
EPISODES = 10000
TEST = 10
n_particle=3
independent_flag=1

def main():
    # tensorflow session
    sess=tf.InteractiveSession()

    # set agents per each particle
    envs=np.zeros(n_particle,dtype=object);
    agents=np.zeros(n_particle,dtype=object);
    states=np.zeros(n_particle,dtype=object);
    dones=np.zeros(n_particle,dtype=bool);
    actor_nets=np.zeros(n_particle,dtype=object);
    actor_pg_list=np.zeros(n_particle,dtype=object);
    ave_reward = np.zeros(n_particle,dtype=float);
    for i in range(n_particle):
      envs[i] = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
      agents[i] = DDPG(sess,envs[i],i)
      dones[i] = False;
      actor_nets[i] = agents[i].actor_network.net;
      actor_pg_list[i] = agents[i].actor_network.pg_list;
    actor_nets=np.array(list(actor_nets));
    actor_pg_list=np.array(list(actor_pg_list));
    svpg = SVPG(sess,actor_nets,actor_pg_list,envs[0].observation_space.shape[0],envs[0].action_space.shape[0],independent_flag);

    # session initialization and target NN update
    sess.run(tf.global_variables_initializer());
    for par in range(n_particle):
      agents[par].update_target();

    for episode in xrange(EPISODES):
      for par in range(n_particle):
        states[par] = envs[par].reset()
      # Train
      for par in range(n_particle):
        dones[par]=False;
      for step in xrange(envs[0].spec.timestep_limit):
        flag=0;
        for par in range(n_particle):
          if not dones[par]:
            action = agents[par].noise_action(states[par])
            next_state,reward,done,_ = envs[par].step(action)
            agents[par].save_to_buffer(states[par],action,reward,next_state,done)
            states[par] = next_state
            if done:
              dones[par]=True;
            if agents[par].can_train():
              flag+=1;
        if(flag == n_particle):
          for par in range(n_particle):
            # train critic NN and get policy gradient
            agents[par].train();
          # svpg
          svpg.run();
          for par in range(n_particle):
            agents[par].update_target();
      
      # Testing:
      if episode % 100 == 0 and episode > 100:
        for par in range(n_particle):
          total_reward=0;
	  for i in xrange(TEST):
  	    state = envs[par].reset()
	    for j in xrange(envs[0].spec.timestep_limit):
	      action = agents[par].action(state) # direct action for test
	      state,reward,done,_ = envs[par].step(action)
	      total_reward += reward
	      if done:
	        break
	  ave_reward[par] = total_reward/TEST
	print 'episode: ',episode,'Evaluation Average Reward:',np.max(ave_reward)
        if np.max(ave_reward)>950.0:
          print 'solved';
          exit(1);

if __name__ == '__main__':
    main()
