# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
import time
import sys
import argparse
import filter_env
from gym import wrappers
from ddpg import *
from svpg import SVPG
import gc
gc.enable()

FLAGS=None;
#ENV_NAME = 'Reacher-v1'
ENV_NAME = 'MountainCarContinuous-v0'
EPISODES = 500
local_step=10
TEST=10
sleep_time=0.1

def wait_flag_on(sess,flag_list,idx_list,on_value):
  while True:
    # check flag value
    sum_v=0;
    for idx in idx_list:
      sum_v+=int(sess.run(flag_list[idx]));
    if(sum_v==on_value*len(idx_list)):
      break;
    else:
      time.sleep(sleep_time);

def set_flag(sess,flag_ph_list,flag_ops_list,idx_list,value):
    for i in idx_list:
      sess.run(flag_ops_list[i],feed_dict={flag_ph_list[i]:value});

def train():
  # parameter server and worker information
  ps_hosts = np.zeros(FLAGS.ps_hosts_num,dtype=object);
  worker_hosts = np.zeros(FLAGS.worker_hosts_num,dtype=object);
  port_num=FLAGS.st_port_num;
  for i in range(FLAGS.ps_hosts_num):
    ps_hosts[i]=str(FLAGS.hostname)+":"+str(port_num);
    port_num+=1;
  for i in range(FLAGS.worker_hosts_num):
    worker_hosts[i]=str(FLAGS.hostname)+":"+str(port_num);
    port_num+=1;
  ps_hosts=list(ps_hosts);
  worker_hosts=list(worker_hosts);
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join();
  elif FLAGS.job_name == "worker":
    device=tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % FLAGS.task_index,
          cluster=cluster);

    #tf.set_random_seed(1);
    # env and model call
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env,device,FLAGS.task_index,FLAGS.worker_hosts_num)
    # actor params lists (size is n_particles)
    a_list = agent.a_list;
    # policy gradient for actor params list (size is n_particles)
    pg_list = agent.pg_list;
    # svpg
    # 1 worker(task index = worker_hosts_num-1) is to SVGD 
    svpg = SVPG(a_list,pg_list,agent.state_dim,agent.action_dim,FLAGS.independent_flag)

    # prepare session
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
      # global step
      global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False);
      global_step_ph=tf.placeholder(global_step.dtype,shape=global_step.get_shape());
      global_step_ops=global_step.assign(global_step_ph);
      # score
      score = tf.get_variable('score',[],initializer=tf.constant_initializer(-150),trainable=False);
      score_ph=tf.placeholder(score.dtype,shape=score.get_shape());
      score_ops=score.assign(score_ph);
      # score for each particles
      score_p=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      score_p_ph=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      score_p_ops=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      for i in range(FLAGS.worker_hosts_num):
        score_p[i] = tf.get_variable('score_p_'+str(i),[],initializer=tf.constant_initializer(0),trainable=False);
        score_p_ph[i] = tf.placeholder(score_p[i].dtype,shape=score_p[i].get_shape());
        score_p_ops[i] =score_p[i].assign(score_p_ph[i]);
      # flags
      flag=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      flag_ph=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      flag_ops=np.zeros(FLAGS.worker_hosts_num,dtype=object);
      for i in range(FLAGS.worker_hosts_num):
        flag[i] = tf.get_variable('flag_'+str(i),[],initializer=tf.constant_initializer(0),trainable=False);
        flag_ph[i] = tf.placeholder(flag[i].dtype,shape=flag[i].get_shape());
        flag_ops[i] =flag[i].assign(flag_ph[i]);
      # initialization
      init_op=tf.global_variables_initializer();
      # summary for tensorboard
      tf.summary.scalar("score", score);
      summary_op = tf.summary.merge_all()
      saver = tf.train.Saver();
    
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                   global_step=global_step,
                                   logdir=FLAGS.log_dir,
                                   summary_op=summary_op,
                                   saver=saver,
                                   init_op=init_op)

    with sv.managed_session(server.target) as sess:
      agent.set_sess(sess);
      while True:
        if sess.run(global_step) > EPISODES:
          break
        score=0;
        for ls in range(local_step):
          state = env.reset();
          for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent_flag=agent.perceive(state,action,reward,next_state,done)
            if(agent_flag):
              # updating flag for stopping to not go next step before svpg
              sess.run(flag_ops[FLAGS.task_index],{flag_ph[FLAGS.task_index]:1.0});
              # last worker does svpg
              if(FLAGS.task_index==(FLAGS.worker_hosts_num-1)):
                # wait other workers
                wait_flag_on(sess,flag,range(FLAGS.worker_hosts_num-1),1);
                svpg.svgd_run(sess);
                # turn off the flags
                set_flag(sess,flag_ph,flag_ops,range(FLAGS.worker_hosts_num),0.0);
              # wait to finish svpg
              wait_flag_on(sess,flag,[FLAGS.task_index],0);
              # after sampling, updating target NN
              agent.update_target();
            state = next_state
            if done:
              break;
        for i in xrange(TEST):
          state = env.reset()
          for j in xrange(env.spec.timestep_limit):
            action = agent.action(state) # direct action for test
            state,reward,done,_ = env.step(action)
            score += reward
            if done:
              break
        # save the score to each particles score tf variable
        sess.run(score_p_ops[FLAGS.task_index],feed_dict={score_p_ph[FLAGS.task_index]:score/TEST});
        set_flag(sess,flag_ph,flag_ops,[FLAGS.task_index],1.0);
        # last worker updates score and global step
        if(FLAGS.task_index==(FLAGS.worker_hosts_num-1)):
          wait_flag_on(sess,flag,range(FLAGS.worker_hosts_num-1),1);
          # find maximum score
          max_score=-500;
          for i in range(FLAGS.worker_hosts_num):
            if(sess.run(score_p[i])>max_score):
              max_score=sess.run(score_p[i]);
          sess.run(score_ops,{score_ph:max_score});
          # global step update
          sess.run(global_step_ops,{global_step_ph:sess.run([global_step])[0]+local_step});
          print(str(sess.run([global_step])[0])+","+str(max_score));
      # for checking parallelization
      """
      print(FLAGS.task_index);
      while True:
        print(FLAGS.task_index);
        time.sleep(10);
      """
    sv.stop();
    print("Done");

def main(_):
  #os.system("rm -rf "+FLAGS.log_dir);
  FLAGS.ps_hosts_num+=1;
  FLAGS.worker_hosts_num+=1;
  train()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts_num",
      type=int,
      default=5,
      help="The Number of Parameter Servers"
  )
  parser.add_argument(
      "--worker_hosts_num",
      type=int,
      default=10,
      help="The Number of Workers"
  )
  parser.add_argument(
      "--hostname",
      type=str,
      default="localhost",
      help="The Hostname of the machine"
  )
  parser.add_argument(
      "--st_port_num",
      type=int,
      default=2222,
      help="The start port number of ps and worker servers"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  # the flag for independently learning or not
  parser.add_argument(
      "--independent_flag",
      type=int,
      default=0,
      help="the flag for independently learning or not"
  )
  # Log folder 
  parser.add_argument(
      "--log_dir",
      type=str,
      default="/tmp/svpg_sddpg/MountainCarContinuous/independent(a=1.0,stepsize=1e-4,particle_n=1)",
      help="log folder name"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
