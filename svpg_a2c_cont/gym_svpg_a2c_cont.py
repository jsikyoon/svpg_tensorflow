# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time
import sys
import argparse

from a2c_training_thread import A2CTrainingThread
from rmsprop_applier import RMSPropApplier
from svpg import SVPG
from constants import ACTION_SIZE
from constants import N_PARTICLES
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_LSTM

FLAGS=None;
independent_flag=1;

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

def train():
  initial_learning_rate=np.zeros(N_PARTICLES,dtype=object);
  learning_rate_input=np.zeros(N_PARTICLES,dtype=object);
  grad_applier=np.zeros(N_PARTICLES,dtype=object);
  training_thread=np.zeros(N_PARTICLES,dtype=object);
  net_list=np.zeros(N_PARTICLES,dtype=object);
  grad_list=np.zeros(N_PARTICLES,dtype=object);
  score_list=np.zeros(N_PARTICLES,dtype=object);
  # define each particles
  for par_idx in range(N_PARTICLES):
    #initial learning rate
    initial_learning_rate[par_idx] = log_uniform(INITIAL_ALPHA_LOW,INITIAL_ALPHA_HIGH,INITIAL_ALPHA_LOG_RATE)
    learning_rate_input[par_idx] = tf.placeholder("float")
    #rms grad applier
    grad_applier[par_idx] = RMSPropApplier(learning_rate = learning_rate_input[par_idx],decay = RMSP_ALPHA,momentum = 0.0,epsilon = RMSP_EPSILON,clip_norm = GRAD_NORM_CLIP)
    #a2c class
    training_thread[par_idx] = A2CTrainingThread(initial_learning_rate[par_idx],learning_rate_input[par_idx],grad_applier[par_idx], MAX_TIME_STEP,par_idx)
    net_list[par_idx] = training_thread[par_idx].vars_for_svpg;
    grad_list[par_idx] = training_thread[par_idx].policy_grad_for_svpg;

  # prepare session
  score = tf.get_variable('score',[],initializer=tf.constant_initializer(-21),trainable=False);
  score_ph=tf.placeholder(score.dtype,shape=score.get_shape());
  score_ops=score.assign(score_ph);
  # summary for tensorboard
  tf.summary.scalar("score", score);
  summary_op = tf.summary.merge_all()
  saver = tf.train.Saver();
  
  sess=tf.InteractiveSession();
  
  # SVPG
  svpg = SVPG(sess,net_list,grad_list,independent_flag);
  
  # initialization
  sess.run(tf.global_variables_initializer());
  
  # set start time
  for par_idx in range(N_PARTICLES):
    training_thread[par_idx].set_start_time(time.time())
  global_step=0;
  while True:
    if global_step > MAX_TIME_STEP:
      break
    for par_idx in range(N_PARTICLES):
      score_list[par_idx]=training_thread[par_idx].process(sess, global_step, "",summary_op, "",score_ph,score_ops)
    svpg.run();
    global_step+=1;
    print("Iteration: "+str(global_step)+", score: "+str(np.max(score_list)));

def main(_):
  train()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
