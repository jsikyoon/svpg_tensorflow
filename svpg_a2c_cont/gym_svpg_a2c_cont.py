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

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
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

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

def train():
  #initial learning rate
  initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,INITIAL_ALPHA_HIGH,INITIAL_ALPHA_LOG_RATE)
  learning_rate_input = tf.placeholder("float")
  
  #rms grad applier
  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,decay = RMSP_ALPHA,momentum = 0.0,epsilon = RMSP_EPSILON,clip_norm = GRAD_NORM_CLIP)
    
  #a2c class
  training_thread = A2CTrainingThread(initial_learning_rate,learning_rate_input,grad_applier, MAX_TIME_STEP)
    
  # prepare session
  score = tf.get_variable('score',[],initializer=tf.constant_initializer(-21),trainable=False);
  score_ph=tf.placeholder(score.dtype,shape=score.get_shape());
  score_ops=score.assign(score_ph);
  init_op=tf.global_variables_initializer();
  # summary for tensorboard
  tf.summary.scalar("score", score);
  summary_op = tf.summary.merge_all()
  saver = tf.train.Saver();
  
  sess=tf.InteractiveSession();
  sess.run(init_op);
  
  training_thread.set_start_time(time.time())
  global_step=0;
  while True:
    if global_step > MAX_TIME_STEP:
      break
    diff_global_step = training_thread.process(sess, global_step, "",
                                            summary_op, "",score_ph,score_ops)
    global_step+=1;

def main(_):
  train()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
