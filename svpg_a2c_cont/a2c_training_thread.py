# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from game_state import GameState
from game_state import ACTION_SIZE
from game_state import STATE_SIZE
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A2CTrainingThread(object):
  def __init__(self,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               par_idx):

    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step
    self.game_state = GameState()
    state=self.game_state.reset();
    self.game_state.reset_gs(state);
    self.par_idx=par_idx;

    if USE_LSTM:
      self.local_network = GameACLSTMNetwork(ACTION_SIZE,str(par_idx))
    else:
      self.local_network = GameACFFNetwork(ACTION_SIZE,str(par_idx))

    self.local_network.prepare_loss(ENTROPY_BETA)

    var_refs = [v._ref() for v in self.local_network.get_vars()]
    # for total loss
    self.gradients = tf.gradients(
      self.local_network.total_loss, var_refs,
      gate_gradients=False,
      aggregation_method=None,
      colocate_gradients_with_ops=False)
    # for policy loss
    self.policy_gradients = tf.gradients(
      self.local_network.policy_loss, var_refs,
      gate_gradients=False,
      aggregation_method=None,
      colocate_gradients_with_ops=False)
    # for value loss
    self.value_gradients = tf.gradients(
      self.local_network.value_loss, var_refs,
      gate_gradients=False,
      aggregation_method=None,
      colocate_gradients_with_ops=False)

    # get parms and gradient except None ones
    vars_for_svpg=[];
    policy_gradients2=[];
    for params,grad in zip(self.local_network.get_vars(),self.policy_gradients):
      if not grad == None:
        vars_for_svpg.append(params);
        policy_gradients2.append(grad);
    self.vars_for_svpg=vars_for_svpg;
    self.policy_gradients2=policy_gradients2;

    # get origin shape (2 is for flat version)
    origin_shape=np.zeros(len(self.policy_gradients2),dtype=object);
    origin_shape2=np.zeros(len(self.policy_gradients2),dtype=object);
    for i in range(len(self.policy_gradients2)):
      params_shape=self.policy_gradients2[i].get_shape().as_list();
      total_len=1;
      for j in params_shape:
        total_len*=j;
      origin_shape[i]=params_shape;
      origin_shape2[i]=total_len;

    # make store variable for gradients
    policy_grad_for_svpg=np.zeros(len(self.policy_gradients2),dtype=object);
    policy_grad_for_svpg_ph=np.zeros(len(self.policy_gradients2),dtype=object);
    policy_grad_for_svpg_op=np.zeros(len(self.policy_gradients2),dtype=object);
    for i in range(len(self.policy_gradients2)):
      store_var = tf.get_variable(str(i)+"_"+str(self.par_idx),origin_shape[i],initializer=tf.constant_initializer(0.0),trainable=False);
      store_var_ph = tf.placeholder(store_var.dtype,shape=store_var.get_shape());
      store_var_op = store_var.assign(store_var_ph);
      policy_grad_for_svpg[i]=store_var;
      policy_grad_for_svpg_ph[i]=store_var_ph;
      policy_grad_for_svpg_op[i]=store_var_op;
    self.policy_grad_for_svpg=policy_grad_for_svpg;
    self.policy_grad_for_svpg_ph=policy_grad_for_svpg_ph;
    self.policy_grad_for_svpg_op=policy_grad_for_svpg_op;

    #self.apply_gradients = grad_applier.apply_gradients(
    #  self.local_network.get_vars(),
    #  self.gradients )
    self.apply_gradients = grad_applier.apply_gradients(
      self.local_network.get_vars(),
      self.value_gradients )
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input,score_ph="",score_ops=""):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    start_local_t = self.local_t

    if USE_LSTM:
      pstart_lstm_state = self.local_network.plstm_state_out
      vstart_lstm_state = self.local_network.vlstm_state_out

    # t_max times loop
    for i in range(LOCAL_T_MAX):
      action, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward

      # clip reward
      rewards.append( np.clip(reward, -1, 1) )

      # s_t1 -> s_t
      self.game_state.update()
      if terminal:
        terminal_end = True
        score=self.episode_reward/self.game_state.r_sc;
        """
        print("episode: "+str(global_t+1)+", score={}".format(self.episode_reward/self.game_state.r_sc))
        if summary_writer:
          self._record_score(sess, summary_writer, summary_op, score_input,
            self.episode_reward/self.game_state.r_sc, global_t)
        else:
          sess.run(score_ops,{score_ph:self.episode_reward/self.game_state.r_sc});
        """
        self.episode_reward = 0
        state=self.game_state.reset()
        self.game_state.reset_gs(state);
        if USE_LSTM:
          self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.game_state.s_t)
      score=self.episode_reward/self.game_state.r_sc;

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi

      batch_si.append(si)
      batch_R.append(R)
      batch_td.append(td);

    cur_learning_rate = self._anneal_learning_rate(global_t)

    if USE_LSTM:
      batch_si.reverse()
      batch_td.reverse()
      batch_R.reverse()
      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.pinitial_lstm_state: pstart_lstm_state,
                  self.local_network.pstep_size : [len(batch_a)],
                  self.local_network.vinitial_lstm_state: vstart_lstm_state,
                  self.local_network.vstep_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )
      # get gradient
      pg_list=sess.run( self.policy_gradients2,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.pinitial_lstm_state: pstart_lstm_state,
                  self.local_network.pstep_size : [len(batch_a)],
                  self.local_network.vinitial_lstm_state: vstart_lstm_state,
                  self.local_network.vstep_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )
      for pg_idx in range(len(self.policy_gradients2)):
        pg_list[pg_idx]=pg_list[pg_idx].tolist();
        sess.run(self.policy_grad_for_svpg_op[pg_idx],feed_dict={self.policy_grad_for_svpg_ph[pg_idx]:pg_list[pg_idx]});
    else:
      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.r: batch_R,
                  self.local_network.td: batch_td,
                  self.learning_rate_input: cur_learning_rate} )
      # get gradient
      pg_list=sess.run( self.policy_gradients2,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.r: batch_R,
                  self.local_network.td: batch_td,
                  self.learning_rate_input: cur_learning_rate} )
      for pg_idx in range(len(self.policy_gradients2)):
        pg_list[pg_idx]=pg_list[pg_idx].tolist();
        sess.run(self.policy_grad_for_svpg_op[pg_idx],feed_dict={self.policy_grad_for_svpg_ph[pg_idx]:pg_list[pg_idx]});
      
    if ((global_t+1) - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} EPISODES in {:.0f} sec. {:.0f} EPISODES/sec. {:.2f}M EPISODES/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
    return score;

