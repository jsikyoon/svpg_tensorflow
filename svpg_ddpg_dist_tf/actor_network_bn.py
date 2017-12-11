import tensorflow as tf 
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64
PARAMS_NUM=6

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,state_dim,action_dim,task_index,worker_hosts_num):

		self.state_dim = state_dim
		self.action_dim = action_dim
                self.task_index = task_index
                self.worker_hosts_num = worker_hosts_num
                self.params_num = PARAMS_NUM
                # actor params list
                self.a_list=self.make_particle_list(self.worker_hosts_num,state_dim,action_dim,self.params_num);
                # policy gradient list
                self.pg_list,self.pg_list_ph,self.pg_list_ops=self.make_pg_list(self.worker_hosts_num,state_dim,action_dim,self.params_num,prefix="pg_");
		# create actor network
		self.state_input,self.action_output,self.net,self.is_training = self.create_network(state_dim,self.a_list,self.task_index)
		# create target actor network
                self.target_state_input,self.target_action_output,self.target_update,self.target_is_training = self.create_target_network(state_dim,action_dim,self.a_list,self.task_index)

		# define policy gradient
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
        
        def make_particle_list(self,worker_hosts_num,state_dim,action_dim,params_num,prefix=""):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
                #particle lists
                a_list=np.zeros((worker_hosts_num,params_num),dtype=object);
                # W1,b1,W2,b2,W3,b3
                for i in range(worker_hosts_num):
                  a_list[i,0] = self.variable([state_dim,layer1_size],state_dim)
                  a_list[i,1] = self.variable([layer1_size],state_dim)
                  a_list[i,2] = self.variable([layer1_size,layer2_size],layer1_size)
                  a_list[i,3] = self.variable([layer2_size],layer1_size)
                  a_list[i,4] = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
                  a_list[i,5] = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

                return a_list;

        def make_pg_list(self,worker_hosts_num,state_dim,action_dim,params_num,prefix=""):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
                #particle lists
                # W1,b1,W2,b2,W3,b3
                a_list=np.zeros((worker_hosts_num,params_num),dtype=object);
                a_list_ph=np.zeros((worker_hosts_num,params_num),dtype=object);
                a_list_ops=np.zeros((worker_hosts_num,params_num),dtype=object);
                for i in range(worker_hosts_num):
                  #W1
                  W1=a_list[i,0]=tf.get_variable(prefix+"W1"+"_"+str(i),[state_dim,layer1_size],initializer=tf.constant_initializer(0.0),trainable=False);
                  a_list_ph[i,0]=tf.placeholder(W1.dtype,shape=W1.get_shape());
                  a_list_ops[i,0]=W1.assign(a_list_ph[i,0]);
                  #b1
                  b1=a_list[i,1]=tf.get_variable(prefix+"b1"+"_"+str(i),[layer1_size],initializer=tf.constant_initializer(0.0),trainable=False);
                  a_list_ph[i,1]=tf.placeholder(b1.dtype,shape=b1.get_shape());
                  a_list_ops[i,1]=b1.assign(a_list_ph[i,1]);
                  #W2
                  W2=a_list[i,2]=tf.get_variable(prefix+"W2"+"_"+str(i),[layer1_size,layer2_size],initializer=tf.constant_initializer(0.0),trainable=False);
                  a_list_ph[i,2]=tf.placeholder(W2.dtype,shape=W2.get_shape());
                  a_list_ops[i,2]=W2.assign(a_list_ph[i,2]);
                  #b2
                  b2=a_list[i,3]=tf.get_variable(prefix+"b2"+"_"+str(i),[layer2_size],initializer=tf.constant_initializer(0.0),trainable=False);
                  a_list_ph[i,3]=tf.placeholder(b2.dtype,shape=b2.get_shape());
                  a_list_ops[i,3]=b2.assign(a_list_ph[i,3]);
                  #W3
                  W3=a_list[i,4]=tf.get_variable(prefix+"W3"+"_"+str(i),[layer2_size,action_dim],initializer=tf.constant_initializer(0.0),trainable=False);
                  a_list_ph[i,4]=tf.placeholder(W3.dtype,shape=W3.get_shape());
                  a_list_ops[i,4]=W3.assign(a_list_ph[i,4]);
                  #b3
                  b3=a_list[i,5]=tf.get_variable(prefix+"b3"+"_"+str(i),[action_dim],initializer=tf.constant_initializer(0.0),trainable=False);
                  a_list_ph[i,5]=tf.placeholder(b3.dtype,shape=b3.get_shape());
                  a_list_ops[i,5]=b3.assign(a_list_ph[i,5]);

                return a_list,a_list_ph, a_list_ops;
                
        def set_sess(self,sess):
                self.sess=sess;

	def create_network(self,state_dim,a_list,task_index):

		state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)

		W1 = a_list[task_index,0];
		b1 = a_list[task_index,1];
		W2 = a_list[task_index,2];
		b2 = a_list[task_index,3];
		W3 = a_list[task_index,4];
		b3 = a_list[task_index,5];

		layer0_bn = self.batch_norm_layer(state_input,training_phase=is_training,scope_bn='batch_norm_0',activation=tf.identity)
		layer1 = tf.matmul(layer0_bn,W1) + b1
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='batch_norm_1',activation=tf.nn.relu)
		layer2 = tf.matmul(layer1_bn,W2) + b2
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='batch_norm_2',activation=tf.nn.relu)

		action_output = tf.tanh(tf.matmul(layer2_bn,W3) + b3)

		return state_input,action_output,[W1,b1,W2,b2,W3,b3],is_training

        def create_target_network(self,state_dim,action_dim,a_list,task_index):
                state_input = tf.placeholder("float",[None,state_dim])
                is_training = tf.placeholder(tf.bool)
                ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
                target_update = ema.apply(np.reshape(a_list,[-1]))
                target_net = [ema.average(x) for x in np.reshape(a_list,[-1])][task_index*len(a_list[0]):(task_index+1)*len(a_list[0])];
                
                layer0_bn = self.batch_norm_layer(state_input,training_phase=is_training,scope_bn='target_batch_norm_0',activation=tf.identity)
                layer1 = tf.matmul(layer0_bn,target_net[0]) + target_net[1]
                layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='target_batch_norm_1',activation=tf.nn.relu)
                layer2 = tf.matmul(layer1_bn,target_net[2]) + target_net[3]
                layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='target_batch_norm_2',activation=tf.nn.relu)
                action_output = tf.tanh(tf.matmul(layer2_bn,target_net[4]) + target_net[5])
                return state_input,action_output,target_update,is_training

	def update_target(self):
		self.sess.run(self.target_update)
	
        def save_gradient(self,q_gradient_batch,state_batch):
		gradient=self.sess.run(self.parameters_gradients,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch,
			self.is_training: True
			})
                for i in range(len(gradient)):
                  gradient[i]=gradient[i].tolist();
                  self.sess.run(self.pg_list_ops[self.task_index,i],feed_dict={self.pg_list_ph[self.task_index,i]:gradient[i]});

	def train(self,q_gradient_batch,state_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch,
			self.is_training: True
			})

	def actions(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch,
			self.is_training: True
			})

	def action(self,state):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:[state],
			self.is_training: False
			})[0]


	def target_actions(self,state_batch):
		return self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input: state_batch,
			self.target_is_training: True
			})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
		return tf.cond(training_phase, 
		lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
		updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
		lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
		updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))

		
