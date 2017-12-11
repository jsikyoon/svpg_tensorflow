import tensorflow as tf
import numpy as np
import tf_utils

# hyper parameters
ALPHA=1.
STEP_SIZE=1e-4
LAYER1_SIZE = 400
LAYER2_SIZE = 300

class SVPG:
  def __init__(self,p_list,l_list,state_dim,action_dim,independent_flag=0):
    self.alpha=ALPHA;
    self.step_size=STEP_SIZE;
    self.n_particles=len(p_list);
    self.params_num=len(p_list[0]);
    self.state_dim=state_dim;
    self.action_dim=action_dim;
    self.independent_flag=independent_flag;
    
    # make svgd
    self.svgd_set(p_list,l_list);

  def svgd_run(self,sess):
    sess.run(self.optimizer);

  def svgd_set(self,p_list,l_list):
    layer1_size=LAYER1_SIZE;
    layer2_size=LAYER2_SIZE;
    p_flat_list=self.make_flat(p_list);
    l_flat_list=self.make_flat(l_list);
    # gradients
    if(self.n_particles==1):
      grad=(1/self.alpha)*l_flat_list[0];
    else:
      kernel_mat,grad_kernel=self.kernel(p_flat_list);
      # independently learning or not
      if(self.independent_flag!=0):
        # delta prior is assumed as 1.0
        grad=(tf.matmul(kernel_mat,((1/self.alpha)*l_flat_list+1.0))+grad_kernel)/(self.n_particles);
      else:
        # when independently learning, each particle is just learned as topology of original DDPG
        grad=((1/self.alpha)*l_flat_list)/(self.n_particles);

    # reshape gradient
    if(self.n_particles>1):
      grad=tf.unstack(grad,axis=0);
    else:
      grad=[grad];
    grad_list=np.zeros((self.n_particles,self.params_num),dtype=object);
    for i in range(self.n_particles):
      # W1
      st_idx=0;length=self.state_dim*layer1_size;
      grad_list[i,0]=tf.reshape(tf.slice(grad[i],[st_idx],[length]),[self.state_dim,layer1_size]);
      # b1
      st_idx+=length;length=layer1_size;
      grad_list[i,1]=tf.slice(grad[i],[st_idx],[length]);
      # W2
      st_idx+=length;length=layer1_size*layer2_size;
      grad_list[i,2]=tf.reshape(tf.slice(grad[i],[st_idx],[length]),[layer1_size,layer2_size]);
      # b2
      st_idx+=length;length=layer2_size;
      grad_list[i,3]=tf.slice(grad[i],[st_idx],[length]);
      # W3
      st_idx+=length;length=layer2_size*self.action_dim;
      grad_list[i,4]=tf.reshape(tf.slice(grad[i],[st_idx],[length]),[layer2_size,self.action_dim]);
      # b3
      st_idx+=length;length=self.action_dim;
      grad_list[i,5]=tf.slice(grad[i],[st_idx],[length]);

    # optimizer
    grad_list=list(np.reshape(grad_list,[-1]));
    p_list=list(np.reshape(p_list,[-1]));
    self.optimizer=tf.train.AdamOptimizer(self.step_size).apply_gradients(zip(grad_list,p_list));
    
  def make_flat(self,p_list):
    p_list2=np.zeros((len(p_list),len(p_list[0])),dtype=object);
    for i in range(len(p_list)):
      for j in range(len(p_list[0])):
        p_list2[i,j]=tf.reshape(p_list[i,j],[-1]);
    p_flat_list=[];
    for i in range(len(p_list2)):
      p_flat_list.append(tf.concat(list(p_list2[i]),axis=0));
    return tf.stack(p_flat_list,axis=0);

  def update(self,sess,v_list,v_list_ph,v_list_ops,v_flat_size,v_size):
    for i in range(self.n_particles):
      for j in range(len(v_flat_size)):
        if(j==0):
          sess.run(v_list_ops[i,j],feed_dict={v_list_ph[i,j]:np.reshape(v_list[i,:v_flat_size[j]],v_size[j])});
        else:
          sess.run(v_list_ops[i,j],feed_dict={v_list_ph[i,j]:np.reshape(v_list[i,v_flat_size[j-1]:(v_flat_size[j-1]+v_flat_size[j])],v_size[j])});

  def kernel(self, particle_tensor):
    # kernel
    h = -1
    euclidean_dists = tf_utils.pdist(particle_tensor)
    pairwise_dists = tf_utils.squareform(euclidean_dists) ** 2
    # kernel trick
    h = tf.sqrt(0.5 * tf_utils.median(pairwise_dists) / tf.log(self.n_particles + 1.))
    kernel_matrix = tf.exp(-pairwise_dists / h ** 2 / 2)
    kernel_sum = tf.reduce_sum(kernel_matrix, axis=1)
    grad_kernel = tf.add(-tf.matmul(kernel_matrix, particle_tensor),tf.multiply(particle_tensor, tf.expand_dims(kernel_sum, axis=1))) / (h ** 2)
    return kernel_matrix, grad_kernel
