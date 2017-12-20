import tensorflow as tf
import numpy as np
import tf_utils

# hyper parameters
ALPHA=1.
STEP_SIZE=1e-4

class SVPG:
  def __init__(self,sess,net_list,grad_list,independent_flag=0):
    self.alpha=ALPHA;
    self.step_size=STEP_SIZE;
    self.n_particles=len(net_list);
    self.independent_flag=independent_flag;
    self.sess=sess;

    """
    # get valid ones
    # NOTE: every particle need to have same shape
    net_list2=np.zeros(self.n_particles,dtype=object);
    grad_list2=np.zeros(self.n_particles,dtype=object);
    for par_idx in range(self.n_particles):
      net_plist=[];grad_plist=[];
      for net, grad in zip(net_list[par_idx],grad_list[par_idx]):
        if not grad == None:
          net_plist.append(net);grad_plist.append(grad);
      net_list2[par_idx]=net_plist;
      grad_list2[par_idx]=grad_plist;
    """
    net_list2=net_list;
    grad_list2=grad_list;
    # get the number of params per each particle
    self.params_num=len(net_list2[0]);

    # make svgd
    self.svgd_set(net_list2,grad_list2);
    
  def run(self):
    self.sess.run(self.optimizer);

  def svgd_set(self,net_list2,grad_list2):
    p_flat_list=self.make_flat(net_list2);
    l_flat_list=self.make_flat(grad_list2);
    
    # gradients
    if(self.n_particles==1):
      grad=(1/self.alpha)*l_flat_list[0];
    else:
      kernel_mat,grad_kernel=self.kernel(p_flat_list);
      # independently learning or not
      if(self.independent_flag!=1):
        # delta prior is assumed as 1.0
        grad=(tf.matmul(kernel_mat,((1/self.alpha)*l_flat_list))-grad_kernel)/(self.n_particles);
      else:
        # when independently learning, each particle is just learned as topology of original DDPG
        grad=l_flat_list;

    # get original shape (2 is for flat version)
    origin_shape=np.zeros(self.params_num,dtype=object);
    origin_shape2=np.zeros(self.params_num,dtype=object);
    for i in range(self.params_num):
      params_shape=grad_list2[0][i].get_shape().as_list();
      total_len=1;
      for j in params_shape:
        total_len*=j;
      origin_shape[i]=params_shape;
      origin_shape2[i]=total_len;
    
    # reshape gradient
    if(self.n_particles>1):
      grad=tf.unstack(grad,axis=0);
    else:
      grad=[grad];
    grad_list3=np.zeros((self.n_particles,self.params_num),dtype=object);
    for i in range(self.n_particles):
      st_idx=0;length=0;
      for j in range(self.params_num):
        st_idx+=length;length=origin_shape2[j];
        grad_list3[i,j]=tf.reshape(tf.slice(grad[i],[st_idx],[length]),origin_shape[j]);

    # optimizer
    grad_list3=list(np.reshape(grad_list3,[-1]));
    net_list2=list(np.reshape(net_list2.tolist(),[-1]));
    
    self.optimizer=tf.train.AdamOptimizer(self.step_size).apply_gradients(zip(grad_list3,net_list2));
    
  def make_flat(self,p_list):
    p_list2=np.zeros((len(p_list),len(p_list[0])),dtype=object);
    for i in range(len(p_list)):
      for j in range(len(p_list[0])):
        p_list2[i,j]=tf.reshape(p_list[i][j],[-1]);
    p_flat_list=[];
    for i in range(len(p_list2)):
      p_flat_list.append(tf.concat(list(p_list2[i]),axis=0));
    return tf.stack(p_flat_list,axis=0);

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
