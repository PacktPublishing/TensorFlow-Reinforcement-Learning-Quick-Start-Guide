import tensorflow as tf
import numpy as np
import gym
import sys


nhidden1 = 64 # 400
nhidden2 = 64 #300


xavier = tf.contrib.layers.xavier_initializer()
bias_const = tf.constant_initializer(0.05)
rand_unif = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)



class PPO(object):

    def __init__(self, sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, CLIP_METHOD):
        self.sess = sess
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.A_UPDATE_STEPS = A_UPDATE_STEPS
        self.C_UPDATE_STEPS = C_UPDATE_STEPS
        self.CLIP_METHOD = CLIP_METHOD

        # tf placeholders
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')


        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, nhidden1, activation=None, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l1 = tf.nn.relu(l1)
            l2 = tf.layers.dense(l1, nhidden2, activation=None, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l2 = tf.nn.relu(l2)
            
            self.v = tf.layers.dense(l2, 1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)            
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        # actor
        self.pi, self.pi_params = self._build_anet('pi', trainable=True)
        self.oldpi, self.oldpi_params = self._build_anet('oldpi', trainable=False)

        self.pi_mean = self.pi.mean()
        self.pi_sigma = self.pi.stddev()

 
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)  
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.oldpi_params)]
        
        with tf.variable_scope('loss'):
            self.ratio = tf.exp(self.pi.log_prob(self.tfa) - self.oldpi.log_prob(self.tfa))
            self.clipped_ratio = tf.clip_by_value(self.ratio, 1.-self.CLIP_METHOD['epsilon'], 1.+self.CLIP_METHOD['epsilon'])
            self.aloss = -tf.reduce_mean(tf.minimum(self.ratio*self.tfadv, self.clipped_ratio*self.tfadv))

            # entropy 
            entropy = -tf.reduce_sum(self.pi.prob(self.tfa) * tf.log(tf.clip_by_value(self.pi.prob(self.tfa),1e-10,1.0)),axis=1)
            entropy = tf.reduce_mean(entropy,axis=0)      
            self.aloss -= 0.0 #0.01 * entropy


        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)        
     

    def update(self, s, a, r):
      
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
 

        # update actor
        for _ in range(self.A_UPDATE_STEPS):
            self.sess.run(self.atrain_op, feed_dict={self.tfs: s, self.tfa: a, self.tfadv: adv})
                 
        # update critic
        for _ in range(self.C_UPDATE_STEPS):
           self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) 
     
    
    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, nhidden1, activation=None, trainable=trainable, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l1 = tf.nn.relu(l1)
            l2 = tf.layers.dense(l1, nhidden2, activation=None, trainable=trainable, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l2 = tf.nn.relu(l2)
            
            mu = tf.layers.dense(l2, self.A_DIM, activation=tf.nn.tanh, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)

            small = tf.constant(1e-6)
            mu = tf.clip_by_value(mu,-1.0+small,1.0-small)        

            sigma = tf.layers.dense(l2, self.A_DIM, activation=None, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            sigma = tf.nn.softplus(sigma) + 0.1 

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})
        return a[0]

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        vv = self.sess.run(self.v, {self.tfs: s})
        return vv[0,0]
