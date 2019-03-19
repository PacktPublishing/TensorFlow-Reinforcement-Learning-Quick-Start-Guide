import tensorflow as tf
import numpy as np
import gym
import sys


nhidden1 = 400
nhidden2 = 300
nhidden3 = 300


xavier = tf.contrib.layers.xavier_initializer()
bias_const = tf.constant_initializer(0.05)
rand_unif = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)



class PPO(object):

    def __init__(self, sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, METHOD):
        self.sess = sess
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.A_UPDATE_STEPS = A_UPDATE_STEPS
        self.C_UPDATE_STEPS = C_UPDATE_STEPS
        self.METHOD = METHOD

        # tf placeholders
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.tflam = tf.placeholder(tf.float32, None, 'lambda')


        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, nhidden1, activation=None, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l1 = tf.nn.relu(l1)
            l2 = tf.layers.dense(l1, nhidden2, activation=None, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l2 = tf.nn.relu(l2)
            l3 = tf.layers.dense(l2, nhidden3, activation=None, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer) 
            l3 = tf.nn.relu(l3)

            self.v = tf.layers.dense(l3, 1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)            
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        # actor
        self.pi, self.pi_params = self._build_anet('pi', trainable=True)
        self.oldpi, self.oldpi_params = self._build_anet('oldpi', trainable=False)

        self.pi_mean = self.pi.mean()
        self.pi_sigma = self.pi.stddev()

 
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.oldpi_params)]
        
        with tf.variable_scope('loss'):
            self.ratio = tf.exp(self.pi.log_prob(self.tfa) - self.oldpi.log_prob(self.tfa))

            self.clipped_ratio = tf.clip_by_value(self.ratio, 1.-self.METHOD['epsilon'], 1.+self.METHOD['epsilon'])
            self.aloss = -tf.reduce_mean(tf.minimum(self.ratio*self.tfadv, self.clipped_ratio*self.tfadv))

            # entropy loss
            entropy = -tf.reduce_sum(self.pi.prob(self.tfa) * tf.log(tf.clip_by_value(self.pi.prob(self.tfa),1e-10,1.0)),axis=1)
            entropy = tf.reduce_mean(entropy,axis=0)      
            self.aloss -= 0.001 * entropy


        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)        

   
    def screen_out(self, s, a, r):

        print("ratio: ", self.sess.run(self.ratio, {self.tfs: s, self.tfa: a}))
        print("clipped_ratio: ", self.sess.run(self.clipped_ratio, {self.tfs: s, self.tfa: a}))

        print("mu: ", self.sess.run(self.pi_mean, {self.tfs: s, self.tfa: a}))
        print("sigma: ", self.sess.run(self.pi_sigma, {self.tfs: s, self.tfa: a}))
        
        print("sample action: ", self.sess.run(self.sample_op, {self.tfs: s}))

     

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
            l3 = tf.layers.dense(l2, nhidden3, activation=None, trainable=trainable, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            l3 = tf.nn.relu(l3)

            mu_st = tf.layers.dense(l3, 1, activation=tf.nn.tanh, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            mu_acc = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const) 
            mu_br = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)

            small = tf.constant(1e-6)
            mu_st = tf.clip_by_value(mu_st,-1.0+small,1.0-small)
            mu_acc = tf.clip_by_value(mu_acc,0.0+small,1.0-small) 
            mu_br = tf.clip_by_value(mu_br,0.0+small,1.0-small)
            mu_br = tf.scalar_mul(0.1,mu_br) # scalar mult
            mu = tf.concat([mu_st, mu_acc, mu_br], axis=1)          

            sigma_st = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            sigma_acc = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            sigma_br = tf.layers.dense(l3, 1, activation=tf.nn.sigmoid, trainable=trainable, kernel_initializer=rand_unif, bias_initializer=bias_const)
            sigma_st = tf.scalar_mul(0.2,sigma_st) # scalar mult            
            sigma_acc = tf.scalar_mul(0.2,sigma_acc) # scalar mult 
            sigma_br = tf.scalar_mul(0.05,sigma_br) # scalar mult 
            sigma = tf.concat([sigma_st, sigma_acc, sigma_br], axis=1)          
            sigma = tf.clip_by_value(sigma,0.0+small,1.0-small)

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
