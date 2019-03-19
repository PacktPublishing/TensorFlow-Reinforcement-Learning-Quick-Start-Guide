import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import sys

from replay_buffer import ReplayBuffer



winit = tf.contrib.layers.xavier_initializer()
binit = tf.constant_initializer(0.01)
rand_unif = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)



class ActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # actor 
        self.state, self.out, self.scaled_out = self.create_actor_network(scope='actor')

        # actor params
        self.network_params = tf.trainable_variables()

        # target network
        self.target_state, self.target_out, self.target_scaled_out = self.create_actor_network(scope='act_target')
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # update target using tau and 1-tau as weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                            tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # gradient (this is provided by the critic)
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # actor gradients
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # adam optimization 
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        # num trainable vars
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)


      

    def create_actor_network(self, scope):
      with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        state = tf.placeholder(name='a_states', dtype=tf.float32, shape=[None, self.s_dim])
  
        net = tf.layers.dense(inputs=state, units=400, activation=None, kernel_initializer=winit, bias_initializer=binit, name='anet1') 
        net = tf.nn.relu(net)

        net = tf.layers.dense(inputs=net, units=300, activation=None, kernel_initializer=winit, bias_initializer=binit, name='anet2')
        net = tf.nn.relu(net)

        out = tf.layers.dense(inputs=net, units=self.a_dim, activation=None, kernel_initializer=rand_unif, bias_initializer=binit, name='anet_out')     
        out = tf.nn.tanh(out)
        scaled_out = tf.multiply(out, self.action_bound)
        return state, out, scaled_out

      
        

    def train(self, state, a_gradient):
        self.sess.run(self.optimize, feed_dict={self.state: state, self.action_gradient: a_gradient})

    def predict(self, state):
        return self.sess.run(self.scaled_out, feed_dict={
            self.state: state})

    def predict_target(self, state):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_state: state})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


#-----------------------------------------------------------------------------------

class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # critic
        self.state, self.action, self.out = self.create_critic_network(scope='critic')

        # critic params  
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # target Network
        self.target_state, self.target_action, self.target_out = self.create_critic_network(scope='crit_target')

        # target network params 
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # update target using tau and 1 - tau as weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # network target (y_i in the paper)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # adam optimization; minimize L2 loss function
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # gradient of Q w.r.t. action
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
           state = tf.placeholder(name='c_states', dtype=tf.float32, shape=[None, self.s_dim])  
           action = tf.placeholder(name='c_action', dtype=tf.float32, shape=[None, self.a_dim]) 

           net = tf.concat([state, action],1) 

           net = tf.layers.dense(inputs=net, units=400, activation=None, kernel_initializer=winit, bias_initializer=binit, name='cnet1') 
           net = tf.nn.relu(net)

           net = tf.layers.dense(inputs=net, units=300, activation=None, kernel_initializer=winit, bias_initializer=binit, name='cnet2') 
           net = tf.nn.relu(net)

           out = tf.layers.dense(inputs=net, units=1, activation=None, kernel_initializer=rand_unif, bias_initializer=binit, name='cnet_out')     
           return state, action, out


    def train(self, state, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={self.state: state, self.action: action, self.predicted_q_value: predicted_q_value})

    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={self.state: state, self.action: action})

    def predict_target(self, state, action):
        return self.sess.run(self.target_out, feed_dict={self.target_state: state, self.target_action: action})

    def action_gradients(self, state, actions):
        return self.sess.run(self.action_grads, feed_dict={self.state: state, self.action: actions})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

