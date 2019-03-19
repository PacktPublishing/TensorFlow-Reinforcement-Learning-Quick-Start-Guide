import numpy as np
import sys
import os
import random
import tensorflow as tf


NET = 'bigger'  # 'smaller'

LOSS = 'huber' # 'L2'


winit = tf.variance_scaling_initializer(scale=2) # tf.contrib.layers.xavier_initializer()

#--------------------------------------------------------------------------------------------------


class QNetwork():
    def __init__(self, scope="QNet", VALID_ACTIONS=[0, 1, 2, 3]):
        self.scope = scope
        self.VALID_ACTIONS = VALID_ACTIONS
        with tf.variable_scope(scope):
            self._build_model()
            
    def _build_model(self):
        # input placeholders; input is 4 frames of shape 84x84 
        self.tf_X = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # TD
        self.tf_y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # action
        self.tf_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # normalize input
        X = tf.to_float(self.tf_X) / 255.0
        batch_size = tf.shape(self.tf_X)[0]

#-------------
        
        if (NET == 'bigger'):
 
           # bigger net

           # 3 conv layers
           conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)
           conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)
           conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)

           # fully connected layers
           flattened = tf.contrib.layers.flatten(conv3)
           fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu, weights_initializer=winit)


        elif (NET == 'smaller'): 
 
           # smaller net
   
           # 2 conv layers
           conv1 = tf.contrib.layers.conv2d(X, 16, 8, 4, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)
           conv2 = tf.contrib.layers.conv2d(conv1, 32, 4, 2, padding='VALID', activation_fn=tf.nn.relu, weights_initializer=winit)

           # fully connected layers
           flattened = tf.contrib.layers.flatten(conv2)
           fc1 = tf.contrib.layers.fully_connected(flattened, 256, activation_fn=tf.nn.relu, weights_initializer=winit)  
#-------------         

       

        # Q(s,a)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(self.VALID_ACTIONS), activation_fn=None, weights_initializer=winit)


        action_one_hot = tf.one_hot(self.tf_actions, tf.shape(self.predictions)[1], 1.0, 0.0, name='action_one_hot')
        self.action_predictions = tf.reduce_sum(self.predictions * action_one_hot, reduction_indices=1, name='act_pred')
 
        if (LOSS == 'L2'):
           # L2 loss
           self.loss = tf.reduce_mean(tf.squared_difference(self.tf_y, self.action_predictions), name='loss')
        elif (LOSS == 'huber'):
           # Huber loss
           self.loss = tf.reduce_mean(huber_loss(self.tf_y-self.action_predictions), name='loss')
        

        # optimizer 
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=2e-5)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
        
    def predict(self, sess, s):
        return sess.run(self.predictions, { self.tf_X: s})

    def update(self, sess, s, a, y):
        feed_dict = { self.tf_X: s, self.tf_y: y, self.tf_actions: a }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss




# huber loss
def huber_loss(x):
  condition = tf.abs(x) < 1.0
  output1 = 0.5 * tf.square(x)
  output2 = tf.abs(x) - 0.5
  return tf.where(condition, output1, output2)

