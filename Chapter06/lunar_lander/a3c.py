import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import threading
import multiprocessing

from random import choice
from time import sleep
from time import time
from threading import Lock

from utils import *


xavier = tf.contrib.layers.xavier_initializer()
bias_const = tf.constant_initializer(0.05)
rand_unif = tf.keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3)
regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)



def reward_shaping(r, s, s1):
     # check if y-coord < 0; implies lander crashed
     if (s1[1] < 0.0):
       print('-----lander crashed!----- ')
       d = True  
       r -= 1.0

     # check if lander is stuck
     xx = s[0] - s1[0]
     yy = s[1] - s1[1]
     dist = np.sqrt(xx*xx + yy*yy)  
     if (dist < 1.0e-4):
       print('-----lander stuck!----- ')
       d = True 
       r -= 0.5
     return r, d




class AC():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
           
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
           
            # 2 FC layers  
            net = tf.layers.dense(self.inputs, 256, activation=tf.nn.elu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
            net = tf.layers.dense(net, 128, activation=tf.nn.elu, kernel_initializer=xavier, bias_initializer=bias_const, kernel_regularizer=regularizer)
                          
            # policy
            self.policy = tf.layers.dense(net, a_size, activation=tf.nn.softmax, kernel_initializer=xavier, bias_initializer=bias_const)

            # value
            self.value = tf.layers.dense(net, 1, activation=None, kernel_initializer=rand_unif, bias_initializer=bias_const)
            
            # only workers need ops for loss functions and gradient updating
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.policy_times_a = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # loss 
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1.0e-8))
                self.policy_loss = -tf.reduce_sum(tf.log(self.policy_times_a + 1.0e-8) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.005

                # get gradients from local networks using local losses; clip them to avoid exploding gradients
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                # apply local gradients to global network using tf.apply_gradients()
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))



#-----------------------------------------------------------------------------------------

class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)

        # local copy of the AC network 
        self.local_AC = AC(s_size,a_size,self.name,trainer)

        # tensorflow op to copy global params to local network
        self.update_local_ops = update_target_graph('global',self.name)        
        
        self.actions = np.identity(a_size,dtype=bool).tolist()
        self.env = env
        
    # train function
    def train(self,experience,sess,gamma,bootstrap_value):
        experience = np.array(experience)
        observations = experience[:,0]
        actions = experience[:,1]
        rewards = experience[:,2]
        next_observations = experience[:,3]
        values = experience[:,5]
        
        # discounted rewards
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]

        # value  
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])

        # advantage function 
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # lock for updating global params
        lock = Lock()
        lock.acquire() 

        # update global network params by calling apply_grads
        feed_dict = {self.local_AC.target_v:discounted_rewards, self.local_AC.inputs:np.vstack(observations), 
                     self.local_AC.actions:actions, self.local_AC.advantages:advantages}
        value_loss, policy_loss, entropy, _, _, _ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss, self.local_AC.entropy, self.local_AC.grad_norms, 
            self.local_AC.var_norms, self.local_AC.apply_grads], feed_dict=feed_dict)
 
        # release lock
        lock.release() 

        return value_loss / len(experience), policy_loss / len(experience), entropy / len(experience)
        
    # worker's work function
    def work(self,max_episode_steps, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
          
                # copy global params to local network   
                sess.run(self.update_local_ops)

                # lists for book keeping
                episode_buffer = []
                episode_values = []
                episode_frames = []

                episode_reward = 0
                episode_step_count = 0
                d = False
                
                s = self.env.reset()
                episode_frames.append(s)

                
                while not d:
            
                    # action and value
                    a_dist, v = sess.run([self.local_AC.policy,self.local_AC.value], feed_dict={self.local_AC.inputs:[s]})
                    a = np.random.choice(np.arange(len(a_dist[0])), p=a_dist[0])


                    if (self.name == 'worker_0'):
                       self.env.render()
                      
                    # step
                    s1, r, d, info = self.env.step(a)
                      
                    # normalize reward
                    r = r/100.0

                    # reward shaping for lunar lander
                    r, d = reward_shaping(r, s, s1) 

                    if d == False:
                        episode_frames.append(s1)
                    else:
                        s1 = s
                        
                    # collect experience in buffer 
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])

                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # if buffer has 25 entries, time for an update 
                    if len(episode_buffer) == 25 and d != True and episode_step_count != max_episode_steps - 1:
                        v1 = sess.run(self.local_AC.value, feed_dict={self.local_AC.inputs:[s]})[0,0]
                        value_loss, policy_loss, entropy = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
 
                    # idiot check to ensure we did not miss update for some unforseen reason 
                    if (len(episode_buffer) > 30):
                        print(self.name, "buffer full ", len(episode_buffer))
                        sys.exit()

                    if d == True:
                        break

                
                print("episode: ", episode_count, "| worker: ", self.name, "| episode reward: ", episode_reward, "| step count: ", episode_step_count)
                

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    value_loss, policy_loss, entropy = self.train(episode_buffer,sess,gamma,0.0)
                                
                
                print("loss: ", self.name, value_loss, policy_loss, entropy)

                # write to file for worker_0
                if (self.name == 'worker_0'):  
                   with open("performance.txt", "a") as myfile:
                      myfile.write(str(episode_count) + " " + str(episode_reward) + " " + str(episode_step_count) + "\n")

                # save model params for worker_0
                if (episode_count % 25 == 0 and self.name == 'worker_0' and episode_count != 0):
                    saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                    print ("Saved Model")
                    
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
