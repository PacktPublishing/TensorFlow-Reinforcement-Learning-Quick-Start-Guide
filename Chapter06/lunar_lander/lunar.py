import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import os
import threading
import multiprocessing

from random import choice
from time import sleep
from time import time

from a3c import *
from utils import *



max_episode_steps = 1000
gamma = 0.999
s_size = 8 
a_size = 4 
load_model = False
model_path = './model'


#--------------------------------------------------------------------------------------------


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    


with tf.device("/cpu:0"): 

    # keep count of global episodes
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)

    # number of worker threads
    num_workers = multiprocessing.cpu_count() 

    # Adam optimizer
    trainer = tf.train.AdamOptimizer(learning_rate=2e-4, use_locking=True) 
   
    # global network
    master_network = AC(s_size,a_size,'global',None) 
    
    workers = []
    for i in range(num_workers):
        env = gym.make('LunarLander-v2')
        workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
 
    # tf saver
    saver = tf.train.Saver(max_to_keep=5)


with tf.Session() as sess:

    # tf coordinator for threads
    coord = tf.train.Coordinator()

    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        

    # start the worker threads
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_steps, gamma, sess, coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)


#--------------------------------------------------------------------------------------------


