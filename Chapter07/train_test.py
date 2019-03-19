import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import time

from class_ppo import *

#------------------------------------------------------------------------------------

def reward_shaping(s_):

     r = 0.0

     if s_[0] > -0.4:
          r += 5.0*(s_[0] + 0.4)
     if s_[0] > 0.1: 
          r += 100.0*s_[0]
     if s_[0] < -0.7:
          r += 5.0*(-0.7 - s_[0])
     if s_[0] < 0.3 and np.abs(s_[1]) > 0.02:
          r += 4000.0*(np.abs(s_[1]) - 0.02)

     return r


#----------------------------------------------------------------------------------------

env = gym.make('MountainCarContinuous-v0')


EP_MAX = 1000
GAMMA = 0.9

A_LR = 2e-4
C_LR = 2e-4

BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]

print("S_DIM: ", S_DIM, "| A_DIM: ", A_DIM)

CLIP_METHOD = dict(name='clip', epsilon=0.1)

# train_test = 0 for train; =1 for test
train_test = 0

# irestart = 0 for fresh restart; =1 for restart from ckpt file
irestart = 0

iter_num = 0

if (irestart == 0):
  iter_num = 0

#----------------------------------------------------------------------------------------

sess = tf.Session()

ppo = PPO(sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, CLIP_METHOD)

saver = tf.train.Saver()


if (train_test == 0 and irestart == 0):
  sess.run(tf.global_variables_initializer())
else:
  saver.restore(sess, "ckpt/model")  

#----------------------------------------------------------------------------------------

for ep in range(iter_num, EP_MAX):

    print("-"*70)
   
    s = env.reset()

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0

    max_pos = -1.0
    max_speed = 0.0
    done = False
    t = 0

    while not done:    
       
        env.render()

        # sticky actions
        #if (t == 0 or np.random.uniform() < 0.125): 
        if (t % 8 ==0):
          a = ppo.choose_action(s) 

        # small noise for exploration
        a += 0.1 * np.random.randn()  

        # clip
        a = np.clip(a, -1.0, 1.0)

        # take step  
        s_, r, done, _ = env.step(a)
       
        if s_[0] > 0.4:
            print("nearing flag: ", s_, a) 

        if s_[0] > 0.45:
          print("reached flag on mountain! ", s_, a) 
          if done == False:
             print("something wrong! ", s_, done, r, a)
             sys.exit()   

        # reward shaping 
        if train_test == 0:
          r += reward_shaping(s_)

        if s_[0] > max_pos:
           max_pos = s_[0]
        if s_[1] > max_speed:
           max_speed = s_[1]   


        if (train_test == 0):
          buffer_s.append(s)
          buffer_a.append(a)
          buffer_r.append(r)    

        s = s_
        ep_r += r
        t += 1

        if (train_test == 0):
          if (t+1) % BATCH == 0 or done == True:
              v_s_ = ppo.get_v(s_)
              discounted_r = []
              for r in buffer_r[::-1]:
                  v_s_ = r + GAMMA * v_s_
                  discounted_r.append(v_s_)
              discounted_r.reverse()

              bs = np.array(np.vstack(buffer_s))
              ba = np.array(np.vstack(buffer_a))  
              br = np.array(discounted_r)[:, np.newaxis]

              buffer_s, buffer_a, buffer_r = [], [], []
             
              ppo.update(bs, ba, br)

        if (train_test == 1):
              time.sleep(0.1)

        if (done  == True):
             print("values at done: ", s_, a)
             break

    print("episode: ", ep, "| episode reward: ", round(ep_r,4), "| time steps: ", t)
    print("max_pos: ", max_pos, "| max_speed:", max_speed)

    if (train_test == 0):
      with open("performance.txt", "a") as myfile:
        myfile.write(str(ep) + " " + str(round(ep_r,4)) + " " + str(round(max_pos,4)) + " " + str(round(max_speed,4)) + "\n")

    if (train_test == 0 and ep%10 == 0):
      saver.save(sess, "ckpt/model")




