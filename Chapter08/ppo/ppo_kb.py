import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from class_ppo import *
from gym_torcs import TorcsEnv

#----------------------------------------------------------------------------------------

EP_MAX = 2000
EP_LEN = 1000
GAMMA = 0.95


A_LR = 1e-4
C_LR = 1e-4

BATCH = 128 
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 29, 3
METHOD = dict(name='clip', epsilon=0.1)

# train_test = 0 for train; =1 for test
train_test = 0

# irestart = 0 for fresh restart; =1 for restart from ckpt file
irestart = 0

iter_num = 0

if (irestart == 0):
  iter_num = 0

#----------------------------------------------------------------------------------------

sess = tf.Session()

ppo = PPO(sess, S_DIM, A_DIM, A_LR, C_LR, A_UPDATE_STEPS, C_UPDATE_STEPS, METHOD)

saver = tf.train.Saver()

env = TorcsEnv(vision=False, throttle=True, gear_change=False)

#----------------------------------------------------------------------------------------


if (train_test == 0 and irestart == 0):
  sess.run(tf.global_variables_initializer())
else:
  saver.restore(sess, "ckpt/model")  


for ep in range(iter_num, EP_MAX):

    print("-"*50)
    print("episode: ", ep)

    if np.mod(ep, 100) == 0:
        ob = env.reset(relaunch=True)   #relaunch TORCS every N episode because of the memory leak error
    else:
        ob = env.reset()

    s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0

    for t in range(EP_LEN):    # in one episode
       
        a = ppo.choose_action(s)

        
        a[0] = np.clip(a[0],-1.0,1.0)
        a[1] = np.clip(a[1],0.0,1.0)
        a[2] = np.clip(a[2],0.0,1.0)  

        #print("a: ", a)


        ob, r, done, _ = env.step(a)
        s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))  


        if (train_test == 0):
          buffer_s.append(s)
          buffer_a.append(a)
          buffer_r.append(r)    

        s = s_
        ep_r += r

        if (train_test == 0):
          # update ppo
          if (t+1) % BATCH == 0 or t == EP_LEN-1 or done == True:
          #if t == EP_LEN-1 or done == True:
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

              print("ppo update")              
              ppo.update(bs, ba, br)
             
              #print("screen out: ")
              #ppo.screen_out(bs, ba, br)
              #print("-"*50)


        if (done  == True):
             break

    print('Ep: %i' % ep,"|Ep_r: %i" % ep_r,("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',)

    if (train_test == 0):
      with open("performance.txt", "a") as myfile:
        myfile.write(str(ep) + " " + str(t) + " " + str(round(ep_r,4)) + "\n")

    if (train_test == 0 and ep%25 == 0):
      saver.save(sess, "ckpt/model")

