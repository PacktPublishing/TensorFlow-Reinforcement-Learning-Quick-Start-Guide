#---------------------------------
# AUTHOR: KAUSHIK BALAKRISHNAN
#---------------------------------

import tensorflow as tf
import numpy as np
import argparse
import pprint as pp
import sys

from replay_buffer import ReplayBuffer
from AandC import *
from noise import *
from gym_torcs import TorcsEnv


mu = np.array([0.0, 0.5, 0.01])
theta = np.array([0.0, 0.0, 0.0])
sigma = np.array([0.1, 0.1, 0.1]) 


# irestart = 0 for fresh start; = 1 for restart from restart_step
irestart = 0
restart_step = 0

if (irestart == 0):
  restart_step = 0

#---------------------------------------------------------------------


def trainDDPG(sess, args, actor, critic):

    saver = tf.train.Saver()

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    if (irestart == 0):
      sess.run(tf.global_variables_initializer())
    else:
      saver.restore(sess, "ckpt/model")  


    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    episode_count = args['episode_count']
    max_steps = args['max_steps']

    epsilon = 1.0
 
   
    for i in range(restart_step, episode_count):
        
        if np.mod(i, 100) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every N episodes due to a memory leak error
        else:
            ob = env.reset()


        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

      
        ep_reward = 0
        ep_ave_max_q = 0


        msteps = max_steps
        if (i < 100):
            msteps = 100
        elif (i >=100 and i < 200):
            msteps = 100 + (i-100)*9
        else:  
            msteps = 1000 + (i-200)*5
        msteps = min(msteps, max_steps)


        for j in range(msteps):

       
            # action noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))   
            a[0,:] += OU(x=a[0,:], mu=mu, sigma=sigma, theta=theta)*max(epsilon,0.0) 

           
            # first few episodes step on gas! 
            if (i < 10):
               a[0][0] = 0.0
               a[0][1] = 1.0
               a[0][2] = 0.0


            print("episode: ", i, "step: ", j, "action: ", a)
 
            ob, r, terminal, info = env.step(a[0])
            s2 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            # ob.track is 19 dimensional; ob.wheelSpinVel is 4 dimensional



            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            
            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                
                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                
                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))
                
                ep_ave_max_q += np.amax(predicted_q_value)

               
                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

              
                # Update target networks
                actor.update_target_network()
                critic.update_target_network()
 
                
            s = s2
            ep_reward += r

            if terminal:

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                
                with open("analysis_file.txt", "a") as myfile:
                   myfile.write(str(i) + " " + str(j) + " " + str(ep_reward) + " " + str(ep_ave_max_q / float(j))  + "\n")
                break


        if (np.mod(i,100) == 0 and i > 1):    
         saver.save(sess, "ckpt/model")
         print("saved model after ", i, " episodes ")


#-----------------------------------------------------------------------------

def testDDPG(sess, args, actor, critic):

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)    


    episode_count = args['episode_count']
    max_steps = args['max_steps']


    for i in range(episode_count):

        if np.mod(i, 100) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every N episode because of the memory leak error
        else:
            ob = env.reset()


        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        ep_reward = 0
        ep_ave_max_q = 0

        
        for j in range(max_steps):

            
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) 
            # NOISE AT TEST TIME MAY BE REQUIRED TO STABILIZE ACTIONS
            a[0,:] += OU(x=a[0,:], mu=mu, sigma=sigma, theta=theta)

            ob, r, terminal, info = env.step(a[0])
            s2 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))


            s = s2
            ep_reward += r

            if terminal:

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                break
