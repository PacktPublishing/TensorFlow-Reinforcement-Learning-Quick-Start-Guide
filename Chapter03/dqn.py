import gym
import itertools
import numpy as np
import os
import random
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque, namedtuple

from model import *
from funcs import *


#----------------------------------------------------------------------------------

GAME = "BreakoutDeterministic-v4" # "BreakoutDeterministic-v0"

# Atari Breakout actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) 
VALID_ACTIONS = [0, 1, 2, 3]


#----------------------------------------------------------------------------------

# set parameters for running

train_or_test = 'train' #'test' #'train'
train_from_scratch = True
start_iter = 0
start_episode = 0
epsilon_start = 1.0

#----------------------------------------------------------------------------------

env = gym.envs.make(GAME)

print("Action space size: {}".format(env.action_space.n))
observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

plt.figure()
plt.imsave("Atari_Breakout1.png", env.render(mode='rgb_array'))

env.step(2)
env.step(1)

plt.figure()
plt.imsave("Atari_Breakout2.png", env.render(mode='rgb_array'))
env.close() 

env.step(2)
env.step(0)

plt.figure()
plt.imsave("Atari_Breakout3.png", env.render(mode='rgb_array'))
env.close() 


#----------------------------------------------------------------------------------


# experiment dir
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))


# create ckpt directory    
checkpoint_dir = os.path.join(experiment_dir, "ckpt")
checkpoint_path = os.path.join(checkpoint_dir, "model")
    
if not os.path.exists(checkpoint_dir):
   os.makedirs(checkpoint_dir)

#----------------------------------------------------------------------------------





def deep_q_learning(sess, env, q_net, target_net, state_processor, num_episodes, train_or_test='train', train_from_scratch=True,
                    start_iter=0, start_episode=0, replay_memory_size=250000, replay_memory_init_size=50000, update_target_net_every=10000,
                    gamma=0.99, epsilon_start=1.0, epsilon_end=[0.1,0.01], epsilon_decay_steps=[1e6,1e6], batch_size=32):
                   
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # policy 
    policy = epsilon_greedy_policy(q_net, len(VALID_ACTIONS))


    # populate replay memory
    if (train_or_test == 'train'):
      print("populating replay memory")
      replay_memory = populate_replay_mem(sess, env, state_processor, replay_memory_init_size, policy, epsilon_start, 
                                                       epsilon_end[0], epsilon_decay_steps[0], VALID_ACTIONS, Transition)


    # epsilon start
    if (train_or_test == 'train'):
       delta_epsilon1 = (epsilon_start - epsilon_end[0])/float(epsilon_decay_steps[0])     
       delta_epsilon2 = (epsilon_end[0] - epsilon_end[1])/float(epsilon_decay_steps[1])    
       if (train_from_scratch == True):
          epsilon = epsilon_start
       else:
          if (start_iter <= epsilon_decay_steps[0]):
             epsilon = max(epsilon_start - float(start_iter) * delta_epsilon1, epsilon_end[0])
          elif (start_iter > epsilon_decay_steps[0] and start_iter < epsilon_decay_steps[0]+epsilon_decay_steps[1]):
             epsilon = max(epsilon_end[0] - float(start_iter) * delta_epsilon2, epsilon_end[1])
          else:
             epsilon = epsilon_end[1]      
    elif (train_or_test == 'test'):
       epsilon = epsilon_end[1]


    # total number of time steps 
    total_t = start_iter


    for ep in range(start_episode, num_episodes):

        # save ckpt
        saver.save(tf.get_default_session(), checkpoint_path)

        # env reset
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)

        loss = 0.0
        time_steps = 0
        episode_rewards = 0.0
    
        ale_lives = 5
        info_ale_lives = ale_lives
        steps_in_this_life = 1000000
        num_no_ops_this_life = 0



        while True:
            
            if (train_or_test == 'train'):
              #epsilon = max(epsilon - delta_epsilon, epsilon_end) 
              if (total_t <= epsilon_decay_steps[0]):
                    epsilon = max(epsilon - delta_epsilon1, epsilon_end[0]) 
              elif (total_t >= epsilon_decay_steps[0] and total_t <= epsilon_decay_steps[0]+epsilon_decay_steps[1]):
                    epsilon = epsilon_end[0] - (epsilon_end[0]-epsilon_end[1]) / float(epsilon_decay_steps[1]) * float(total_t-epsilon_decay_steps[0]) 
                    epsilon = max(epsilon, epsilon_end[1])           
              else:
                    epsilon = epsilon_end[1]


              # update target net
              if total_t % update_target_net_every == 0:
                 copy_model_parameters(sess, q_net, target_net)
                 print("\n copied params from Q net to target net ")

                   
            time_to_fire = False
            if (time_steps == 0 or ale_lives != info_ale_lives):
               # new game or new life 
               steps_in_this_life = 0
               num_no_ops_this_life = np.random.randint(low=0,high=7)
               action_probs = [0.0, 1.0, 0.0, 0.0]  # fire
               time_to_fire = True
               if (ale_lives != info_ale_lives):
                  ale_lives = info_ale_lives
            else:
               action_probs = policy(sess, state, epsilon)

            steps_in_this_life += 1 
            if (steps_in_this_life < num_no_ops_this_life and not time_to_fire):
               # no-op
               action_probs = [1.0, 0.0, 0.0, 0.0] # no-op



            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
             
            env.render()
            next_state_img, reward, done, info = env.step(VALID_ACTIONS[action]) 
            
            info_ale_lives = int(info['ale.lives'])

            # rewards = -1,0,+1 as done in the paper
            #reward = np.sign(reward)

            next_state_img = state_processor.process(sess, next_state_img)

            # state is of size [84,84,4]; next_state_img is of size[84,84]
            #next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            next_state = np.zeros((84,84,4),dtype=np.uint8)
            next_state[:,:,0] = state[:,:,1] 
            next_state[:,:,1] = state[:,:,2]
            next_state[:,:,2] = state[:,:,3]
            next_state[:,:,3] = next_state_img    


            episode_rewards += reward  
            time_steps += 1

 
            if (train_or_test == 'train'):

                # if replay memory is full, pop the first element
                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)

                # save transition to replay memory
                # done = True in replay memory for every loss of life 
                if (ale_lives == info_ale_lives):
                   replay_memory.append(Transition(state, action, reward, next_state, done))   
                else:
                   #print('loss of life ')
                   replay_memory.append(Transition(state, action, reward, next_state, True))               

                # sample a minibatch from replay memory
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))


                # calculate q values and targets 
                q_values_next = target_net.predict(sess, next_states_batch)
                greedy_q = np.amax(q_values_next, axis=1) 
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * gamma * greedy_q
               

                # update net 
                if (total_t % 4 == 0):
                   states_batch = np.array(states_batch)
                   loss = q_net.update(sess, states_batch, action_batch, targets_batch)

            if done:
                #print("done: ", done)
                break

            state = next_state
            total_t += 1
            

        if (train_or_test == 'train'): 
           print('\n Eisode: ', ep, '| time steps: ', time_steps, '| total episode reward: ', episode_rewards, '| total_t: ', total_t, '| epsilon: ', epsilon, '| replay mem size: ', len(replay_memory))
        elif (train_or_test == 'test'):
           print('\n Eisode: ', ep, '| time steps: ', time_steps, '| total episode reward: ', episode_rewards, '| total_t: ', total_t, '| epsilon: ', epsilon)


        if (train_or_test == 'train'):
            f = open("experiments/" + str(env.spec.id) + "/performance.txt", "a+")
            f.write(str(ep) + " " + str(time_steps) + " " + str(episode_rewards) + " " + str(total_t) + " " + str(epsilon) + '\n')  
            f.close()

#----------------------------------------------------------------------------------

tf.reset_default_graph()


# Q and target networks 
q_net = QNetwork(scope="q",VALID_ACTIONS=VALID_ACTIONS)
target_net = QNetwork(scope="target_q", VALID_ACTIONS=VALID_ACTIONS)

# state processor
state_processor = ImageProcess()

# tf saver
saver = tf.train.Saver()


with tf.Session() as sess:
 
      # load model/ initialize model
      if ((train_or_test == 'train' and train_from_scratch == False) or train_or_test == 'test'):
                 latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                 print("loading model ckpt {}...\n".format(latest_checkpoint))
                 saver.restore(sess, latest_checkpoint)
      elif (train_or_test == 'train' and train_from_scratch == True):
                 sess.run(tf.global_variables_initializer())    



      # run
      deep_q_learning(sess, env, q_net=q_net, target_net=target_net, state_processor=state_processor, num_episodes=25000,
                            train_or_test=train_or_test, train_from_scratch=train_from_scratch, start_iter=start_iter, start_episode=start_episode,
                                    replay_memory_size=300000, replay_memory_init_size=5000, update_target_net_every=10000,
                                    gamma=0.99, epsilon_start=epsilon_start, epsilon_end=[0.1,0.01], epsilon_decay_steps=[1e6,1e6], batch_size=32)
        



