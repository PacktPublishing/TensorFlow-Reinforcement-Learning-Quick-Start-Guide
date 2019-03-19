import sys
import numpy as np
import gym

env = gym.make('MountainCarContinuous-v0')


for _ in range(100):
  s = env.reset()
  done = False

  max_pos = -1.0
  max_speed = 0.0 
  ep_reward = 0.0

  while not done:
    env.render()  
    a = [-1.0 + 2.0*np.random.uniform()] 
    s_, r, done, _ = env.step(a)

    if s_[0] > max_pos: max_pos = s_[0]
    if s_[1] > max_speed: max_speed = s_[1]   
    ep_reward += r

  print("ep_reward: ", ep_reward, "| max_pos: ", max_pos, "| max_speed: ", max_speed)
  

