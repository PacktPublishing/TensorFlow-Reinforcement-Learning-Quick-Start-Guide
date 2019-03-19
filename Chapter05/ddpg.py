#--------------------------------------------
# AUTHOR: KAUSHIK BALAKRISHNAN
#--------------------------------------------


import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import sys

from replay_buffer import ReplayBuffer
from AandC import *
from TrainOrTest import *



def train(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))
        env._max_episode_steps = int(args['max_episode_len'])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                float(args['actor_lr']), float(args['tau']), int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                 float(args['critic_lr']), float(args['tau']), float(args['gamma']), actor.get_num_trainable_vars())
        

        trainDDPG(sess, env, args, actor, critic)

        saver = tf.train.Saver()
        saver.save(sess, "ckpt/model")
        print("saved model ")


def test(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))
        env._max_episode_steps = int(args['max_episode_len'])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                float(args['actor_lr']), float(args['tau']), int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                 float(args['critic_lr']), float(args['tau']), float(args['gamma']), actor.get_num_trainable_vars())

        saver = tf.train.Saver()
        saver.restore(sess, "ckpt/model")

        testDDPG(sess, env, args, actor, critic)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for Bellman updates', default=0.99)
    parser.add_argument('--tau', help='target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch', default=64)

    # run parameters
    parser.add_argument('--env', help='gym env', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed', default=258)
    parser.add_argument('--max-episodes', help='max num of episodes', default=250)
    parser.add_argument('--max-episode-len', help='max length of each episode', default=1000)
    parser.add_argument('--render-env', help='render gym env', action='store_true')
    parser.add_argument('--mode', help='train/test', default='train')
    
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    if (args['mode'] == 'train'):
      train(args)
    elif (args['mode'] == 'test'):
      test(args)

