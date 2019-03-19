
import tensorflow as tf
import numpy as np
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer
from AandC import *
from TrainOrTest import *


def train(args):

    with tf.Session() as sess:

        
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        

        state_dim = 29
        action_dim = 3
        action_bound = 1.0 
        

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        
        trainDDPG(sess, args, actor, critic)
     


def test(args):

    with tf.Session() as sess:


        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        
        state_dim = 29
        action_dim = 3
        action_bound = 1.0 
        

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                    float(args['actor_lr']), float(args['tau']), int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                    float(args['critic_lr']), float(args['tau']), float(args['gamma']), actor.get_num_trainable_vars())

        saver = tf.train.Saver()
        saver.restore(sess, "ckpt/model")

        testDDPG(sess, args, actor, critic)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters    
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--episode_count', help='max num of episodes to do while training', default=2000)
    parser.add_argument('--max_steps', help='max length of 1 episode', default=10000)


    
    args = vars(parser.parse_args())
    
    pp.pprint(args)


    train_test = input("enter 1 for train / 0 for test ")
    train_test = int(train_test)
  
    if (train_test == 1):
      train(args)
    elif (train_test == 0):
      test(args)

