import numpy as np
import sys
import tensorflow as tf


# convert raw Atari RGB image of size 210x160x3 into 84x84 grayscale image
class ImageProcess():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
    def process(self, sess, state):
        return sess.run(self.output, { self.input_state: state })



# copy params from qnet1 to qnet2
def copy_model_parameters(sess, qnet1, qnet2):
    q1_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet1.scope)]
    q1_params = sorted(q1_params, key=lambda v: v.name)
    q2_params = [t for t in tf.trainable_variables() if t.name.startswith(qnet2.scope)]
    q2_params = sorted(q2_params, key=lambda v: v.name)
    update_ops = []
    for q1_v, q2_v in zip(q1_params, q2_params):
        op = q2_v.assign(q1_v)
        update_ops.append(op)
    sess.run(update_ops)


# epsilon-greedy
def epsilon_greedy_policy(qnet, num_actions):
    def policy_fn(sess, observation, epsilon):
        if (np.random.rand() < epsilon):  
          # explore: equal probabiities for all actions
          A = np.ones(num_actions, dtype=float) / float(num_actions)
        else:
          # exploit 
          q_values = qnet.predict(sess, np.expand_dims(observation, 0))[0]
          max_Q_action = np.argmax(q_values)
          A = np.zeros(num_actions, dtype=float)
          A[max_Q_action] = 1.0 
        return A
    return policy_fn



# populate replay memory
def populate_replay_mem(sess, env, state_processor, replay_memory_init_size, policy, epsilon_start, epsilon_end, epsilon_decay_steps, VALID_ACTIONS, Transition):
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)

    delta_epsilon = (epsilon_start - epsilon_end)/float(epsilon_decay_steps)

    replay_memory = []

    for i in range(replay_memory_init_size):
        epsilon = max(epsilon_start - float(i) * delta_epsilon, epsilon_end)
        action_probs = policy(sess, state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        env.render()   
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])

        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state
    return replay_memory
