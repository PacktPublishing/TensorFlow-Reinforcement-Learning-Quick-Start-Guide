import numpy as np
import tensorflow as tf
from random import choice



# copy model params 
def update_target_graph(from_scope,to_scope):
    from_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    copy_ops = []
    for from_param,to_param in zip(from_params,to_params):
        copy_ops.append(to_param.assign(from_param))
    return copy_ops



# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    dsr = np.zeros_like(x,dtype=np.float32)
    running_sum = 0.0
    for i in reversed(range(0, len(x))):
       running_sum = gamma * running_sum + x[i]
       dsr[i] = running_sum 
    return dsr


