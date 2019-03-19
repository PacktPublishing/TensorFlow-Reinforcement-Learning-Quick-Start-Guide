import tensorflow as tf
import numpy as np

# Torcs uses dt = 0.2 seconds

def OU(x, mu, sigma, theta, dt = 0.2):
   return theta * (mu - x) + sigma * np.random.randn(sigma.shape[0])
