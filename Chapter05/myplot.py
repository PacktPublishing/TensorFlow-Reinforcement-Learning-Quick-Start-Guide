import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('pendulum.txt')

plt.plot(data[:,0], data[:,1])
plt.xlabel('episode number', fontsize=12)
plt.ylabel('episode reward', fontsize=12)
#plt.show()
plt.savefig("ddpg_pendulum.png")
