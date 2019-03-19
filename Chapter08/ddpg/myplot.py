import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('analysis_file.txt')

plt.plot(data[:,0], data[:,1])
plt.xlabel('episode number', fontsize=12)
plt.ylabel('# of time steps ', fontsize=12)
plt.savefig("ddpg_torcs.png")
