import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("performance.txt")

plt.plot(data[:,0],data[:,1],'o',markersize=2)
plt.xlabel("episode count", fontsize=12)
plt.ylabel("episode reward", fontsize=12)
#plt.show()
plt.savefig("cartpole_rewards.png")
