import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('performance.txt')

print(data.shape)


# moving average
alpha = 0.02
mva = np.zeros((data.shape[0]),dtype=np.float)
for i in range(data.shape[0]):
   if i == 0:
     mva[i] = data[i,1]
   else:
     mva[i] = alpha * data[i,1] + (1.0 - alpha) * mva[i-1] 

plt.plot(data[:,0], data[:,1])
plt.plot(data[:,0], mva)
plt.xlabel('episode #', fontsize=20)
plt.ylabel('# of time steps', fontsize=20)
plt.savefig('ddqn1.png')


plt.clf()
plt.close()


# moving average
alpha = 0.02
mvar = np.zeros((data.shape[0]),dtype=np.float)
for i in range(data.shape[0]):
   if i == 0:
     mvar[i] = data[i,2]
   else:
     mvar[i] = alpha * data[i,2] + (1.0 - alpha) * mvar[i-1] 

plt.plot(data[:,3], data[:,2])
plt.plot(data[:,3], mvar)
plt.xlabel('time step #', fontsize=20)
plt.ylabel('episode reward', fontsize=20)
plt.savefig('ddqn2.png')
