import numpy as np

a= np.load('data/PEMS03/test.npz')
# b = np.sort(np.concatenate((np.arange(-(12 - 1), 1, 1),)))
# c = np.sort(np.arange(1, (12 + 1), 1))
ax= np.squeeze(a['x'], axis=3)
ay = np.squeeze(a['y'], axis=3)
# import pdb
# pdb.set_trace()
ax = ax[: , :, 1]
ay= ay[: , :, 1]

t= 0
for i in range(36):
    print(ax[i, 1])
    t+=1
    if t%12==0: print(" ")

print(" after   ")
t= 0
for i in range(24):
    print(ay[i, 1])
    t+=1
    if t%12==0:print(" ")