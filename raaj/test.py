import numpy as np

d_min = 1.
d_max = 60.
nDepth = 256

d_candi = np.linspace(d_min, d_max, nDepth)
#d_candi_up = np.linspace(d_min, d_max, nDepth * 4)

# print(d_candi)
# print(d_candi.shape)
# stop

def expf(d_min, d_max, nDepth, power):
    candi = []
    v = np.log(d_max) / (nDepth-1)
    for i in range(0, nDepth):
        dk = np.exp(float(i)*v)
        candi.append(dk)

    return np.array(candi)

def powerf(d_min, d_max, nDepth, power):
    f = lambda x: d_min + (d_max - 1) * x
    x = np.linspace(start=0, stop=1, num=nDepth)
    x = np.power(x, power)
    candi = [f(v) for v in x]
    return np.array(candi)
    #stop


#d_candi_log = powerf(d_min, d_max, nDepth)

#d_candi_up_log = logf(d_min, d_max, nDepth * 4)

#print(d_candi_log.shape)
#print(d_candi.shape)

import matplotlib.pyplot as plt
#plt.plot(expf(d_min, d_max, nDepth, 4), linestyle='--', marker='o', color='r', markersize=1)
#plt.plot(powerf(d_min, d_max, nDepth, 8), linestyle='--', marker='o', color='b', markersize=1)
#plt.eventplot(powerf(d_min, d_max, nDepth, 4), color='r')
#plt.eventplot(d_candi, color='g')
#plt.eventplot(powerf(d_min, d_max, nDepth, 4), color='g')
plt.scatter(expf(d_min, d_max, nDepth, 4), np.ones(d_candi.shape[0]), s=1, color='r')

plt.scatter(powerf(d_min, d_max, nDepth, 2), np.zeros(d_candi.shape[0]), s=1, color='g')

plt.show()


# from numpy import linspace, power, exp
# from matplotlib.pyplot import axis, eventplot, show, yticks
#
# f = lambda x: 1 + 59 * x
#
# x = linspace(start=0, stop=1, num=128)
# x = exp(x)
#
# data = [f(v) for v in x]
# print(data)
#
# eventplot(data)
# axis("image")
# #yticks(ticks=[0, 2], labels=[None, None])
# show()