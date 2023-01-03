### Exercise 3.3.4

import numpy as np
import matplotlib.pyplot as plt

# def f(x, k): return k*x*(1-x)

# x = np.linspace(0,4)

# for k in np.arange(1,10): 
#     plt.plot(x,f(x,k), label = f"k = {k}")

# plt.xlabel("x");plt.ylabel("kx(x-1)")
# plt.legend()
# plt.show()


interval = (2.4, 4)  # start, end
accuracy = 0.001
reps = 200  # number of repetitions
numtoplot = 100 # number of lines to plor
lims = np.zeros(reps)

fig, biax = plt.subplots()

lims[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps-1):
        lims[i+1] = r*lims[i]*(1-lims[i])

    biax.plot([r]*numtoplot, lims[reps-numtoplot:], 'b.',markersize = 0.02)

biax.set(xlabel='k', ylabel='x', title='logistic map')
plt.show()