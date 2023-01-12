"""
Problem 3.3.5 - Logistic map.
Discrete-time demographic model with logistic map (chaos).
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt


interval = (2.4, 4)     # start, end
accuracy = 0.001        # precision
reps = 200              # number of repetitions
numtoplot = 100         # number of lines to plor
lims = np.zeros(reps)

fig, biax = plt.subplots()

lims[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps-1):
        lims[i+1] = r*lims[i]*(1-lims[i])

    biax.plot([r]*numtoplot, lims[reps-numtoplot:], 'b.',markersize = 0.02)

biax.set(xlabel='k', ylabel='x', title='logistic map')
plt.show()