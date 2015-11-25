"""
File:   plotexamples.py

Visualization of a propagation failure event using the
modified Hodgkin-Huxley model
"""

import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt
import spatialextneuron

# Set time discretization
dt = 0.01
# Set the input signal received via the boundary condition
# (rectangular pulse of length tsignal and height I)
tsignal = 0.5
I = 0.001
vmax = 100
vmin = -20

# Define the neuron
M = 1
N = 500
length_axon = 1 # 1cm
extension = 0.
d = 0.00005 #0.5 um
sq2 = 1 / np.sqrt(2)
sigma = np.linspace(0.024,1.2,50)[20]
'Sigma is '+str(sigma)
refarea = 1.535
#refarea = 3.60

# Original HH parameters
#init_state = [0.0, 0.053, 0.596, 0.318]
#alpha_m = lambda v: 0.1 * (v - 25) / (1 - np.exp(-0.1 * (v - 25)))
#beta_h = lambda v: 1 / (1 + np.exp(-0.1 * (v - 30)))
# Original HH parameters with alpha_m changed
init_state = [-0.820, 0.022, 0.429, 0.305]
alpha_m = lambda v: 0.1 * (v - 36) / (1 - np.exp(-0.1 * (v - 36)))
beta_h = lambda v: 1 / (1 + np.exp(-0.1 * (v - 21.5)))


neuron = spatialextneuron.HodgkinHuxley(dt=dt, L=length_axon, N=N,
        sigma_curr=sigma, M=M, d=d, extension=extension, init_state=init_state,
        alpha_m=alpha_m, beta_h=beta_h)
neuron2 = spatialextneuron.HodgkinHuxley(dt=dt, L=length_axon, N=N,
        sigma_curr=sigma, M=M, d=d, extension=extension, init_state=init_state,
        alpha_m=alpha_m, beta_h=beta_h)


# Initialize the figure
fig = plt.figure(figsize=[12,10])
ax = plt.subplot(411, xlim=(0,length_axon+extension), ylim=(vmin,vmax))
ax2 = plt.subplot(412, xlim=(0,length_axon+extension), ylim=(vmin, vmax))
ax3 = plt.subplot(413, xlim=(0,length_axon+extension), ylim=(vmin, vmax))
ax4 = plt.subplot(414)
    
# Set the seed such that the event takes place
#seed = rnd.randint(100000)
#print seed
rnd.seed(95097)
area = []
area2 = []

for t in xrange(50):
    noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                       rnd.normal(0,1,[N-1,M]),
                       rnd.normal(0,sq2,[1,M])])
    neuron.timestep(I, curr_noise=noise)
    area.append((neuron.area() - init_state[0])/refarea)
    noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                       rnd.normal(0,1,[N-1,M]),
                       rnd.normal(0,sq2,[1,M])])
    neuron2.timestep(I, curr_noise=noise)
    area2.append((neuron2.area() - init_state[0])/refarea)

for t in xrange(1400):
    noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                       rnd.normal(0,1,[N-1,M]),
                       rnd.normal(0,sq2,[1,M])])
    neuron.timestep(0, curr_noise=noise)
    area.append((neuron.area() - init_state[0])/refarea)
    noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                       rnd.normal(0,1,[N-1,M]),
                       rnd.normal(0,sq2,[1,M])])
    neuron2.timestep(0, curr_noise=noise)
    area2.append((neuron2.area() - init_state[0])/refarea)
    
ax.plot(neuron.axis_vector, neuron.state[0].copy())
ax.plot(neuron.axis_vector, neuron2.state[0].copy())
for t in xrange(150):
    noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                       rnd.normal(0,1,[N-1,M]),
                       rnd.normal(0,sq2,[1,M])])
    neuron.timestep(0, curr_noise=noise)
    area.append((neuron.area() - init_state[0])/refarea)
    noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                       rnd.normal(0,1,[N-1,M]),
                       rnd.normal(0,sq2,[1,M])])
    neuron2.timestep(0, curr_noise=noise)
    area2.append((neuron2.area() - init_state[0])/refarea)
    
ax2.plot(neuron.axis_vector, neuron.state[0].copy())
ax2.plot(neuron.axis_vector, neuron2.state[0].copy())
for t in xrange(300):
    noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                       rnd.normal(0,1,[N-1,M]),
                       rnd.normal(0,sq2,[1,M])])
    neuron.timestep(0, curr_noise=noise)
    area.append((neuron.area() - init_state[0])/refarea)
    noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                       rnd.normal(0,1,[N-1,M]),
                       rnd.normal(0,sq2,[1,M])])
    neuron2.timestep(0, curr_noise=noise)
    area2.append((neuron2.area() - init_state[0])/refarea)
    
ax3.plot(neuron.axis_vector, neuron.state[0].copy())
ax3.plot(neuron.axis_vector, neuron2.state[0].copy())

ax4.plot(area)
ax4.plot(area2)

plt.show()