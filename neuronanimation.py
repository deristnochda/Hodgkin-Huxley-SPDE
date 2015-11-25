"""
File:   neuronanimation.py

Animation of spatially extended stochastic neurons using the
module spatialextneuron.
"""

import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt
from matplotlib import animation
import spatialextneuron
import argparse

# Parse the given arguments
parser = argparse.ArgumentParser()
parser.add_argument("id", help="specify noise intensity as {1, ..., 50}", type=int)
args = parser.parse_args()

# Set time discretization
dt = 0.01
# Set the input signal received via the boundary condition
# (rectangular pulse of length tsignal and height I)
tsignal = 0.5
I = 0.001
# axis limits for plotting
vmax = 120
vmin = -30

# Define the neuron
M = 1
N = 500
length_axon = 1 # 1cm
extension = 0.5 # 0.5cm
d = 0.00005 #0.5 um
sq2 = 1 / np.sqrt(2)
sigma = np.linspace(0.0, 2.5, 51)[args.id-1]
print 'Sigma is '+str(sigma)

# Original HH parameters
init_state = [0.0, 0.053, 0.596, 0.318]
alpha_m = lambda v: 0.1 * (v - 25) / (1 - np.exp(-0.1 * (v - 25)))
beta_h = lambda v: 1 / (1 + np.exp(-0.1 * (v - 30)))
# Modified HH parameters exhibiting propagation failure
#init_state = [-0.820, 0.022, 0.429, 0.305]
#alpha_m = lambda v: 0.1 * (v - 36) / (1 - np.exp(-0.1 * (v - 36)))
#beta_h = lambda v: 1 / (1 + np.exp(-0.1 * (v - 21.5)))

neuron = spatialextneuron.HodgkinHuxley(dt=dt, L=length_axon, N=N,
        sigma_curr=sigma, M=M, d=d, extension=extension, init_state=init_state,
        alpha_m=alpha_m, beta_h=beta_h)

refarea = 1.535
shift = init_state[0] * (length_axon + extension)
shift2 = init_state[0] * extension

# Initialize the figure
fig = plt.figure(figsize=[12,10])
ax = plt.subplot(311, xlim=(0,length_axon+extension), ylim=(vmin,vmax))
ax_gating = plt.subplot(312, xlim=(0,length_axon+extension), ylim=(0,1))
ax_area = plt.subplot(313)
line = ax.plot([], [])[0]
m_line = ax_gating.plot([], [], label='m')[0]
h_line = ax_gating.plot([], [], label='h')[0]
n_line = ax_gating.plot([], [], label='n')[0]
area_line = ax_area.plot([], [])[0]
area2_line = ax_area.plot([], [])[0]
ax_gating.legend()

# Functions for animation
def init():
    line.set_data([], [])
    area_line.set_data([], [])
    area2_line.set_data([], [])
    m_line.set_data([], [])
    h_line.set_data([], [])
    n_line.set_data([], [])
    return [line, area_line, area2_line, m_line, h_line, n_line]
    
def animate(t):
    global neuron, I, tsignal
    # Switch signal off
    if neuron.time_elapsed > tsignal:
        I = 0
    print t
    # Update neuron
    # Speed up animation
    for s in xrange(10):
        noise = np.vstack([rnd.normal(0,sq2,[1,M]), 
                           rnd.normal(0,1,[N-1,M]),
                           rnd.normal(0,sq2,[1,M])])
        neuron.timestep(I, curr_noise=noise)
    # Update voltage plot
    line.set_data(neuron.axis_vector, neuron.state[0][:,0].T)
    # Update plot of gating variables
    m_line.set_data(neuron.axis_vector, neuron.state[1])
    h_line.set_data(neuron.axis_vector, neuron.state[2])
    n_line.set_data(neuron.axis_vector, neuron.state[3])
    # Update plot area vs time
    area = (neuron.area()+neuron.area_ext()-shift) / refarea
    area_ext = (neuron.area_ext() - shift2) / refarea
    area_line.set_data(np.append(area_line.get_xdata(), neuron.time_elapsed),
            np.append(area_line.get_ydata(), area))
    area2_line.set_data(np.append(area2_line.get_xdata(), neuron.time_elapsed),
            np.append(area2_line.get_ydata(), area_ext))
    ax_area.relim()
    ax_area.autoscale_view(True,True,True)
    return [line, area_line, area2_line, m_line, h_line, n_line]


# The actual animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=6000,
                              interval=10, blit=False)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('animation.mp4', fps=30, 
#          extra_args=['-vcodec', 'h264', 
#                      '-pix_fmt', 'yuv420p'])
plt.show()

