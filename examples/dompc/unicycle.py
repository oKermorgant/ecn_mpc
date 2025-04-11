#!/usr/bin/env python3

import numpy as np
from ecn_mpc.dompc import Model, MPC
from time import time
import sys

varying = '-v' in sys.argv


def as_array(l, dim):
    return np.array(l).reshape(-1,dim).T


def deltatime(t0):
    return f'{1000*(time() - t0):.2f} ms'


def phi_ref(t):
    if varying:
        return [np.cos(t), np.sin(t), 0]
    return [1,2,np.pi]


class Unicycle(Model):
    def __init__(self):
        super().__init__(3, 2, x_names=['x','y','theta'], u_names=['v', 'w'])

        x, y, theta = self.x
        v, w = self.u

        self.set_xdot([v*np.cos(theta),v*np.sin(theta),w])

        # bounds on u
        self.u_lower[0] = -3.
        self.u_upper[0] = 3.

        # arbitrary bounds on x with another method
        self.set_xbound('x', lower=-0.5)


tp = Unicycle()
dt = 0.1

mpc = MPC(tp, ref_idx= [0,1,2])
mpc.configure(10, 0.1)

mpc.set_ref(phi_ref)
mpc.set_Q([100, 100, 0])
mpc.set_R([1e-2, 1e-2])

# simulation
x0 = np.pi*np.array([1, 1, -1]).reshape(-1,1)

mpc.setup(x0)

#sys.exit(0)

import do_mpc


simulator = do_mpc.simulator.Simulator(tp.model)
simulator.set_param(t_step = dt)


tvp_sim = simulator.get_tvp_template()
def tvp_fun(t_now):
    tvp_sim['ref'] = phi_ref(t_now)
    return tvp_sim

simulator.set_tvp_fun(tvp_fun)
simulator.setup()

# sim

simulator.x0 = x0


import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

plt.close('all')


#%%capture
# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(3, sharex=True, figsize=(16,9))
fig.align_ylabels()

ax[0].set_ylabel('Position error [m]')
ax[1].set_ylabel('Orientation error [rad]')
ax[2].set_ylabel('u [m/s,rad/s]')
ax[2].set_xlabel('time [s]')

u0 = np.zeros((2,1))
simulator.reset_history()
simulator.x0 = x0
mpc.mpc.reset_history()

imax = 100

err = []
u = []

xy = []
xyr = []

t = 0.
times = []
for i in range(imax):

    t0 = time()
    u0 = mpc.compute_from(x0, t0 = t)
    ref = np.array(phi_ref(t)).reshape(3,-1)
    xy.append(x0[:2,:])
    xyr.append(ref[:2,:])
    err.append(x0 - ref)
    u.append(u0)


    print(f'{i} / {imax} : {deltatime(t0)}')
    x0 = simulator.make_step(u0)
    t += dt

for i,e in enumerate(as_array(err, 3)):
    ax[max(i-1,0)].plot(e, label = tp.x_names[i])

for i,u in enumerate(as_array(u,2)):
    ax[2].plot(u, label = tp.u_names[i])

for a in ax:
    a.legend()

if varying:
    ax[0].set_title('Varying')
else:
    ax[0].set_title('Constant')


fxy,xyp = plt.subplots(1, figsize=(16,9))
xy = as_array(xy,2)
xyr = as_array(xyr,2)

plt.plot(xy[0],xy[1],'C0')
plt.plot(xyr[0],xyr[1],'C1D')
xyp.set_aspect('equal')
plt.tight_layout()



plt.show()

