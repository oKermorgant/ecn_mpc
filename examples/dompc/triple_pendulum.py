#!/usr/bin/env python3

import numpy as np
from ecn_mpc.dompc import Model, MPC
import time
varying = 1


def phi_ref(t):
    if varying:
        return .5*np.sin(2*t)
    return 0.


class TriplePendulum(Model):
    def __init__(self):
        super().__init__(8, 2, u_names=['p1', 'p2'])

        x, u = self.get_x_u()
        # x = (phi1, phi2, phi3, phidot1, phidot2, phidot3, phi1m, phi2m)

        Theta = 2.25*1e-4
        c = np.array([2.697, 2.66, 3.05, 2.86])*1e-3
        d = np.array([6.78, 8.01, 8.82])*1e-5
        tau = 1e-2

        self.set_xdot([x[3],
                       x[4],
                       x[5],
                      -c[0]/Theta*(x[0]-x[6])-c[1]/Theta*(x[0]-x[1])-d[0]/Theta*x[3],
                      -c[1]/Theta*(x[1]-x[0])-c[2]/Theta*(x[1]-x[2])-d[1]/Theta*x[4],
                      -c[2]/Theta*(x[2]-x[1])-c[3]/Theta*(x[2]-x[7])-d[2]/Theta*x[5],
                       1/tau*(u[0] - x[6]),
                       1/tau*(u[1] - x[7])])

        # bounds
        pi2 = 2*np.pi
        for i in range(3):
            self.x_lower[i] = -pi2
            self.x_upper[i] = pi2
        for i in range(2):
            self.u_lower[i] = -pi2
            self.u_upper[i] = pi2


tp = TriplePendulum()

mpc = MPC(tp, ref_idx= [0,1,2])
mpc.configure(20, 0.1)

mpc.set_ref(phi_ref)
#mpc.set_ref(0.)
mpc.set_Q([100, 100, 100])
mpc.set_R([1e-2, 1e-2])

# simulation
x0 = np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1,1)

mpc.setup(x0)

#sys.exit(0)

import do_mpc


simulator = do_mpc.simulator.Simulator(tp.model)
simulator.set_param(t_step = 0.1)


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

#plt.close('all')

mpc_graphics = do_mpc.graphics.Graphics(mpc.mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)
unames = tp.u_names

#%%capture
# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()

#%%capture
for g in [sim_graphics, mpc_graphics]:
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    g.add_line(var_type='_x', var_name='x_0', axis=ax[0])
    g.add_line(var_type='_x', var_name='x_1', axis=ax[0])
    g.add_line(var_type='_x', var_name='x_2', axis=ax[0])

    # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
    g.add_line(var_type='_u', var_name=unames[0], axis=ax[1])
    g.add_line(var_type='_u', var_name=unames[1], axis=ax[1])


ax[0].set_ylabel('angle position [rad]')
ax[1].set_ylabel('motor angle [rad]')
ax[1].set_xlabel('time [s]')

# Change the color for the three states:

for line_i in mpc_graphics.pred_lines['_x', 'x_0']: line_i.set_color('C0') # blue
for line_i in mpc_graphics.pred_lines['_x', 'x_0']: line_i.set_color('C1') # orange
for line_i in mpc_graphics.pred_lines['_x', 'x_0']: line_i.set_color('C2') # green
# Change the color for the two inputs:
for line_i in mpc_graphics.pred_lines['_u', unames[0]]: line_i.set_color('C0')
for line_i in mpc_graphics.pred_lines['_u', unames[1]]: line_i.set_color('C1')

# Make all predictions transparent:
for line_i in mpc_graphics.pred_lines.full: line_i.set_alpha(0.2)

lines = sim_graphics.result_lines['_x', 'x_0']+sim_graphics.result_lines['_x', 'x_2']+sim_graphics.result_lines['_x', 'x_1']

ax[0].legend(lines,'123',title='disc')

# also set legend for second subplot:
lines = sim_graphics.result_lines['_u', unames[0]]+sim_graphics.result_lines['_u', unames[1]]
ax[1].legend(lines,'12',title='motor')


u0 = np.zeros((2,1))
simulator.reset_history()
simulator.x0 = x0
mpc.mpc.reset_history()

imax = 100

t = 0.
T = []
for i in range(imax):
    print(f'{i} / {imax}')
    t0 = time.perf_counter()
    u0 = mpc.compute_from(x0, t0 = t)
    T.append(time.perf_counter() - t0)
    x0 = simulator.make_step(u0)
    t += 0.1

# Plot predictions from t=0
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()

if varying:
    plt.title('Varying')
else:
    plt.title('Constant')

plt.show()


x,u = mpc.get_prediction(['x_0', 'x_1', 'x_2'])
