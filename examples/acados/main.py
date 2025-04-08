#!/usr/bin/env python3
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import numpy as np
import scipy.linalg
from time import time
import sys
from casadi import SX, vertcat, sin, cos
import matplotlib.pyplot as plt


generate = '-g' in sys.argv
build = '-b' in sys.argv or generate
varying = '-v' in sys.argv


def phi_ref(t, include_u = True):
    # [x,y,x_d,y_d,th,th_d, u]
    idx_max = 5 if include_u else 3
    if varying:
        return np.array([np.cos(t), np.sin(t), 0, 0, 0])[:idx_max]
    return np.array([1,2,np.pi, 0, 0])[:idx_max]


X0 = np.array([1, 1, -1])  # Initial state
v_max = 3.  # Define the max linear velocity
w_max = 1.  # Define the max angular velocity
T_horizon = 2.0  # Define the prediction horizon
N_horizon = 10  # Define the number of discretization steps


def export_robot_model() -> AcadosModel:
    model_name = "unicycle"

    # set up states & controls
    x = SX.sym("x")
    y = SX.sym("y")
    theta = SX.sym("theta")

    x = vertcat(x, y, theta)

    v = SX.sym("v")
    w = SX.sym("w")
    u = vertcat(v, w)

    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    theta_dot = SX.sym("theta_dot")

    xdot = vertcat(x_dot, y_dot, theta_dot)

    # dynamics
    f_expl = vertcat(v * cos(theta), v * sin(theta), w)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$\\theta$"]
    model.u_labels = ['$v$', "$\\omega$"]

    return model


def deltatime(t0):
    return f'{1000*(time() - t0):.2f} ms'


def create_ocp_solver_description() -> AcadosOcp:


    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost
    Q_mat = np.diag([1e3, 1e3, 0])  # [x,y,x_d,th]
    R_mat = np.diag([1e-2, 1e-2])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx:(nx + nu), 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([-v_max, -w_max])
    ocp.constraints.ubu = np.array([v_max, w_max])
    ocp.constraints.idxbu = np.array([0,1])

    ocp.constraints.x0 = X0

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

    return ocp


# main code

# create solvers
ocp = create_ocp_solver_description()
model = ocp.model
t0 = time()
try:
    mpc_solver = AcadosOcpSolver(ocp, generate = generate, build = build)
    simulator = AcadosSimSolver(ocp, generate = generate, build = build)
except FileNotFoundError:
    print('Code not generated, doing it now')
    mpc_solver = AcadosOcpSolver(ocp)
    simulator = AcadosSimSolver(ocp)
print(f'Getting generated code... {deltatime(t0)}')

N_horizon = mpc_solver.N
dt = T_horizon/N_horizon

# prepare simulation
Nsim = 100
nx = ocp.model.x.rows()
nu = ocp.model.u.rows()

simX = np.zeros((Nsim + 1, nx))
simU = np.zeros((Nsim, nu))
simXref = np.zeros((Nsim + 1, nx))


xcurrent = X0
simX[0, :] = xcurrent
simXref[0,:] = phi_ref(0.,False)

yref = np.array([1, 1, 0, 0, 0])
yref_N = np.array([1, 1, 0])

# initialize solver
for stage in range(N_horizon + 1):
    mpc_solver.set(stage, "x", 0.0 * np.ones(xcurrent.shape))
for stage in range(N_horizon):
    mpc_solver.set(stage, "u", np.zeros((nu,)))

# closed loop
t_sim = 0
for i in range(Nsim):
    # update yref
    t0 = time()
    for j in range(N_horizon+1):
        mpc_solver.set(j, "yref", phi_ref(t_sim+j*dt,j < N_horizon))
    # mpc_solver.set(N_horizon, "yref", phi_ref(t_sim+T_horizon, False))

    # register reference
    simXref[i+1,:] = phi_ref(t_sim+dt, False)

    # solve ocp

    simU[i, :] = mpc_solver.solve_for_x0(xcurrent)
    status = mpc_solver.get_status()
    print(f'{i} / {Nsim} : {deltatime(t0)}')

    if status not in [0, 2]:
        mpc_solver.print_statistics()
        raise Exception(
            f"acados mpc_solver returned status {status} in closed loop instance {i} with {xcurrent}"
        )

    # simulate system
    simX[i + 1, :] = xcurrent = simulator.simulate(xcurrent, simU[i, :])
    t_sim += dt

# plot results
plt.close('all')
plt.ion()

fig, ax = plt.subplots(2, sharex = True, figsize=(16,9))
fig.align_ylabels()
ax[0].set_ylabel('Position error [m]')
ax[0].set_xlabel('time [s]')
ax[1].set_ylabel('u [m/s,rad/s]')
ax[1].set_xlabel('time [s]')

t = np.linspace(0, dt*Nsim, Nsim + 1)
xy = simX[:,:2].T
xyr = simXref[:,:2].T

ax[0].plot(t, (xyr-xy).T)
ax[0].legend(model.x_labels[:2])
ax[1].plot(t[1:],simU)
ax[1].legend(model.u_labels)
plt.tight_layout()

# top view
fxy,xyp = plt.subplots(1, figsize=(16,9))


plt.plot(xy[0],xy[1],'C0')
plt.plot(xyr[0],xyr[1],'C1D')
xyp.set_aspect('equal')
plt.tight_layout()

plt.show()
plt.waitforbuttonpress()
