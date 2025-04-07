#!/usr/bin/env python3
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import numpy as np
import scipy.linalg
from utils import plot_robot
from time import time
import sys
from casadi import SX, vertcat, sin, cos
import matplotlib.pyplot as plt


generate = '-g' in sys.argv
build = '-b' in sys.argv or generate
varying = '-v' in sys.argv


def phi_ref(t):
    # [x,y,x_d,y_d,th,th_d, u]
    if varying:
        return np.array([np.cos(t), np.sin(t), 0, 0, 0, 0, 0])
    return np.array([1,2,0, np.pi, 0, 0, 0])


X0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Initial state
F_max = 10  # Define the max force allowed
T_horizon = 2.0  # Define the prediction horizon


def export_robot_model() -> AcadosModel:
    model_name = "unicycle"

    # set up states & controls
    x = SX.sym("x")
    y = SX.sym("y")
    v = SX.sym("x_d")
    theta = SX.sym("theta")
    theta_d = SX.sym("theta_d")

    x = vertcat(x, y, v, theta, theta_d)

    F = SX.sym("F")
    T = SX.sym("T")
    u = vertcat(F, T)

    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    v_dot = SX.sym("v_dot")
    theta_dot = SX.sym("theta_dot")
    theta_ddot = SX.sym("theta_ddot")

    xdot = vertcat(x_dot, y_dot, v_dot, theta_dot, theta_ddot)

    # dynamics
    f_expl = vertcat(v * cos(theta), v * sin(theta), F, theta_d, T)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$v$", "$\\theta$", "$\\omega$"]
    model.u_labels = ["$F$", "$T$"]

    return model


def deltatime(t0):
    return f'{1000*(time() - t0):.2f} ms'


def create_ocp_solver_description() -> AcadosOcp:
    N_horizon = 10  # Define the number of discretization steps

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model = export_robot_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost
    Q_mat = 2 * np.diag([1e3, 1e3, 0, 0, 0])  # [x,y,x_d,y_d,th,th_d]
    R_mat = 2 * 5 * np.diag([1e-1, 1e-2])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([-F_max])
    ocp.constraints.ubu = np.array([+F_max])
    ocp.constraints.idxbu = np.array([0])

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
    acados_ocp_solver = AcadosOcpSolver(ocp, generate = generate, build = build)
    acados_integrator = AcadosSimSolver(ocp, generate = generate, build = build)
except FileNotFoundError:
    print('Code not generated, doing it now')
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)
print(f'Getting generated code... {deltatime(t0)}')

N_horizon = acados_ocp_solver.N
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
simXref[0,:] = phi_ref(time())[:5]

yref = np.array([1, 1, 0, 0, 0, 0, 0])
yref_N = np.array([1, 1, 0, 0, 0])

# initialize solver
for stage in range(N_horizon + 1):
    acados_ocp_solver.set(stage, "x", 0.0 * np.ones(xcurrent.shape))
for stage in range(N_horizon):
    acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

# closed loop
for i in range(Nsim):
    # update yref
    t0 = time()
    for j in range(N_horizon):
        acados_ocp_solver.set(j, "yref", phi_ref(t0+(j+1)*dt))
    acados_ocp_solver.set(N_horizon, "yref", phi_ref(t0+T_horizon)[:5])

    # register reference
    simXref[i+1,:] = phi_ref(t0+dt)[:5]

    # solve ocp

    simU[i, :] = acados_ocp_solver.solve_for_x0(xcurrent)
    status = acados_ocp_solver.get_status()
    print(f'{i} / {Nsim} : {deltatime(t0)}')

    if status not in [0, 2]:
        acados_ocp_solver.print_statistics()
        plot_robot(
            np.linspace(0, T_horizon / N_horizon * i, i + 1),
            F_max,
            simU[:i, :],
            simX[: i + 1, :],
        )
        raise Exception(
            f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
        )

    # simulate system
    xcurrent = acados_integrator.simulate(xcurrent, simU[i, :])
    simX[i + 1, :] = xcurrent

# plot results
plt.close('all')
plot_robot(
    np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1), [F_max, None], simU, simX,
    x_labels=model.x_labels, u_labels=model.u_labels, time_label=model.t_label
)

# top view
fxy,xyp = plt.subplots(1, figsize=(16,9))
xy = simX[:,:2].T
xyr = simXref[:,:2].T

plt.plot(xy[0],xy[1],'C0')
plt.plot(xyr[0],xyr[1],'C1D')
xyp.set_aspect('equal')
plt.tight_layout()
