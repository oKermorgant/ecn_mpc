#!/usr/bin/env python3

import do_mpc
import casadi
import numpy as np

'''
A wrapper around do_mpc to make it more straightforward during labs
'''


class Model:
    def __init__(self, x_dim, u_dim, x_names = [], u_names = []):
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        if not x_names:
            self.x_names = [f'x_{i}' for i in range(x_dim)]
        else:
            self.x_names = x_names
            x_dim = len(x_names)
        self.x = [self.model.set_variable(var_type='_x', var_name=self.x_names[i], shape = (1, 1)) for i in range(x_dim)]

        self.u = []
        if not u_names:
            self.u_names = [f'u_{i}' for i in range(u_dim)]
        else:
            self.u_names = u_names
            u_dim = len(u_names)
        self.u = [self.model.set_variable(var_type='_u', var_name=self.u_names[i], shape = (1, 1)) for i in range(u_dim)]

        inf = casadi.casadi.DM(casadi.inf)
        self.x_lower = [-inf for _ in range(x_dim)]
        self.x_upper = [inf for _ in range(x_dim)]
        self.u_lower = [-inf for _ in range(u_dim)]
        self.u_upper = [inf for _ in range(u_dim)]

    def set_xbound(self, label, lower = None, upper = None):
        if lower is None and upper is None:
            return
        try:
            idx = self.x_names.index(label)
            if lower is not None:
                self.x_lower[idx] = lower
            if upper is not None:
                self.x_upper[idx] = upper
        except ValueError:
            raise ValueError(f'dompc.Model.set_xbound: {label} is not a valid state component')

    def set_ubound(self, label, lower = None, upper = None):
        if lower is None and upper is None:
            return
        try:
            idx = self.u_names.index(label)
            if lower is not None:
                self.u_lower[idx] = lower
            if upper is not None:
                self.u_upper[idx] = upper
        except ValueError:
            raise ValueError(f'dompc.Model.set_ubound: {label} is not a valid control component')

    def xdim(self):
        return len(self.x)

    def udim(self):
        return len(self.u)

    def set_xdot(self, xdot):

        if isinstance(xdot, dict):
            # assume keys are x-keys
            for key, val in xdot.items():
                self.model.set_rhs(key, val)
        else:
            # ordering is key ordering
            for idx, key in enumerate(self.x_names):
                self.model.set_rhs(key, xdot[idx])

    def get_x_u(self):
        return self.x, self.u


class MPC:
    def __init__(self, model: Model, ref_idx = None):
        self.model = model

        # finish setup the model
        if ref_idx is None:
            ref_idx = list(range(self.model.xdim()))
        self.ref_idx = ref_idx
        self.ref = self.model.model.set_variable(var_type='_tvp',
                                                 var_name='ref',
                                                 shape=(len(self.ref_idx),1))
        self.model.model.setup()
        self.mpc = do_mpc.controller.MPC(self.model.model)
        self.mpc.settings.supress_ipopt_output()

        # store bounds
        for i in range(self.model.xdim()):
            tag = self.model.x_names[i]
            self.mpc.bounds['lower','_x', tag] = self.model.x_lower[i]
            self.mpc.bounds['upper','_x', tag] = self.model.x_upper[i]
        for i in range(self.model.udim()):
            tag = self.model.u_names[i]
            self.mpc.bounds['lower','_u', tag] = self.model.u_lower[i]
            self.mpc.bounds['upper','_u', tag] = self.model.u_upper[i]

    def configure(self, n_horizon = 10, t_step = 0.1, store_full_solution = True):
        setup_mpc = {'n_horizon': n_horizon,
                     't_step': t_step,
                     'store_full_solution': store_full_solution}
        self.mpc.set_param(**setup_mpc)

    def set_ref(self, ref):
        # set the state reference either as a float, list or function

        tvp_mpc = self.mpc.get_tvp_template()
        n = self.mpc.settings.n_horizon

        if isinstance(ref, (float, list, tuple)):
            def tvp_fun(t):
                for k in range(n+1):
                    tvp_mpc['_tvp',k,'ref'] = ref
                return tvp_mpc
        else:
            # function
            step = self.mpc.settings.t_step

            def tvp_fun(t):
                for k in range(n+1):
                    tvp_mpc['_tvp',k,'ref'] = ref(t + step*k)
                return tvp_mpc

        self.mpc.set_tvp_fun(tvp_fun)
        return self.ref

    def set_Q(self, Q):
        if isinstance(Q, float):
            Q = [Q for _ in self.ref_idx]
        if len(Q) != len(self.ref_idx):
            print(f'Q should have the size of the reference terms ({len(Q)} vs {len(self.ref_idx)}')

        mterm = 0
        x = self.model.x
        for ref,xi in enumerate(self.ref_idx):
            mterm += Q[ref] * (x[xi] - self.ref[ref])**2
        self.mpc.set_objective(mterm=mterm, lterm=mterm)

    def set_R(self, R):
        u_dim = self.model.udim()
        if isinstance(R, float):
            R = [R for _ in range(u_dim)]
        if len(R) != u_dim:
            print(f'R should have the size of the control terms ({len(R)} vs {u_dim}')
        rterm = dict((self.model.u_names[i], R[i]) for i in range(u_dim))
        self.mpc.set_rterm(**rterm)

    def setup(self, x0):
        self.mpc.setup()
        self.mpc.x0 = np.array(x0)
        self.mpc.set_initial_guess()

    def compute_from(self, x0, t0 = None):
        if t0 is not None:
            self.mpc._t0 = t0
        return self.mpc.make_step(np.array(x0))

    def get_prediction(self, states = None):

        if states is not None:
            if isinstance(states[0], str):
                names = states
            else:
                names = [self.model.x_names[i] for i in states]
        else:
            names = self.model.x_names

        x_pred = np.vstack([self.mpc.data.prediction(('_x', name, 0))[:,:,0] for name in names])
        u_pred = np.vstack([self.mpc.data.prediction(('_u', name, 0))[:,:,0] for name in self.model.u_names])
        return x_pred, u_pred
