"""
controller.py  –  NN-augmented piecewise tracking LQR
Control law:  u = u_eq - K*(x - x_des) - alpha_nn * f_hat(x)

alpha_nn scales the NN contribution to prevent destabilisation.
The NN learns to cancel the nonlinear residual; it is only activated
after a brief warm-up to let LQR bring the system close to the reference.

Author: Muhammed Aslam A (2025EEA2819), IIT Delhi
"""
import numpy as np
from neural_network import MLNN
from piecewise_linearisation import select_region


class NNPiecewiseLQRController:
    def __init__(self, regions, nn_hidden=20,
                 gamma_W=0.1, gamma_V=0.1,
                 k_w=0.02, k_v=0.02,
                 alpha_nn=0.3,          # NN contribution scaling
                 nn_warmup=2.0,         # seconds before NN is active
                 u_min=None, u_max=None):

        self.regions  = regions
        self.alpha_nn = alpha_nn
        self.nn_warmup = nn_warmup

        self.nn = MLNN(n_input=12, n_hidden=nn_hidden, n_output=4,
                       gamma_W=gamma_W, gamma_V=gamma_V,
                       k_w=k_w, k_v=k_v, w_max=1.0, seed=42)
        self.nn.W_hat[:] = 0.0
        self.nn.V_hat[:] = 0.0

        self.u_min = np.array([0.0,-2.0,-2.0,-2.0]) if u_min is None else np.asarray(u_min)
        self.u_max = np.array([20., 2.0, 2.0, 2.0]) if u_max is None else np.asarray(u_max)

        self.active_idx = 0
        self.t_prev = None

    def compute(self, t, x, x_des, update_weights=True):
        dt = 0.01 if self.t_prev is None else max(t - self.t_prev, 1e-6)
        self.t_prev = t

        self.active_idx = select_region(x, self.regions)
        reg  = self.regions[self.active_idx]
        K    = reg['K']
        u_eq = reg['u_eq']

        e = x - x_des                       # (12,) tracking error

        # NN is only active after warm-up period
        nn_on = (t >= self.nn_warmup)
        f_hat, cache = self.nn.forward(x)   # always forward (for adaptation)
        nn_contribution = self.alpha_nn * f_hat if nn_on else np.zeros(4)

        # u = u_eq - K*e - alpha_nn * f_hat
        u_raw = u_eq - K @ e - nn_contribution
        u     = np.clip(u_raw, self.u_min, self.u_max)

        # Adapt weights only when NN is active and error is not too large
        if update_weights and nn_on and np.linalg.norm(e[0:3]) < 2.0:
            e4 = np.array([e[2], e[3], e[4], e[5]])
            self.nn.update(e4, cache, dt)

        return u, {
            'region'    : self.active_idx,
            'error_pos' : float(np.linalg.norm(e[0:3])),
            'error_att' : float(np.linalg.norm(e[6:9])),
            'f_hat_norm': float(np.linalg.norm(f_hat)),
            'nn_active' : nn_on,
            'u'         : u.copy(),
        }
