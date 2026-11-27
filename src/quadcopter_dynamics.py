"""
quadcopter_dynamics.py
Nonlinear quadcopter dynamics model for the NN-based piecewise linearisation controller.

State vector:  x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]  (12-dim)
Control input: u = [T, tau_phi, tau_theta, tau_psi]                       (4-dim)

Author: Muhammed Aslam A (2025EEA2819), IIT Delhi
Course: Learning Based Control
Date  : April 2026
"""

import numpy as np

# ── Physical parameters ────────────────────────────────────────────────────────
class QuadcopterParams:
    m   = 0.5        # mass [kg]
    g   = 9.81       # gravity [m/s²]
    Ixx = 4.856e-3   # moment of inertia X [kg·m²]
    Iyy = 4.856e-3   # moment of inertia Y [kg·m²]
    Izz = 8.801e-3   # moment of inertia Z [kg·m²]
    l   = 0.225      # arm length [m]
    kT  = 2.98e-6    # thrust coefficient
    kD  = 1.14e-7    # drag coefficient


def quadcopter_dynamics(t, x, u, params=None):
    """
    Continuous-time nonlinear quadcopter dynamics.
    ẋ = f(x) + g(x)·u   (control-affine form)

    Parameters
    ----------
    t      : float        – current time (unused, kept for ODE interface)
    x      : (12,) array  – state vector
    u      : (4,)  array  – control input [T, tau_phi, tau_theta, tau_psi]
    params : QuadcopterParams or None

    Returns
    -------
    xdot : (12,) array – state derivative
    """
    if params is None:
        params = QuadcopterParams()

    m, g   = params.m, params.g
    Ixx, Iyy, Izz = params.Ixx, params.Iyy, params.Izz

    px, py, pz      = x[0], x[1], x[2]
    vx, vy, vz      = x[3], x[4], x[5]
    phi, theta, psi  = x[6], x[7], x[8]   # roll, pitch, yaw
    p,   q,   r      = x[9], x[10], x[11]  # body angular rates

    T        = u[0]          # total thrust
    tau_phi  = u[1]          # roll torque
    tau_theta= u[2]          # pitch torque
    tau_psi  = u[3]          # yaw torque

    # Trig shortcuts
    cphi, sphi     = np.cos(phi),   np.sin(phi)
    cth,  sth      = np.cos(theta), np.sin(theta)
    cpsi, spsi     = np.cos(psi),   np.sin(psi)
    tth            = np.tan(theta)

    # Position kinematics  ẋ = v  (already in world frame)
    dpx = vx
    dpy = vy
    dpz = vz

    # Translational dynamics (Euler-angle rotation of thrust)
    dvx = (T / m) * (cpsi*sth*cphi + spsi*sphi)
    dvy = (T / m) * (spsi*sth*cphi - cpsi*sphi)
    dvz = (T / m) * (cth*cphi) - g

    # Euler-angle kinematics (ZYX convention)
    dphi   = p + (q*sphi + r*cphi) * tth
    dtheta = q*cphi - r*sphi
    dpsi   = (q*sphi + r*cphi) / (cth + 1e-8)

    # Rotational dynamics (Euler equations)
    dp = (tau_phi  - (Izz - Iyy)*q*r) / Ixx
    dq = (tau_theta- (Ixx - Izz)*p*r) / Iyy
    dr = (tau_psi  - (Iyy - Ixx)*p*q) / Izz

    return np.array([dpx, dpy, dpz,
                     dvx, dvy, dvz,
                     dphi, dtheta, dpsi,
                     dp, dq, dr])


def linearise_at(x_eq, u_eq, params=None, eps=1e-5):
    """
    Numerically compute (A, B) Jacobians at an equilibrium (x_eq, u_eq)
    via central finite differences.

    Returns
    -------
    A : (12,12)  state matrix
    B : (12, 4)  input matrix
    """
    if params is None:
        params = QuadcopterParams()

    n, m_in = len(x_eq), len(u_eq)
    A = np.zeros((n, n))
    B = np.zeros((n, m_in))

    f0 = quadcopter_dynamics(0, x_eq, u_eq, params)

    for i in range(n):
        xp, xm = x_eq.copy(), x_eq.copy()
        xp[i] += eps; xm[i] -= eps
        A[:, i] = (quadcopter_dynamics(0, xp, u_eq, params) -
                   quadcopter_dynamics(0, xm, u_eq, params)) / (2*eps)

    for i in range(m_in):
        up, um = u_eq.copy(), u_eq.copy()
        up[i] += eps; um[i] -= eps
        B[:, i] = (quadcopter_dynamics(0, x_eq, up, params) -
                   quadcopter_dynamics(0, x_eq, um, params)) / (2*eps)

    return A, B
