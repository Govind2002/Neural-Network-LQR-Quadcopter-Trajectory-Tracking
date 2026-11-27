"""
piecewise_linearisation.py
Partition the quadcopter state-space into operating regions,
linearise at each equilibrium, and design LQR gains per region.

Author: Muhammed Aslam A (2025EEA2819), IIT Delhi
Course: Learning Based Control
Date  : April 2026
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from quadcopter_dynamics import QuadcopterParams, linearise_at


# ── LQR design ────────────────────────────────────────────────────────────────
def lqr(A, B, Q, R):
    """
    Solve the continuous-time LQR problem.
    Returns optimal gain K such that u = -K(x - x_eq).
    """
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
        return K, P
    except Exception as e:
        # Fallback: return zero gain if ARE fails (degenerate region)
        print(f"  [LQR warning] ARE failed: {e}. Using zero gain.")
        return np.zeros((B.shape[1], A.shape[0])), np.eye(A.shape[0])


# ── Operating-region definition ───────────────────────────────────────────────
def build_operating_regions(waypoints, params=None):
    """
    Build one linearisation region per waypoint.

    Each region is centred on a hover equilibrium at the waypoint's (x,y,z).
    Hover equilibrium:
        - zero velocity and angular rate
        - zero attitude (level flight)
        - thrust = m*g, zero torques

    Parameters
    ----------
    waypoints : list of (3,) arrays   – [px, py, pz] targets
    params    : QuadcopterParams

    Returns
    -------
    regions : list of dicts, each containing:
        'x_eq'  : (12,) equilibrium state
        'u_eq'  : (4,)  equilibrium input
        'A','B' : linearised matrices
        'K'     : LQR gain (12→4)
        'P'     : Lyapunov matrix
        'center': (3,) position centre
        'radius': float – switching radius
    """
    if params is None:
        params = QuadcopterParams()

    # LQR weights  (position, velocity, attitude, angular-rate, actuation)
    Q = np.diag([10, 10, 20,           # px, py, pz  – emphasise altitude
                  2,  2,  2,           # vx, vy, vz
                  5,  5,  2,           # phi, theta, psi
                  1,  1,  1])          # p, q, r
    R = np.diag([0.1, 0.5, 0.5, 1.0]) # T, tau_phi, tau_theta, tau_psi

    regions = []
    for wp in waypoints:
        px, py, pz = wp

        # Hover equilibrium
        x_eq = np.zeros(12)
        x_eq[0:3] = [px, py, pz]
        u_eq = np.array([params.m * params.g, 0.0, 0.0, 0.0])

        A, B = linearise_at(x_eq, u_eq, params)
        K, P = lqr(A, B, Q, R)

        # Switching radius: half the distance to nearest other waypoint
        # (set large for single-region case)
        regions.append({
            'x_eq'  : x_eq,
            'u_eq'  : u_eq,
            'A'     : A,
            'B'     : B,
            'K'     : K,
            'P'     : P,
            'center': np.array(wp, dtype=float),
        })

    # Assign switching radii
    for i, reg in enumerate(regions):
        dists = []
        for j, other in enumerate(regions):
            if i != j:
                dists.append(np.linalg.norm(reg['center'] - other['center']))
        reg['radius'] = (min(dists) / 2.0) if dists else 5.0

    return regions


def select_region(x, regions):
    """
    Select the active linearisation region based on the current position.
    Returns the index of the closest region centre.
    """
    pos = x[0:3]
    dists = [np.linalg.norm(pos - r['center']) for r in regions]
    return int(np.argmin(dists))
