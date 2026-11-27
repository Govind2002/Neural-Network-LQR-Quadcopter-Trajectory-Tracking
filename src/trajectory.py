"""
trajectory.py
Minimum-jerk polynomial trajectory generator for point-to-point tracking.

Reference: Mellinger & Kumar (ICRA 2011).
Each segment is a 5th-order polynomial in each axis ensuring:
  - continuous position, velocity, acceleration at segment boundaries
  - zero velocity and acceleration at start and end of each segment.

Author: Muhammed Aslam A (2025EEA2819), IIT Delhi
"""

import numpy as np


def min_jerk_segment(p0, p1, T):
    """
    Generate coefficients for a minimum-jerk polynomial from p0 to p1 in time T.
    s(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵

    Boundary conditions: s(0)=p0, ṡ(0)=0, s̈(0)=0
                         s(T)=p1, ṡ(T)=0, s̈(T)=0

    Returns c : (6,) coefficient vector
    """
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    # Derived analytically from BC matrix
    c = np.array([
        p0,
        0.0,
        0.0,
        10*(p1-p0)/T3,
        -15*(p1-p0)/T4,
         6*(p1-p0)/T5,
    ])
    return c


def eval_poly(c, t):
    """Evaluate polynomial and its first two derivatives at time t."""
    t2, t3, t4, t5 = t**2, t**3, t**4, t**5
    pos  = c[0] + c[1]*t + c[2]*t2 + c[3]*t3  + c[4]*t4   + c[5]*t5
    vel  =        c[1]   + 2*c[2]*t + 3*c[3]*t2 + 4*c[4]*t3 + 5*c[5]*t4
    acc  =                 2*c[2]   + 6*c[3]*t   + 12*c[4]*t2 + 20*c[5]*t3
    return pos, vel, acc


class MinJerkTrajectory:
    """
    Multi-segment minimum-jerk trajectory through a list of 3D waypoints.

    Parameters
    ----------
    waypoints  : list of (3,) arrays  – [px, py, pz] waypoints including start
    seg_times  : list of floats       – duration of each segment [s]
    """

    def __init__(self, waypoints, seg_times):
        assert len(waypoints) == len(seg_times) + 1, \
            "Need N+1 waypoints for N segments."
        self.waypoints = [np.array(w, dtype=float) for w in waypoints]
        self.seg_times = seg_times
        self.T_total   = sum(seg_times)
        self.t_breaks  = np.concatenate([[0.0], np.cumsum(seg_times)])

        # Pre-compute per-axis polynomial coefficients for each segment
        self.coeffs = []           # coeffs[seg][axis] = (6,) array
        for k in range(len(seg_times)):
            seg = []
            for axis in range(3):
                c = min_jerk_segment(self.waypoints[k][axis],
                                     self.waypoints[k+1][axis],
                                     self.seg_times[k])
                seg.append(c)
            self.coeffs.append(seg)

    def evaluate(self, t):
        """
        Evaluate the trajectory at time t.

        Returns
        -------
        x_des    : (12,) desired state  [pos, vel, phi=0, theta=0, psi=0, p=0, q=0, r=0]
        xdot_des : (12,) desired state derivative
        """
        t = np.clip(t, 0.0, self.T_total)

        # Find active segment
        seg_idx = np.searchsorted(self.t_breaks[1:], t, side='right')
        seg_idx = min(seg_idx, len(self.seg_times) - 1)
        t_local = t - self.t_breaks[seg_idx]

        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)

        for axis in range(3):
            p, v, a = eval_poly(self.coeffs[seg_idx][axis], t_local)
            pos[axis] = p
            vel[axis] = v
            acc[axis] = a

        # Full 12-dim desired state (hover attitude assumed)
        x_des    = np.zeros(12)
        x_des[0:3] = pos
        x_des[3:6] = vel

        xdot_des    = np.zeros(12)
        xdot_des[0:3] = vel
        xdot_des[3:6] = acc

        return x_des, xdot_des

    def get_full_reference(self, t_array):
        """Evaluate trajectory at multiple time instants (for plotting)."""
        positions = []
        for t in t_array:
            xd, _ = self.evaluate(t)
            positions.append(xd[0:3])
        return np.array(positions)
