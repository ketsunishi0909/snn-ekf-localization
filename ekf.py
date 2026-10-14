"""
Dead reckoning and EKF for the planar unicycle model in sim.py.

The EKF uses the noisy (v, omega) readings as control inputs in the prediction
step and corrects with whatever measurements arrive. The measurement update is
written against a generic (z, h(x), H, R) interface so a learned
pseudo-measurement (ANN / SNN output) can be plugged in later the same way.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2

from sim import propagate, wrap


@dataclass
class FilterConfig:
    # What the filter *believes* the noise is. Set these differently from the
    # simulator to test model mismatch.
    sigma_v: float = 0.05
    sigma_w: float = 0.02
    sigma_pos: float = 1.0
    sigma_r: float = 0.5
    sigma_b: float = np.deg2rad(2.0)
    q_extra: float = 1e-6          # small additive process noise for stability
    gate: bool = False             # chi-square innovation gating
    gate_prob: float = 0.997


def dead_reckoning(x0, v_meas, w_meas, dt):
    n = len(v_meas)
    xs = np.zeros((n, 3))
    xs[0] = x0
    for k in range(1, n):
        xs[k] = propagate(xs[k - 1], v_meas[k - 1], w_meas[k - 1], dt)
    return xs


class EKF:
    def __init__(self, x0, P0, fcfg: FilterConfig, beacons=None):
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)
        self.c = fcfg
        self.beacons = beacons
        self.n_rejected = 0

    def predict(self, v, w, dt):
        th = self.x[2]
        F = np.array([[1, 0, -v * np.sin(th) * dt],
                      [0, 1,  v * np.cos(th) * dt],
                      [0, 0, 1]])
        G = np.array([[np.cos(th) * dt, 0],
                      [np.sin(th) * dt, 0],
                      [0, dt]])
        M = np.diag([self.c.sigma_v ** 2, self.c.sigma_w ** 2])
        Q = G @ M @ G.T + self.c.q_extra * np.eye(3)
        self.x = propagate(self.x, v, w, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, z, h, H, R, angle_idx=()):
        """Generic EKF update. angle_idx = innovation components to wrap."""
        y = z - h
        for i in angle_idx:
            y[i] = wrap(y[i])
        S = H @ self.P @ H.T + R
        if self.c.gate:
            d2 = y @ np.linalg.solve(S, y)
            if d2 > chi2.ppf(self.c.gate_prob, df=len(z)):
                self.n_rejected += 1
                return
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form

    def update_position(self, z):
        H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        R = np.eye(2) * self.c.sigma_pos ** 2
        self.update(z, H @ self.x, H, R)

    def update_range_bearing(self, z, beacon_idx):
        bx, by = self.beacons[beacon_idx]
        px, py, th = self.x
        dx, dy = bx - px, by - py
        q = dx ** 2 + dy ** 2
        r = np.sqrt(q)
        h = np.array([r, wrap(np.arctan2(dy, dx) - th)])
        H = np.array([[-dx / r, -dy / r, 0],
                      [dy / q, -dx / q, -1]])
        R = np.diag([self.c.sigma_r ** 2, self.c.sigma_b ** 2])
        self.update(z, h, H, R, angle_idx=(1,))


def run_ekf(data, dt, fcfg: FilterConfig, beacons=None,
            x0=np.zeros(3), P0=np.diag([1.0, 1.0, np.deg2rad(5) ** 2])):
    n = len(data["t"])
    ekf = EKF(x0, P0, fcfg, beacons)
    xs = np.zeros((n, 3))
    Ps = np.zeros((n, 3, 3))
    xs[0], Ps[0] = ekf.x, ekf.P
    for k in range(1, n):
        ekf.predict(data["v_meas"][k - 1], data["w_meas"][k - 1], dt)
        if data["meas"][k]:
            for kind, z, idx in data["meas"][k]:
                if kind == "pos":
                    ekf.update_position(z)
                else:
                    ekf.update_range_bearing(z, idx)
        xs[k], Ps[k] = ekf.x, ekf.P
    return xs, Ps, ekf.n_rejected


# ---------------- metrics ----------------
def pos_error(x_est, x_true):
    return np.linalg.norm(x_est[:, :2] - x_true[:, :2], axis=1)


def rmse(x_est, x_true):
    return float(np.sqrt(np.mean(pos_error(x_est, x_true) ** 2)))


def nees(x_est, Ps, x_true):
    """Normalized estimation error squared. For a consistent 3-state filter the
    average should be close to 3. Much larger = filter is overconfident."""
    e = x_true - x_est
    e[:, 2] = wrap(e[:, 2])
    return np.array([e[k] @ np.linalg.solve(Ps[k], e[k]) for k in range(len(e))])
