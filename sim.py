"""
Synthetic 2D trajectory + sensor noise generator.

Vehicle model (unicycle / planar AUV at constant depth):
    state  x = [px, py, theta]
    inputs u = [v, omega]   (forward speed, yaw rate)

Sensors:
    - "odometry/IMU": noisy v and omega every dt (optionally with a gyro bias)
    - position fixes: noisy [px, py] every meas_period seconds, with random dropout
    - range-bearing to known beacons (acoustic-beacon style), same schedule

Everything is seeded so runs are repeatable.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimConfig:
    dt: float = 0.1               # IMU/odometry rate (s)
    T: float = 300.0              # total duration (s)
    speed: float = 1.0            # nominal forward speed (m/s)
    leg_time: float = 40.0        # straight leg duration in the survey pattern (s)
    turn_time: float = 10.0       # duration of each 180-degree turn (s)

    # odometry / IMU noise
    sigma_v: float = 0.05         # m/s
    sigma_w: float = 0.02         # rad/s
    gyro_bias: float = 0.0        # rad/s, constant, NOT known to the filter

    # external measurements
    meas_mode: str = "position"   # "position" or "range_bearing"
    meas_period: float = 1.0      # s between measurement attempts
    dropout: float = 0.0          # probability a measurement is lost
    sigma_pos: float = 1.0        # m (position fixes)
    sigma_r: float = 0.5          # m (range)
    sigma_b: float = np.deg2rad(2.0)  # rad (bearing)
    max_range: float = 60.0       # beacons farther than this aren't heard
    beacons: np.ndarray = field(default_factory=lambda: np.array(
        [[0.0, -20.0], [80.0, -20.0], [0.0, 60.0], [80.0, 60.0]]))
    outage: tuple | None = None   # (t_start, t_end) with no measurements at all
    outlier_prob: float = 0.0     # chance a measurement is a gross outlier
    outlier_scale: float = 10.0   # outlier noise = scale * nominal sigma

    seed: int = 0


def survey_controls(cfg: SimConfig):
    """Lawnmower pattern: straight legs joined by alternating 180-degree turns."""
    n = int(round(cfg.T / cfg.dt))
    t = np.arange(n) * cfg.dt
    v = np.full(n, cfg.speed)
    w = np.zeros(n)
    period = cfg.leg_time + cfg.turn_time
    turn_rate = np.pi / cfg.turn_time
    for k, tk in enumerate(t):
        cycle = int(tk // period)
        if (tk % period) >= cfg.leg_time:
            w[k] = turn_rate if cycle % 2 == 0 else -turn_rate
    return t, v, w


def propagate(x, v, w, dt):
    px, py, th = x
    return np.array([px + v * np.cos(th) * dt,
                     py + v * np.sin(th) * dt,
                     wrap(th + w * dt)])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    t, v, w = survey_controls(cfg)
    n = len(t)

    x_true = np.zeros((n, 3))
    for k in range(1, n):
        x_true[k] = propagate(x_true[k - 1], v[k - 1], w[k - 1], cfg.dt)

    v_meas = v + rng.normal(0, cfg.sigma_v, n)
    w_meas = w + cfg.gyro_bias + rng.normal(0, cfg.sigma_w, n)

    every = int(round(cfg.meas_period / cfg.dt))
    meas = [None] * n   # meas[k] = list of (kind, z, beacon_index) or None
    for k in range(every, n, every):
        if rng.random() < cfg.dropout:
            continue
        if cfg.outage and cfg.outage[0] <= t[k] <= cfg.outage[1]:
            continue
        outlier = rng.random() < cfg.outlier_prob
        s = cfg.outlier_scale if outlier else 1.0
        px, py, th = x_true[k]
        if cfg.meas_mode == "position":
            z = np.array([px, py]) + rng.normal(0, cfg.sigma_pos * s, 2)
            meas[k] = [("pos", z, None)]
        else:
            obs = []
            for i, (bx, by) in enumerate(cfg.beacons):
                dx, dy = bx - px, by - py
                r = np.hypot(dx, dy)
                if r > cfg.max_range:
                    continue
                b = wrap(np.arctan2(dy, dx) - th)
                z = np.array([r + rng.normal(0, cfg.sigma_r * s),
                              wrap(b + rng.normal(0, cfg.sigma_b * s))])
                obs.append(("rb", z, i))
            meas[k] = obs or None

    return dict(t=t, x_true=x_true, v_meas=v_meas, w_meas=w_meas, meas=meas)
