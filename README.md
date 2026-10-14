# Week 2 (Oct 19-25): 2D synthetic trajectory + EKF baseline

This is the classical baseline that the ANN and SNN versions will be compared against in the following weeks.

## Files

- `sim.py`: generates a 2D survey trajectory (lawnmower pattern, like an AUV at constant depth), noisy odometry/IMU readings, and intermittent external measurements.
- `ekf.py`: dead reckoning, the EKF, and the metrics (RMSE, NEES).
- `run_experiments.py`: runs everything and writes the figures and `results/results.md`.

Requirements: numpy, scipy, matplotlib. Run `python run_experiments.py` (about a minute), or add `--quick` for a faster run.

## Setup

**State:** x = [px, py, θ] (position and heading).

**Motion model:** a unicycle model driven by measured forward speed v and yaw rate ω:

```
px' = px + v cos θ Δt
py' = py + v sin θ Δt
θ'  = θ + ω Δt
```

The noisy v and ω readings are used as control inputs in the prediction step. Process noise comes from mapping the input noise through the Jacobian G: Q = G·diag(σv², σω²)·Gᵀ.

**Two measurement types:**

1. Position fixes [px, py] with σ = 1 m. The model is linear, so H is constant. This stands in for something like a surface GPS fix or a USBL position.
2. Range and bearing to 4 known beacons, with σr = 0.5 m and σb = 2°, and a maximum range of 60 m. This model is nonlinear, so it is the case that actually exercises the "extended" part of the EKF.

Measurements are attempted every 1 s and can be dropped at random. Odometry runs at 10 Hz.

**Covariance update:** Joseph form, for numerical stability.

**Optional chi-square innovation gating:** a measurement is rejected if its Mahalanobis distance exceeds the 99.7% threshold.

**Metrics:**
- Position RMSE, maximum error and final drift.
- NEES (normalized estimation error squared), which checks whether the filter's covariance is honest. For 3 states it should average about 3.
- Runtime per step.

## Assumptions

These are the things the baseline takes for granted, and that later experiments could break on purpose:

- Planar motion only: no depth, roll or pitch, and no currents pushing the vehicle sideways.
- Noise is white, Gaussian and zero-mean, and the filter knows its true variance unless a test says otherwise.
- Beacon positions are known exactly, and there is no data association problem: every measurement comes labeled with its beacon.
- Measurements arrive on time, with no latency or timestamp offset.
- The initial state is known to within a small covariance (1 m, 5°).
- This is pure state estimation. There is no mapping, so it is not SLAM yet.

## Results (30 Monte Carlo seeds per row)

Full tables are in `results/results.md`.

**Nominal case, position fixes:**
- Dead reckoning RMSE is 2.23 m. Its error keeps growing over time.
- EKF RMSE is 0.37 m, and NEES is 2.7. That NEES value means the filter's uncertainty estimate matches its real error.

**Range-bearing case:**
- EKF RMSE is 0.16 m.
- Part of the reason it beats position fixes is that up to 4 beacons each give 2 values per second, and the bearings also tell the filter about heading. So this comparison depends on the noise values I picked. It is not a general result.

**Dropout:**
- The EKF degrades gracefully. With position fixes, RMSE goes from 0.37 m at 0% dropout to 0.88 m at 90% dropout.
- At 97% dropout (roughly one fix every 30 s) it is still better than dead reckoning.

## Failure cases

**Unmodeled gyro bias (0.01 rad/s)**
- RMSE rises to 1.42 m, and NEES jumps to about 43.
- The filter assumes zero-mean noise, so it is both wrong and confident about it.
- With 75% dropout, RMSE gets worse still: 3.2 m, with a maximum error of 8.4 m.
- Of all the cases, this is the one that hurts most.
- The standard fix is to add the bias to the state vector.

**100 s outage**
- The error grows to about 2 m during the gap, then recovers within a few seconds once measurements return.
- NEES stays near 3, because the 3σ bound grows along with the error.
- In other words, the filter fails "honestly" here: it knows it is uncertain.
- See `fig4_outage.png`.

**Wrong measurement noise, filter too confident (assumes σ = 0.2 m)**
- RMSE rises to 0.51 m, and NEES is about 40.
- The filter chases noisy fixes.

**Wrong measurement noise, filter too cautious (assumes σ = 5 m)**
- RMSE rises to 0.57 m, and NEES is about 1.
- The filter leans too much on drifting odometry.
- Both mistunings cost accuracy. Only the NEES value tells you which direction the tuning is wrong in.

**5% outliers (10× noise)**
- RMSE roughly doubles, to 0.76 m, and NEES is about 10.
- Chi-square gating brings RMSE back to the nominal 0.38 m and NEES back to 2.7.
- The same pattern holds for range-bearing measurements.

**Runtime**
- About 0.04 to 0.09 ms per step in plain Python/NumPy.
- This is the cost that any learned measurement module has to be compared against.

## How the ANN/SNN step plugs in next week

`EKF.update(z, h, H, R)` is generic. A learned model that outputs a pseudo-measurement z, plus ideally its own uncertainty R, can be fed through the same call without touching the prediction step.

That keeps the comparison fair: same filter, same data, and only the measurement source changes. The failure cases above (bias, outliers, outages) are natural test conditions for checking whether a learned measurement actually helps.
