# Results

Monte Carlo: 30 seeds per row, 300 s lawnmower survey, dt = 0.1 s, measurements attempted every 1 s.

## Dropout sweep (position RMSE, m)

| dropout | dead reckoning | EKF (position fixes) | EKF (range-bearing) |
|---|---|---|---|
| 0.00 | 2.23 | 0.37 | 0.16 |
| 0.25 | 2.23 | 0.39 | 0.17 |
| 0.50 | 2.23 | 0.47 | 0.20 |
| 0.75 | 2.23 | 0.63 | 0.29 |
| 0.90 | 2.23 | 0.88 | 0.46 |
| 0.97 | 2.23 | 1.34 | 0.85 |

## Failure cases

NEES: average normalized error squared. ~3 means the filter's covariance is honest; much higher means it is overconfident, much lower means underconfident.

| case | EKF RMSE (m) | EKF max err (m) | DR RMSE (m) | mean NEES | rejected meas. | ms/step |
|---|---|---|---|---|---|---|
| Nominal | 0.37 | 1.08 | 2.23 | 2.7 | 0 | 0.038 |
| Unmodeled gyro bias 0.01 rad/s | 1.42 | 2.53 | 29.18 | 43.0 | 0 | 0.040 |
| Gyro bias + 75% dropout | 3.22 | 8.39 | 29.18 | 68.7 | 0 | 0.038 |
| 100 s outage (t=100-200 s) | 0.80 | 2.29 | 2.23 | 2.8 | 0 | 0.041 |
| Filter overconfident (assumes σ_pos=0.2 m, true 1.0 m) | 0.51 | 1.74 | 2.23 | 40.4 | 0 | 0.039 |
| Filter underconfident (assumes σ_pos=5 m) | 0.57 | 1.36 | 2.23 | 1.0 | 0 | 0.042 |
| 5% outliers (10x noise), no gating | 0.76 | 2.85 | 2.23 | 9.9 | 0 | 0.044 |
| 5% outliers, chi-square gating | 0.38 | 1.09 | 2.23 | 2.7 | 15 | 0.061 |
| Range-bearing, nominal | 0.16 | 0.65 | 2.23 | 2.9 | 0 | 0.052 |
| Range-bearing, 5% outliers, no gating | 0.33 | 1.38 | 2.23 | 11.8 | 0 | 0.055 |
| Range-bearing, 5% outliers, gating | 0.17 | 0.69 | 2.23 | 2.9 | 28 | 0.088 |
