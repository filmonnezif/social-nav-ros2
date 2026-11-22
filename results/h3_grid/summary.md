# H3 results (30 paired scene/direction runs per set-up)

| setup                                          |   runs |   success_% |   time_to_goal_s (successful) |   min_dist_m (mean) |   time_within_0.5m_s (mean) |   time_within_1m_s (mean) |   robot-caused within 0.5m_s (mean) |   robot-caused within 1m_s (mean) |
|:-----------------------------------------------|-------:|------------:|------------------------------:|--------------------:|----------------------------:|--------------------------:|------------------------------------:|----------------------------------:|
| Nav2                                           |     30 |       96.67 |                         41.13 |                0.13 |                        2.11 |                      7.46 |                                1.04 |                              3.64 |
| Nav2 + CV prediction                           |     30 |      100    |                         47.08 |                0.15 |                        2.06 |                      7.71 |                                0.99 |                              3.68 |
| Nav2 + LSTM prediction                         |     30 |       93.33 |                         48.92 |                0.16 |                        2.13 |                      8.05 |                                0.99 |                              3.88 |
| Social (LSTM + yielding + human-fitted limits) |     30 |       96.67 |                         54.02 |                0.13 |                        2.7  |                      8.89 |                                1.34 |                              4.37 |

## Nav2 + CV prediction vs Nav2

- time within 0.5 m: -3% change in the mean (paired difference -0.05 s, 95% CI [-0.40, +0.26])
- minimum distance: +0.021 m (95% CI [-0.028, +0.074])
- time to goal (runs both completed): +15%
- robot-caused time within 1 m: +1% (paired difference +0.04 s, 95% CI [-0.51, +0.58])
- failures: 0 new, 1 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Nav2 + LSTM prediction vs Nav2

- time within 0.5 m: +1% change in the mean (paired difference +0.02 s, 95% CI [-0.57, +0.65])
- minimum distance: +0.036 m (95% CI [-0.016, +0.091])
- time to goal (runs both completed): +20%
- robot-caused time within 1 m: +7% (paired difference +0.24 s, 95% CI [-0.41, +0.91])
- failures: 2 new, 1 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Social (LSTM + yielding + human-fitted limits) vs Nav2

- time within 0.5 m: +28% change in the mean (paired difference +0.59 s, 95% CI [-0.03, +1.20])
- minimum distance: +0.006 m (95% CI [-0.045, +0.058])
- time to goal (runs both completed): +30%
- robot-caused time within 1 m: +20% (paired difference +0.73 s, 95% CI [+0.14, +1.32])
- failures: 1 new, 1 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)
