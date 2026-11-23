# H3 results (32 paired scene/direction runs per set-up)

| setup                                          |   runs |   success_% |   time_to_goal_s (successful) |   min_dist_m (mean) |   time_within_0.5m_s (mean) |   time_within_1m_s (mean) |   robot-caused within 0.5m_s (mean) |   robot-caused within 1m_s (mean) |
|:-----------------------------------------------|-------:|------------:|------------------------------:|--------------------:|----------------------------:|--------------------------:|------------------------------------:|----------------------------------:|
| Nav2                                           |     32 |      100    |                         44.5  |                0.17 |                        2.1  |                      8.26 |                                0.99 |                              4.08 |
| Nav2 + LSTM prediction                         |     32 |      100    |                         49.64 |                0.14 |                        1.96 |                      8.15 |                                0.93 |                              3.92 |
| Social (LSTM + yielding + human-fitted limits) |     32 |       96.88 |                         56.33 |                0.15 |                        2.66 |                      9.66 |                                1.26 |                              4.64 |

## Nav2 + LSTM prediction vs Nav2

- time within 0.5 m: -7% change in the mean (paired difference -0.14 s, 95% CI [-0.71, +0.45])
- minimum distance: -0.026 m (95% CI [-0.078, +0.025])
- time to goal (runs both completed): +12%
- robot-caused time within 1 m: -4% (paired difference -0.17 s, 95% CI [-0.77, +0.51])
- failures: 0 new, 0 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Social (LSTM + yielding + human-fitted limits) vs Nav2

- time within 0.5 m: +27% change in the mean (paired difference +0.56 s, 95% CI [+0.02, +1.12])
- minimum distance: -0.018 m (95% CI [-0.062, +0.020])
- time to goal (runs both completed): +27%
- robot-caused time within 1 m: +14% (paired difference +0.56 s, 95% CI [-0.09, +1.27])
- failures: 1 new, 0 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)
