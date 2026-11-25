# H3 results (32 paired scene/direction runs per set-up)

| setup                                          |   runs |   success_% |   time_to_goal_s (successful) |   min_dist_m (mean) |   time_within_0.5m_s (mean) |   time_within_1m_s (mean) |   robot-caused within 0.5m_s (mean) |   robot-caused within 1m_s (mean) |
|:-----------------------------------------------|-------:|------------:|------------------------------:|--------------------:|----------------------------:|--------------------------:|------------------------------------:|----------------------------------:|
| Nav2                                           |     32 |       93.75 |                         41.15 |                0.12 |                        2.34 |                      7.87 |                                1.15 |                              3.84 |
| Nav2 + CV prediction                           |     32 |      100    |                         47.06 |                0.14 |                        2.01 |                      7.71 |                                0.97 |                              3.68 |
| Nav2 + LSTM prediction                         |     32 |       93.75 |                         49.65 |                0.15 |                        2.16 |                      8.05 |                                1.01 |                              3.89 |
| Social (LSTM + yielding + human-fitted limits) |     32 |       96.88 |                         54.14 |                0.13 |                        2.69 |                      8.97 |                                1.34 |                              4.41 |

## Nav2 + CV prediction vs Nav2

- time within 0.5 m: -14% change in the mean (paired difference -0.32 s, 95% CI [-0.88, +0.14])
- minimum distance: +0.023 m (95% CI [-0.025, +0.071])
- time to goal (runs both completed): +14%
- robot-caused time within 1 m: -4% (paired difference -0.16 s, 95% CI [-0.80, +0.45])
- failures: 0 new, 2 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Nav2 + LSTM prediction vs Nav2

- time within 0.5 m: -7% change in the mean (paired difference -0.17 s, 95% CI [-0.80, +0.49])
- minimum distance: +0.030 m (95% CI [-0.020, +0.083])
- time to goal (runs both completed): +21%
- robot-caused time within 1 m: +1% (paired difference +0.05 s, 95% CI [-0.66, +0.77])
- failures: 2 new, 2 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Social (LSTM + yielding + human-fitted limits) vs Nav2

- time within 0.5 m: +15% change in the mean (paired difference +0.36 s, 95% CI [-0.30, +1.03])
- minimum distance: +0.004 m (95% CI [-0.044, +0.054])
- time to goal (runs both completed): +30%
- robot-caused time within 1 m: +15% (paired difference +0.57 s, 95% CI [-0.01, +1.18])
- failures: 1 new, 2 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)
