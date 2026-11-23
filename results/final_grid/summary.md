# H3 results (32 paired scene/direction runs per set-up)

| setup                                          |   runs |   success_% |   time_to_goal_s (successful) |   min_dist_m (mean) |   time_within_0.5m_s (mean) |   time_within_1m_s (mean) |   robot-caused within 0.5m_s (mean) |   robot-caused within 1m_s (mean) |
|:-----------------------------------------------|-------:|------------:|------------------------------:|--------------------:|----------------------------:|--------------------------:|------------------------------------:|----------------------------------:|
| Nav2                                           |     32 |      100    |                         43.41 |                0.22 |                        1.46 |                      7.21 |                                0.66 |                              3.31 |
| Nav2 blind to people (bulldozer)               |     32 |      100    |                         37.75 |                0.2  |                        1.52 |                      6.03 |                                0.65 |                              2.77 |
| Nav2 + CV prediction                           |     32 |      100    |                         47.82 |                0.25 |                        1.56 |                      7.18 |                                0.69 |                              3.23 |
| Nav2 + LSTM prediction                         |     32 |      100    |                         49.6  |                0.22 |                        1.57 |                      7.04 |                                0.71 |                              3.27 |
| Social (LSTM + yielding + human-fitted limits) |     32 |       96.88 |                         58.25 |                0.18 |                        2.39 |                      9.11 |                                1.08 |                              4.22 |
| Learned driver (LSTM + cloned human speed)     |     32 |       87.5  |                         94.09 |                0.17 |                        2.76 |                     12.46 |                                1.21 |                              5.66 |

## Nav2 blind to people (bulldozer) vs Nav2

- time within 0.5 m: +4% change in the mean (paired difference +0.06 s, 95% CI [-0.25, +0.38])
- minimum distance: -0.024 m (95% CI [-0.081, +0.034])
- time to goal (runs both completed): -13%
- robot-caused time within 1 m: -17% (paired difference -0.55 s, 95% CI [-0.81, -0.29])
- failures: 0 new, 0 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Nav2 + CV prediction vs Nav2

- time within 0.5 m: +7% change in the mean (paired difference +0.10 s, 95% CI [-0.24, +0.42])
- minimum distance: +0.033 m (95% CI [-0.007, +0.074])
- time to goal (runs both completed): +10%
- robot-caused time within 1 m: -2% (paired difference -0.08 s, 95% CI [-0.44, +0.31])
- failures: 0 new, 0 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Nav2 + LSTM prediction vs Nav2

- time within 0.5 m: +8% change in the mean (paired difference +0.11 s, 95% CI [-0.26, +0.45])
- minimum distance: +0.004 m (95% CI [-0.049, +0.056])
- time to goal (runs both completed): +14%
- robot-caused time within 1 m: -1% (paired difference -0.04 s, 95% CI [-0.44, +0.33])
- failures: 0 new, 0 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Social (LSTM + yielding + human-fitted limits) vs Nav2

- time within 0.5 m: +64% change in the mean (paired difference +0.93 s, 95% CI [+0.46, +1.40])
- minimum distance: -0.043 m (95% CI [-0.099, +0.009])
- time to goal (runs both completed): +35%
- robot-caused time within 1 m: +27% (paired difference +0.90 s, 95% CI [+0.35, +1.48])
- failures: 1 new, 0 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)

## Learned driver (LSTM + cloned human speed) vs Nav2

- time within 0.5 m: +88% change in the mean (paired difference +1.29 s, 95% CI [+0.85, +1.74])
- minimum distance: -0.051 m (95% CI [-0.115, +0.015])
- time to goal (runs both completed): +114%
- robot-caused time within 1 m: +71% (paired difference +2.35 s, 95% CI [+1.64, +3.02])
- failures: 4 new, 0 fixed
- **H3 NOT SUPPORTED** (needs: min distance up, time within 0.5 m down >= 30%, time to goal <= +20%, no extra failures)
