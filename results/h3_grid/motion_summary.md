# Smoothness, comfort and human-likeness

| setup                                          |   runs |   mean_speed |   rms_jerk |   heading_change_per_m |   stop_starts_per_min |   time_ttc_under_2s |   mean_robot_approach_close |   human_likeness_distance |
|:-----------------------------------------------|-------:|-------------:|-----------:|-----------------------:|----------------------:|--------------------:|----------------------------:|--------------------------:|
| Nav2                                           |     32 |        0.36  |      0.974 |                  0.78  |                 4.235 |               6.597 |                      -0.023 |                     0.797 |
| Nav2 + CV prediction                           |     32 |        0.334 |      0.944 |                  0.86  |                 5.75  |               6.609 |                      -0.042 |                     0.839 |
| Nav2 + LSTM prediction                         |     32 |        0.327 |      1.052 |                  0.928 |                 5.839 |               6.819 |                      -0.025 |                     0.87  |
| Social (LSTM + yielding + human-fitted limits) |     30 |        0.294 |      0.776 |                  0.864 |                 6.739 |               8.083 |                      -0.024 |                     0.885 |
| Human driver                                   |     10 |        0.238 |      1.229 |                  1.095 |                 3.151 |               1.974 |                      -0.025 |                     0     |

Burden on people (reactive crowds only): person-seconds pushed > 0.2 m off their path, displacement integral (m*s), mean peak displacement (m):

| setup                                          |   people_displaced_s |   people_displacement_ms |   mean_peak_displacement_m |
|:-----------------------------------------------|---------------------:|-------------------------:|---------------------------:|
| Nav2                                           |                  nan |                      nan |                        nan |
| Nav2 + CV prediction                           |                  nan |                      nan |                        nan |
| Nav2 + LSTM prediction                         |                  nan |                      nan |                        nan |
| Social (LSTM + yielding + human-fitted limits) |                  nan |                      nan |                        nan |
| Human driver                                   |                  nan |                      nan |                        nan |

Human-likeness distance per feature (Wasserstein / human IQR; lower is more human-like):

| setup                                          |   W_speed |   W_accel |   W_jerk |   W_turn_rate |   W_robot_approach_close |   W_pass_clearance |
|:-----------------------------------------------|----------:|----------:|---------:|--------------:|-------------------------:|-------------------:|
| Nav2                                           |      0.62 |      1.3  |     0.9  |          0.4  |                     0.24 |               1.32 |
| Nav2 + CV prediction                           |      0.54 |      1.61 |     1.06 |          0.41 |                     0.12 |               1.29 |
| Nav2 + LSTM prediction                         |      0.47 |      1.77 |     1.11 |          0.48 |                     0.09 |               1.3  |
| Social (LSTM + yielding + human-fitted limits) |      0.33 |      1.79 |     1.17 |          0.5  |                     0.2  |               1.32 |

## Nav2 + CV prediction vs Nav2 (paired over 30 scene/direction pairs)

- rms_jerk: +0.030 (+3%), 95% CI [-0.277, +0.285]
- heading_change_per_m: +0.108 (+14%), 95% CI [-0.027, +0.210]
- stop_starts_per_min: +1.391 (+33%), 95% CI [+0.585, +2.245]
- time_ttc_under_2s: +0.417 (+7%), 95% CI [-0.473, +1.293]

## Nav2 + LSTM prediction vs Nav2 (paired over 30 scene/direction pairs)

- rms_jerk: +0.168 (+19%), 95% CI [-0.181, +0.532]
- heading_change_per_m: +0.183 (+24%), 95% CI [+0.032, +0.321]
- stop_starts_per_min: +1.679 (+40%), 95% CI [+0.614, +2.728]
- time_ttc_under_2s: +0.567 (+9%), 95% CI [-0.630, +1.847]

## Social (LSTM + yielding + human-fitted limits) vs Nav2 (paired over 30 scene/direction pairs)

- rms_jerk: -0.131 (-14%), 95% CI [-0.397, +0.028]
- heading_change_per_m: +0.115 (+15%), 95% CI [-0.007, +0.214]
- stop_starts_per_min: +2.572 (+62%), 95% CI [+1.428, +3.682]
- time_ttc_under_2s: +1.880 (+30%), 95% CI [+0.607, +3.160]
