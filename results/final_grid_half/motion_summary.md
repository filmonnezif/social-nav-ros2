# Smoothness, comfort and human-likeness

| setup                                          |   runs |   mean_speed |   rms_jerk |   heading_change_per_m |   stop_starts_per_min |   time_ttc_under_2s |   mean_robot_approach_close |   human_likeness_distance |
|:-----------------------------------------------|-------:|-------------:|-----------:|-----------------------:|----------------------:|--------------------:|----------------------------:|--------------------------:|
| Nav2                                           |     32 |        0.344 |      0.728 |                  0.68  |                 4.676 |               7.219 |                      -0.008 |                     0.822 |
| Nav2 + LSTM prediction                         |     32 |        0.318 |      0.817 |                  0.811 |                 5.119 |               7.006 |                      -0.013 |                     0.807 |
| Social (LSTM + yielding + human-fitted limits) |     32 |        0.28  |      0.772 |                  0.855 |                 6.208 |               8.778 |                      -0.01  |                     0.815 |
| Human driver                                   |     10 |        0.238 |      1.229 |                  1.095 |                 3.151 |               1.974 |                      -0.025 |                     0     |

Burden on people (reactive crowds only): person-seconds pushed > 0.2 m off their path, displacement integral (m*s), mean peak displacement (m):

| setup                                          |   people_displaced_s |   people_displacement_ms |   mean_peak_displacement_m |
|:-----------------------------------------------|---------------------:|-------------------------:|---------------------------:|
| Nav2                                           |                 0.41 |                     4.12 |                       0.13 |
| Nav2 + LSTM prediction                         |                 0.47 |                     4.33 |                       0.13 |
| Social (LSTM + yielding + human-fitted limits) |                 0.48 |                     5.05 |                       0.14 |
| Human driver                                   |               nan    |                   nan    |                     nan    |

Human-likeness distance per feature (Wasserstein / human IQR; lower is more human-like):

| setup                                          |   W_speed |   W_accel |   W_jerk |   W_turn_rate |   W_robot_approach_close |   W_pass_clearance |
|:-----------------------------------------------|----------:|----------:|---------:|--------------:|-------------------------:|-------------------:|
| Nav2                                           |      0.6  |      1.37 |     0.99 |          0.43 |                     0.22 |               1.32 |
| Nav2 + LSTM prediction                         |      0.45 |      1.48 |     1.02 |          0.49 |                     0.1  |               1.31 |
| Social (LSTM + yielding + human-fitted limits) |      0.26 |      1.53 |     0.98 |          0.52 |                     0.25 |               1.35 |

## Nav2 + LSTM prediction vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: +0.090 (+12%), 95% CI [-0.058, +0.229]
- heading_change_per_m: +0.131 (+19%), 95% CI [+0.074, +0.181]
- stop_starts_per_min: +0.443 (+9%), 95% CI [-0.275, +1.238]
- time_ttc_under_2s: -0.212 (-3%), 95% CI [-1.622, +1.225]
- people_displaced_s: +0.062 (+15%), 95% CI [-0.134, +0.314]
- people_displacement_ms: +0.209 (+5%), 95% CI [-0.281, +0.806]

## Social (LSTM + yielding + human-fitted limits) vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: +0.044 (+6%), 95% CI [-0.148, +0.296]
- heading_change_per_m: +0.176 (+26%), 95% CI [+0.085, +0.279]
- stop_starts_per_min: +1.532 (+33%), 95% CI [+0.759, +2.308]
- time_ttc_under_2s: +1.559 (+22%), 95% CI [+0.166, +2.944]
- people_displaced_s: +0.069 (+17%), 95% CI [-0.142, +0.322]
- people_displacement_ms: +0.927 (+22%), 95% CI [+0.362, +1.551]
