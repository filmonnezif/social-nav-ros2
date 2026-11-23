# Smoothness, comfort and human-likeness

| setup                                          |   runs |   mean_speed |   rms_jerk |   heading_change_per_m |   stop_starts_per_min |   time_ttc_under_2s |   mean_robot_approach_close |   human_likeness_distance |
|:-----------------------------------------------|-------:|-------------:|-----------:|-----------------------:|----------------------:|--------------------:|----------------------------:|--------------------------:|
| Nav2                                           |     32 |        0.353 |      0.785 |                  0.697 |                 4.842 |               5.975 |                      -0.005 |                     0.806 |
| Nav2 blind to people (bulldozer)               |     32 |        0.393 |      0.741 |                  0.575 |                 3.277 |               5.647 |                       0.012 |                     0.804 |
| Nav2 + CV prediction                           |     32 |        0.327 |      0.856 |                  0.787 |                 5.033 |               6.909 |                      -0.021 |                     0.784 |
| Nav2 + LSTM prediction                         |     32 |        0.32  |      0.885 |                  0.822 |                 5.277 |               6.334 |                      -0.007 |                     0.752 |
| Social (LSTM + yielding + human-fitted limits) |     32 |        0.277 |      0.774 |                  0.854 |                 6.858 |               9.156 |                      -0.003 |                     0.786 |
| Learned driver (LSTM + cloned human speed)     |     32 |        0.168 |      0.533 |                  1.255 |                 6.567 |              11.431 |                      -0.024 |                     0.746 |
| Human driver                                   |     10 |        0.238 |      1.229 |                  1.095 |                 3.151 |               1.974 |                      -0.025 |                     0     |

Burden on people (reactive crowds only): person-seconds pushed > 0.2 m off their path, displacement integral (m*s), mean peak displacement (m):

| setup                                          |   people_displaced_s |   people_displacement_ms |   mean_peak_displacement_m |
|:-----------------------------------------------|---------------------:|-------------------------:|---------------------------:|
| Nav2                                           |                 9.49 |                     7.73 |                       0.24 |
| Nav2 blind to people (bulldozer)               |                 8.43 |                     6.84 |                       0.24 |
| Nav2 + CV prediction                           |                 9.7  |                     8.12 |                       0.24 |
| Nav2 + LSTM prediction                         |                 9.28 |                     8.08 |                       0.24 |
| Social (LSTM + yielding + human-fitted limits) |                12.49 |                    10.17 |                       0.26 |
| Learned driver (LSTM + cloned human speed)     |                16.82 |                    14.3  |                       0.28 |
| Human driver                                   |               nan    |                   nan    |                     nan    |

Human-likeness distance per feature (Wasserstein / human IQR; lower is more human-like):

| setup                                          |   W_speed |   W_accel |   W_jerk |   W_turn_rate |   W_robot_approach_close |   W_pass_clearance |
|:-----------------------------------------------|----------:|----------:|---------:|--------------:|-------------------------:|-------------------:|
| Nav2                                           |      0.66 |      1.33 |     0.99 |          0.4  |                     0.23 |               1.23 |
| Nav2 blind to people (bulldozer)               |      0.9  |      1.06 |     1    |          0.47 |                     0.3  |               1.09 |
| Nav2 + CV prediction                           |      0.49 |      1.49 |     0.98 |          0.47 |                     0.09 |               1.18 |
| Nav2 + LSTM prediction                         |      0.44 |      1.43 |     0.98 |          0.44 |                     0.12 |               1.09 |
| Social (LSTM + yielding + human-fitted limits) |      0.26 |      1.59 |     0.93 |          0.43 |                     0.27 |               1.23 |
| Learned driver (LSTM + cloned human speed)     |      0.45 |      1.06 |     0.92 |          0.56 |                     0.29 |               1.2  |

## Nav2 blind to people (bulldozer) vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: -0.044 (-6%), 95% CI [-0.175, +0.078]
- heading_change_per_m: -0.122 (-17%), 95% CI [-0.184, -0.061]
- stop_starts_per_min: -1.565 (-32%), 95% CI [-2.448, -0.736]
- time_ttc_under_2s: -0.328 (-5%), 95% CI [-1.006, +0.275]
- people_displaced_s: -1.057 (-11%), 95% CI [-2.346, +0.571]
- people_displacement_ms: -0.895 (-12%), 95% CI [-1.599, -0.117]

## Nav2 + CV prediction vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: +0.070 (+9%), 95% CI [-0.076, +0.234]
- heading_change_per_m: +0.089 (+13%), 95% CI [+0.028, +0.147]
- stop_starts_per_min: +0.191 (+4%), 95% CI [-0.522, +0.930]
- time_ttc_under_2s: +0.934 (+16%), 95% CI [+0.147, +1.753]
- people_displaced_s: +0.208 (+2%), 95% CI [-0.951, +1.576]
- people_displacement_ms: +0.383 (+5%), 95% CI [-0.175, +1.024]

## Nav2 + LSTM prediction vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: +0.100 (+13%), 95% CI [-0.079, +0.273]
- heading_change_per_m: +0.125 (+18%), 95% CI [+0.061, +0.186]
- stop_starts_per_min: +0.435 (+9%), 95% CI [-0.394, +1.249]
- time_ttc_under_2s: +0.359 (+6%), 95% CI [-0.456, +1.184]
- people_displaced_s: -0.213 (-2%), 95% CI [-1.478, +1.025]
- people_displacement_ms: +0.351 (+5%), 95% CI [-0.213, +0.921]

## Social (LSTM + yielding + human-fitted limits) vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: -0.011 (-1%), 95% CI [-0.164, +0.133]
- heading_change_per_m: +0.157 (+23%), 95% CI [+0.073, +0.244]
- stop_starts_per_min: +2.017 (+42%), 95% CI [+1.014, +3.067]
- time_ttc_under_2s: +3.181 (+53%), 95% CI [+1.828, +4.669]
- people_displaced_s: +3.001 (+32%), 95% CI [+1.153, +5.037]
- people_displacement_ms: +2.437 (+32%), 95% CI [+1.375, +3.594]

## Learned driver (LSTM + cloned human speed) vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: -0.253 (-32%), 95% CI [-0.381, -0.150]
- heading_change_per_m: +0.558 (+80%), 95% CI [+0.476, +0.643]
- stop_starts_per_min: +1.725 (+36%), 95% CI [+0.637, +2.803]
- time_ttc_under_2s: +5.456 (+91%), 95% CI [+3.828, +7.059]
- people_displaced_s: +7.332 (+77%), 95% CI [+5.218, +9.367]
- people_displacement_ms: +6.571 (+85%), 95% CI [+5.266, +7.739]
