# Smoothness, comfort and human-likeness

| setup                  |   runs |   mean_speed |   rms_jerk |   heading_change_per_m |   stop_starts_per_min |   time_ttc_under_2s |   mean_robot_approach_close |   human_likeness_distance |
|:-----------------------|-------:|-------------:|-----------:|-----------------------:|----------------------:|--------------------:|----------------------------:|--------------------------:|
| Nav2                   |     32 |        0.36  |      0.974 |                  0.78  |                 4.235 |               6.597 |                      -0.023 |                     0.797 |
| Nav2 + CV prediction   |     32 |        0.334 |      0.944 |                  0.86  |                 5.75  |               6.609 |                      -0.042 |                     0.839 |
| Nav2 + LSTM prediction |     32 |        0.327 |      1.052 |                  0.928 |                 5.839 |               6.819 |                      -0.025 |                     0.87  |
| Human driver           |     10 |        0.238 |      1.229 |                  1.095 |                 3.151 |               1.974 |                      -0.025 |                     0     |

Human-likeness distance per feature (Wasserstein / human IQR; lower is more human-like):

| setup                  |   W_speed |   W_accel |   W_jerk |   W_turn_rate |   W_robot_approach_close |   W_pass_clearance |
|:-----------------------|----------:|----------:|---------:|--------------:|-------------------------:|-------------------:|
| Nav2                   |      0.62 |      1.3  |     0.9  |          0.4  |                     0.24 |               1.32 |
| Nav2 + CV prediction   |      0.54 |      1.61 |     1.06 |          0.41 |                     0.12 |               1.29 |
| Nav2 + LSTM prediction |      0.47 |      1.77 |     1.11 |          0.48 |                     0.09 |               1.3  |

## Nav2 + CV prediction vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: -0.030 (-3%), 95% CI [-0.345, +0.240]
- heading_change_per_m: +0.079 (+10%), 95% CI [-0.062, +0.193]
- stop_starts_per_min: +1.515 (+36%), 95% CI [+0.704, +2.387]
- time_ttc_under_2s: +0.012 (+0%), 95% CI [-1.056, +1.056]

## Nav2 + LSTM prediction vs Nav2 (paired over 32 scene/direction pairs)

- rms_jerk: +0.078 (+8%), 95% CI [-0.301, +0.454]
- heading_change_per_m: +0.148 (+19%), 95% CI [-0.010, +0.294]
- stop_starts_per_min: +1.604 (+38%), 95% CI [+0.632, +2.636]
- time_ttc_under_2s: +0.222 (+3%), 95% CI [-1.050, +1.566]
