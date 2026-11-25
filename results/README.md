# Results

Summary tables, per-episode metrics and plots are committed here. Full rosbags and videos are attached to the GitHub Releases.

| Folder | What |
| --- | --- |
| `predictor/` | Predictor test metrics, exported weights (MAT, ONNX), figures |
| `reaction/` | Reaction-model fits: the direct force fit (failed), occupancy-only, final, half-strength |
| `driver/` | Cloned-driver offline metrics and weights (BC-blind, BC-social, BC-nocopy, BC-people) |
| `human_reference/` | Motion statistics of the human-driven robot in THÖR-MAGNI Scenario 3A |
| `final_grid/` | **Final experiment**: DARKO-sized robot, reactive crowd, 6 set-ups × 32 runs (learned = BC v2) |
| `final_grid_learned_copycat/` | BC v1 (own velocity as input) on the same 32 runs |
| `final_grid_half/` | Sensitivity: people react at half strength (Nav2, LSTM, Social) |
| `h3_grid/` | Earlier experiment: TurtleBot-sized robot among non-reacting people (4 set-ups × 32 runs) |
| `tuning/` | Yielding-governor profile sweep |
| `clean_run/`, `actors_run_v2/`, `first_run/`, `actors_run/` | First single episodes (walking figures, data cleaning) |
| `*_smoke/` | Smoke tests of each set-up |

Each grid has `summary.md` (success, proximity, paired CIs) and `motion_summary.md` (smoothness, burden, human-likeness).
