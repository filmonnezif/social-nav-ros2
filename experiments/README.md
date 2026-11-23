# Experiments

| Script | Does |
| --- | --- |
| `rank_windows.py` | Scores every 110 s window of every THÖR-MAGNI recording by tracking quality |
| `build_scene.py` | Turns a window into a Gazebo world, a Nav2 map and a people file (`--people actors\|bodies`) |
| `make_nav2_params.py` | Writes the Nav2 parameter files for each set-up from the stock defaults |
| `run_episode.sh` | Runs, records (rosbag + MP4) and scores one episode in the container |
| `run_grid.py` | Runs a grid of scenes × directions × set-ups; resumable (skips runs with `metrics.json`) |
| `render_run.py` | Metrics and the demo video from a bag |
| `motion_metrics.py`, `motion_from_bags.py` | Smoothness, burden and human-likeness metrics |
| `human_reference.py` | The same motion features for the human-driven robot in THÖR-MAGNI |
| `analyze_h3.py`, `analyze_motion.py` | Summary tables, paired bootstrap CIs and plots for a grid |
| `tune_governor.sh` | Small sweep used to pick the yielding governor's profile |

Each episode directory holds `metrics.json`, `motion.json`, `episode.json` and, locally, the bag and video (these go to Releases, not git).
