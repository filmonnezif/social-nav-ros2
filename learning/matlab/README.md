# Predictor training (MATLAB)

`train_predictor.m` trains and evaluates the pedestrian trajectory predictors for H1 and H2.

1. Build the data file (Python, a few seconds):
   `python learning/data/make_dataset.py <tracks_dir> thor_magni_sc3_snippets.mat`
2. Put `train_predictor.m` and `thor_magni_sc3_snippets.mat` in the same folder of MATLAB Drive.
3. Open MATLAB Online, go to that folder, run `train_predictor`. CPU is enough; set `QUICK = true` for a one-minute smoke test.

Needs the Deep Learning Toolbox (R2024a or newer for `trainnet` / `minibatchpredict`).

Results land in `results_predictor/`: `metrics.csv`, `metrics_per_run.csv`, `ade_fde.png`, `examples.png`, and `weights_*.mat` (plain LSTM weights the ROS 2 node loads with numpy).
