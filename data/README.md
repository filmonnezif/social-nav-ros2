# Data

Raw datasets are not committed.

| Dataset | Used for | Source | Licence |
| --- | --- | --- | --- |
| THÖR-MAGNI (Scenario 3) | Pedestrian predictor, reaction calibration, cloned driver, human-driver reference, Gazebo scenes | [Zenodo 10.5281/zenodo.10407223](https://doi.org/10.5281/zenodo.10407223) | CC-BY-4.0 |

Download the THÖR-MAGNI CSVs and maps from Zenodo, then:

```bash
python learning/data/thor_magni.py <THOR_MAGNI>/CSVs_Scenarios/Scenario_3/<run>.csv tracks.csv   # cleaned tracks
python learning/data/make_dataset.py <tracks_dir> predictor.mat                                # predictor snippets
python learning/data/make_bc_dataset.py <tracks_dir> bc_out/                                   # cloned-driver samples
```

The processed training files are attached to the `training-data-v1` release.

## Future work

[SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html) (human-teleoperated robots in public spaces) and [MuSoHu](https://arxiv.org/abs/2303.14880) (humans navigating crowds with robot-like sensors) could give a driver whose speed and steering depend on the people around it. SCAND's repository could not be reached during this work.
