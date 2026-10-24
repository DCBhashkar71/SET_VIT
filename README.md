# Between-subject baseline shift in wearable stress detection: analysis code

Code for the study on subject-independent stress detection with WESAD (chest ECG, EDA, temperature) and Stress-Predict
(wrist heart rate, EDA, temperature). All analyses use leave-one-subject-out validation with participant-level inference.

## Data (not included)
* WESAD: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html  (put `S2.pkl` ... `S17.pkl` in `data/wesad_raw/`)
* Stress-Predict: dataset released with Iqbal et al., Sensors 2022, 22(21):8135 (unzip to `data/Stress-Predict-Dataset-main/`)
* SWELL-KW derived release (Kaggle `qiriro/swell-heart-rate-variability-hrv`; `data/swell/train.csv`, `test.csv`)
* Room Occupancy Detection (Kaggle `kukuroo3/room-occupancy-detection-data-iot-sensor`; `data/room_occupancy.csv`)

## Requirements
Python 3.11, see `requirements.txt` (`neurokit2` is optional and only used for the detector-agreement check).

## Run order (from the project folder)
```
python extract_wesad.py      --raw data\wesad_raw --out outputs_revised
python run_experiments.py    --features outputs_revised\wesad_features.csv --room data\room_occupancy.csv --out outputs_revised
python cost_profile_only.py  --features outputs_revised\wesad_features.csv --out outputs_revised
python extra_experiments.py  --features outputs_revised\wesad_features.csv --out outputs_extra --loso-per-subject outputs_revised\loso_per_subject.csv
python extra_experiments.py  --features outputs_revised\wesad_features.csv --out outputs_extra_stack --stacking --only e2 e5
python sensitivity_fairness.py --features outputs_revised\wesad_features.csv --out outputs_sens
python build_stress_predict.py --root data\Stress-Predict-Dataset-main --out outputs_sp
python extra_experiments.py  --features outputs_sp\stress_predict_features.csv --cols hr eda temp_wrist --tag stress_predict --out outputs_sp_extra --only e1 e2 e3 e5 --n-cals 5 10 25 --models LR
python combine_datasets.py   --a outputs_extra\e2_calibration_per_subject.csv --b outputs_sp_extra\e2_calibration_per_subject.csv --names WESAD StressPredict --out outputs_pooled
python cross_dataset.py      --wesad-raw data\wesad_raw --swell data\swell --out outputs_revised
```
Window-length sensitivity: `python extract_wesad.py --raw data\wesad_raw --out sens_45 --win 45 --step 22.5` then
`python extra_experiments.py --features sens_45\wesad_features.csv --out sens_45_out --only e1 e3`.

## Figures
```
python make_figures.py          --out outputs_revised
python make_figures_extra.py    --extra outputs_extra --features outputs_revised\wesad_features.csv --main outputs_revised
python make_figures_final.py    --features outputs_revised\wesad_features.csv --oof outputs_revised\oof_predictions.csv --sp outputs_sp\stress_predict_features.csv --out figs_final
python make_manuscript_figures.py            (or open make_manuscript_figures.ipynb)
```

## Files
| File | Purpose |
|---|---|
| `stresslib.py` | ECG processing, models, leakage-controlled stacking, participant-level statistics, profiling |
| `extract_wesad.py` | WESAD feature extraction with a per-window quality log |
| `run_experiments.py` | main comparison, cascade, sensor failure, personalisation, alarms, environmental negative control |
| `extra_experiments.py` | baseline shift, calibration, smoothing, learning curve, protocol variants, power |
| `sensitivity_fairness.py` | class-weighting and hyper-parameter-tuning sensitivity |
| `build_stress_predict.py` | harmonises Stress-Predict into 30 s windows |
| `combine_datasets.py` | pooled calibration estimate across both datasets |
| `cross_dataset.py` | exploratory WESAD / SWELL-KW transfer |
| `cost_profile_only.py` | single-thread latency and size of fitted models |
| `make_*figures*.py/.ipynb` | figure generation |

Seeds, software versions and participant lists are written to `run_manifest.json` by `run_experiments.py`.

## Citation
To be added.
