# Predicting Biological Age from Neuromuscular Features Using Wearable EMG and IMU Sensor Data

📄 **Full thesis report:** [125100304_Thesis _report.pdf](125100304_Thesis%20_report.pdf) (MSc Data Science and Analytics, University College Cork)

Can the way we move tell us how old we are? This project uses muscle activity (surface EMG) and foot motion (Xsens IMU) recordings from wearable sensors to try two things: predict a person's age, and classify people as younger or older adults.

## Study overview

- **Participants:** 29 adults aged 21 to 63 (16 under 50, 13 aged 50 or over)
- **Sensors:** surface EMG on lower-limb muscles (1000 Hz) and Xsens foot-mounted IMUs (100 Hz, upsampled to 1000 Hz)
- **Tasks:** flat walking, incline walking, forward stepping and lateral stepping
- **Features:** foot range of motion and peak angles (pitch and roll), stride-timing variability, muscle co-contraction indices (knee and ankle), and muscle synergy VAF (variance accounted for). Ratios between tasks are also used.
- **Validation:** leave-one-out cross-validation (LOOCV). Feature selection is nested inside each fold so the held-out participant never influences model building.

## Key results

### Younger (<50) vs older (≥50) classification

| Model | Accuracy | AUC | Permutation p |
|---|---|---|---|
| Logistic Regression (3 features) | **62.1%** | **0.68** (95% CI 0.45 to 0.88) | **0.032** |
| XGBoost (3 features) | 37.9% | 0.36 | 0.584 |

Logistic regression separated the two age groups better than chance, and a permutation test confirmed it (p = 0.032). This is the strongest result in the thesis. It uses a fixed set of three features chosen on the whole sample, though. When the features are re-selected inside each fold, accuracy drops to 48.3% (AUC 0.50), which is chance level. XGBoost did worse than chance (training accuracy 79% vs 38% under LOOCV), which shows clear overfitting.

### Predicting age in years (nested LOOCV)

| Model | MAE (years) | Pearson r | Wilcoxon p vs baseline |
|---|---|---|---|
| Baseline (predict the mean) | 14.47 | n/a | n/a |
| Lasso (top 3 features) | 13.47 | 0.21 | 0.38 |
| Ridge (top 3 features) | 13.58 | 0.19 | 0.44 |
| Symbolic regression (gplearn) | 13.72 | 0.16 | 0.38 |
| Gaussian Process (Matérn) | 13.92 | 0.22 | 0.61 |
| Linear (top 3 features) | 14.25 | 0.19 | 0.92 |

No regression model was significantly better than simply predicting the mean age.

### Finding: data leakage

When features were chosen once using the whole dataset, the models looked promising (linear model: MAE 12.3 years, r = 0.45, p = 0.014). After feature selection was moved inside each cross-validation fold, the improvement disappeared. This shows how easily leakage can inflate results on small biomechanical datasets.

### Most informative features

| Feature | Spearman ρ with age | p |
|---|---|---|
| Peak foot roll, incline walking | -0.49 | 0.008 |
| Stride-timing variability, flat walking | +0.49 | 0.032 |
| Foot roll ROM ratio, step-forward vs walk | +0.48 | 0.008 |
| Peak foot roll, flat walking | -0.42 | 0.025 |

### Further analyses

- **Data augmentation** (Gaussian noise, SMOTE, GMM) lowered regression MAE to about 11.9 to 12.5 years. Most of these gains were not statistically significant.
- **Bias correction** showed strong regression-to-the-mean bias in every model (bias correlation -0.74 to -0.91).
- **Group interaction models** looked at whether feature-age relationships differ between younger and older adults.

## Repository structure

```
├── MATLAB Code/      Original MATLAB pipeline: data structuring, EMG/IMU processing,
│                     muscle synergy extraction (NMF vs PCA), co-contraction (DMCI)
├── Python_Code/
│   ├── 01_Pipeline_and_Data_prep.py          Raw EMG/IMU loading, filtering, stride segmentation
│   ├── 02_feature_engineering.py             Feature table and age correlations
│   ├── 03_baseline_and_symbolic_regression.py  Baseline, Linear/Ridge/Lasso, gplearn (nested LOOCV)
│   ├── 04_gaussian_process_regression.py     GPR with RBF and Matérn kernels
│   ├── 05_bias_correction_and_diagnostics.py Bias correction, Bland-Altman, statistical tests
│   ├── 06_group_interaction_model.py         Age-group × feature interaction models
│   ├── 07_classification.py                  Younger vs older classification (LR, XGBoost)
│   ├── 08_lr_improvement_attempts.py         Logistic regression variants
│   ├── 09_data_augmentation.py               Noise, SMOTE and GMM augmentation
│   └── final_consolidation.py                Combines results into summary tables
├── Outputs/          Result CSVs (metrics, per-participant predictions, statistical tests)
└── Plots/            Bland-Altman plots and GPR kernel comparison
```

## Data

The raw sensor data (about 3.2 GB) is **not included** in this repository, to protect participant privacy. The `Outputs/` folder holds derived, de-identified results, where participants are referred to only by number.

## Running the code

```bash
pip install -r requirements.txt
```

The scripts use absolute file paths (for example `DATA_ROOT`, `IN_PATH`, `OUT_PATH` at the top of each script). Change these to point to your local copy of the data, then run the scripts in numerical order.

## Limitations

- The sample is small (n = 29), so the confidence intervals are wide and the statistical power is limited.
- Some trials were excluded or corrected because of sensor dropout or mislabelled files (documented in `01_Pipeline_and_Data_prep.py`).
- The results need to be tested on a larger, independent cohort.

## Author

Deepak Babu P K
