# Power Plant Energy Output Prediction

CLA-3 Machine Learning case study: a **regression** problem.

A combined cycle power plant produces less electricity on hot days than on cold ones. Using only four readings around the plant (air temperature, ambient pressure, relative humidity and the steam turbine's exhaust vacuum), how accurately can we predict the plant's net hourly electrical output in MW, and which reading matters most?

## Dataset
UCI Machine Learning Repository: [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant). It has 9,568 hourly readings from 2006–2011, recorded with the plant at full load.

| Column | Meaning | Range |
|---|---|---|
| AT | Ambient temperature (°C) | 1.81 – 37.11 |
| V | Exhaust vacuum (cm Hg) | 25.36 – 81.56 |
| AP | Ambient pressure (mbar) | 992.89 – 1033.30 |
| RH | Relative humidity (%) | 25.56 – 100.16 |
| **PE** | **Net hourly energy output (MW), the target** | 420.26 – 495.76 |

The Excel file has 5 sheets with the same rows shuffled 5 times, provided for the paper's 5×2 cross-validation. I use Sheet1 and drop its 41 exact duplicate rows, which leaves 9,527.

## Results (1,906 unseen test hours)
| Model | MAE (MW) | RMSE (MW) | R² |
|---|---|---|---|
| Baseline (predict mean) | 14.973 | 17.143 | 0.0000 |
| Simple Linear Regression (temperature only) | 4.283 | 5.420 | 0.9000 |
| Linear Regression | 3.621 | 4.587 | 0.9284 |
| Polynomial Regression (deg 2) | 3.401 | 4.346 | 0.9357 |
| K-Nearest Neighbours | 2.755 | 3.872 | 0.9490 |
| SVR (RBF kernel) | 2.991 | 4.056 | 0.9440 |
| Decision Tree | 3.141 | 4.707 | 0.9246 |
| Gradient Boosting | 3.007 | 3.984 | 0.9460 |
| Random Forest | 2.406 | 3.431 | 0.9599 |
| **Random Forest (tuned)** | **2.331** | **3.334** | **0.9622** |

- The tuned Random Forest's average error is **0.52%** of the output, and **90.7%** of its predictions are within ±5 MW.
- **Temperature matters most.** Shuffling it raises the error by 14.04 MW; exhaust vacuum comes next at 4.50 MW.
- With the paper's own 5×2 CV protocol, the Random Forest's mean RMSE is **3.409 MW**, against 4.562 MW for Linear Regression.

## Course topics covered
| Topic | Where in the code |
|---|---|
| Training / testing / validation, cross-validation | 80/20 split, 5-fold CV, the paper's 5×2 CV |
| Simple, multiple and polynomial linear regression | Step 4 (simple: `PE = 497.053 − 2.172·AT`) |
| Evaluating regression fit | MAE, RMSE, R², residual plot |
| Decision trees; overfitting vs underfitting | Step 10: depth sweep (best depth 8; full depth overfits) |
| K-nearest neighbours | Step 4 |
| Feature selection (univariate and multivariate) | Correlations, permutation importance, feature subsets |
| PCA feature reduction | Step 11: 3 components keep 97.39% of the variance but raise the RMSE |
| SVM with a kernel | Step 4: SVR with an RBF kernel |

The full output is in `outputs/results_log.txt`.

## How to run
```bash
pip install -r requirements.txt
python power_output_prediction.py
```
A full run takes about 3–4 minutes, mostly for the grid search. The random seed is fixed (42), so every run gives the same numbers. Tested with Python 3.13.

## Folder structure
```
├── data/
│   ├── Folds5x2_pp.xlsx           dataset (UCI)
│   └── CCPP_Readme.txt            original dataset notes
├── outputs/                       figures, tables, log (the ~400 MB trained model is not uploaded;
│                                  running the script recreates it)
├── power_output_prediction.py     main code
├── requirements.txt
└── README.md
```

## Reference
P. Tüfekci, "Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods," *International Journal of Electrical Power & Energy Systems*, vol. 60, pp. 126–140, 2014. doi: 10.1016/j.ijepes.2014.02.027
