"""
CLA-3 Case Study
Power Plant Energy Output Prediction
-------------------------------------
Problem type : REGRESSION - predict the net hourly electrical energy output (MW)
               of a Combined Cycle Power Plant from four ambient readings.
Dataset      : UCI "Combined Cycle Power Plant" (9568 hourly records, 2006-2011)
               https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
Base paper   : P. Tufekci, "Prediction of full load electrical power output of a
               base load operated combined cycle power plant using machine learning
               methods", Int. J. Electrical Power & Energy Systems, 60, 126-140, 2014.

Question I want to answer:
    "Using only the weather around the plant (temperature, pressure, humidity)
     and the steam-turbine exhaust vacuum, how accurately can we predict the
     plant's hourly power output - and which reading matters most?"

How to run:
    pip install -r requirements.txt
    python power_output_prediction.py
Everything (tables, figures, a text log, saved model) goes into the "outputs" folder.
"""

import os
import sys
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"  # keeps parallel worker processes quiet too

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "Folds5x2_pp.xlsx")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")


class Tee:
    """Shows output on screen and also saves it to outputs/results_log.txt."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


sys.stdout = Tee(os.path.join(OUT_DIR, "results_log.txt"))


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


NAMES = {"AT": "Temperature (C)", "V": "Exhaust vacuum (cm Hg)",
         "AP": "Ambient pressure (mbar)", "RH": "Relative humidity (%)", "PE": "Energy output (MW)"}
FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
banner("STEP 1 : Loading the dataset")
sheets = pd.read_excel(DATA_PATH, sheet_name=None)
print(f"Sheets in the Excel file : {list(sheets)}")
same = all(s.sort_values(FEATURES + [TARGET]).reset_index(drop=True)
           .equals(sheets["Sheet1"].sort_values(FEATURES + [TARGET]).reset_index(drop=True))
           for s in sheets.values())
print(f"All 5 sheets hold the same rows, only shuffled differently : {same}")
print("-> I use Sheet1 for the main experiment (and all 5 sheets later for the paper's 5x2 CV).")

df = sheets["Sheet1"].copy()
print(f"\nRows    : {len(df)}")
print(f"Columns : {list(df.columns)}")
print(f"Missing values       : {int(df.isna().sum().sum())}")
print(f"Exact duplicate rows : {int(df.duplicated().sum())}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Rows after removing duplicates : {len(df)}")

print("\nSummary statistics:")
print(df.describe().T.round(2).to_string())

# ------------------------------------------------------------------
# 2. Explore
# ------------------------------------------------------------------
banner("STEP 2 : Exploring the data")
corr = df.corr()
print("Pearson correlation matrix:")
print(corr.round(3).to_string())

df["temp_band"] = pd.cut(df["AT"], bins=[0, 10, 20, 30, 40], labels=["<10 C", "10-20 C", "20-30 C", ">30 C"])
print("\nAverage power output by temperature band:")
print(df.groupby("temp_band", observed=True)["PE"].agg(["count", "mean", "min", "max"]).round(2).to_string())
df = df.drop(columns="temp_band")

# Figure 1: each input vs output
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]
for ax, f, c in zip(axes, FEATURES, colors):
    ax.scatter(df[f], df[TARGET], s=3, alpha=0.25, color=c)
    ax.set_xlabel(NAMES[f])
    ax.set_ylabel(NAMES[TARGET])
    ax.set_title(f"r = {corr.loc[f, TARGET]:.3f}")
fig.suptitle("How each ambient reading relates to the plant's energy output (n = 9527)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_features_vs_output.png"), dpi=150)
plt.close(fig)

# Figure 2: correlation heat-map
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr.rename(index=NAMES, columns=NAMES), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation between variables")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_correlation_heatmap.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 3. Train / test split
# ------------------------------------------------------------------
banner("STEP 3 : Train / test split")
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
print(f"Training rows : {len(X_train)}")
print(f"Test rows     : {len(X_test)}")

# ------------------------------------------------------------------
# 4. Compare models
# ------------------------------------------------------------------
banner("STEP 4 : Comparing regression models (5-fold CV on train + hold-out test)")
models = {
    "Baseline (predict mean)": DummyRegressor(),
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Polynomial Regression (deg 2)": make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),
    "K-Nearest Neighbours (k=5)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    # Support Vector Regression with an RBF kernel (SVM idea: fit inside an epsilon-tube, maximum margin)
    "SVR (RBF kernel)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, epsilon=1.0, gamma="scale")),
    "Decision Tree": DecisionTreeRegressor(random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []

# Simple linear regression: one input only (temperature), PE = b0 + b1 * AT
slr = LinearRegression()
cv_slr = -cross_val_score(slr, X_train[["AT"]], y_train, cv=cv, scoring="neg_root_mean_squared_error")
slr.fit(X_train[["AT"]], y_train)
p_slr = slr.predict(X_test[["AT"]])
rows.append({"Model": "Simple Linear Regression (AT only)", "CV RMSE": round(cv_slr.mean(), 3),
             "Test MAE": round(mean_absolute_error(y_test, p_slr), 3),
             "Test RMSE": round(rmse(y_test, p_slr), 3), "Test R2": round(r2_score(y_test, p_slr), 4)})
print(f"Simple linear regression equation : PE = {slr.intercept_:.3f} {slr.coef_[0]:+.3f} x AT")
print(f"   (each extra 1 C of air temperature changes the output by {slr.coef_[0]:.3f} MW)")
r = rows[-1]
print(f"{'Simple Linear Regression (AT)':<31} CV-RMSE={r['CV RMSE']:.3f}  MAE={r['Test MAE']:.3f}  "
      f"RMSE={r['Test RMSE']:.3f}  R2={r['Test R2']:.4f}")

for name, model in models.items():
    t0 = time.time()
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({"Model": name, "CV RMSE": round(cv_rmse.mean(), 3),
                 "Test MAE": round(mean_absolute_error(y_test, pred), 3),
                 "Test RMSE": round(rmse(y_test, pred), 3),
                 "Test R2": round(r2_score(y_test, pred), 4)})
    r = rows[-1]
    print(f"{name:<31} CV-RMSE={r['CV RMSE']:.3f}  MAE={r['Test MAE']:.3f}  "
          f"RMSE={r['Test RMSE']:.3f}  R2={r['Test R2']:.4f}  ({time.time() - t0:.1f}s)")

lin = models["Linear Regression"].named_steps["linearregression"]
print("\nLinear Regression coefficients (on standardised inputs, MW per 1 std):")
for f, c in zip(FEATURES, lin.coef_):
    print(f"   {NAMES[f]:<26} {c:+.3f}")
print(f"   Intercept                  {lin.intercept_:.3f}")

# ------------------------------------------------------------------
# 5. Tune the Random Forest
# ------------------------------------------------------------------
banner("STEP 5 : Tuning the Random Forest with GridSearchCV")
grid = GridSearchCV(
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    {"n_estimators": [300, 600], "max_features": [1, 2, 4], "min_samples_leaf": [1, 2, 4]},
    cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
pred = best.predict(X_test)
err = y_test - pred
print(f"Best parameters : {grid.best_params_}")
print(f"Best CV RMSE    : {-grid.best_score_:.3f}")
print(f"Test MAE        : {mean_absolute_error(y_test, pred):.3f} MW")
print(f"Test RMSE       : {rmse(y_test, pred):.3f} MW")
print(f"Test R2         : {r2_score(y_test, pred):.4f}")
print(f"Mean absolute % error : {np.mean(np.abs(err) / y_test) * 100:.3f}%")
for k in (2, 5, 10):
    print(f"Predictions within +/-{k} MW : {np.mean(np.abs(err) <= k) * 100:.1f}%")
rows.append({"Model": "Random Forest (tuned)", "CV RMSE": round(-grid.best_score_, 3),
             "Test MAE": round(mean_absolute_error(y_test, pred), 3),
             "Test RMSE": round(rmse(y_test, pred), 3), "Test R2": round(r2_score(y_test, pred), 4)})
results = pd.DataFrame(rows)
# keep the baseline as the first row of the table
results = pd.concat([results[results.Model == "Baseline (predict mean)"],
                     results[results.Model != "Baseline (predict mean)"]]).reset_index(drop=True)
results.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
banner("FINAL COMPARISON TABLE (test set = 1906 hours)")
print(results.to_string(index=False))

# Figure 3: model comparison
plot_df = results[results.Model != "Baseline (predict mean)"]
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.barplot(data=plot_df, y="Model", x="Test RMSE", ax=ax, color="#4C72B0")
ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
ax.set_xlabel("Test RMSE (MW) - lower is better")
ax.set_ylabel("")
ax.set_title(f"Test error of each model (baseline that predicts the mean: "
             f"{results.iloc[0]['Test RMSE']:.2f} MW)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_model_comparison.png"), dpi=150)
plt.close(fig)

# Figure 4: actual vs predicted + residuals
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_test, pred, s=6, alpha=0.4)
lims = [y.min() - 2, y.max() + 2]
axes[0].plot(lims, lims, "r--", label="perfect prediction")
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].set_xlabel("Actual output (MW)")
axes[0].set_ylabel("Predicted output (MW)")
axes[0].set_title(f"Actual vs predicted  (R2 = {r2_score(y_test, pred):.4f}, RMSE = {rmse(y_test, pred):.2f} MW)")
axes[0].legend()
sns.histplot(err, bins=60, ax=axes[1], color="#55A868")
axes[1].axvline(0, color="r", ls="--")
axes[1].set_xlabel("Error = actual - predicted (MW)")
axes[1].set_title("Distribution of prediction errors")
fig.suptitle("Tuned Random Forest on the 1906 unseen test hours")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_actual_vs_predicted.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 6. Which reading matters most?
# ------------------------------------------------------------------
banner("STEP 6 : Which ambient reading matters most?")
perm = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=SEED,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
imp = pd.DataFrame({"Feature": [NAMES[f] for f in FEATURES],
                    "Impurity importance": best.feature_importances_.round(4),
                    "Permutation: RMSE increase (MW)": perm.importances_mean.round(3)}
                   ).sort_values("Permutation: RMSE increase (MW)", ascending=False)
print(imp.to_string(index=False))
imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

print("\nFeature-subset study (Random Forest with the tuned settings, same split):")
subsets = [["AT"], ["V"], ["AT", "V"], ["AT", "V", "AP"], ["AT", "V", "RH"], ["AT", "V", "AP", "RH"]]
sub_rows = []
for s in subsets:
    params = dict(grid.best_params_)
    params["max_features"] = min(params["max_features"], len(s))
    m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params).fit(X_train[s], y_train)
    p = m.predict(X_test[s])
    sub_rows.append({"Inputs": " + ".join(s), "Test MAE": round(mean_absolute_error(y_test, p), 3),
                     "Test RMSE": round(rmse(y_test, p), 3), "Test R2": round(r2_score(y_test, p), 4)})
    print(f"   {sub_rows[-1]['Inputs']:<18} MAE={sub_rows[-1]['Test MAE']:.3f}  "
          f"RMSE={sub_rows[-1]['Test RMSE']:.3f}  R2={sub_rows[-1]['Test R2']:.4f}")
pd.DataFrame(sub_rows).to_csv(os.path.join(OUT_DIR, "feature_subsets.csv"), index=False)

# Figure 5: permutation importance + subsets
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.barplot(data=imp, y="Feature", x="Permutation: RMSE increase (MW)", ax=axes[0], color="#C44E52")
axes[0].bar_label(axes[0].containers[0], fmt="%.2f", padding=3)
axes[0].set_title("Permutation importance (how much RMSE grows when a reading is shuffled)")
axes[0].set_ylabel("")
sub = pd.DataFrame(sub_rows)
sns.barplot(data=sub, y="Inputs", x="Test RMSE", ax=axes[1], color="#4C72B0")
axes[1].bar_label(axes[1].containers[0], fmt="%.2f", padding=3)
axes[1].set_title("Test RMSE (MW) using only some of the readings")
axes[1].set_ylabel("")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_importance_and_subsets.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 7. One correct and one incorrect prediction
# ------------------------------------------------------------------
banner("STEP 7 : One good and one bad prediction (tuned Random Forest)")
res = X_test.copy()
res["actual"] = y_test
res["predicted"] = pred.round(2)
res["abs_error"] = np.abs(err).round(2)
good, bad = res["abs_error"].idxmin(), res["abs_error"].idxmax()
for label, i in [("CORRECT (smallest error)", good), ("INCORRECT (largest error)", bad)]:
    print(f"\n{label} - row {i}")
    print(res.loc[[i]].to_string())
print("\nFive largest errors in the test set:")
print(res.sort_values("abs_error", ascending=False).head(5).to_string())

# how unusual is the worst hour? find training hours with similar conditions
near = X_train[(abs(X_train.AT - res.loc[bad, "AT"]) < 1.0) & (abs(X_train.V - res.loc[bad, "V"]) < 1.0)]
print(f"\nTraining hours with temperature and vacuum within 1 unit of the worst-case hour: {len(near)}")
if len(near):
    print(f"   their actual output: mean {y_train.loc[near.index].mean():.2f} MW, "
          f"min {y_train.loc[near.index].min():.2f}, max {y_train.loc[near.index].max():.2f}")

# ------------------------------------------------------------------
# 8. What-if: a cold winter hour vs a hot summer hour
# ------------------------------------------------------------------
banner("STEP 8 : What-if - winter hour vs summer hour")
what_if = pd.DataFrame([
    {"AT": 8.0, "V": 40.0, "AP": 1018.0, "RH": 85.0},
    {"AT": 30.0, "V": 70.0, "AP": 1008.0, "RH": 55.0},
], index=["Cold winter hour", "Hot summer hour"])
what_if["predicted MW"] = best.predict(what_if[FEATURES]).round(2)
print(what_if.to_string())
print(f"Drop in output from winter to summer: "
      f"{what_if['predicted MW'].iloc[0] - what_if['predicted MW'].iloc[1]:.2f} MW")

# ------------------------------------------------------------------
# 9. The paper's own protocol: 5x2 cross-validation on the 5 shuffled sheets
# ------------------------------------------------------------------
banner("STEP 9 : Paper-style 5x2 cross-validation (5 shuffles x 2 folds)")
proto = {"Linear Regression": LinearRegression(),
         "Random Forest (tuned)": RandomForestRegressor(random_state=SEED, n_jobs=-1, **grid.best_params_)}
rows5 = []
for name, model in proto.items():
    maes, rmses = [], []
    for sname, s in sheets.items():
        half = len(s) // 2
        A, B = s.iloc[:half], s.iloc[half:]
        for tr, te in [(A, B), (B, A)]:
            model.fit(tr[FEATURES], tr[TARGET])
            p = model.predict(te[FEATURES])
            maes.append(mean_absolute_error(te[TARGET], p))
            rmses.append(rmse(te[TARGET], p))
    rows5.append({"Model": name, "Mean MAE": round(np.mean(maes), 3), "Mean RMSE": round(np.mean(rmses), 3),
                  "RMSE std": round(np.std(rmses), 3)})
    print(f"{name:<24} mean MAE = {np.mean(maes):.3f}   mean RMSE = {np.mean(rmses):.3f} "
          f"(std {np.std(rmses):.3f}) over 10 folds")
pd.DataFrame(rows5).to_csv(os.path.join(OUT_DIR, "paper_protocol_5x2cv.csv"), index=False)

# ------------------------------------------------------------------
# 10. Overfitting vs underfitting: decision-tree depth sweep
# ------------------------------------------------------------------
banner("STEP 10 : Overfitting and underfitting - decision tree depth")
depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
dt_rows = []
for d in depths:
    s = cross_validate(DecisionTreeRegressor(max_depth=d, random_state=SEED), X_train, y_train, cv=cv,
                       scoring="neg_root_mean_squared_error", return_train_score=True, n_jobs=-1)
    dt_rows.append({"max_depth": "None (full)" if d is None else d,
                    "Train RMSE": round(abs(s["train_score"].mean()), 3),
                    "Validation RMSE": round(-s["test_score"].mean(), 3)})
dt = pd.DataFrame(dt_rows)
dt["Gap"] = (dt["Validation RMSE"] - dt["Train RMSE"]).round(3)
print("5-fold CV on the training set (validation = held-out fold):")
print(dt.to_string(index=False))
best_row = dt.loc[dt["Validation RMSE"].idxmin()]
print(f"\nBest depth = {best_row['max_depth']} (validation RMSE {best_row['Validation RMSE']:.3f} MW)")
print("Small depth -> both errors high (underfitting). Full depth -> train error ~0 but validation "
      "error rises (overfitting).")
dt.to_csv(os.path.join(OUT_DIR, "tree_depth_overfitting.csv"), index=False)

# ------------------------------------------------------------------
# 11. Feature reduction with PCA
# ------------------------------------------------------------------
banner("STEP 11 : Feature reduction with Principal Component Analysis (PCA)")
pca_full = make_pipeline(StandardScaler(), PCA()).fit(X_train)
evr = pca_full.named_steps["pca"].explained_variance_ratio_
print("Explained variance ratio of each principal component:")
for i, v in enumerate(evr, 1):
    print(f"   PC{i}: {v * 100:6.2f}%   (cumulative {evr[:i].sum() * 100:6.2f}%)")
print("Loadings (how each standardised reading builds PC1 and PC2):")
load = pd.DataFrame(pca_full.named_steps["pca"].components_[:2].T, index=FEATURES, columns=["PC1", "PC2"])
print(load.round(3).to_string())

pca_rows = []
for k in [1, 2, 3, 4]:
    for name, reg in [("Linear Regression", LinearRegression()),
                      ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))]:
        m = make_pipeline(StandardScaler(), PCA(n_components=k), reg).fit(X_train, y_train)
        p = m.predict(X_test)
        pca_rows.append({"Components": k, "Variance kept %": round(evr[:k].sum() * 100, 2), "Model": name,
                         "Test RMSE": round(rmse(y_test, p), 3), "Test R2": round(r2_score(y_test, p), 4)})
pca_df = pd.DataFrame(pca_rows)
print("\nTest results using only the first k principal components:")
print(pca_df.to_string(index=False))
pca_df.to_csv(os.path.join(OUT_DIR, "pca_results.csv"), index=False)

# Figure 6: depth sweep + PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
xlab = [str(d) for d in dt["max_depth"]]
axes[0].plot(xlab, dt["Train RMSE"], "o-", label="Training error")
axes[0].plot(xlab, dt["Validation RMSE"], "s-", label="Validation error (5-fold CV)")
axes[0].axvspan(-0.4, 2.5, color="#DD8452", alpha=0.12)
axes[0].axvspan(8.5, 11.4, color="#C44E52", alpha=0.12)
axes[0].text(0.75, 7.6, "underfitting", color="#DD8452")
axes[0].text(9.3, 7.6, "overfitting", color="#C44E52")
axes[0].set_xlabel("Decision tree max_depth")
axes[0].set_ylabel("RMSE (MW)")
axes[0].set_title("Overfitting vs underfitting (decision tree)")
axes[0].legend(loc="upper center")
for name, mk in [("Linear Regression", "o-"), ("Random Forest", "s-")]:
    sub = pca_df[pca_df.Model == name]
    axes[1].plot(sub["Components"], sub["Test RMSE"], mk, label=name)
axes[1].set_xticks([1, 2, 3, 4])
axes[1].set_xticklabels([f"{k}\n({evr[:k].sum() * 100:.1f}% var.)" for k in [1, 2, 3, 4]])
axes[1].set_xlabel("Number of principal components kept")
axes[1].set_ylabel("Test RMSE (MW)")
axes[1].set_title("PCA feature reduction")
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_overfitting_and_pca.png"), dpi=150)
plt.close(fig)

joblib.dump(best, os.path.join(OUT_DIR, "rf_power_output_model.joblib"))
banner("DONE")
print("All tables, figures and the trained model are saved in the 'outputs' folder.")
