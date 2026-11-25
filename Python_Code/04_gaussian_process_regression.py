import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)

CONFIGS = {
    "A_RBF": lambda: ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2)),
    "B_Matern": lambda: ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5) + WhiteKernel(1.0, (1e-5, 1e2))}


def run_nested_gpr(kernel_fn, age_labels, verbose=False):
    pred_age = np.empty(N)
    pred_std = np.empty(N)
    for i in range(N):
        train_df = df.drop(index=i)
        test_row = df.iloc[[i]]
        y_train = age_labels[train_df.index.values]

        medians = train_df[FEATURES].median()
        X_train_raw = train_df[FEATURES].fillna(medians).values.astype(float)
        X_test_raw = test_row[FEATURES].fillna(medians).values.astype(float)

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        gpr = GaussianProcessRegressor(kernel=kernel_fn(), n_restarts_optimizer=10,
                                        normalize_y=True, random_state=0)
        gpr.fit(X_train, y_train)
        mean, std = gpr.predict(X_test, return_std=True)
        pred_age[i] = mean[0]
        pred_std[i] = std[0]
        if verbose:
            print(f"  [{i+1:2d}/{N}] actual={age_labels[i]:.1f} pred={mean[0]:.2f} std={std[0]:.2f}")
    return pred_age, pred_std


def metrics(pred_age, actual):
    abs_err = np.abs(pred_age - actual)
    mae = abs_err.mean()
    rmse = np.sqrt(((pred_age - actual) ** 2).mean())
    r, p = stats.pearsonr(actual, pred_age)
    return mae, rmse, r, p, abs_err


def bootstrap_ci_mae_diff(abs_err_a, abs_err_b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(abs_err_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(abs_err_a[idx].mean() - abs_err_b[idx].mean())
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


# Load existing baselines for comparison 
baseline_pp = pd.read_csv(f"{BASE}\\per_participant_baseline.csv")
baseline_pp = baseline_pp[baseline_pp["participant_id"] != "MEAN"].copy()
baseline_pp["participant_id"] = baseline_pp["participant_id"].astype(int)
baseline_pp = baseline_pp.set_index("participant_id")

ridge_pp = pd.read_csv(f"{BASE}\\per_participant_ridge.csv")
ridge_pp = ridge_pp[ridge_pp["participant_id"] != "MEAN"].copy()
ridge_pp["participant_id"] = ridge_pp["participant_id"].astype(int)
ridge_pp = ridge_pp.set_index("participant_id")

participant_ids = df["participant"].astype(int).values
baseline_err = baseline_pp.loc[participant_ids, "absolute_error"].values
ridge_err = ridge_pp.loc[participant_ids, "absolute_error"].values

results = {}
for cfg_name, kernel_fn in CONFIGS.items():
    t0 = time.time()
    pred_age, pred_std = run_nested_gpr(kernel_fn, actual_age, verbose=True)
    elapsed = time.time() - t0
    mae, rmse, r, p, abs_err = metrics(pred_age, actual_age)
    avg_std = pred_std.mean()
    print(f"\nDone in {elapsed:.1f}s ({elapsed/N:.2f}s/fold).")
    print(f"MAE={mae:.4f} RMSE={rmse:.4f} r={r:.4f} p={p:.4f} avg_predictive_std={avg_std:.4f}")

    # Wilcoxon vs baseline & Ridge
    _, wp_base = stats.wilcoxon(abs_err, baseline_err)
    _, wp_ridge = stats.wilcoxon(abs_err, ridge_err)
    print(f"Wilcoxon vs baseline: p={wp_base:.4f}   Wilcoxon vs Ridge: p={wp_ridge:.4f}")

    ci_lo, ci_hi = bootstrap_ci_mae_diff(abs_err, baseline_err)
    print(f"Bootstrap 95% CI (MAE diff vs baseline, 5000 resamples): [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Calibration check
    calib_corr, calib_p = stats.pearsonr(pred_std, abs_err)
    lower = pred_age - 1.96 * pred_std
    upper = pred_age + 1.96 * pred_std
    within_95 = ((actual_age >= lower) & (actual_age <= upper)).mean()
    print(f"Calibration: corr(pred_std, abs_err)={calib_corr:.4f} (p={calib_p:.4f})"
          f"fraction within 95% PI = {within_95:.4f}")

    results[cfg_name] = dict(pred_age=pred_age, pred_std=pred_std, mae=mae, rmse=rmse, r=r, p=p,
                              avg_std=avg_std, abs_err=abs_err, wp_base=wp_base, wp_ridge=wp_ridge,
                              ci_lo=ci_lo, ci_hi=ci_hi, calib_corr=calib_corr, calib_p=calib_p,
                              within_95=within_95, elapsed=elapsed)

# Save per-participant predictions 
for cfg_name in CONFIGS:
    r = results[cfg_name]
    out = pd.DataFrame({
        "participant_id": df["participant"].astype(int),
        "actual_age": actual_age,
        "predicted_age": r["pred_age"],
        "predictive_std": r["pred_std"],
        "absolute_error": r["abs_err"],
        "within_95pct_interval": (actual_age >= r["pred_age"] - 1.96 * r["pred_std"]) &
                                  (actual_age <= r["pred_age"] + 1.96 * r["pred_std"])})
    out_path = f"{BASE}\\per_participant_gpr_{cfg_name}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

master = pd.read_csv(f"{BASE}\\final_results_summary.csv")
new_rows = []
for cfg_name, label, kernel_desc in [
    ("A_RBF", "Gaussian Process Regressor (RBF kernel)", "ConstantKernel*RBF+WhiteKernel"),
    ("B_Matern", "Gaussian Process Regressor (Matern kernel, nu=1.5)", "ConstantKernel*Matern(nu=1.5)+WhiteKernel")]:
    r = results[cfg_name]
    new_rows.append({
        "model_name": label,
        "feature_selection_method": "Fixed primary 3-feature set (nested standardisation + imputation)",
        "n_features": 3, "n_participants": 29.0, "MAE": round(r["mae"], 4), "RMSE": round(r["rmse"], 4),
        "RMSE_MAE_ratio": round(r["rmse"] / r["mae"], 4), "pearson_r": round(r["r"], 4), "p_value": round(r["p"], 4),
        "wilcoxon_p_vs_baseline": round(r["wp_base"], 4), "bootstrap_ci_lower": round(r["ci_lo"], 4),
        "bootstrap_ci_upper": round(r["ci_hi"], 4), "significant_at_0.05": bool(r["wp_base"] < 0.05)})

updated = pd.concat([master, pd.DataFrame(new_rows)], ignore_index=True)
updated.to_csv(f"{BASE}\\final_results_summary.csv", index=False)
print(f"\nAppended 2 GPR rows -> final_results_summary.csv (now {len(updated)} rows)")


import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats

import importlib.util
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)

CONFIGS = {
    "A_RBF": lambda: ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2)),
    "B_Matern": lambda: ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5) + WhiteKernel(1.0, (1e-5, 1e2)),
}

OBSERVED_R = {"A_RBF": 0.1723, "B_Matern": 0.2152}


def run_nested_gpr(kernel_fn, age_labels):
    pred_age = np.empty(N)
    for i in range(N):
        train_df = df.drop(index=i)
        test_row = df.iloc[[i]]
        y_train = age_labels[train_df.index.values]
        medians = train_df[FEATURES].median()
        X_train_raw = train_df[FEATURES].fillna(medians).values.astype(float)
        X_test_raw = test_row[FEATURES].fillna(medians).values.astype(float)
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        gpr = GaussianProcessRegressor(kernel=kernel_fn(), n_restarts_optimizer=10,
                                        normalize_y=True, random_state=0)
        gpr.fit(X_train, y_train)
        mean = gpr.predict(X_test)
        pred_age[i] = mean[0]
    return pred_age


N_PERMS = 500
for cfg_name, kernel_fn in CONFIGS.items():
    rng = np.random.default_rng(42)
    null_r = []
    t0 = time.time()
    for p in range(N_PERMS):
        shuffled_age = rng.permutation(actual_age)
        pred = run_nested_gpr(kernel_fn, shuffled_age)
        r, _ = stats.pearsonr(shuffled_age, pred)
        null_r.append(r)
        if (p + 1) % 50 == 0:
            print(f"  {p+1}/{N_PERMS} done ({time.time()-t0:.0f}s elapsed)", flush=True)
    elapsed = time.time() - t0
    null_r = np.array(null_r)
    observed = OBSERVED_R[cfg_name]
    perm_p = float(np.mean(null_r >= observed))
    print(f"Done in {elapsed:.1f}s. Observed r={observed:.4f}. "
          f"Null r: mean={null_r.mean():.4f} sd={null_r.std():.4f}. Permutation p={perm_p:.4f}")
    pd.DataFrame({"permutation": range(1, N_PERMS + 1), "null_pearson_r": null_r}).to_csv(
        f"{BASE}\\gpr_{cfg_name}_permutation_null.csv", index=False)
    print(f"Saved -> {BASE}\\gpr_{cfg_name}_permutation_null.csv")


import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)

kernel_fn = lambda: ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5) + WhiteKernel(1.0, (1e-5, 1e2))
OBSERVED_R = 0.2152


def run_nested_gpr(kfn, age_labels):
    pred_age = np.empty(N)
    for i in range(N):
        train_df = df.drop(index=i)
        test_row = df.iloc[[i]]
        y_train = age_labels[train_df.index.values]
        medians = train_df[FEATURES].median()
        X_train_raw = train_df[FEATURES].fillna(medians).values.astype(float)
        X_test_raw = test_row[FEATURES].fillna(medians).values.astype(float)
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        gpr = GaussianProcessRegressor(kernel=kfn(), n_restarts_optimizer=10,
                                        normalize_y=True, random_state=0)
        gpr.fit(X_train, y_train)
        pred_age[i] = gpr.predict(X_test)[0]
    return pred_age


N_PERMS = 500
rng = np.random.default_rng(42)
null_r = []
t0 = time.time()
for p in range(N_PERMS):
    shuffled_age = rng.permutation(actual_age)
    pred = run_nested_gpr(kernel_fn, shuffled_age)
    r, _ = stats.pearsonr(shuffled_age, pred)
    null_r.append(r)
    if (p + 1) % 50 == 0:
        print(f"  {p+1}/{N_PERMS} done ({time.time()-t0:.0f}s elapsed)", flush=True)
elapsed = time.time() - t0
null_r = np.array(null_r)
perm_p = float(np.mean(null_r >= OBSERVED_R))
print(f"Done in {elapsed:.1f}s. Observed r={OBSERVED_R:.4f}. "
      f"Null r: mean={null_r.mean():.4f} sd={null_r.std():.4f}. Permutation p={perm_p:.4f}")
pd.DataFrame({"permutation": range(1, N_PERMS + 1), "null_pearson_r": null_r}).to_csv(
    f"{BASE}\\gpr_B_Matern_permutation_null.csv", index=False)
print(f"Saved -> {BASE}\\gpr_B_Matern_permutation_null.csv")
print("MATERN PERMUTATION TEST COMPLETE")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
OUT_PNG = f"{BASE}\\plots\\GPR_kernel_comparison.png"

configs = [
    ("A_RBF", "RBF kernel", f"{BASE}\\per_participant_gpr_A_RBF.csv"),
    ("B_Matern", "Matérn kernel (ν=1.5)", f"{BASE}\\per_participant_gpr_B_Matern.csv"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, (key, label, path) in zip(axes, configs):
    df = pd.read_csv(path)
    std = df["predictive_std"].values
    err = df["absolute_error"].values
    r, p = stats.pearsonr(std, err)

    ax.scatter(std, err, color="#2a6f7f", alpha=0.8, s=45, zorder=3)
    slope, intercept = np.polyfit(std, err, 1)
    x_line = np.array([std.min(), std.max()])
    ax.plot(x_line, slope * x_line + intercept, "--", color="#c0392b", linewidth=2, zorder=2)

    ax.set_xlabel("GP predictive std. dev. (years)")
    ax.set_ylabel("Actual absolute error (years)")
    ax.set_title(f"{label}\nr = {r:.3f}, p = {p:.4f}")

fig.suptitle("Figure 3.4 — GPR calibration: predictive uncertainty vs. actual error\n"
             "(negative slope = model most confident where most wrong)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved -> {OUT_PNG}")


