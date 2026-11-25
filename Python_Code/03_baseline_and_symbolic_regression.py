import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import LinearRegression, Ridge, Lasso

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\baseline_comparison.csv"

TOP_N = 3
MIN_N_FOR_CORR = 5

df = pd.read_csv(DATA)
EXCLUDE = {"participant", "age", "group"}
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]
df = df.dropna(subset=["age"]).reset_index(drop=True)
N = len(df)

print(f"Participants: {N}")
print(f"Candidate features ({len(CANDIDATE_FEATURES)}), top-{TOP_N} reselected every fold\n")


def select_top3(train_df: pd.DataFrame):
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < MIN_N_FOR_CORR:
            continue
        rho, _ = stats.spearmanr(valid["age"], valid[col])
        rows.append((col, rho))
    rho_df = pd.DataFrame(rows, columns=["feature", "rho"])
    rho_df["abs_rho"] = rho_df["rho"].abs()
    return rho_df.sort_values("abs_rho", ascending=False).head(TOP_N)["feature"].tolist()


models = {
    "baseline": None,
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=10.0),
    "lasso": Lasso(alpha=1.0)}

preds = {name: np.full(N, np.nan) for name in models}
fold_selected_features = []

for i in range(N):
    test_row = df.iloc[[i]]
    train_df = df.drop(index=df.index[i])
    y_train = train_df["age"].values.astype(float)
    y_test = float(test_row["age"].values[0])

    # Baseline: training-only mean (properly nested - excludes the held-out participant).
    preds["baseline"][i] = y_train.mean()

    # Nested top-3 feature selection, training data only.
    selected = select_top3(train_df)
    fold_selected_features.append(";".join(selected))

    train_medians = train_df[selected].median()
    X_train = train_df[selected].fillna(train_medians).values.astype(float)
    X_test = test_row[selected].fillna(train_medians).values.astype(float)

    for name, model in models.items():
        if model is None:
            continue
        model.fit(X_train, y_train)
        preds[name][i] = float(model.predict(X_test)[0])

actual = df["age"].values.astype(float)

results = []
for name in models:
    p = preds[name]
    mae = float(np.mean(np.abs(p - actual)))
    rmse = float(np.sqrt(np.mean((p - actual) ** 2)))
    if np.std(p) > 1e-9:
        r, pval = stats.pearsonr(actual, p)
    else:
        r, pval = 0.0, 1.0
    results.append({"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "Pearson_r": round(r, 3), "p_value": round(pval, 4)})

results_df = pd.DataFrame(results)
print("MODEL COMPARISON (all nested LOOCV, top-3 features re-selected every fold except baseline)")
print(results_df.to_string(index=False))


import os
import re
import time
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from gplearn.genetic import SymbolicRegressor

# Config 
DATA    = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
PLOTS   = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_predictions.csv"
NULL_OUT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\null_mae_distribution.csv"
os.makedirs(PLOTS, exist_ok=True)

FEATURE_COLS = [
    "peak_roll_walkingincline",
    "stride_timing_cv_walking",
    "rom_roll_ratio_stepforward_walk"]

FEATURE_RHO = {
    "peak_roll_walkingincline": -0.494,
    "stride_timing_cv_walking": +0.493,
    "rom_roll_ratio_stepforward_walk": +0.484}

N_SEEDS      = 3   
N_GENS       = 25     
POPULATION   = 1500   
PARSIMONY    = 0.05
FUNCTION_SET = ("add", "sub", "mul", "div")
RANDOM_STATE = 42
N_PERMS = 0

# Load data 
df = pd.read_csv(DATA)
df_model = df[["participant", "age"] + FEATURE_COLS].copy()
df_model = df_model[df_model["age"].notna()].reset_index(drop=True)
N = len(df_model)

print(f"Participants: {N}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")


def make_gp_model(seed):
    return SymbolicRegressor(
        population_size=POPULATION,
        generations=N_GENS,
        tournament_size=20,
        function_set=FUNCTION_SET,
        parsimony_coefficient=PARSIMONY,
        metric="mean absolute error",
        max_samples=1.0,
        n_jobs=1,
        verbose=0,
        random_state=seed)


def run_loocv(df_in: pd.DataFrame, verbose: bool = False):
    n = len(df_in)
    gp_preds = np.full(n, np.nan)
    lin_preds = np.full(n, np.nan)
    fold_records = []

    for i in range(n):
        test_row = df_in.iloc[[i]]
        train_df = df_in.drop(index=df_in.index[i])
        train_medians = train_df[FEATURE_COLS].median()
        X_train_raw = train_df[FEATURE_COLS].fillna(train_medians).values.astype(float)
        X_test_raw = test_row[FEATURE_COLS].fillna(train_medians).values.astype(float)
        y_train = train_df["age"].values.astype(float)
        y_test = float(test_row["age"].values[0])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        fold_gp_preds, fold_gp_eqs = [], []
        for s in range(N_SEEDS):
            model = make_gp_model(seed=RANDOM_STATE + i * N_SEEDS + s)
            model.fit(X_train, y_train)
            fold_gp_preds.append(float(model.predict(X_test)[0]))
            fold_gp_eqs.append(str(model._program))
        gp_pred = float(np.median(fold_gp_preds))
        eq_idx = int(np.argmin(np.abs(np.array(fold_gp_preds) - gp_pred)))
        gp_eq = fold_gp_eqs[eq_idx]

        lin = LinearRegression()
        lin.fit(X_train, y_train)
        lin_pred = float(lin.predict(X_test)[0])

        gp_preds[i] = gp_pred
        lin_preds[i] = lin_pred

        fold_records.append({
            "fold": i,
            "participant": test_row["participant"].values[0],
            "actual_age": y_test,
            "gp_predicted_age": gp_pred,
            "linear_predicted_age": lin_pred,
            "gp_abs_error": abs(gp_pred - y_test),
            "linear_abs_error": abs(lin_pred - y_test),
            "gp_equation": gp_eq})

        if verbose:
            print(f"  [{i+1:2d}/{n}] participant={test_row['participant'].values[0]:<4} "
                  f"actual={y_test:.0f}  gp={gp_pred:.1f}  lin={lin_pred:.1f}  "
                  f"gp_err={abs(gp_pred-y_test):.1f}  lin_err={abs(lin_pred-y_test):.1f}")

    return gp_preds, lin_preds, fold_records

t0 = time.time()
gp_preds, lin_preds, fold_records = run_loocv(df_model, verbose=True)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/N:.2f}s/fold average).")


def readable(eq: str, feature_names) -> str:
    for idx in range(len(feature_names) - 1, -1, -1):
        eq = eq.replace(f"X{idx}", feature_names[idx])
    return eq


results_df = pd.DataFrame(fold_records)
results_df["gp_equation_readable"] = [readable(eq, FEATURE_COLS) for eq in results_df["gp_equation"]]
results_df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")

actual = results_df["actual_age"].values
gp_mae = float(np.mean(np.abs(gp_preds - actual)))
lin_mae = float(np.mean(np.abs(lin_preds - actual)))
baseline_mae = float(np.mean(np.abs(actual - actual.mean())))

gp_r, gp_p = stats.pearsonr(actual, gp_preds)
lin_r, lin_p = stats.pearsonr(actual, lin_preds)
gp_rmse = float(np.sqrt(np.mean((gp_preds - actual) ** 2)))
lin_rmse = float(np.sqrt(np.mean((lin_preds - actual) ** 2)))

best_name = "gplearn" if gp_mae <= lin_mae else "linear regression"
if gp_mae <= lin_mae:
    best_mae, best_r, best_p, best_rmse, best_preds = gp_mae, gp_r, gp_p, gp_rmse, gp_preds
else:
    best_mae, best_r, best_p, best_rmse, best_preds = lin_mae, lin_r, lin_p, lin_rmse, lin_preds


perm_block = None
if N_PERMS > 0:
    rng = np.random.default_rng(RANDOM_STATE)
    null_maes = []
    t_perm0 = time.time()
    for p in range(N_PERMS):
        df_shuf = df_model.copy()
        df_shuf["age"] = rng.permutation(df_shuf["age"].values)
        gp_p_preds, lin_p_preds, _ = run_loocv(df_shuf, verbose=False)
        best_p_preds = gp_p_preds if best_name == "gplearn" else lin_p_preds
        mae_p = float(np.mean(np.abs(best_p_preds - df_shuf["age"].values)))
        null_maes.append(mae_p)
        print(f"  permutation {p+1}/{N_PERMS}: null MAE = {mae_p:.2f}  "
              f"(elapsed {time.time()-t_perm0:.0f}s)")
        pd.DataFrame({"permutation": range(1, len(null_maes)+1), "null_mae": null_maes}).to_csv(NULL_OUT, index=False)

    null_maes = np.array(null_maes)
    pctl_95 = float(np.percentile(null_maes, 95))
    perm_p = float(np.mean(null_maes <= best_mae))
    perm_block = (null_maes, pctl_95, perm_p)
else:
    print(f"\nPermutation test skipped (N_PERMS=0)")
    
# FINAL REPORT 
print(f"\n\n{'RESULTS SUMMARY':^60}")
print("=" * 60)
print(f"Features used: {FEATURE_COLS}")
print(f"Participants: N={N}")

print(f"\nMODEL COMPARISON:")
print(f"  gplearn MAE:           {gp_mae:.2f} years")
print(f"  Linear regression MAE: {lin_mae:.2f} years")
print(f"  Trivial baseline MAE:  {baseline_mae:.2f} years")

print(f"\nBEST MODEL: {best_name}")
print(f"  MAE:       {best_mae:.2f} years")
print(f"  Pearson r: {best_r:.3f} (p = {best_p:.4f})")
print(f"  RMSE:      {best_rmse:.2f} years")



import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_reselected_features.csv"

RHO_THRESHOLD  = 0.3
TOP_N_FALLBACK = 3
MIN_N_FOR_CORR = 5

# Previous fixed-3-feature result, for direct comparison at the end.
PREV_MAE, PREV_R, PREV_P = 12.28, 0.451, 0.0140

df = pd.read_csv(DATA)
EXCLUDE = {"participant", "age", "group"}
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]

df = df.dropna(subset=["age"]).reset_index(drop=True)
N = len(df)

print(f"Participants: {N}")
print(f"Candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}\n")


def select_features(train_df: pd.DataFrame):
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < MIN_N_FOR_CORR:
            rows.append((col, np.nan))
            continue
        rho, _ = stats.spearmanr(valid["age"], valid[col])
        rows.append((col, rho))

    rho_df = pd.DataFrame(rows, columns=["feature", "rho"])
    rho_df["abs_rho"] = rho_df["rho"].abs()

    passing = rho_df[rho_df["abs_rho"] > RHO_THRESHOLD].dropna(subset=["abs_rho"])
    if len(passing) < 2:
        passing = rho_df.dropna(subset=["abs_rho"]).sort_values("abs_rho", ascending=False).head(TOP_N_FALLBACK)

    return passing.sort_values("abs_rho", ascending=False)["feature"].tolist()


preds = np.full(N, np.nan)
fold_records = []
selection_counts = {f: 0 for f in CANDIDATE_FEATURES}

for i in range(N):
    test_row = df.iloc[[i]]
    train_df = df.drop(index=df.index[i])

    selected = select_features(train_df)
    for f in selected:
        selection_counts[f] += 1

    train_medians = train_df[selected].median()
    X_train_raw = train_df[selected].fillna(train_medians).values.astype(float)
    X_test_raw = test_row[selected].fillna(train_medians).values.astype(float)
    y_train = train_df["age"].values.astype(float)
    y_test = float(test_row["age"].values[0])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = float(model.predict(X_test)[0])
    preds[i] = pred

    fold_records.append({
        "fold": i,
        "participant": test_row["participant"].values[0],
        "actual_age": y_test,
        "predicted_age": pred,
        "abs_error": abs(pred - y_test),
        "n_features_selected": len(selected),
        "selected_features": ";".join(selected)})

    print(f"  [{i+1:2d}/{N}] participant={test_row['participant'].values[0]:<4} "
          f"actual={y_test:.0f}  pred={pred:.1f}  err={abs(pred-y_test):.1f}  "
          f"n_feat={len(selected)}  features={selected}")

results_df = pd.DataFrame(fold_records)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

# Performance 
actual = results_df["actual_age"].values
mae = float(np.mean(np.abs(preds - actual)))
r, p = stats.pearsonr(actual, preds)
rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
baseline_mae = float(np.mean(np.abs(actual - actual.mean())))

print(f"  MAE            : {mae:.2f} years")
print(f"  RMSE           : {rmse:.2f} years")
print(f"  Pearson r      : {r:.3f}  (p={p:.4f})")
print(f"  Baseline MAE   : {baseline_mae:.2f} years (predict-mean)")

# Feature selection stability 
stability = pd.Series(selection_counts).sort_values(ascending=False)
stable_thresh = N / 2.0


for feat, count in stability.items():
    if count > 0:
        tag = "  <-- STABLE (>50%)" if count > stable_thresh else ""

# Direct comparison 
print("COMPARISON: fixed 3-feature (leaky selection) vs. nested (nested selection)")
print(f"  Fixed 3-feature linear regression : MAE={PREV_MAE:.2f}  r={PREV_R:.3f}  p={PREV_P:.4f}")
print(f"  Nested (re-selected per fold)      : MAE={mae:.2f}  r={r:.3f}  p={p:.4f}")
delta = mae - PREV_MAE
print(f"  Delta MAE (nested - fixed)         : {delta:+.2f} years "
      f"({'nested is WORSE' if delta > 0 else 'nested is BETTER'})")




import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor

# Config
DATA  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
os.makedirs(PLOTS, exist_ok=True)

N_SEEDS      = 3          
N_GENS       = 30         
POPULATION   = 3000      
N_PERMS      = 500       
RANDOM_STATE = 42


FEATURE_COLS = [
    "peak_roll_walkingincline",
    "stride_timing_cv_walking",
    "rom_roll_ratio_stepforward_walk",
    "peak_roll_walking",
    "rom_pitch_walkingincline",
    "stride_timing_cv_walkingincline",
    "rom_pitch_stepforward",
    "vaf_stepforward",
]

# Load data
df = pd.read_csv(DATA)

df_model = df[["participant", "age"] + FEATURE_COLS].copy()
df_model = df_model[df_model["age"].notna()].copy()
# Keep rows with at least half the features present
min_features = len(FEATURE_COLS) // 2
df_model = df_model[df_model[FEATURE_COLS].notna().sum(axis=1) >= min_features].copy()
df_model = df_model.reset_index(drop=True)

print(f"Modelling dataset: n={len(df_model)} participants")
print(f"  Age range: {df_model['age'].min():.0f} – {df_model['age'].max():.0f} yrs")
print(f"  Features : {', '.join(FEATURE_COLS)}")

# Median for imputation (will be recomputed per fold to avoid leakage)
global_medians = df_model[FEATURE_COLS].median()

# GP model factory 
def make_model(seed=0):
    return SymbolicRegressor(
        population_size=POPULATION,
        generations=N_GENS,
        tournament_size=20,
        stopping_criteria=0.0,
        const_range=(-10.0, 10.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "abs", "neg", "inv"),
        metric="mean absolute error",
        parsimony_coefficient=0.0005,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        warm_start=False,
        n_jobs=1, 
        verbose=0,
        random_state=seed)

# LOOCV 
n = len(df_model)
preds = np.full(n, np.nan)
best_eqs = []


for i in range(n):
    test_row  = df_model.iloc[[i]]
    train_set = df_model.drop(index=i).reset_index(drop=True)

    # Impute with training-set median (no leakage from test row)
    train_medians = train_set[FEATURE_COLS].median()
    X_train = train_set[FEATURE_COLS].fillna(train_medians).values.astype(float)
    X_test  = test_row[FEATURE_COLS].fillna(train_medians).values.astype(float)
    y_train = train_set["age"].values.astype(float)
    y_test  = float(test_row["age"].values[0])

    # Multi-seed ensemble
    fold_preds = []
    fold_eqs   = []
    for seed_offset in range(N_SEEDS):
        model = make_model(seed=RANDOM_STATE + i * N_SEEDS + seed_offset)
        model.fit(X_train, y_train)
        fold_preds.append(float(model.predict(X_test)[0]))
        fold_eqs.append(str(model._program))

    preds[i] = float(np.median(fold_preds))
    # Pick equation with prediction closest to median
    eq_idx = int(np.argmin(np.abs(np.array(fold_preds) - preds[i])))
    best_eqs.append(fold_eqs[eq_idx])

    err = abs(preds[i] - y_test)
    print(f"  [{i+1:2d}/{n}] {str(test_row['participant'].values[0]):22s}  "
          f"true={y_test:.0f}  pred={preds[i]:.1f}  |err|={err:.1f} yrs")

# Evaluation
true_ages = df_model["age"].values.astype(float)
valid     = ~np.isnan(preds)
mae       = float(np.mean(np.abs(preds[valid] - true_ages[valid])))
r, p_r    = stats.pearsonr(true_ages[valid], preds[valid])
rmse      = float(np.sqrt(np.mean((preds[valid] - true_ages[valid])**2)))

print(f"LOOCV Results  (n={int(valid.sum())} participants)")
print(f"  MAE            : {mae:.2f} years")
print(f"  RMSE           : {rmse:.2f} years")
print(f"  Pearson r      : {r:.3f}  (p={p_r:.4f})")
print(f"  Baseline MAE   : {float(np.mean(np.abs(true_ages - np.mean(true_ages[valid])))):.2f} yrs (predict-mean baseline)")


# Permutation test 
print(f"\nPermutation test ({N_PERMS} label shuffles)")
rng       = np.random.default_rng(RANDOM_STATE)
perm_maes = []
for _ in range(N_PERMS):
    shuffled = rng.permutation(true_ages[valid])
    perm_maes.append(float(np.mean(np.abs(preds[valid] - shuffled))))
perm_p = float(np.mean(np.array(perm_maes) <= mae))
print(f"  Permutation p-value: {perm_p:.4f}")
print(f"  (fraction of {N_PERMS} shuffles with MAE <= observed {mae:.2f} yrs)")

# Save results 
def readable_equation(eq: str, feature_names) -> str:
    for idx in range(len(feature_names) - 1, -1, -1):
        eq = eq.replace(f"X{idx}", feature_names[idx])
    return eq

results_df = df_model[["participant", "age"]].copy()
results_df["predicted_age"] = preds
results_df["abs_error"]     = np.abs(preds - true_ages)
results_df["loocv_equation"] = best_eqs
results_df["loocv_equation_readable"] = [readable_equation(eq, FEATURE_COLS) for eq in best_eqs]
results_df.to_csv(os.path.join(PLOTS, "loocv_results.csv"), index=False)

summary = {
    "n_participants": int(valid.sum()),
    "mae_years":      round(mae, 2),
    "rmse_years":     round(rmse, 2),
    "pearson_r":      round(r, 3),
    "pearson_p":      round(p_r, 4),
    "perm_p":         round(perm_p, 4),
}
pd.DataFrame([summary]).to_csv(os.path.join(PLOTS, "loocv_summary.csv"), index=False)
print(f"\nSummary: {summary}")


import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gplearn.genetic import SymbolicRegressor

DATA  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
os.makedirs(PLOTS, exist_ok=True)

# Same 8-feature screened + pruned set as 03_symbolic_regression.py (see that
# file's header comment for the screening/pruning rationale).
FEATURE_COLS = [
    "peak_roll_walkingincline",
    "stride_timing_cv_walking",
    "rom_roll_ratio_stepforward_walk",
    "peak_roll_walking",
    "rom_pitch_walkingincline",
    "stride_timing_cv_walkingincline",
    "rom_pitch_stepforward",
    "vaf_stepforward",
]

df = pd.read_csv(DATA)

df_model = df[["participant", "age"] + FEATURE_COLS].dropna(subset=["age"]).copy()
df_model = df_model[df_model[FEATURE_COLS].notna().sum(axis=1) >= len(FEATURE_COLS)//2].copy()
df_model = df_model.reset_index(drop=True)

X = df_model[FEATURE_COLS].fillna(df_model[FEATURE_COLS].median()).values.astype(float)
y = df_model["age"].values.astype(float)

print(f"Full dataset: n={len(df_model)} participants")


def readable_equation(eq: str, feature_names) -> str:
    for idx in range(len(feature_names) - 1, -1, -1):
        eq = eq.replace(f"X{idx}", feature_names[idx])
    return eq

N_SEEDS     = 5
N_GENS      = 60
POPULATION  = 3000

best_mae    = np.inf
best_model  = None
best_seed   = None

for seed in range(N_SEEDS):
    model = SymbolicRegressor(
        population_size=POPULATION,
        generations=N_GENS,
        tournament_size=20,
        const_range=(-10.0, 10.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "abs", "neg", "inv"),
        metric="mean absolute error",
        parsimony_coefficient=0.0005,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        n_jobs=1,
        verbose=0,
        random_state=seed)
    model.fit(X, y)
    preds = model.predict(X)
    mae   = float(np.mean(np.abs(preds - y)))
    r, _  = stats.pearsonr(y, preds)
    print(f"  Seed {seed}: MAE={mae:.2f} yrs  r={r:.3f}  eq={readable_equation(str(model._program), FEATURE_COLS)[:90]}")
    if mae < best_mae:
        best_mae   = mae
        best_model = model
        best_seed  = seed

best_eq_str = str(best_model._program)
best_eq_readable = readable_equation(best_eq_str, FEATURE_COLS)

print(f"\nBest seed: {best_seed}  (MAE={best_mae:.2f} yrs)")
print(f"Best equation (exact):    {best_eq_str}")
print(f"Best equation (readable): {best_eq_readable}")
print(f"  Complexity (# nodes): {best_model._program.length_}")
print(f"  Depth               : {best_model._program.depth_}")

# Training metrics
preds_final = best_model.predict(X)
r_final, p_final = stats.pearsonr(y, preds_final)
print(f"  Train r             : {r_final:.3f}  (p={p_final:.4f})")
print(f"  Train MAE           : {best_mae:.2f} yrs  (note: not cross-validated)")


import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gplearn.genetic import SymbolicRegressor

# Config
DATA  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
os.makedirs(PLOTS, exist_ok=True)

PRED_OUT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_predictions.csv"

RHO_THRESHOLD  = 0.3
TOP_N_FALLBACK = 3    
MIN_N_FOR_CORR = 5  

GP_PARAMS = dict(
    population_size=1000,
    generations=20,
    tournament_size=20,
    function_set=("add", "sub", "mul", "div", "sqrt"),
    max_samples=0.9,
    parsimony_coefficient=0.01,
    metric="mean absolute error")

NULL_MAE_OUT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\null_mae_distribution.csv"

# Load data
df_raw = pd.read_csv(DATA)
EXCLUDE = {"participant", "age", "group"}
dmci_cols = [c for c in df_raw.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df_raw.columns if c not in EXCLUDE and c not in dmci_cols]

df_raw = df_raw.dropna(subset=["age"]).reset_index(drop=True)
N = len(df_raw)

print(f"Participants: {N}")
print(f"Candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}")


# Inner feature selection (training fold only) 
def select_features(train_df: pd.DataFrame):
    
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < MIN_N_FOR_CORR:
            rows.append((col, np.nan))
            continue
        rho, _ = stats.spearmanr(valid["age"], valid[col])
        rows.append((col, rho))

    rho_lookup = dict(rows)
    rho_df = pd.DataFrame(rows, columns=["feature", "rho"])
    rho_df["abs_rho"] = rho_df["rho"].abs()

    passing = rho_df[rho_df["abs_rho"] > RHO_THRESHOLD].dropna(subset=["abs_rho"])
    if len(passing) < 2:
        passing = rho_df.dropna(subset=["abs_rho"]).sort_values("abs_rho", ascending=False).head(TOP_N_FALLBACK)

    selected = passing.sort_values("abs_rho", ascending=False)["feature"].tolist()
    return selected, rho_lookup


def readable_equation(eq: str, feature_names) -> str:
    for idx in range(len(feature_names) - 1, -1, -1):
        eq = eq.replace(f"X{idx}", feature_names[idx])
    return eq


# One full nested-LOOCV pass (used for both the real run and each permutation) 
def run_nested_loocv(df: pd.DataFrame, verbose_progress: bool = False):
    n = len(df)
    preds = np.full(n, np.nan)
    fold_records = []
    selection_log = []

    for i in range(n):
        test_row = df.iloc[[i]]
        train_df = df.drop(index=df.index[i])

        selected, rho_lookup = select_features(train_df)

        train_medians = train_df[selected].median()
        X_train = train_df[selected].fillna(train_medians).values.astype(float)
        y_train = train_df["age"].values.astype(float)
        X_test = test_row[selected].fillna(train_medians).values.astype(float)
        y_test = float(test_row["age"].values[0])

        model = SymbolicRegressor(
            population_size=GP_PARAMS["population_size"],
            generations=GP_PARAMS["generations"],
            tournament_size=GP_PARAMS["tournament_size"],
            function_set=GP_PARAMS["function_set"],
            max_samples=GP_PARAMS["max_samples"],
            parsimony_coefficient=GP_PARAMS["parsimony_coefficient"],
            metric=GP_PARAMS["metric"],
            random_state=i,   # "random_state = fold number"
            n_jobs=1,
            verbose=0,
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(X_test)[0])
        preds[i] = pred

        train_pred = model.predict(X_train)
        train_mae = float(np.mean(np.abs(train_pred - y_train)))

        fold_records.append({
            "fold": i,
            "participant": test_row["participant"].values[0],
            "actual_age": y_test,
            "predicted_age": pred,
            "abs_error": abs(pred - y_test),
            "n_features_selected": len(selected),
            "selected_features": ";".join(selected),
            "equation": str(model._program),
            "equation_readable": readable_equation(str(model._program), selected),
            "train_mae": train_mae})

        for feat in CANDIDATE_FEATURES:
            selection_log.append({
                "fold": i,
                "feature": feat,
                "rho": rho_lookup.get(feat, np.nan),
                "selected": feat in selected})

        if verbose_progress:
            print(f"  [{i+1:2d}/{n}] participant={test_row['participant'].values[0]:<4} "
                  f"actual={y_test:.0f}  pred={pred:.1f}  |err|={abs(pred-y_test):.1f}  "
                  f"n_feat={len(selected)}  train_mae={train_mae:.2f}")

    return preds, fold_records, selection_log

# Run the real nested LOOCV 
t0 = time.time()
preds, fold_records, selection_log = run_nested_loocv(df_raw, verbose_progress=True)
elapsed = time.time() - t0
per_fold = elapsed / N
print(f"\nDone in {elapsed:.1f}s ({per_fold:.2f}s/fold average).")

pred_df = pd.DataFrame(fold_records)
pred_df.to_csv(PRED_OUT, index=False)
print(f"Saved -> {PRED_OUT}")

# Performance
actual = pred_df["actual_age"].values
predicted = pred_df["predicted_age"].values
mae = float(np.mean(np.abs(predicted - actual)))
r, p_r = stats.pearsonr(actual, predicted)
rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

print("PERFORMANCE (nested LOOCV)")
print(f"  MAE       : {mae:.2f} years")
print(f"  RMSE      : {rmse:.2f} years")
print(f"  Pearson r : {r:.3f}  (p={p_r:.4f})")


fig, ax = plt.subplots(figsize=(6.5, 6))

# Feature stability 
stability = (sel_log_df[sel_log_df["selected"]]
             .groupby("feature")["fold"].count()
             .reindex(CANDIDATE_FEATURES, fill_value=0)
             .sort_values(ascending=False))
stable_thresh = N / 2.0


print(f"FEATURE STABILITY (selected in how many of {N} folds)")

for feat, count in stability.items():
    tag = "  <-- STABLE (>50%)" if count > stable_thresh else ""

# Permutation test 
if N_PERMS > 0:
    rng = np.random.default_rng(42)
    null_maes = []
    t_perm0 = time.time()
    for p in range(N_PERMS):
        df_shuf = df_raw.copy()
        df_shuf["age"] = rng.permutation(df_shuf["age"].values)
        preds_p, records_p, _ = run_nested_loocv(df_shuf, verbose_progress=False)
        mae_p = float(np.mean(np.abs(np.array([r["predicted_age"] for r in records_p]) -
                                      np.array([r["actual_age"] for r in records_p]))))
        null_maes.append(mae_p)
        print(f"  permutation {p+1}/{N_PERMS}: null MAE = {mae_p:.2f}  "
              f"(elapsed {time.time()-t_perm0:.0f}s)")
        pd.DataFrame({"permutation": range(1, len(null_maes) + 1), "null_mae": null_maes}) \
            .to_csv(NULL_MAE_OUT, index=False)

    perm_elapsed = time.time() - t_perm0

    null_maes = np.array(null_maes)
    pctl_95 = float(np.percentile(null_maes, 95))
    empirical_p = float(np.mean(null_maes <= mae))  # standard permutation p-value

    print("PERMUTATION TEST")
    print(f"  Shuffles run           : {N_PERMS}  (took {perm_elapsed:.1f}s total)")
    print(f"  Null MAE distribution  : mean={null_maes.mean():.2f}  sd={null_maes.std():.2f}")
    print(f"  95th percentile of null MAE: {pctl_95:.2f} years")
    print(f"  Observed real MAE      : {mae:.2f} years")
    beats = mae < pctl_95
    print(f"  Real MAE {'BEATS' if beats else 'does NOT beat'} the 95th-percentile null threshold "
          f"({mae:.2f} {'<' if beats else '>='} {pctl_95:.2f})")
    print(f"  Empirical permutation p-value (fraction of null MAE <= observed): {empirical_p:.4f}")

    print(f"Saved -> {NULL_MAE_OUT}")

    
import os
import re
import time
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from gplearn.genetic import SymbolicRegressor

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_gplearn_top3_perfold.csv"

TOP_N = 3
MIN_N_FOR_CORR = 5
N_SEEDS = 3
RANDOM_STATE = 42

GP_PARAMS = dict(
    population_size=3000,
    generations=30,
    tournament_size=20,
    function_set=("add", "sub", "mul", "div"),
    parsimony_coefficient=0.05,
    const_range=(-10.0, 10.0),
    metric="mean absolute error")

COMPARISON = {
    "Run 2 (nested, 8 features)":            dict(mae=13.72, r=0.160, p=0.41),
    "Run 3 (nested, fixed 3)":                dict(mae=14.26, r=None, p=None),
    "Run 5 (nested linear, 11-12 features)":  dict(mae=15.35, r=0.125, p=0.518),
    "Baseline (predict mean)":                dict(mae=13.97, r=None, p=None)}

df = pd.read_csv(DATA)
EXCLUDE = {"participant", "age", "group"}
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]
df = df.dropna(subset=["age"]).reset_index(drop=True)
N = len(df)

print(f"Participants: {N}")
print(f"Candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}\n")


def select_top3(train_df: pd.DataFrame):
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < MIN_N_FOR_CORR:
            continue
        rho, _ = stats.spearmanr(valid["age"], valid[col])
        rows.append((col, rho))
    rho_df = pd.DataFrame(rows, columns=["feature", "rho"])
    rho_df["abs_rho"] = rho_df["rho"].abs()
    top3 = rho_df.sort_values("abs_rho", ascending=False).head(TOP_N)
    return top3["feature"].tolist(), dict(zip(rho_df["feature"], rho_df["rho"]))


def make_gp_model(seed):
    return SymbolicRegressor(
        population_size=GP_PARAMS["population_size"],
        generations=GP_PARAMS["generations"],
        tournament_size=GP_PARAMS["tournament_size"],
        function_set=GP_PARAMS["function_set"],
        parsimony_coefficient=GP_PARAMS["parsimony_coefficient"],
        const_range=GP_PARAMS["const_range"],
        metric=GP_PARAMS["metric"],
        n_jobs=1,
        verbose=0,
        random_state=seed)


fold_records = []
selection_counts = {f: 0 for f in CANDIDATE_FEATURES}

t0 = time.time()
for i in range(N):
    test_row = df.iloc[[i]]
    train_df = df.drop(index=df.index[i])

    selected, rho_lookup = select_top3(train_df)
    for f in selected:
        selection_counts[f] += 1

    train_medians = train_df[selected].median()
    X_train = train_df[selected].fillna(train_medians).values.astype(float)
    X_test = test_row[selected].fillna(train_medians).values.astype(float)
    y_train = train_df["age"].values.astype(float)
    y_test = float(test_row["age"].values[0])

    fold_preds, fold_eqs = [], []
    for s in range(N_SEEDS):
        model = make_gp_model(seed=RANDOM_STATE + i * N_SEEDS + s)
        model.fit(X_train, y_train)
        fold_preds.append(float(model.predict(X_test)[0]))
        fold_eqs.append(str(model._program))
    pred = float(np.median(fold_preds))
    eq_idx = int(np.argmin(np.abs(np.array(fold_preds) - pred)))
    eq = fold_eqs[eq_idx]

    def readable_equation(eq_str, feature_names):
        for idx in range(len(feature_names) - 1, -1, -1):
            eq_str = eq_str.replace(f"X{idx}", feature_names[idx])
        return eq_str

    fold_records.append({
        "fold": i,
        "participant": test_row["participant"].values[0],
        "actual_age": y_test,
        "predicted_age": pred,
        "abs_error": abs(pred - y_test),
        "selected_features": ";".join(selected),
        "feature_1": selected[0], "feature_2": selected[1], "feature_3": selected[2],
        "equation": eq,
        "equation_readable": readable_equation(eq, selected)})

    elapsed_so_far = time.time() - t0
    print(f"  [{i+1:2d}/{N}] participant={test_row['participant'].values[0]:<4} "
          f"actual={y_test:.0f}  pred={pred:.1f}  err={abs(pred-y_test):.1f}  "
          f"features={selected}  (elapsed {elapsed_so_far:.0f}s, "
          f"avg {elapsed_so_far/(i+1):.1f}s/fold)")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/N:.2f}s/fold average).")

results_df = pd.DataFrame(fold_records)
results_df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")

# Performance 
actual = results_df["actual_age"].values
preds = results_df["predicted_age"].values
mae = float(np.mean(np.abs(preds - actual)))
rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
r, p = stats.pearsonr(actual, preds)
baseline_mae = float(np.mean(np.abs(actual - actual.mean())))


print("RESULTS: nested gplearn, exactly top-3 features per fold")

print(f"  N          : {N}")
print(f"  MAE        : {mae:.2f} years")
print(f"  RMSE       : {rmse:.2f} years")
print(f"  Pearson r  : {r:.3f}  (p={p:.4f})")

print(f"\nCOMPARISON:")
print(f"  {'This run (top-3 per fold)':40s}: MAE={mae:.2f}  r={r:.3f}  p={p:.4f}")
for name, vals in COMPARISON.items():
    r_str = f"{vals['r']:.3f}" if vals['r'] is not None else "n/a"
    p_str = f"{vals['p']:.4f}" if vals['p'] is not None else "n/a"
    print(f"  {name:40s}: MAE={vals['mae']:.2f}  r={r_str}  p={p_str}")

# Feature stability
stability = pd.Series(selection_counts).sort_values(ascending=False)
print(f"FEATURE STABILITY (how many of {N} folds each feature was in the top 3)")
for feat, count in stability.items():
    if count > 0:
        print(f"  {feat:35s}  {count:2d}/{N}")

top_feat, top_count = stability.index[0], stability.iloc[0]
same3_folds = (results_df[["feature_1", "feature_2", "feature_3"]]
               .apply(lambda row: frozenset(row), axis=1))
same3_counts = Counter(same3_folds)
most_common_set, most_common_freq = same3_counts.most_common(1)[0]
print(f"\nMost frequent single feature: '{top_feat}' in {top_count}/{N} folds.")
print(f"Most frequent EXACT top-3 SET: {sorted(most_common_set)} in {most_common_freq}/{N} folds.")
if most_common_freq > N / 2:
    print("The SAME 3 features were selected in a majority of folds (stable selection).")
else:
    print("The exact top-3 set VARIED across folds (not dominated by one fixed triple).")

# Equation stability
def normalize_equation(eq: str) -> str:
    return re.sub(r"-?\d+\.\d+|-?\d+", "C", eq)

templates = [normalize_equation(e) for e in results_df["equation"]]
template_counts = Counter(templates)
most_common_template, freq = template_counts.most_common(1)[0]

print(f"Most common equation structure occurred in {freq}/{N} folds.")
matching_idx = [i for i, t in enumerate(templates) if t == most_common_template]
rep_row = results_df.iloc[matching_idx].loc[results_df.iloc[matching_idx]["abs_error"].idxmin()]
print(f"Representative equation (fold {int(rep_row['fold'])}, lowest error in that group):")
print(f"  Exact:    {rep_row['equation']}")
print(f"  Readable: {rep_row['equation_readable']}")


import numpy as np
import pandas as pd
from scipy import stats
from gplearn.genetic import SymbolicRegressor

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\final_model_constrained_seeds.csv"

FEATURE_COLS = [
    "peak_roll_walkingincline",
    "stride_timing_cv_walking",
    "rom_roll_ratio_stepforward_walk",
    "peak_roll_walking",
    "rom_pitch_walkingincline",
    "stride_timing_cv_walkingincline",
    "rom_pitch_stepforward",
    "vaf_stepforward"]

N_SEEDS = 5
POPULATION = 3000
GENERATIONS = 60
PARSIMONY = 0.5
MAX_NODES = 15

df = pd.read_csv(DATA)
df_model = df[["participant", "age"] + FEATURE_COLS].dropna(subset=["age"]).copy()
df_model = df_model[df_model[FEATURE_COLS].notna().sum(axis=1) >= len(FEATURE_COLS) // 2].copy()
df_model = df_model.reset_index(drop=True)

X = df_model[FEATURE_COLS].fillna(df_model[FEATURE_COLS].median()).values.astype(float)
y = df_model["age"].values.astype(float)
baseline_mae = float(np.mean(np.abs(y - y.mean())))

print(f"n={len(df_model)} participants, baseline (predict-mean) MAE={baseline_mae:.2f} yr")
print(f"parsimony_coefficient={PARSIMONY}  hard ceiling={MAX_NODES} nodes (post-hoc filter, see docstring)\n")

results = []
for seed in range(N_SEEDS):
    model = SymbolicRegressor(
        population_size=POPULATION,
        generations=GENERATIONS,
        tournament_size=20,
        const_range=(-10.0, 10.0),
        init_depth=(2, 3),  
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "abs", "neg", "inv"),
        metric="mean absolute error",
        parsimony_coefficient=PARSIMONY,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        n_jobs=1,
        verbose=0,
        random_state=seed)
    model.fit(X, y)
    preds = model.predict(X)
    mae = float(np.mean(np.abs(preds - y)))
    r, _ = stats.pearsonr(y, preds)
    n_nodes = int(model._program.length_)
    depth = int(model._program.depth_)
    eq = str(model._program)
    passes = n_nodes <= MAX_NODES

    def _readable(eq_str, feature_names):
        for idx in range(len(feature_names) - 1, -1, -1):
            eq_str = eq_str.replace(f"X{idx}", feature_names[idx])
        return eq_str

    eq_readable = _readable(eq, FEATURE_COLS)

    results.append({
        "seed": seed, "mae": mae, "r": r, "n_nodes": n_nodes,
        "depth": depth, "passes_ceiling": passes, "equation": eq,
        "equation_readable": eq_readable})

    verdict = "OK  (<=15 nodes)" if passes else "REJECTED (>15 nodes -- hard ceiling)"
    print(f"Seed {seed}: MAE={mae:.2f} yr  r={r:.3f}  nodes={n_nodes}  depth={depth}  [{verdict}]")
    print(f"  eq (exact):    {eq}")
    print(f"  eq (readable): {eq_readable}\n")

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")

valid = [res for res in results if res["passes_ceiling"]]

print(f"RESULT: {len(valid)}/{N_SEEDS} seeds produced an equation at or under {MAX_NODES} nodes.")


if valid:
    best = min(valid, key=lambda res: res["mae"])
    readable = best["equation"]
    for idx in range(len(FEATURE_COLS) - 1, -1, -1):
        readable = readable.replace(f"X{idx}", FEATURE_COLS[idx])

    print(f"\nBEST VALID EQUATION (seed {best['seed']}, confirmed {best['n_nodes']} <= {MAX_NODES} nodes):")
    print(f"  Training MAE: {best['mae']:.2f} yr   r: {best['r']:.3f}   nodes: {best['n_nodes']}")
    print(f"\n  Exact:    {best['equation']}")
    print(f"\n  Readable: {readable}")

    delta = baseline_mae - best["mae"]
    print(f"\nComparison to baseline: {best['mae']:.2f} yr vs {baseline_mae:.2f} yr baseline "
          f"(improvement: {delta:+.2f} yr)")
    if delta < 1.0:
        print("-> NOT a meaningful improvement over baseline. Even under a hard 15-node ceiling, "
              "no simple equation was found that beats predicting the mean age by a practically "
              "useful margin. This confirms no simple equation exists in this feature set either.")
    else:
        print("-> This is a meaningful improvement over baseline, but note this is still an "
              "IN-SAMPLE training MAE with no cross-validation, so it should not be taken as "
              "evidence of a generalizable relationship without a LOOCV check.")
else:
    print("\nNO seed produced an equation at or under 15 nodes, even with parsimony_coefficient=0.5.")
    print("All 5 discovered equations exceeded the hard ceiling and are REJECTED")
    print("valid 'best equation' to report from this run.")

