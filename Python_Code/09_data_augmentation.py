import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]


def _prep_real(df_real):
    d = df_real.copy().reset_index(drop=True)
    medians = d[FEATURES].median()
    d[FEATURES] = d[FEATURES].fillna(medians)
    return d, medians


def generate_noise(df_real, n_per_real=3, noise_frac_range=(0.05, 0.10), seed=0):
    d, _ = _prep_real(df_real)
    rng = np.random.default_rng(seed)
    sds = d[FEATURES].std(ddof=1)

    rows = []
    for _, r in d.iterrows():
        for copy_i in range(n_per_real):
            frac = rng.uniform(*noise_frac_range, size=len(FEATURES))
            noise = rng.normal(0, 1, size=len(FEATURES)) * frac * sds.values
            new_feats = r[FEATURES].values.astype(float) + noise
            rows.append({
                "participant": f"{int(r['participant'])}_noise{copy_i}",
                "age": r["age"], "group": r["group"],
                **dict(zip(FEATURES, new_feats)),
            })
    out = pd.DataFrame(rows)
    out["is_synthetic"] = True
    out["synthetic_method"] = "noise"
    return out


def generate_smote(df_real, target_total=100, k_neighbors=5, seed=0):
    d, _ = _prep_real(df_real)
    rng = np.random.default_rng(seed)
    n_real = len(d)
    n_synth_needed = max(target_total - n_real, 0)

    rows = []
    for group_val, gdf in d.groupby("group"):
        gdf = gdf.reset_index(drop=True)
        n_group = len(gdf)
        n_synth_group = round(n_synth_needed * n_group / n_real)
        if n_group < 2 or n_synth_group == 0:
            continue
        k = min(k_neighbors, n_group - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(gdf[FEATURES].values)
        _, neighbor_idx = nn.kneighbors(gdf[FEATURES].values)

        for i in range(n_synth_group):
            anchor_i = rng.integers(0, n_group)
            candidate_neighbors = neighbor_idx[anchor_i][1:]  # exclude self
            neighbor_i = rng.choice(candidate_neighbors)
            ratio = rng.uniform(0, 1)

            anchor = gdf.iloc[anchor_i]
            neighbor = gdf.iloc[neighbor_i]
            new_feats = anchor[FEATURES].values.astype(float) + ratio * (
                neighbor[FEATURES].values.astype(float) - anchor[FEATURES].values.astype(float))
            new_age = anchor["age"] + ratio * (neighbor["age"] - anchor["age"])

            rows.append({
                "participant": f"{int(anchor['participant'])}x{int(neighbor['participant'])}_smote{i}",
                "age": new_age, "group": group_val,
                **dict(zip(FEATURES, new_feats))})
    out = pd.DataFrame(rows)
    out["is_synthetic"] = True
    out["synthetic_method"] = "smote"
    return out


def generate_gmm(df_real, target_total=100, n_components=1, seed=0):
    d, _ = _prep_real(df_real)
    rng_seed = seed
    n_real = len(d)
    n_synth_needed = max(target_total - n_real, 0)
    dims = ["age"] + FEATURES

    rows = []
    for group_val, gdf in d.groupby("group"):
        gdf = gdf.reset_index(drop=True)
        n_group = len(gdf)
        n_synth_group = round(n_synth_needed * n_group / n_real)
        if n_group < 2 or n_synth_group == 0:
            continue
        nc = min(n_components, max(1, n_group - 1))
        gmm = GaussianMixture(n_components=nc, covariance_type="full",
                               reg_covar=1e-3, random_state=rng_seed)
        gmm.fit(gdf[dims].values)
        sampled, _ = gmm.sample(n_synth_group)

        for i, s in enumerate(sampled):
            row = dict(zip(dims, s))
            rows.append({
                "participant": f"gmm_g{int(group_val)}_{i}",
                "age": row["age"], "group": group_val,
                **{f: row[f] for f in FEATURES}})
    out = pd.DataFrame(rows)
    out["is_synthetic"] = True
    out["synthetic_method"] = "gmm"
    return out


def format_real(df_real):
    out = df_real[["participant", "age", "group"] + FEATURES].copy()
    out["is_synthetic"] = False
    out["synthetic_method"] = "none"
    return out


import pandas as pd
import augment_lib as A

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
real_tagged = A.format_real(df)
print(f"Real participants: {len(df)}")

for method_name, gen_fn, kwargs in [
    ("noise", A.generate_noise, dict(n_per_real=3, seed=1)),
    ("smote", A.generate_smote, dict(target_total=100, seed=1)),
    ("gmm", A.generate_gmm, dict(target_total=100, seed=1)),
]:
    synth = gen_fn(df, **kwargs)
    combined = pd.concat([real_tagged, synth], ignore_index=True)
    out_path = f"{BASE}\\augmented_data_{method_name}.csv"
    combined.to_csv(out_path, index=False)
    print(f"{method_name}: {len(synth)} synthetic + {len(df)} real = {len(combined)} total -> {out_path}")

import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

import augment_lib as A

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = A.FEATURES

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_group = df["group"].values.astype(int)
actual_age = df["age"].values.astype(float)

GEN_CONFIG = {
    "noise": dict(fn=A.generate_noise, kwargs=dict(n_per_real=3, seed=1)),
    "smote": dict(fn=A.generate_smote, kwargs=dict(target_total=round(100 * 28 / 29), seed=1)),
    "gmm":   dict(fn=A.generate_gmm,   kwargs=dict(target_total=round(100 * 28 / 29), n_components=1, seed=1))}


def build_training_set(train_real_df, method):
    medians = train_real_df[FEATURES].median()
    real_part = train_real_df.copy()
    real_part[FEATURES] = real_part[FEATURES].fillna(medians)

    cfg = GEN_CONFIG[method]
    synth_part = cfg["fn"](train_real_df, **cfg["kwargs"])  
    combined = pd.concat([
        real_part[["participant", "age", "group"] + FEATURES],
        synth_part[["participant", "age", "group"] + FEATURES]], ignore_index=True)
    return combined, medians, len(synth_part)


def run_augmented_loocv(method):
    lr_pred_group = np.empty(N, dtype=int)
    lr_pred_proba = np.empty(N)
    ridge_pred_age = np.empty(N)
    lasso_pred_age = np.empty(N)
    n_synth_list = []

    for i in range(N):
        train_real = df.drop(index=i)
        test_row = df.iloc[[i]]

        train_combined, medians, n_synth = build_training_set(train_real, method)
        n_synth_list.append(n_synth)

        X_train = train_combined[FEATURES].values.astype(float)
        X_test = test_row[FEATURES].fillna(medians).values.astype(float)

        # Logistic regression (classification)
        y_train_bin = (train_combined["group"].values.astype(int) == 2).astype(int)
        if len(np.unique(y_train_bin)) < 2:
            lr_pred_proba[i] = 0.5
            lr_pred_group[i] = 1
        else:
            clf = LogisticRegression()
            clf.fit(X_train, y_train_bin)
            classes = list(clf.classes_)
            old_idx = classes.index(1) if 1 in classes else 0
            proba_old = clf.predict_proba(X_test)[0][old_idx]
            lr_pred_proba[i] = proba_old
            lr_pred_group[i] = 2 if proba_old >= 0.5 else 1

        # Ridge / Lasso (regression)
        y_train_age = train_combined["age"].values.astype(float)
        ridge = Ridge(alpha=10.0).fit(X_train, y_train_age)
        lasso = Lasso(alpha=1.0).fit(X_train, y_train_age)
        ridge_pred_age[i] = ridge.predict(X_test)[0]
        lasso_pred_age[i] = lasso.predict(X_test)[0]

    return dict(
        lr_pred_group=lr_pred_group, lr_pred_proba=lr_pred_proba,
        ridge_pred_age=ridge_pred_age, lasso_pred_age=lasso_pred_age,
        n_synth_per_fold=n_synth_list)


def permutation_test_lr(method, observed_auc, n_perms=500, seed=42):
    rng = np.random.default_rng(seed)
    null_aucs = []
    t0 = time.time()
    for p in range(n_perms):
        shuffled = rng.permutation(df["group"].values)
        d2 = df.copy()
        d2["group"] = shuffled
        proba = np.empty(N)
        for i in range(N):
            train_real = d2.drop(index=i)
            test_row = d2.iloc[[i]]
            train_combined, medians, _ = build_training_set(train_real, method)
            X_train = train_combined[FEATURES].values.astype(float)
            X_test = test_row[FEATURES].fillna(medians).values.astype(float)
            y_train_bin = (train_combined["group"].values.astype(int) == 2).astype(int)
            if len(np.unique(y_train_bin)) < 2:
                proba[i] = 0.5
            else:
                clf = LogisticRegression()
                clf.fit(X_train, y_train_bin)
                classes = list(clf.classes_)
                old_idx = classes.index(1) if 1 in classes else 0
                proba[i] = clf.predict_proba(X_test)[0][old_idx]
        null_aucs.append(roc_auc_score((shuffled == 2).astype(int), proba))
    elapsed = time.time() - t0
    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))
    return p_value, elapsed, null_aucs


# Load existing baselines (no re-run, per Task 3 instructions)
baseline_class = pd.read_csv(f"{BASE}\\classification_young_vs_old.csv")
baseline_class = baseline_class[baseline_class["participant_id"] != "SUMMARY"].copy()
baseline_class["participant_id"] = baseline_class["participant_id"].astype(int)
baseline_class = baseline_class.set_index("participant_id")

baseline_ridge = pd.read_csv(f"{BASE}\\per_participant_ridge.csv")
baseline_ridge = baseline_ridge[baseline_ridge["participant_id"] != "MEAN"].copy()
baseline_ridge["participant_id"] = baseline_ridge["participant_id"].astype(int)
baseline_ridge = baseline_ridge.set_index("participant_id")

baseline_lasso = pd.read_csv(f"{BASE}\\per_participant_lasso.csv")
baseline_lasso = baseline_lasso[baseline_lasso["participant_id"] != "MEAN"].copy()
baseline_lasso["participant_id"] = baseline_lasso["participant_id"].astype(int)
baseline_lasso = baseline_lasso.set_index("participant_id")

participant_ids = df["participant"].astype(int).values

comparison_rows = []

# Baseline (no augmentation) 
comparison_rows.append(dict(
    method="none", model="Logistic Regression", n_real=29, n_synthetic_per_fold=0,
    accuracy_or_MAE=0.6207, AUC_or_r=0.6779, permutation_p=0.032, wilcoxon_p_vs_no_augmentation=None))
comparison_rows.append(dict(
    method="none", model="Ridge (nested top-3)", n_real=29, n_synthetic_per_fold=0,
    accuracy_or_MAE=13.58, AUC_or_r=0.194, permutation_p=None, wilcoxon_p_vs_no_augmentation=None))
comparison_rows.append(dict(
    method="none", model="Lasso (nested top-3)", n_real=29, n_synthetic_per_fold=0,
    accuracy_or_MAE=13.47, AUC_or_r=0.208, permutation_p=None, wilcoxon_p_vs_no_augmentation=None))

honesty_report = []

for method in ["noise", "smote", "gmm"]:
    t0 = time.time()
    res = run_augmented_loocv(method)
    elapsed = time.time() - t0
    avg_n_synth = np.mean(res["n_synth_per_fold"])
    print(f"Done in {elapsed:.1f}s. Avg synthetic participants per fold: {avg_n_synth:.1f}")

    # -- Logistic regression --
    lr_acc = float((res["lr_pred_group"] == actual_group).mean())
    lr_auc = roc_auc_score(actual_group, res["lr_pred_proba"])
    baseline_correct = (baseline_class.loc[participant_ids, "predicted_group"].values.astype(int)
                         == baseline_class.loc[participant_ids, "actual_group"].values.astype(int)).astype(int)
    aug_correct = (res["lr_pred_group"] == actual_group).astype(int)
    if np.any(baseline_correct != aug_correct):
        _, wp_lr = stats.wilcoxon(aug_correct, baseline_correct)
    else:
        wp_lr = 1.0
    print(f"LR: accuracy={lr_acc:.4f} AUC={lr_auc:.4f}  Wilcoxon vs baseline p={wp_lr:.4f}")

    acc_gain_pp = (lr_acc - 0.6207) * 100
    suspicious = abs(acc_gain_pp) > 15
    confirmed_lr = (lr_acc > 0.6207) and (wp_lr < 0.05)

    perm_p_lr = None
    if confirmed_lr:
        print("Wilcoxon shows a significant, positive result: running full 500-shuffle permutation test")
        perm_p_lr, perm_elapsed, _ = permutation_test_lr(method, lr_auc, n_perms=500)
        print(f"  Permutation p={perm_p_lr:.4f} (took {perm_elapsed:.1f}s)")
    else:
        print("No positive, significant signal vs baseline, permutation test skipped (matches established session practice).")

    comparison_rows.append(dict(
        method=method, model="Logistic Regression", n_real=29,
        n_synthetic_per_fold=round(avg_n_synth), accuracy_or_MAE=round(lr_acc, 4),
        AUC_or_r=round(lr_auc, 4), permutation_p=perm_p_lr, wilcoxon_p_vs_no_augmentation=round(wp_lr, 4),
        notes=f"Fixed primary 3-feature set (matches Row 1). {'SUSPICIOUS GAIN >15pp -- investigate' if suspicious else ''}"))

    honesty_report.append(dict(method=method, model="Logistic Regression", accuracy=lr_acc, auc=lr_auc,
                                wilcoxon_p=wp_lr, permutation_p=perm_p_lr, confirmed=confirmed_lr,
                                suspicious=suspicious, gain_pp=acc_gain_pp))

    # Ridge
    ridge_mae = mean_absolute_error(actual_age, res["ridge_pred_age"])
    ridge_rmse = np.sqrt(mean_squared_error(actual_age, res["ridge_pred_age"]))
    ridge_r, ridge_p = stats.pearsonr(actual_age, res["ridge_pred_age"])
    baseline_ridge_err = baseline_ridge.loc[participant_ids, "absolute_error"].values
    aug_ridge_err = np.abs(res["ridge_pred_age"] - actual_age)
    _, wp_ridge = stats.wilcoxon(aug_ridge_err, baseline_ridge_err)
    print(f"Ridge: MAE={ridge_mae:.4f} r={ridge_r:.4f} p={ridge_p:.4f}  Wilcoxon vs baseline p={wp_ridge:.4f}")
    confirmed_ridge = (ridge_mae < 13.58) and (wp_ridge < 0.05)

    comparison_rows.append(dict(
        method=method, model="Ridge (fixed primary-3, augmented)", n_real=29,
        n_synthetic_per_fold=round(avg_n_synth), accuracy_or_MAE=round(ridge_mae, 4),
        AUC_or_r=round(ridge_r, 4), permutation_p=None, wilcoxon_p_vs_no_augmentation=round(wp_ridge, 4),
        notes="Fixed primary 3-feature set (synthetic data has no other candidates to reselect from) -- "
              "compared against the nested-top-3 baseline; feature-set difference noted."))
    honesty_report.append(dict(method=method, model="Ridge", accuracy=ridge_mae, auc=ridge_r,
                                wilcoxon_p=wp_ridge, permutation_p=None, confirmed=confirmed_ridge,
                                suspicious=False, gain_pp=None))

    # Lasso
    lasso_mae = mean_absolute_error(actual_age, res["lasso_pred_age"])
    lasso_r, lasso_p = stats.pearsonr(actual_age, res["lasso_pred_age"])
    baseline_lasso_err = baseline_lasso.loc[participant_ids, "absolute_error"].values
    aug_lasso_err = np.abs(res["lasso_pred_age"] - actual_age)
    _, wp_lasso = stats.wilcoxon(aug_lasso_err, baseline_lasso_err)
    print(f"Lasso: MAE={lasso_mae:.4f} r={lasso_r:.4f} p={lasso_p:.4f}  Wilcoxon vs baseline p={wp_lasso:.4f}")
    confirmed_lasso = (lasso_mae < 13.47) and (wp_lasso < 0.05)

    comparison_rows.append(dict(
        method=method, model="Lasso (fixed primary-3, augmented)", n_real=29,
        n_synthetic_per_fold=round(avg_n_synth), accuracy_or_MAE=round(lasso_mae, 4),
        AUC_or_r=round(lasso_r, 4), permutation_p=None, wilcoxon_p_vs_no_augmentation=round(wp_lasso, 4),
        notes="Fixed primary 3-feature set (synthetic data has no other candidates to reselect from) -- "
              "compared against the nested-top-3 baseline; feature-set difference noted."))
    honesty_report.append(dict(method=method, model="Lasso", accuracy=lasso_mae, auc=lasso_r,
                                wilcoxon_p=wp_lasso, permutation_p=None, confirmed=confirmed_lasso,
                                suspicious=False, gain_pp=None))

comparison = pd.DataFrame(comparison_rows)
out_path = f"{BASE}\\augmentation_comparison.csv"
comparison.to_csv(out_path, index=False)
print(comparison.to_string(index=False))

# Final summary
any_confirmed = [h for h in honesty_report if h["confirmed"]]
if any_confirmed:
    for h in any_confirmed:
        print(f"CONFIRMED improvement: {h['method']} / {h['model']}")
else:
    print("No augmentation method produced a confirmed improvement (metric better AND Wilcoxon p<0.05) "
          "over the real-data-only baseline, for any model.")

suspicious_any = [h for h in honesty_report if h["suspicious"]]
if suspicious_any:
    for h in suspicious_any:
        print(f"SUSPICIOUS GAIN FLAGGED: {h['method']} / {h['model']} -- gain={h['gain_pp']:+.1f}pp, investigate before trusting.")
else:
    print("No method showed a suspiciously large (>15pp) accuracy gain.")

import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

import augment_lib as A

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = A.FEATURES

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)
participant_ids = df["participant"].astype(int).values

CONFIGS = {
    "A_RBF": lambda: ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2)),
    "B_Matern": lambda: ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5) + WhiteKernel(1.0, (1e-5, 1e2))}
BASELINE_MAE = {"A_RBF": 14.1998, "B_Matern": 13.9207}

GEN_CONFIG = {
    "noise": dict(fn=A.generate_noise, kwargs=dict(n_per_real=3, seed=1)),
    "smote": dict(fn=A.generate_smote, kwargs=dict(target_total=round(100 * 28 / 29), seed=1)),
    "gmm":   dict(fn=A.generate_gmm,   kwargs=dict(target_total=round(100 * 28 / 29), n_components=1, seed=1))}

baseline_gpr = {
    "A_RBF": pd.read_csv(f"{BASE}\\per_participant_gpr_A_RBF.csv").set_index("participant_id"),
    "B_Matern": pd.read_csv(f"{BASE}\\per_participant_gpr_B_Matern.csv").set_index("participant_id")}


def build_training_set(train_real_df, method):
    medians = train_real_df[FEATURES].median()
    real_part = train_real_df.copy()
    real_part[FEATURES] = real_part[FEATURES].fillna(medians)
    cfg = GEN_CONFIG[method]
    synth_part = cfg["fn"](train_real_df, **cfg["kwargs"])
    combined = pd.concat([
        real_part[["participant", "age", "group"] + FEATURES],
        synth_part[["participant", "age", "group"] + FEATURES]], ignore_index=True)
    return combined, medians


def run_augmented_gpr(kernel_fn, method):
    pred_age = np.empty(N)
    for i in range(N):
        train_real = df.drop(index=i)
        test_row = df.iloc[[i]]
        train_combined, medians = build_training_set(train_real, method)

        X_train_raw = train_combined[FEATURES].values.astype(float)
        X_test_raw = test_row[FEATURES].fillna(medians).values.astype(float)
        y_train = train_combined["age"].values.astype(float)

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        gpr = GaussianProcessRegressor(kernel=kernel_fn(), n_restarts_optimizer=10, normalize_y=True, random_state=0)
        gpr.fit(X_train, y_train)
        pred_age[i] = gpr.predict(X_test)[0]
    return pred_age


all_rows = []
for cfg_name, kernel_fn in CONFIGS.items():
    baseline_err = np.abs(baseline_gpr[cfg_name].loc[participant_ids, "predicted_age"].values - actual_age)
    for method in ["noise", "smote", "gmm"]:
        t0 = time.time()
        pred_age = run_augmented_gpr(kernel_fn, method)
        elapsed = time.time() - t0
        mae = float(np.mean(np.abs(pred_age - actual_age)))
        rmse = float(np.sqrt(np.mean((pred_age - actual_age) ** 2)))
        r, p = stats.pearsonr(actual_age, pred_age)
        aug_err = np.abs(pred_age - actual_age)
        _, wp = stats.wilcoxon(aug_err, baseline_err)
        confirmed = (mae < BASELINE_MAE[cfg_name]) and (wp < 0.05)
        print(f"Done in {elapsed:.1f}s. MAE={mae:.4f} RMSE={rmse:.4f} r={r:.4f} p={p:.4f}  "
              f"Wilcoxon vs baseline p={wp:.4f}  ({'CONFIRMED' if confirmed else 'not confirmed'})", flush=True)
        all_rows.append(dict(kernel=cfg_name, method=method, mae=round(mae, 4), rmse=round(rmse, 4),
                              pearson_r=round(r, 4), p_value=round(p, 4), wilcoxon_p_vs_baseline=round(wp, 4),
                              baseline_mae=BASELINE_MAE[cfg_name], confirmed_improvement=bool(confirmed),
                              elapsed_s=round(elapsed, 1)))

out = pd.DataFrame(all_rows)
out.to_csv(f"{BASE}\\gpr_augmentation_comparison.csv", index=False)
print(f"\nSaved -> {BASE}\\gpr_augmentation_comparison.csv")
print(out.to_string(index=False))

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from gplearn.genetic import SymbolicRegressor

import augment_lib as A

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = A.FEATURES  # peak_roll_walkingincline, stride_timing_cv_walking, rom_roll_ratio_stepforward_walk
OUT_CSV = f"{BASE}\\gplearn_augmented_noise_perfold.csv"

N_SEEDS = 3
RANDOM_STATE = 42
GP_PARAMS = dict(
    population_size=3000, generations=30, tournament_size=20,
    function_set=("add", "sub", "mul", "div"), parsimony_coefficient=0.05,
    const_range=(-10.0, 10.0), metric="mean absolute error")

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)


def make_gp_model(seed):
    return SymbolicRegressor(**GP_PARAMS, n_jobs=1, verbose=0, random_state=seed)


t0_all = time.time()
records = []
for i in range(N):
    fold_t0 = time.time()
    train_real = df.drop(index=i)
    test_row = df.iloc[[i]]

    medians = train_real[FEATURES].median()
    real_part = train_real.copy()
    real_part[FEATURES] = real_part[FEATURES].fillna(medians)
    synth_part = A.generate_noise(train_real, n_per_real=3, seed=1)

    train_combined = pd.concat([
        real_part[["participant", "age", "group"] + FEATURES],
        synth_part[["participant", "age", "group"] + FEATURES]], ignore_index=True)

    X_train = train_combined[FEATURES].values.astype(float)
    y_train = train_combined["age"].values.astype(float)
    X_test = test_row[FEATURES].fillna(medians).values.astype(float)
    y_test = float(test_row["age"].values[0])

    fold_preds = []
    for s in range(N_SEEDS):
        model = make_gp_model(seed=RANDOM_STATE + i * N_SEEDS + s)
        model.fit(X_train, y_train)
        fold_preds.append(float(model.predict(X_test)[0]))
    pred = float(np.median(fold_preds))

    records.append({
        "fold": i, "participant": int(test_row["participant"].values[0]),
        "actual_age": y_test, "predicted_age": pred, "abs_error": abs(pred - y_test),
        "n_synthetic": len(synth_part)})

    fold_elapsed = time.time() - fold_t0
    total_elapsed = time.time() - t0_all
    print(f"  [{i+1:2d}/{N}] participant={int(test_row['participant'].values[0]):<4} "
          f"actual={y_test:.0f} pred={pred:.1f} err={abs(pred-y_test):.1f}  "
          f"(fold {fold_elapsed:.0f}s, total {total_elapsed/60:.1f}min, "
          f"avg {total_elapsed/(i+1):.0f}s/fold, "
          f"ETA {(total_elapsed/(i+1))*(N-i-1)/60:.0f}min remaining)", flush=True)

elapsed = time.time() - t0_all
results_df = pd.DataFrame(records)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min). Saved -> {OUT_CSV}")

preds = results_df["predicted_age"].values
mae = float(np.mean(np.abs(preds - actual_age)))
rmse = float(np.sqrt(np.mean((preds - actual_age) ** 2)))
r, p = stats.pearsonr(actual_age, preds)
print(f"MAE={mae:.4f} RMSE={rmse:.4f} r={r:.4f} p={p:.4f}")

# Compare against existing gplearn (nested, forced top-3) baseline 
baseline = pd.read_csv(f"{BASE}\\loocv_gplearn_top3_perfold.csv")
baseline = baseline.set_index("participant")
participant_ids = df["participant"].astype(int).values
baseline_err = baseline.loc[participant_ids, "abs_error"].values
aug_err = results_df["abs_error"].values

_, wp = stats.wilcoxon(aug_err, baseline_err)

rng = np.random.default_rng(42)
diffs = []
for _ in range(5000):
    idx = rng.integers(0, N, N)
    diffs.append(aug_err[idx].mean() - baseline_err[idx].mean())
diffs = np.array(diffs)
ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
print(f"Bootstrap 95% CI (MAE diff, 5000 resamples): [{ci_lo:.4f}, {ci_hi:.4f}]")

# gplearn_augmented_smote_gmm
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from gplearn.genetic import SymbolicRegressor

import augment_lib as A

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = A.FEATURES

N_SEEDS = 3
RANDOM_STATE = 42
GP_PARAMS = dict(
    population_size=3000, generations=30, tournament_size=20,
    function_set=("add", "sub", "mul", "div"), parsimony_coefficient=0.05,
    const_range=(-10.0, 10.0), metric="mean absolute error")
BASELINE_MAE = 16.4464

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)
participant_ids = df["participant"].astype(int).values

baseline = pd.read_csv(f"{BASE}\\loocv_gplearn_top3_perfold.csv").set_index("participant")
baseline_err = baseline.loc[participant_ids, "abs_error"].values

GEN_CONFIG = {
    "smote": dict(fn=A.generate_smote, kwargs=dict(target_total=round(100 * 28 / 29), seed=1)),
    "gmm":   dict(fn=A.generate_gmm,   kwargs=dict(target_total=round(100 * 28 / 29), n_components=1, seed=1))}


def make_gp_model(seed):
    return SymbolicRegressor(**GP_PARAMS, n_jobs=1, verbose=0, random_state=seed)


def run_method(method, seed_offset):
    cfg = GEN_CONFIG[method]
    pred_age = np.empty(N)
    t0_all = time.time()
    for i in range(N):
        fold_t0 = time.time()
        train_real = df.drop(index=i)
        test_row = df.iloc[[i]]

        medians = train_real[FEATURES].median()
        real_part = train_real.copy()
        real_part[FEATURES] = real_part[FEATURES].fillna(medians)
        synth_part = cfg["fn"](train_real, **cfg["kwargs"])

        train_combined = pd.concat([
            real_part[["participant", "age", "group"] + FEATURES],
            synth_part[["participant", "age", "group"] + FEATURES]], ignore_index=True)

        X_train = train_combined[FEATURES].values.astype(float)
        y_train = train_combined["age"].values.astype(float)
        X_test = test_row[FEATURES].fillna(medians).values.astype(float)
        y_test = float(test_row["age"].values[0])

        fold_preds = []
        for s in range(N_SEEDS):
            model = make_gp_model(seed=RANDOM_STATE + seed_offset + i * N_SEEDS + s)
            model.fit(X_train, y_train)
            fold_preds.append(float(model.predict(X_test)[0]))
        pred_age[i] = float(np.median(fold_preds))

        fold_elapsed = time.time() - fold_t0
        total_elapsed = time.time() - t0_all
        print(f"  [{method}] [{i+1:2d}/{N}] participant={int(test_row['participant'].values[0]):<4} "
              f"actual={y_test:.0f} pred={pred_age[i]:.1f} err={abs(pred_age[i]-y_test):.1f}  "
              f"(fold {fold_elapsed:.0f}s, total {total_elapsed/60:.1f}min, "
              f"ETA {(total_elapsed/(i+1))*(N-i-1)/60:.0f}min remaining)", flush=True)

    elapsed = time.time() - t0_all
    mae = float(np.mean(np.abs(pred_age - actual_age)))
    rmse = float(np.sqrt(np.mean((pred_age - actual_age) ** 2)))
    r, p = stats.pearsonr(actual_age, pred_age)
    aug_err = np.abs(pred_age - actual_age)
    _, wp = stats.wilcoxon(aug_err, baseline_err)
    confirmed = (mae < BASELINE_MAE) and (wp < 0.05)

    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(5000):
        idx = rng.integers(0, N, N)
        diffs.append(aug_err[idx].mean() - baseline_err[idx].mean())
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

    print(f"MAE={mae:.4f} RMSE={rmse:.4f} r={r:.4f} p={p:.4f}")
    print(f"Wilcoxon vs baseline (MAE=16.4464): p={wp:.4f}  ({'CONFIRMED' if confirmed else 'NOT confirmed'})")
    print(f"Bootstrap 95% CI (MAE diff, 5000 resamples): [{ci_lo:.4f}, {ci_hi:.4f}]")

    pd.DataFrame({
        "participant": participant_ids, "actual_age": actual_age, "predicted_age": pred_age,
        "abs_error": aug_err}).to_csv(f"{BASE}\\gplearn_augmented_{method}_perfold.csv", index=False)
    print(f"Saved -> {BASE}\\gplearn_augmented_{method}_perfold.csv")

    return dict(method=method, mae=round(mae, 4), rmse=round(rmse, 4), pearson_r=round(r, 4),
                p_value=round(p, 4), wilcoxon_p_vs_baseline=round(wp, 4), baseline_mae=BASELINE_MAE,
                confirmed_improvement=bool(confirmed), bootstrap_ci_lower=round(ci_lo, 4),
                bootstrap_ci_upper=round(ci_hi, 4), elapsed_min=round(elapsed / 60, 1))


results = []
results.append(run_method("smote", seed_offset=1000))
results.append(run_method("gmm", seed_offset=2000))

out = pd.DataFrame(results)
out.to_csv(f"{BASE}\\gplearn_augmentation_smote_gmm_comparison.csv", index=False)
print(f"\nSaved -> {BASE}\\gplearn_augmentation_smote_gmm_comparison.csv")
print(out.to_string(index=False))

# gplearn_gmm_only

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from gplearn.genetic import SymbolicRegressor

import augment_lib as A

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = A.FEATURES

N_SEEDS = 3
RANDOM_STATE = 42
GP_PARAMS = dict(
    population_size=3000, generations=30, tournament_size=20,
    function_set=("add", "sub", "mul", "div"), parsimony_coefficient=0.05,
    const_range=(-10.0, 10.0), metric="mean absolute error")
BASELINE_MAE = 16.4464

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)
participant_ids = df["participant"].astype(int).values

baseline = pd.read_csv(f"{BASE}\\loocv_gplearn_top3_perfold.csv").set_index("participant")
baseline_err = baseline.loc[participant_ids, "abs_error"].values

cfg = dict(fn=A.generate_gmm, kwargs=dict(target_total=round(100 * 28 / 29), n_components=1, seed=1))


def make_gp_model(seed):
    return SymbolicRegressor(**GP_PARAMS, n_jobs=1, verbose=0, random_state=seed)


pred_age = np.empty(N)
t0_all = time.time()
for i in range(N):
    fold_t0 = time.time()
    train_real = df.drop(index=i)
    test_row = df.iloc[[i]]

    medians = train_real[FEATURES].median()
    real_part = train_real.copy()
    real_part[FEATURES] = real_part[FEATURES].fillna(medians)
    synth_part = cfg["fn"](train_real, **cfg["kwargs"])

    train_combined = pd.concat([
        real_part[["participant", "age", "group"] + FEATURES],
        synth_part[["participant", "age", "group"] + FEATURES],
    ], ignore_index=True)

    X_train = train_combined[FEATURES].values.astype(float)
    y_train = train_combined["age"].values.astype(float)
    X_test = test_row[FEATURES].fillna(medians).values.astype(float)
    y_test = float(test_row["age"].values[0])

    fold_preds = []
    for s in range(N_SEEDS):
        model = make_gp_model(seed=RANDOM_STATE + 2000 + i * N_SEEDS + s)
        model.fit(X_train, y_train)
        fold_preds.append(float(model.predict(X_test)[0]))
    pred_age[i] = float(np.median(fold_preds))

    fold_elapsed = time.time() - fold_t0
    total_elapsed = time.time() - t0_all
    print(f"  [gmm] [{i+1:2d}/{N}] participant={int(test_row['participant'].values[0]):<4} "
          f"actual={y_test:.0f} pred={pred_age[i]:.1f} err={abs(pred_age[i]-y_test):.1f}  "
          f"(fold {fold_elapsed:.0f}s, total {total_elapsed/60:.1f}min, "
          f"ETA {(total_elapsed/(i+1))*(N-i-1)/60:.0f}min remaining)", flush=True)

elapsed = time.time() - t0_all
mae = float(np.mean(np.abs(pred_age - actual_age)))
rmse = float(np.sqrt(np.mean((pred_age - actual_age) ** 2)))
r, p = stats.pearsonr(actual_age, pred_age)
aug_err = np.abs(pred_age - actual_age)
_, wp = stats.wilcoxon(aug_err, baseline_err)
confirmed = (mae < BASELINE_MAE) and (wp < 0.05)

rng = np.random.default_rng(42)
diffs = []
for _ in range(5000):
    idx = rng.integers(0, N, N)
    diffs.append(aug_err[idx].mean() - baseline_err[idx].mean())
diffs = np.array(diffs)
ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

print(f"MAE={mae:.4f} RMSE={rmse:.4f} r={r:.4f} p={p:.4f}")
print(f"Wilcoxon vs baseline (MAE=16.4464): p={wp:.4f}  ({'CONFIRMED' if confirmed else 'NOT confirmed'})")
print(f"Bootstrap 95% CI (MAE diff, 5000 resamples): [{ci_lo:.4f}, {ci_hi:.4f}]")

# Combine with already-saved SMOTE result into the final comparison table
smote_df = pd.read_csv(f"{BASE}\\gplearn_augmented_smote_perfold.csv")
smote_err = smote_df["abs_error"].values
smote_mae = float(smote_df["abs_error"].mean())
smote_rmse = float(np.sqrt((smote_df["abs_error"] ** 2).mean()))
smote_r, smote_p = stats.pearsonr(smote_df["actual_age"], smote_df["predicted_age"])
_, smote_wp = stats.wilcoxon(smote_err, baseline_err)
smote_confirmed = (smote_mae < BASELINE_MAE) and (smote_wp < 0.05)
diffs2 = []
rng2 = np.random.default_rng(42)
for _ in range(5000):
    idx = rng2.integers(0, N, N)
    diffs2.append(smote_err[idx].mean() - baseline_err[idx].mean())
diffs2 = np.array(diffs2)
smote_ci_lo, smote_ci_hi = np.percentile(diffs2, [2.5, 97.5])

results = [
    dict(method="smote", mae=round(smote_mae, 4), rmse=round(smote_rmse, 4), pearson_r=round(smote_r, 4),
         p_value=round(smote_p, 4), wilcoxon_p_vs_baseline=round(smote_wp, 4), baseline_mae=BASELINE_MAE,
         confirmed_improvement=bool(smote_confirmed), bootstrap_ci_lower=round(smote_ci_lo, 4),
         bootstrap_ci_upper=round(smote_ci_hi, 4)),
    dict(method="gmm", mae=round(mae, 4), rmse=round(rmse, 4), pearson_r=round(r, 4),
         p_value=round(p, 4), wilcoxon_p_vs_baseline=round(wp, 4), baseline_mae=BASELINE_MAE,
         confirmed_improvement=bool(confirmed), bootstrap_ci_lower=round(ci_lo, 4),
         bootstrap_ci_upper=round(ci_hi, 4))]
out = pd.DataFrame(results)
out.to_csv(f"{BASE}\\gplearn_augmentation_smote_gmm_comparison.csv", index=False)
print(f"\nSaved -> {BASE}\\gplearn_augmentation_smote_gmm_comparison.csv")
print(out.to_string(index=False))


