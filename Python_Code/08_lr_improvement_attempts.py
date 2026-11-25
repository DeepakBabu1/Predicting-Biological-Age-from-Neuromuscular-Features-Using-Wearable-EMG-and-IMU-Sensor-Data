import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
EXCLUDE = {"participant", "age", "group"}

_df_raw = pd.read_csv(DATA)
_dmci_cols = [c for c in _df_raw.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in _df_raw.columns if c not in EXCLUDE and c not in _dmci_cols]
df = _df_raw.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
ORIGINAL_FIXED = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]
actual_group = df["group"].values.astype(int)


def select_features(train_df, n=3, threshold=None, target_col="age"):
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[[target_col, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 5:
            continue
        rho, _ = stats.spearmanr(valid[target_col], valid[col])
        rows.append((col, rho))
    rho_df = pd.DataFrame(rows, columns=["feature", "rho"]).dropna()
    rho_df["abs_rho"] = rho_df["rho"].abs()
    if threshold is None:
        return rho_df.sort_values("abs_rho", ascending=False).head(n)["feature"].tolist()
    passing = rho_df[rho_df["abs_rho"] > threshold]
    if len(passing) == 0:
        return rho_df.sort_values("abs_rho", ascending=False).head(1)["feature"].tolist()
    return passing.sort_values("abs_rho", ascending=False).head(n)["feature"].tolist()


def to_binary(group_array):
    return (np.asarray(group_array).astype(int) == 2).astype(int)


def inner_loocv_proba(train_df, features, model_kwargs):
    idx_list = train_df.index.tolist()
    inner_actual, inner_proba = [], []
    for inner_test_idx in idx_list:
        inner_train = train_df.drop(index=inner_test_idx)
        inner_test = train_df.loc[[inner_test_idx]]
        med = inner_train[features].median()
        Xtr = inner_train[features].fillna(med).values.astype(float)
        ytr = to_binary(inner_train["group"].values)
        Xte = inner_test[features].fillna(med).values.astype(float)
        if len(np.unique(ytr)) < 2:
            inner_proba.append(0.5)
        else:
            clf = LogisticRegression(**model_kwargs)
            clf.fit(Xtr, ytr)
            classes = list(clf.classes_)
            old_idx = classes.index(1) if 1 in classes else 0
            inner_proba.append(clf.predict_proba(Xte)[0][old_idx])
        inner_actual.append(int(inner_test["group"].values[0] == 2))
    return np.array(inner_actual), np.array(inner_proba)


def select_best_C(train_df, features, C_grid, base_kwargs):
    best_C, best_acc = C_grid[0], -1
    for C in C_grid:
        kw = {**base_kwargs, "C": C}
        actual, proba = inner_loocv_proba(train_df, features, kw)
        acc = ((proba >= 0.5).astype(int) == actual).mean()
        if acc > best_acc:
            best_acc, best_C = acc, C
    return best_C


def select_best_threshold(train_df, features, model_kwargs, threshold_grid):
    actual, proba = inner_loocv_proba(train_df, features, model_kwargs)
    best_t, best_acc = 0.5, -1
    for t in threshold_grid:
        acc = ((proba >= t).astype(int) == actual).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def run_nested(group_labels, feature_kwargs, model_kwargs, decision_threshold=0.5,
               tune_C=False, C_grid=None, tune_threshold=False, threshold_grid=None,
               fixed_features=None, verbose=False):
    d = df.copy()
    d["group"] = group_labels
    pred_group = np.empty(N, dtype=int)
    pred_proba = np.empty(N)

    for i in range(N):
        train_df = d.drop(index=i)
        test_row = d.iloc[[i]]

        selected = fixed_features if fixed_features is not None else select_features(train_df, **feature_kwargs)

        med = train_df[selected].median()
        X_train = train_df[selected].fillna(med).values.astype(float)
        y_train = to_binary(train_df["group"].values)
        X_test = test_row[selected].fillna(med).values.astype(float)

        mk = dict(model_kwargs)
        if tune_C:
            mk["C"] = select_best_C(train_df, selected, C_grid, model_kwargs)

        clf = LogisticRegression(**mk)
        clf.fit(X_train, y_train)
        classes = list(clf.classes_)
        old_idx = classes.index(1) if 1 in classes else 0
        proba_old = clf.predict_proba(X_test)[0][old_idx]

        thresh = decision_threshold
        if tune_threshold:
            thresh = select_best_threshold(train_df, selected, mk, threshold_grid)

        pred_group[i] = 2 if proba_old >= thresh else 1
        pred_proba[i] = proba_old

        if verbose:
            print(f"  [{i+1:2d}/{N}] actual={int(test_row['group'].values[0])} pred={pred_group[i]} "
                  f"proba={proba_old:.3f} thresh={thresh:.2f} feat={selected}")

    return pred_group, pred_proba


def compute_metrics(pred_group, pred_proba, actual):
    accuracy = float((pred_group == actual).mean())
    cm = confusion_matrix(actual, pred_group, labels=[1, 2])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = roc_auc_score(actual, pred_proba)
    return accuracy, sensitivity, specificity, auc, cm


def wilcoxon_vs(pred_a, pred_b, actual):
    ca = (pred_a == actual).astype(int)
    cb = (pred_b == actual).astype(int)
    if np.any(ca != cb):
        _, p = stats.wilcoxon(ca, cb)
        return float(p)
    return 1.0


def permutation_test(feature_kwargs=None, model_kwargs=None, decision_threshold=0.5,
                      tune_C=False, C_grid=None, tune_threshold=False, threshold_grid=None,
                      fixed_features=None, n_perms=500, observed_auc=None, seed=42):
    rng = np.random.default_rng(seed)
    null_aucs = []
    t0 = time.time()
    for p in range(n_perms):
        shuffled = rng.permutation(df["group"].values)
        _, proba = run_nested(shuffled, feature_kwargs or {}, model_kwargs or {}, decision_threshold,
                               tune_C, C_grid, tune_threshold, threshold_grid, fixed_features)
        null_aucs.append(roc_auc_score((shuffled == 2).astype(int), proba))
    elapsed = time.time() - t0
    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))
    return p_value, elapsed, null_aucs

import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\logistic_regression_improvement_attempts.csv"

EXCLUDE = {"participant", "age", "group"}
df = pd.read_csv(DATA)
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
ORIGINAL_FIXED = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]
print("N:", N)


# Feature selection 
def select_features(train_df, n=3, threshold=None, target_col="group"):
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[[target_col, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 5:
            continue
        rho, _ = stats.spearmanr(valid[target_col], valid[col])
        rows.append((col, rho))
    rho_df = pd.DataFrame(rows, columns=["feature", "rho"]).dropna()
    rho_df["abs_rho"] = rho_df["rho"].abs()
    if threshold is None:
        return rho_df.sort_values("abs_rho", ascending=False).head(n)["feature"].tolist()
    passing = rho_df[rho_df["abs_rho"] > threshold]
    if len(passing) == 0:
        return rho_df.sort_values("abs_rho", ascending=False).head(1)["feature"].tolist()
    return passing.sort_values("abs_rho", ascending=False).head(n)["feature"].tolist()


def to_binary(group_array):
    return (np.asarray(group_array).astype(int) == 2).astype(int)


# Inner LOOCV over training fold (for hyperparameter/threshold selection) 
def inner_loocv_proba(train_df, features, model_kwargs):
    idx_list = train_df.index.tolist()
    inner_actual, inner_proba = [], []
    for inner_test_idx in idx_list:
        inner_train = train_df.drop(index=inner_test_idx)
        inner_test = train_df.loc[[inner_test_idx]]
        med = inner_train[features].median()
        Xtr = inner_train[features].fillna(med).values.astype(float)
        ytr = to_binary(inner_train["group"].values)
        Xte = inner_test[features].fillna(med).values.astype(float)
        if len(np.unique(ytr)) < 2:
            inner_proba.append(0.5)
        else:
            clf = LogisticRegression(**model_kwargs)
            clf.fit(Xtr, ytr)
            classes = list(clf.classes_)
            old_idx = classes.index(1) if 1 in classes else 0
            inner_proba.append(clf.predict_proba(Xte)[0][old_idx])
        inner_actual.append(int(inner_test["group"].values[0] == 2))
    return np.array(inner_actual), np.array(inner_proba)


def select_best_C(train_df, features, C_grid, base_kwargs):
    best_C, best_acc = C_grid[0], -1
    for C in C_grid:
        kw = {**base_kwargs, "C": C}
        actual, proba = inner_loocv_proba(train_df, features, kw)
        acc = ((proba >= 0.5).astype(int) == actual).mean()
        if acc > best_acc:
            best_acc, best_C = acc, C
    return best_C


def select_best_threshold(train_df, features, model_kwargs, threshold_grid):
    actual, proba = inner_loocv_proba(train_df, features, model_kwargs)
    best_t, best_acc = 0.5, -1
    for t in threshold_grid:
        acc = ((proba >= t).astype(int) == actual).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


# Main nested LOOCV runner 
def run_nested(group_labels, feature_kwargs, model_kwargs, decision_threshold=0.5,
               tune_C=False, C_grid=None, tune_threshold=False, threshold_grid=None,
               fixed_features=None, verbose=False):
    d = df.copy()
    d["group"] = group_labels
    pred_group = np.empty(N, dtype=int)
    pred_proba = np.empty(N)

    for i in range(N):
        train_df = d.drop(index=i)
        test_row = d.iloc[[i]]

        selected = fixed_features if fixed_features is not None else select_features(train_df, **feature_kwargs)
        med = train_df[selected].median()
        X_train = train_df[selected].fillna(med).values.astype(float)
        y_train = to_binary(train_df["group"].values)
        X_test = test_row[selected].fillna(med).values.astype(float)

        mk = dict(model_kwargs)
        if tune_C:
            mk["C"] = select_best_C(train_df, selected, C_grid, model_kwargs)

        clf = LogisticRegression(**mk)
        clf.fit(X_train, y_train)
        classes = list(clf.classes_)
        old_idx = classes.index(1) if 1 in classes else 0
        proba_old = clf.predict_proba(X_test)[0][old_idx]

        thresh = decision_threshold
        if tune_threshold:
            thresh = select_best_threshold(train_df, selected, mk, threshold_grid)

        pred_group[i] = 2 if proba_old >= thresh else 1
        pred_proba[i] = proba_old

        if verbose:
            print(f"  [{i+1:2d}/{N}] actual={int(test_row['group'].values[0])} pred={pred_group[i]} "
                  f"proba={proba_old:.3f} thresh={thresh:.2f} feat={selected}")

    return pred_group, pred_proba


def compute_metrics(pred_group, pred_proba, actual_group):
    accuracy = float((pred_group == actual_group).mean())
    cm = confusion_matrix(actual_group, pred_group, labels=[1, 2])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = roc_auc_score(actual_group, pred_proba)
    return accuracy, sensitivity, specificity, auc, cm


def permutation_test(feature_kwargs, model_kwargs, decision_threshold=0.5,
                      tune_C=False, C_grid=None, tune_threshold=False, threshold_grid=None,
                      fixed_features=None, n_perms=500, observed_auc=None):
    rng = np.random.default_rng(42)
    null_aucs = []
    t0 = time.time()
    for p in range(n_perms):
        shuffled = rng.permutation(df["group"].values)
        _, proba = run_nested(shuffled, feature_kwargs, model_kwargs, decision_threshold,
                               tune_C, C_grid, tune_threshold, threshold_grid, fixed_features)
        null_aucs.append(roc_auc_score((shuffled == 2).astype(int), proba))
    elapsed = time.time() - t0
    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))
    return p_value, elapsed, null_aucs


actual_group = df["group"].values.astype(int)
results = []

# Row 1: ORIGINAL baseline (fixed features, as previously reported)

print("ROW 1: ORIGINAL BASELINE (fixed features, as previously reported)")
t0 = time.time()
pred1, proba1 = run_nested(df["group"].values, {}, {}, fixed_features=ORIGINAL_FIXED)
acc1, sens1, spec1, auc1, cm1 = compute_metrics(pred1, proba1, actual_group)
print(f"accuracy={acc1:.4f} sensitivity={sens1:.4f} specificity={spec1:.4f} AUC={auc1:.4f}  "
      f"(matches previously reported 0.621/0.678: {abs(acc1-0.6207)<0.001})")
results.append(dict(attempt="Row 1: Original baseline (fixed features)", accuracy=acc1, sensitivity=sens1,
                     specificity=spec1, AUC=auc1, permutation_p=0.032, n_features=3,
                     wilcoxon_p_vs_baseline=None, beats_baseline=None))
# Row 2: CORRECTED baseline (nested top-3, selected against GROUP, not age) 

print("ROW 2: CORRECTED BASELINE (nested top-3 reselection vs GROUP, per fold)")
BASE_FEATURE_KW = dict(n=3, threshold=None, target_col="group")
BASE_MODEL_KW = {}
t0 = time.time()
pred2, proba2 = run_nested(df["group"].values, BASE_FEATURE_KW, BASE_MODEL_KW, verbose=True)
elapsed2 = time.time() - t0
acc2, sens2, spec2, auc2, cm2 = compute_metrics(pred2, proba2, actual_group)
print(f"\naccuracy={acc2:.4f} sensitivity={sens2:.4f} specificity={spec2:.4f} AUC={auc2:.4f}  "
      f"(run took {elapsed2:.2f}s)")

correct1 = (pred1 == actual_group).astype(int)
correct2 = (pred2 == actual_group).astype(int)
diff_12 = correct2 - correct1
if np.any(diff_12 != 0):
    wstat, wp_2v1 = stats.wilcoxon(correct2, correct1)
else:
    wp_2v1 = 1.0
print(f"Wilcoxon (corrected baseline vs original baseline), per-participant correctness: p={wp_2v1:.4f}")

p2, elapsed_p2, null2 = permutation_test(BASE_FEATURE_KW, BASE_MODEL_KW, observed_auc=auc2, n_perms=500)

results.append(dict(attempt="Row 2: Corrected baseline (nested top-3 vs group)", accuracy=acc2, sensitivity=sens2,
                     specificity=spec2, AUC=auc2, permutation_p=round(p2, 4), n_features=3,
                     wilcoxon_p_vs_baseline=round(wp_2v1, 4), beats_baseline=bool(acc2 > acc1 and wp_2v1 < 0.05)))
import time
import lr_improve_lib as L

thresh_grid = [round(0.3 + 0.05 * i, 2) for i in range(9)]
feature_kwargs = dict(n=3, threshold=None, target_col="age")
model_kwargs = {"class_weight": "balanced"}

pred_i5, proba_i5 = L.run_nested(L.df["group"].values, feature_kwargs, model_kwargs,
                                  tune_threshold=True, threshold_grid=thresh_grid)
acc, sens, spec, auc, cm = L.compute_metrics(pred_i5, proba_i5, L.actual_group)
print(f"Improvement 5 real result: acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} auc={auc:.4f}")

t0 = time.time()
p_value, elapsed, null_aucs = L.permutation_test(
    feature_kwargs=feature_kwargs, model_kwargs=model_kwargs,
    tune_threshold=True, threshold_grid=thresh_grid,
    n_perms=500, observed_auc=auc,
)
print(f"Done in {elapsed:.1f}s")
print(f"Real AUC: {auc:.4f}")
print(f"Null AUC: mean={null_aucs.mean():.4f} sd={null_aucs.std():.4f} 95th_pctile={(null_aucs.mean()):.4f}")
import numpy as np
print(f"Null 95th percentile: {np.percentile(null_aucs, 95):.4f}")
print(f"Permutation p-value: {p_value:.4f}  ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})")


import pandas as pd

OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\logistic_regression_improvement_attempts.csv"

rows = [
    dict(attempt="Row 1: Original baseline (fixed 3 features, not re-derived per fold)",
         accuracy=0.6207, sensitivity=0.5385, specificity=0.6875, AUC=0.6779,
         n_features=3, wilcoxon_p_vs_row2b=0.0455, permutation_p=0.032),

    dict(attempt="Row 2: Corrected baseline (nested top-3 vs GROUP, per fold)",
         accuracy=0.2414, sensitivity=0.3077, specificity=0.1875, AUC=0.1731,
         n_features=3, wilcoxon_p_vs_row2b=None, permutation_p=0.834,),

    dict(attempt="Row 2b: Corrected baseline (nested top-3 vs AGE, per fold), TRUE baseline for Improvements 1-5",
         accuracy=0.4828, sensitivity=0.5385, specificity=0.4375, AUC=0.4952,
         n_features=3, wilcoxon_p_vs_row2b=None, permutation_p=None),
    
    dict(attempt="Improvement 1: Up to 5 features, threshold |rho|>0.3 (nested per fold)",
         accuracy=0.4828, sensitivity=0.4615, specificity=0.5000, AUC=0.3894,
         n_features="<=5, varies", wilcoxon_p_vs_row2b=0.1025, permutation_p=None),
    
    dict(attempt="Improvement 2: class_weight='balanced' (nested top-3 vs age)",
         accuracy=0.5172, sensitivity=0.6923, specificity=0.3750, AUC=0.5481,
         n_features=3, wilcoxon_p_vs_row2b=0.2568, permutation_p=None),

    dict(attempt="Improvement 3: C grid search via inner LOOCV (nested top-3 vs age)",
         accuracy=0.4483, sensitivity=0.4615, specificity=0.4375, AUC=0.4615,
         n_features=3, wilcoxon_p_vs_row2b=0.0588, permutation_p=None),

    dict(attempt="Improvement 4: Decision threshold tuned via inner LOOCV (nested top-3 vs age)",
         accuracy=0.5862, sensitivity=0.8462, specificity=0.3750, AUC=0.4952,
         n_features=3, wilcoxon_p_vs_row2b=0.1797, permutation_p=None),

    dict(attempt="Improvement 5: class_weight='balanced' + threshold tuning combined (nested top-3 vs age)",
         accuracy=0.6207, sensitivity=0.9231, specificity=0.3750, AUC=0.5481,
         n_features=3, wilcoxon_p_vs_row2b=0.1025)]

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")
print(f"Rows: {len(df)}")
pd.set_option("display.max_colwidth", 30)
pd.set_option("display.width", 200)
print(df[["attempt", "accuracy", "AUC", "wilcoxon_p_vs_row2b"]].to_string(index=False))


