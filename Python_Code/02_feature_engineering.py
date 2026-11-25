import pandas as pd
import numpy as np

IN_PATH  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\pipeline_outputs\pipeline_results.csv"
OUT_PATH = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"

df = pd.read_csv(IN_PATH)
df.columns = [str(c).strip() for c in df.columns]

# Column selection
VAF_COLS = ["vaf_walking", "vaf_walkingincline", "vaf_stepforward", "vaf_steplateral"]
DMCI_COLS = ["dmci_walking", "dmci_walkingincline", "dmci_stepforward", "dmci_steplateral"]

CCI_COLS = [
    "cci_knee_walking", "cci_knee_walkingincline", "cci_knee_stepforward", "cci_knee_steplateral",
    "cci_ankle_walking", "cci_ankle_walkingincline", "cci_ankle_stepforward", "cci_ankle_steplateral"]

KINEMATIC_COLS_GAIT = []
for ex in ["walking", "walkingincline"]:
    KINEMATIC_COLS_GAIT += [
        f"peak_roll_{ex}", f"rom_roll_{ex}", f"peak_pitch_{ex}", f"rom_pitch_{ex}",
        f"stride_timing_cv_{ex}"]

KINEMATIC_COLS_STEP = []
for ex in ["stepforward", "steplateral"]:
    KINEMATIC_COLS_STEP += [
        f"peak_roll_{ex}", f"rom_roll_{ex}", f"peak_pitch_{ex}", f"rom_pitch_{ex}"]

KINEMATIC_COLS = KINEMATIC_COLS_GAIT + KINEMATIC_COLS_STEP

base_cols = ["participant", "age", "group"] + VAF_COLS + DMCI_COLS + CCI_COLS + KINEMATIC_COLS
missing = [c for c in base_cols if c not in df.columns]
if missing:
    raise ValueError(f"Expected columns missing from {IN_PATH}: {missing}")

feat = df[base_cols].copy()

# Cross-task ratios 
feat["vaf_ratio_stepforward_walk"] = feat["vaf_stepforward"] / feat["vaf_walking"]
feat["cci_knee_ratio_stepforward_walk"] = feat["cci_knee_stepforward"] / feat["cci_knee_walking"]
feat["rom_roll_ratio_stepforward_walk"] = feat["rom_roll_stepforward"] / feat["rom_roll_walking"]

# Save 
feat.to_csv(OUT_PATH, index=False)

print(f"Saved -> {OUT_PATH}")
print(f"\nShape: {feat.shape}")
print(f"\nColumns ({len(feat.columns)}):")
for c in feat.columns:
    print(f"  {c}")


import pandas as pd
import numpy as np
from scipy import stats

IN_PATH  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_PATH = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_correlations.csv"

RHO_THRESHOLD = 0.3
REDUNDANCY_THRESHOLD = 0.8

df = pd.read_csv(IN_PATH)

EXCLUDE_ALWAYS = {"participant", "age", "group"}
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
excluded_cols = EXCLUDE_ALWAYS.union(dmci_cols)

feature_cols = [c for c in df.columns if c not in excluded_cols]

vaf_cols = [c for c in feature_cols if c.startswith("vaf_") and "ratio" not in c]
cci_cols = [c for c in feature_cols if c.startswith("cci_") and "ratio" not in c]
kinematic_cols = [c for c in feature_cols if c not in vaf_cols + cci_cols and "ratio" not in c]
ratio_cols = [c for c in feature_cols if "ratio" in c]

# correlation screening 
rows = []
for col in feature_cols:
    valid = df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid)
    if n < 5:
        rows.append({"feature": col, "n": n, "spearman_rho": np.nan, "spearman_p": np.nan,
                      "pearson_r": np.nan, "pearson_p": np.nan, "passes_0.3": False})
        continue
    rho, p_s = stats.spearmanr(valid["age"], valid[col])
    r, p_p = stats.pearsonr(valid["age"], valid[col])
    rows.append({
        "feature": col, "n": n,
        "spearman_rho": round(rho, 4), "spearman_p": round(p_s, 4),
        "pearson_r": round(r, 4), "pearson_p": round(p_p, 4),
        "passes_0.3": abs(rho) > RHO_THRESHOLD})

corr_df = pd.DataFrame(rows)
corr_df["abs_rho"] = corr_df["spearman_rho"].abs()
corr_df = corr_df.sort_values("abs_rho", ascending=False, na_position="last").drop(columns="abs_rho")
corr_df.to_csv(OUT_PATH, index=False)

print(f"Saved -> {OUT_PATH}\n")
print(corr_df.to_string(index=False))

passed = corr_df[corr_df["passes_0.3"]].copy()
failed = corr_df[~corr_df["passes_0.3"]].copy()

print("FEATURES THAT PASS |rho| > 0.3:")
for _, r in passed.iterrows():
    print(f"  {r['feature']:35s}  rho={r['spearman_rho']:+.3f}  p={r['spearman_p']:.4f}  n={r['n']}")

print(f"\nFEATURES THAT FAIL |rho| > 0.3: {len(failed)} features")

if len(passed):
    complete_mask = df[passed["feature"].tolist()].notna().all(axis=1)
    n_complete = int(complete_mask.sum())
    print(f"\nParticipants with complete data across all {len(passed)} passing features: {n_complete} / {len(df)}")
else:
    n_complete = 0

# Redundancy check among passing features 
redundant_pairs = []
if len(passed) >= 2:
    pass_names = passed["feature"].tolist()
    sub = df[pass_names]
    rmat = sub.corr(method="pearson")
    for i in range(len(pass_names)):
        for j in range(i + 1, len(pass_names)):
            r_ij = rmat.iloc[i, j]
            if pd.notna(r_ij) and abs(r_ij) > REDUNDANCY_THRESHOLD:
                redundant_pairs.append((pass_names[i], pass_names[j], round(float(r_ij), 3)))

# final summary block 
n_participants = len(df)
n_vaf = len(vaf_cols)
n_dmci = len(dmci_cols)
n_cci = len(cci_cols)
n_kinematic = len(kinematic_cols)
n_ratio = len(ratio_cols)
n_total_computed = n_vaf + n_dmci + n_cci + n_kinematic + n_ratio


print(f"Participants in feature table: {n_participants}")
print(f"Features computed: {n_total_computed} total")
print(f"VAF features: {n_vaf}")
print(f"DMCI features: {n_dmci} (excluded from screening)")
print(f"CCI features: {n_cci}")
print(f"Kinematic features: {n_kinematic}")
print(f"Cross-task ratios: {n_ratio}")

for _, r in passed.iterrows():
    print(f"  {r['feature']:35s}  rho={r['spearman_rho']:+.3f}  p={r['spearman_p']:.4f}")
print(f"FAILED: {len(failed)} features")

print("\nRECOMMENDED FEATURE SET FOR MODEL:")
if len(passed) == 0:
    print("(none passed the |rho| > 0.3 threshold)")
else:
    flagged = set()
    for a, b, r_ij in redundant_pairs:
        flagged.add(a)
        flagged.add(b)
    for _, r in passed.iterrows():
        tag = " [REDUNDANT with another passing feature -- see below]" if r["feature"] in flagged else " [independent]"
        print(f"  {r['feature']}{tag}")
    if redundant_pairs:
        print("\n  Redundant pairs (|Pearson r| > {:.1f} between passing features):".format(REDUNDANCY_THRESHOLD))
        for a, b, r_ij in redundant_pairs:
            print(f"{a}  <->  {b}   r={r_ij}")
    else:
        print("\n No pairs of passing features exceed |r| > {:.1f} -- all appear independent.".format(REDUNDANCY_THRESHOLD))

if len(passed):
    incomplete = df.loc[~complete_mask, "participant"].tolist() if "participant" in df.columns else []
    if incomplete:
        print(f"  {len(incomplete)} participant(s) have at least one NaN among the recommended features: {incomplete}")
    else:
        print("No participants have missing data among the recommended feature set.")
else:
    print("N/A -- no features passed screening.")

import pandas as pd
from collections import Counter

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
FIXED_3 = "peak_roll_walkingincline; stride_timing_cv_walking; rom_roll_ratio_stepforward_walk"
FIXED_8 = ("peak_roll_walkingincline; stride_timing_cv_walking; rom_roll_ratio_stepforward_walk; "
           "peak_roll_walking; rom_pitch_walkingincline; stride_timing_cv_walkingincline; "
           "rom_pitch_stepforward; vaf_stepforward")

def freq_string(series_of_lists, n_folds):
    counts = Counter()
    for feats in series_of_lists:
        for f in feats:
            counts[f] += 1
    ordered = sorted(counts.items(), key=lambda x: -x[1])
    return "; ".join(f"{f}: {c}/{n_folds}" for f, c in ordered)

rows = []

# Ridge / Lasso / Linear (nested top-3) 
bc = pd.read_csv(f"{BASE}\\baseline_comparison.csv")
bc_feats = bc["selected_features"].str.split(";")
n29 = len(bc)
freq_bc = freq_string(bc_feats, n29)
final_set_bc = "Varies by fold -- see frequency column (top-3 by |Spearman rho| vs age, re-ranked every fold)"

for model in ["Ridge (nested top-3)", "Lasso (nested top-3)", "Linear (nested top-3)"]:
    rows.append(dict(
        model_name=model,
        feature_selection_method="Per-fold nested",
        n_features_available_as_candidates=33,
        features_selected_final=final_set_bc,
        selection_varies_by_fold=True,
        most_frequently_selected_features_with_counts=freq_bc))

# gplearn (nested, threshold-based, 10-12 features)
d8 = pd.read_csv(f"{BASE}\\loocv_predictions.csv")
d8_feats = d8["selected_features"].str.split(";")
n8 = len(d8)
freq_8 = freq_string(d8_feats, n8)
rows.append(dict(
    model_name="gplearn (nested, threshold-based, 10-12 features)",
    feature_selection_method="Per-fold nested",
    n_features_available_as_candidates=33,
    features_selected_final=f"Varies by fold, {int(d8['n_features_selected'].min())}-{int(d8['n_features_selected'].max())} features -- see frequency column",
    selection_varies_by_fold=True,
    most_frequently_selected_features_with_counts=freq_8,
))

# gplearn (nested, forced top-3)
d12 = pd.read_csv(f"{BASE}\\loocv_gplearn_top3_perfold.csv")
d12_feats = d12[["feature_1", "feature_2", "feature_3"]].values.tolist()
n12 = len(d12)
freq_12 = freq_string(d12_feats, n12)
rows.append(dict(
    model_name="gplearn (nested, forced top-3)",
    feature_selection_method="Per-fold nested",
    n_features_available_as_candidates=33,
    features_selected_final="Varies by fold",
    selection_varies_by_fold=True,
    most_frequently_selected_features_with_counts=freq_12))

# Logistic Regression (classification)
rows.append(dict(
    model_name="Logistic Regression (young/old classification)",
    feature_selection_method="FIXED (not re-derived within this task)",
    n_features_available_as_candidates=33,
    features_selected_final=FIXED_3,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A, fixed every fold (29/29 for all 3)"))

# XGBoost (classification)
import re
xgb_feats = []
with open(f"{BASE}\\run14.log") as f:
    for line in f:
        m = re.search(r"features=\[(.*?)\]", line)
        if m:
            feats = [x.strip().strip("'") for x in m.group(1).split(",")]
            xgb_feats.append(feats)
freq_xgb = freq_string(xgb_feats, len(xgb_feats)) if xgb_feats else "COULD NOT PARSE run14.log"
rows.append(dict(
    model_name="XGBoost (young/old classification)",
    feature_selection_method="Per-fold nested",
    n_features_available_as_candidates=33,
    features_selected_final="Varies by fold",
    selection_varies_by_fold=True,
    most_frequently_selected_features_with_counts=freq_xgb))

# Group interaction model
rows.append(dict(
    model_name="Group interaction model (Model A & B)",
    feature_selection_method="FIXED (not selected at all, pre-specified by hypothesis)",
    n_features_available_as_candidates=33,
    features_selected_final=FIXED_3,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A, fit once on the full (listwise-deleted) cohort, not cross-validated"))

# Leaky / whole-sample models
rows.append(dict(
    model_name="Linear (3-feature, whole-sample, leaky)",
    feature_selection_method="Whole-sample (fixed, leaky)",
    n_features_available_as_candidates=33,
    features_selected_final=FIXED_3,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A, identical every fold by construction (29/29)"))
rows.append(dict(
    model_name="gplearn (8-feature, whole-sample, leaky)",
    feature_selection_method="Whole-sample (fixed, leaky)",
    n_features_available_as_candidates=33,
    features_selected_final=FIXED_8,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A, identical every fold by construction (29/29)"))
rows.append(dict(
    model_name="gplearn (3-feature, whole-sample, leaky)",
    feature_selection_method="Whole-sample (fixed, leaky)",
    n_features_available_as_candidates=33,
    features_selected_final=FIXED_3,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A, identical every fold by construction (29/29)"))

table = pd.DataFrame(rows)
out_path = f"{BASE}\\feature_selection_by_model.csv"
table.to_csv(out_path, index=False)
print(f"Saved -> {out_path}")
print(f"Rows: {len(table)}")
print(table[["model_name", "feature_selection_method", "selection_varies_by_fold"]].to_string(index=False))


