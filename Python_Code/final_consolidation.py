import numpy as np
import pandas as pd

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"

stat_comp = pd.read_csv(f"{BASE}\\statistical_comparison.csv").set_index("comparison")
bc = pd.read_csv(f"{BASE}\\baseline_comparison.csv").set_index("model")
d8 = pd.read_csv(f"{BASE}\\loocv_predictions.csv")
d12 = pd.read_csv(f"{BASE}\\loocv_gplearn_top3_perfold.csv")
loocv_sum03 = pd.read_csv(f"{BASE}\\plots\\loocv_summary.csv").iloc[0]
d09 = pd.read_csv(f"{BASE}\\loocv_results.csv")
clf = pd.read_csv(f"{BASE}\\classification_young_vs_old_metrics.csv").iloc[0]

from scipy import stats as sps
mae8 = d8["abs_error"].mean()
rmse8 = float(np.sqrt(((d8["predicted_age"] - d8["actual_age"]) ** 2).mean()))
r8, p8 = sps.pearsonr(d8["actual_age"], d8["predicted_age"])

mae12 = d12["abs_error"].mean()
rmse12 = float(np.sqrt(((d12["predicted_age"] - d12["actual_age"]) ** 2).mean()))
r12, p12 = sps.pearsonr(d12["actual_age"], d12["predicted_age"])

lin3_mae = d09["linear_abs_error"].mean()
lin3_rmse = float(np.sqrt(((d09["linear_predicted_age"] - d09["actual_age"]) ** 2).mean()))
lin3_r, lin3_p = sps.pearsonr(d09["actual_age"], d09["linear_predicted_age"])

rows = [
    dict(model_name="Ridge (nested top-3)", feature_selection_method="Per-fold nested",
         n_features="3 (varies/fold)", n_participants=29,
         MAE=bc.loc["ridge","MAE"], RMSE=bc.loc["ridge","RMSE"],
         RMSE_MAE_ratio=round(bc.loc["ridge","RMSE"]/bc.loc["ridge","MAE"],4),
         pearson_r=bc.loc["ridge","Pearson_r"], p_value=bc.loc["ridge","p_value"],
         wilcoxon_p_vs_baseline=stat_comp.loc["Ridge vs Baseline","wilcoxon_p"],
         bootstrap_ci_lower=stat_comp.loc["Ridge vs Baseline","ci_lower_95"],
         bootstrap_ci_upper=stat_comp.loc["Ridge vs Baseline","ci_upper_95"]),

    dict(model_name="Lasso (nested top-3)", feature_selection_method="Per-fold nested",
         n_features="3 (varies/fold)", n_participants=29,
         MAE=bc.loc["lasso","MAE"], RMSE=bc.loc["lasso","RMSE"],
         RMSE_MAE_ratio=round(bc.loc["lasso","RMSE"]/bc.loc["lasso","MAE"],4),
         pearson_r=bc.loc["lasso","Pearson_r"], p_value=bc.loc["lasso","p_value"],
         wilcoxon_p_vs_baseline=stat_comp.loc["Lasso vs Baseline","wilcoxon_p"],
         bootstrap_ci_lower=stat_comp.loc["Lasso vs Baseline","ci_lower_95"],
         bootstrap_ci_upper=stat_comp.loc["Lasso vs Baseline","ci_upper_95"]),

    dict(model_name="Linear (nested top-3)", feature_selection_method="Per-fold nested",
         n_features="3 (varies/fold)", n_participants=29,
         MAE=bc.loc["linear","MAE"], RMSE=bc.loc["linear","RMSE"],
         RMSE_MAE_ratio=round(bc.loc["linear","RMSE"]/bc.loc["linear","MAE"],4),
         pearson_r=bc.loc["linear","Pearson_r"], p_value=bc.loc["linear","p_value"],
         wilcoxon_p_vs_baseline=stat_comp.loc["Linear vs Baseline","wilcoxon_p"],
         bootstrap_ci_lower=stat_comp.loc["Linear vs Baseline","ci_lower_95"],
         bootstrap_ci_upper=stat_comp.loc["Linear vs Baseline","ci_upper_95"]),

    dict(model_name="gplearn (nested, 10-12 features)", feature_selection_method="Per-fold nested (threshold |rho|>0.3)",
         n_features="10-12 (varies/fold)", n_participants=len(d8),
         MAE=round(mae8,4), RMSE=round(rmse8,4), RMSE_MAE_ratio=round(rmse8/mae8,4),
         pearson_r=round(r8,4), p_value=round(p8,4),
         wilcoxon_p_vs_baseline=stat_comp.loc["gplearn vs Baseline","wilcoxon_p"],
         bootstrap_ci_lower=stat_comp.loc["gplearn vs Baseline","ci_lower_95"],
         bootstrap_ci_upper=stat_comp.loc["gplearn vs Baseline","ci_upper_95"]),

    dict(model_name="gplearn (nested, forced top-3)", feature_selection_method="Per-fold nested (exact top-3 by rank)",
         n_features=3, n_participants=len(d12),
         MAE=round(mae12,4), RMSE=round(rmse12,4), RMSE_MAE_ratio=round(rmse12/mae12,4),
         pearson_r=round(r12,4), p_value=round(p12,4),
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None),

    dict(model_name="Baseline (predict-mean)", feature_selection_method="N/A",
         n_features=0, n_participants=29,
         MAE=bc.loc["baseline","MAE"], RMSE=bc.loc["baseline","RMSE"],
         RMSE_MAE_ratio=round(bc.loc["baseline","RMSE"]/bc.loc["baseline","MAE"],4),
         pearson_r=None, p_value=None,
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None),

    # whole-sample (leaky)
    dict(model_name="Linear (8-feature, whole-sample, leaky)", feature_selection_method="Whole-sample (fixed, leaky)",
         n_features=8, n_participants=None,
         MAE=None, RMSE=None, RMSE_MAE_ratio=None, pearson_r=None, p_value=None,
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None),

    dict(model_name="Linear (3-feature, whole-sample, leaky)", feature_selection_method="Whole-sample (fixed, leaky)",
         n_features=3, n_participants=len(d09),
         MAE=round(lin3_mae,4), RMSE=round(lin3_rmse,4), RMSE_MAE_ratio=round(lin3_rmse/lin3_mae,4),
         pearson_r=round(lin3_r,4), p_value=round(lin3_p,4),
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None),

    dict(model_name="gplearn (8-feature, whole-sample, leaky)", feature_selection_method="Whole-sample (fixed, leaky)",
         n_features=8, n_participants=int(loocv_sum03["n_participants"]),
         MAE=loocv_sum03["mae_years"], RMSE=loocv_sum03["rmse_years"],
         RMSE_MAE_ratio=round(loocv_sum03["rmse_years"]/loocv_sum03["mae_years"],4),
         pearson_r=loocv_sum03["pearson_r"], p_value=loocv_sum03["pearson_p"],
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None),

    dict(model_name="Logistic Regression (binary young/old classification)",
         feature_selection_method="Fixed (same 3 features, nested LOOCV model fit)",
         n_features=3, n_participants=int(clf["N"]),
         MAE=clf["accuracy"], RMSE=None, RMSE_MAE_ratio=None,
         pearson_r=clf["AUC_ROC"], p_value=0.032,
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None)]

table = pd.DataFrame(rows)
BASELINE_MAE = bc.loc["baseline","MAE"]
table["significant_at_0.05"] = table["p_value"].apply(lambda p: bool(p < 0.05) if pd.notna(p) else False)

table["_sort_mae"] = pd.to_numeric(table["MAE"], errors="coerce")
table = table.sort_values(["significant_at_0.05", "_sort_mae"], ascending=[False, True], na_position="last")
table = table.drop(columns="_sort_mae")

col_order = ["model_name", "feature_selection_method", "n_features", "n_participants",
             "MAE", "RMSE", "RMSE_MAE_ratio", "pearson_r", "p_value",
             "wilcoxon_p_vs_baseline", "bootstrap_ci_lower", "bootstrap_ci_upper",
             "significant_at_0.05", "notes"]

bcr = pd.read_csv(f"{BASE}\\bias_correction_results.csv")
bcr = bcr[bcr["model"] != "Baseline"].copy() 

bias_table = pd.DataFrame({
    "model_name": bcr["model"],
    "MAE_before": bcr["mae_before"],
    "r_before": bcr["r_before"],
    "bias_correlation": bcr["bias_corr_before"],
    "MAE_after_correction": bcr["mae_after"],
    "r_after_correction": bcr["r_after"]})
bias_table.to_csv(f"{BASE}\\bias_correction_summary.csv", index=False)

pp = pd.read_csv(f"{BASE}\\per_participant_all_runs.csv")
model_mae = {
    "ridge": bc.loc["ridge","MAE"], "lasso": bc.loc["lasso","MAE"],
    "linear": bc.loc["linear","MAE"], "gplearn": mae8,
    "baseline": bc.loc["baseline","MAE"],
}
for m, mae_val in model_mae.items():
    pp[f"{m}_outlier"] = pp[f"{m}_error"] > (2 * mae_val)

pp.to_csv(f"{BASE}\\per_participant_all_runs.csv", index=False)

print(f"{BASE}\\final_results_summary.csv")
print(f"{BASE}\\bias_correction_summary.csv")
print(f"{BASE}\\per_participant_all_runs.csv")
print(f"BASELINE_MAE used for outlier/beats-baseline reference: {BASELINE_MAE}")

import pandas as pd
t = pd.read_csv('final_results_summary.csv')
print(t.to_string(index=False))


import numpy as np
import pandas as pd
from scipy import stats

OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\audit_summary.csv"


def p_from_r(r, n):
    if abs(r) >= 1.0 or n <= 2:
        return 0.0
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))


rows = []

# symbolic_regression.py 
s = pd.read_csv(r"plots\loocv_summary.csv").iloc[0]
rows.append({
    "script_name": "03_symbolic_regression.py",
    "rigor_tier": "partial_leaky",
    "feature_selection_method": "whole_sample_once",
    "n_features": 8, "N": int(s["n_participants"]),
    "MAE": s["mae_years"], "pearson_r": s["pearson_r"], "p_value": s["pearson_p"]})

# final_model.py 
eq4 = pd.read_csv(r"plots\final_model_equation.csv").iloc[0]
rows.append({
    "script_name": "04_final_model.py",
    "rigor_tier": "not_validated",
    "feature_selection_method": "whole_sample_no_holdout",
    "n_features": 8, "N": 29,
    "MAE": eq4["train_mae"], "pearson_r": eq4["train_r"], "p_value": None})

# baseline_comparison.py 
b = pd.read_csv(r"baseline_comparison_v2_summary.csv")
per_fold_05 = pd.read_csv(r"baseline_comparison_v2.csv")
n05 = len(per_fold_05)
for _, r in b.iterrows():
    rows.append({
        "script_name": f"05_baseline_comparison.py ({r['model']})",
        "rigor_tier": "fully_rigorous",
        "feature_selection_method": "n/a" if r["model"] == "baseline" else "per_fold",
        "n_features": 0 if r["model"] == "baseline" else "3 (varies/fold)",
        "N": n05, "MAE": r["MAE"], "pearson_r": r["Pearson_r"], "p_value": r["p_value"]})

# nested_loocv.py 
d8 = pd.read_csv(r"loocv_predictions.csv")
mae8 = d8["abs_error"].mean()
rmse8 = np.sqrt(((d8["predicted_age"] - d8["actual_age"]) ** 2).mean())
r8, p8 = stats.pearsonr(d8["actual_age"], d8["predicted_age"])
rows.append({
    "script_name": "08_nested_loocv.py",
    "rigor_tier": "fully_rigorous",
    "feature_selection_method": "per_fold",
    "n_features": f"{int(d8['n_features_selected'].min())}-{int(d8['n_features_selected'].max())} (varies/fold)",
    "N": len(d8), "MAE": round(mae8, 2), "pearson_r": round(r8, 3), "p_value": round(p8, 4)})

# improved_model.py 
d9 = pd.read_csv(r"loocv_results_v2.csv")
gp_mae9 = d9["gp_abs_error"].mean()
gp_r9, gp_p9 = stats.pearsonr(d9["actual_age"], d9["gp_predicted_age"])
lin_mae9 = d9["linear_abs_error"].mean()
lin_r9, lin_p9 = stats.pearsonr(d9["actual_age"], d9["linear_predicted_age"])
rows.append({
    "script_name": "09_improved_model.py (gplearn)",
    "rigor_tier": "partial_leaky",
    "feature_selection_method": "whole_sample_once",
    "n_features": 3, "N": len(d9),
    "MAE": round(gp_mae9, 2), "pearson_r": round(gp_r9, 3), "p_value": round(gp_p9, 4)})
rows.append({
    "script_name": "09_improved_model.py (linear)",
    "rigor_tier": "partial_leaky",
    "feature_selection_method": "whole_sample_once",
    "n_features": 3, "N": len(d9),
    "MAE": round(lin_mae9, 2), "pearson_r": round(lin_r9, 3), "p_value": round(lin_p9, 4)})

# nested_linear_regression
d10 = pd.read_csv(r"loocv_reselected_features.csv")
mae10 = d10["abs_error"].mean()
r10, p10 = stats.pearsonr(d10["actual_age"], d10["predicted_age"])
rows.append({
    "script_name": "10_nested_linear_regression.py",
    "rigor_tier": "fully_rigorous",
    "feature_selection_method": "per_fold",
    "n_features": f"{int(d10['n_features_selected'].min())}-{int(d10['n_features_selected'].max())} (varies/fold)",
    "N": len(d10), "MAE": round(mae10, 2), "pearson_r": round(r10, 3), "p_value": round(p10, 4)})

# ─nested_gplearn_top3
d12 = pd.read_csv(r"loocv_gplearn_top3_perfold_with_readable_eq.csv")
mae12 = d12["abs_error"].mean()
r12, p12 = stats.pearsonr(d12["actual_age"], d12["predicted_age"])
rep = d12.loc[d12["abs_error"].idxmin(), "equation_readable"]
rows.append({
    "script_name": "12_nested_gplearn_top3.py",
    "rigor_tier": "fully_rigorous",
    "feature_selection_method": "per_fold",
    "n_features": 3, "N": len(d12), "MAE": round(mae12, 2), "pearson_r": round(r12, 3), "p_value": round(p12, 4)})

# final_model_constrained
d13 = pd.read_csv(r"final_model_constrained_seeds.csv")
valid13 = d13[d13["passes_ceiling"]]
best13 = valid13.loc[valid13["mae"].idxmin()]
p13 = p_from_r(best13["r"], 29)
rows.append({
    "script_name": "13_final_model_constrained.py",
    "rigor_tier": "not_validated",
    "feature_selection_method": "whole_sample_no_holdout",
    "n_features": 8, "N": 29,
    "MAE": round(best13["mae"], 2), "pearson_r": round(best13["r"], 3), "p_value": round(p13, 4)})

table = pd.DataFrame(rows)
tier_order = {"fully_rigorous": 0, "partial_leaky": 1, "not_validated": 2, "not_yet_run": 3}
table["_sort"] = table["rigor_tier"].map(tier_order)
table = table.sort_values(["_sort", "MAE"]).drop(columns="_sort").reset_index(drop=True)

table.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}\n")

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)
print(table.to_string(index=False))


import numpy as np
import pandas as pd
from scipy import stats

OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\session_summary_all_runs.csv"

BASELINE_MAE = 14.47


def p_from_r(r, n):
    if r is None or abs(r) >= 1.0 or n <= 2:
        return None
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))


FULLY_RIGOROUS = "Per-fold nested (leakage-free)"
WHOLE_SAMPLE = "Whole-sample (fixed, leaky)"
NO_HOLDOUT = "No holdout (not cross-validated)"

rows = []
# baseline_comparison
b = pd.read_csv("baseline_comparison_summary.csv").set_index("model")
n05 = len(pd.read_csv("baseline_comparison.csv"))

r = b.loc["lasso"]
rows.append(dict(Run="Run 1: Lasso regression, nested top-3 features (nested)",
                  N_features="3 (varies/fold)", Feature_selection=FULLY_RIGOROUS,
                  Model="Lasso regression", N=n05, MAE=r["MAE"], RMSE=r["RMSE"],
                  Pearson_r=r["Pearson_r"], p_value=r["p_value"]))

r = b.loc["ridge"]
rows.append(dict(Run="Run 2: Ridge regression, nested top-3 features (nested)",
                  N_features="3 (varies/fold)", Feature_selection=FULLY_RIGOROUS,
                  Model="Ridge regression", N=n05, MAE=r["MAE"], RMSE=r["RMSE"],
                  Pearson_r=r["Pearson_r"], p_value=r["p_value"]))

d8 = pd.read_csv("loocv_predictions.csv")
mae8 = d8["abs_error"].mean()
rmse8 = float(np.sqrt(((d8["predicted_age"] - d8["actual_age"]) ** 2).mean()))
r8, p8 = stats.pearsonr(d8["actual_age"], d8["predicted_age"])
rows.append(dict(Run="Run 3: Nested gplearn, 10-12 features per fold (nested)",
                  N_features=f"{int(d8['n_features_selected'].min())}-{int(d8['n_features_selected'].max())} (varies/fold)",
                  Feature_selection=FULLY_RIGOROUS, Model="gplearn", N=len(d8),
                  MAE=round(mae8, 4), RMSE=round(rmse8, 4), Pearson_r=round(r8, 4), p_value=round(p8, 4)))

r = b.loc["linear"]
rows.append(dict(Run="Run 4: Linear regression, nested top-3 features (nested)",
                  N_features="3 (varies/fold)", Feature_selection=FULLY_RIGOROUS,
                  Model="Linear regression", N=n05, MAE=r["MAE"], RMSE=r["RMSE"],
                  Pearson_r=r["Pearson_r"], p_value=r["p_value"]))

r = b.loc["baseline"]
rows.append(dict(Run="Run 5: Predict-mean baseline, properly nested (nested)",
                  N_features=0, Feature_selection=FULLY_RIGOROUS,
                  Model="Predict-mean baseline", N=n05, MAE=r["MAE"], RMSE=r["RMSE"],
                  Pearson_r=None, p_value=None))

d10 = pd.read_csv("loocv_reselected_features.csv")
mae10 = d10["abs_error"].mean()
rmse10 = float(np.sqrt(((d10["predicted_age"] - d10["actual_age"]) ** 2).mean()))
r10, p10 = stats.pearsonr(d10["actual_age"], d10["predicted_age"])
rows.append(dict(Run="Run 6: Nested linear regression, 10-12 features per fold (nested)",
                  N_features=f"{int(d10['n_features_selected'].min())}-{int(d10['n_features_selected'].max())} (varies/fold)",
                  Feature_selection=FULLY_RIGOROUS, Model="Linear regression", N=len(d10),
                  MAE=round(mae10, 4), RMSE=round(rmse10, 4), Pearson_r=round(r10, 4), p_value=round(p10, 4)))

d12 = pd.read_csv("loocv_gplearn_top3_perfold.csv")
mae12 = d12["abs_error"].mean()
rmse12 = float(np.sqrt(((d12["predicted_age"] - d12["actual_age"]) ** 2).mean()))
r12, p12 = stats.pearsonr(d12["actual_age"], d12["predicted_age"])
rows.append(dict(Run="Run 7: Nested gplearn, exactly top-3 per fold (nested)",
                  N_features=3, Feature_selection=FULLY_RIGOROUS, Model="gplearn", N=len(d12),
                  MAE=round(mae12, 4), RMSE=round(rmse12, 4), Pearson_r=round(r12, 4), p_value=round(p12, 4)))

# Partial leaky

d9 = pd.read_csv("loocv_results.csv")
lin_mae9 = d9["linear_abs_error"].mean()
lin_rmse9 = float(np.sqrt(((d9["linear_predicted_age"] - d9["actual_age"]) ** 2).mean()))
lin_r9, lin_p9 = stats.pearsonr(d9["actual_age"], d9["linear_predicted_age"])
rows.append(dict(Run="Run 8: Linear regression, fixed 3 features (leaky)",
                  N_features=3, Feature_selection=WHOLE_SAMPLE, Model="Linear regression", N=len(d9),
                  MAE=round(lin_mae9, 4), RMSE=round(lin_rmse9, 4), Pearson_r=round(lin_r9, 4), p_value=round(lin_p9, 4)))

s3 = pd.read_csv(r"plots\loocv_summary.csv").iloc[0]
rows.append(dict(Run="Run 9: gplearn, 8 fixed features (leaky)",
                  N_features=8, Feature_selection=WHOLE_SAMPLE, Model="gplearn", N=int(s3["n_participants"]),
                  MAE=s3["mae_years"], RMSE=s3["rmse_years"], Pearson_r=s3["pearson_r"], p_value=s3["pearson_p"]))
gp_mae9 = d9["gp_abs_error"].mean()
gp_rmse9 = float(np.sqrt(((d9["gp_predicted_age"] - d9["actual_age"]) ** 2).mean()))
gp_r9, gp_p9 = stats.pearsonr(d9["actual_age"], d9["gp_predicted_age"])
rows.append(dict(Run="Run 10: gplearn, fixed 3 features (leaky)",
                  N_features=3, Feature_selection=WHOLE_SAMPLE, Model="gplearn", N=len(d9),
                  MAE=round(gp_mae9, 4), RMSE=round(gp_rmse9, 4), Pearson_r=round(gp_r9, 4), p_value=round(gp_p9, 4)))

# Not validated (no holdout)

eq4 = pd.read_csv(r"plots\final_model_equation.csv").iloc[0]
rows.append(dict(Run="Run 11: gplearn, 452-node equation, full-dataset fit (not cross-validated)",
                  N_features=8, Feature_selection=NO_HOLDOUT, Model="gplearn", N=29,
                  MAE=round(eq4["train_mae"], 4), RMSE=None,
                  Pearson_r=round(eq4["train_r"], 4), p_value=round(eq4["train_p"], 4)))

d13 = pd.read_csv("final_model_constrained_seeds.csv")
valid13 = d13[d13["passes_ceiling"]]
best13 = valid13.loc[valid13["mae"].idxmin()]
p13 = p_from_r(best13["r"], 29)
rows.append(dict(Run="Run 12: gplearn, 3-node equation (heavy parsimony), full-dataset fit (not cross-validated)",
                  N_features=8, Feature_selection=NO_HOLDOUT, Model="gplearn", N=29,
                  MAE=round(best13["mae"], 4), RMSE=None,
                  Pearson_r=round(best13["r"], 4), p_value=round(p13, 4)))

# Finalize
table = pd.DataFrame(rows)

def sigfig4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(f"{x:.4g}")

for col in ["MAE", "RMSE", "Pearson_r", "p_value"]:
    table[col] = table[col].apply(lambda v: sigfig4(v) if pd.notna(v) else None)

table["Beats_baseline"] = table["MAE"].apply(lambda m: bool(m < BASELINE_MAE) if pd.notna(m) else None)
table["Significant"] = table["p_value"].apply(lambda p: bool(p < 0.05) if pd.notna(p) else False)

table = table[["Run", "N_features", "Feature_selection", "Model", "N",
               "MAE", "RMSE", "Pearson_r", "p_value", "Beats_baseline", "Significant", "Notes"]]

table.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}\n")

print(table.to_string(index=False))


import pandas as pd

OUT_DIR = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"

b = pd.read_csv(f"{OUT_DIR}\\baseline_comparison.csv")
g = pd.read_csv(f"{OUT_DIR}\\loocv_predictions.csv")
groups = pd.read_csv(f"{OUT_DIR}\\feature_table_full.csv")[["participant", "group"]]
groups["group"] = groups["group"].astype("Int64")


def build_csv(df, pred_col, features_col, filename, features_present=True):
    out = pd.DataFrame({
        "participant_id": df["participant"],
        "actual_age": df["actual_age"],
        "predicted_age": df[pred_col]})
    out["absolute_error"] = (out["predicted_age"] - out["actual_age"]).abs()
    out = out.merge(groups, left_on="participant_id", right_on="participant", how="left").drop(columns="participant")
    if features_present:
        out["features_used_this_fold"] = df[features_col].str.replace(";", ", ", regex=False)
    else:
        out["features_used_this_fold"] = "(none -- predict-mean baseline uses no features)"

    out = out[["participant_id", "actual_age", "predicted_age", "absolute_error", "group", "features_used_this_fold"]]
    out = out.sort_values("absolute_error", ascending=False).reset_index(drop=True)

    mae = out["absolute_error"].mean()
    mean_row = pd.DataFrame([{
        "participant_id": "MEAN", "actual_age": None, "predicted_age": None,
        "absolute_error": mae, "group": None, "features_used_this_fold": None}])
    out = pd.concat([out, mean_row], ignore_index=True)
    out.to_csv(f"{OUT_DIR}\\{filename}", index=False)
    print(f"Saved -> {filename}  (MAE={mae:.4f})")
    return mae


build_csv(b, "ridge_pred", "selected_features", "per_participant_ridge.csv")
build_csv(b, "lasso_pred", "selected_features", "per_participant_lasso.csv")
build_csv(b, "linear_pred", "selected_features", "per_participant_linear.csv")
build_csv(g, "predicted_age", "selected_features", "per_participant_gplearn.csv")
build_csv(b, "baseline_pred", None, "per_participant_baseline.csv", features_present=False)

combined = b[["participant", "actual_age"]].copy()
combined = combined.merge(groups, on="participant", how="left")
combined["ridge_pred"] = b["ridge_pred"]
combined["ridge_error"] = (b["ridge_pred"] - b["actual_age"]).abs()
combined["lasso_pred"] = b["lasso_pred"]
combined["lasso_error"] = (b["lasso_pred"] - b["actual_age"]).abs()
combined["linear_pred"] = b["linear_pred"]
combined["linear_error"] = (b["linear_pred"] - b["actual_age"]).abs()
combined["baseline_pred"] = b["baseline_pred"]
combined["baseline_error"] = (b["baseline_pred"] - b["actual_age"]).abs()

g_indexed = g.set_index("participant")
combined["gplearn_pred"] = combined["participant"].map(g_indexed["predicted_age"])
combined["gplearn_error"] = (combined["gplearn_pred"] - combined["actual_age"]).abs()

combined = combined.rename(columns={"participant": "participant_id"})
combined = combined[["participant_id", "actual_age", "group",
                      "ridge_pred", "ridge_error",
                      "lasso_pred", "lasso_error",
                      "linear_pred", "linear_error",
                      "gplearn_pred", "gplearn_error",
                      "baseline_pred", "baseline_error"]]
combined = combined.sort_values("participant_id", ascending=True).reset_index(drop=True)
combined.to_csv(f"{OUT_DIR}\\per_participant_all_runs.csv", index=False)
print(f"\nSaved -> per_participant_all_runs.csv  ({len(combined)} rows)\n")
print(combined.round(2).to_string(index=False))


import pandas as pd

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
rows = []


def add(category, model_name, metric_1, metric_1_value, metric_2, metric_2_value,
        comparison_test, comparison_p_value, significant, notes):
    rows.append(dict(category=category, model_name=model_name,
                      metric_1=metric_1, metric_1_value=metric_1_value,
                      metric_2=metric_2, metric_2_value=metric_2_value,
                      comparison_test=comparison_test, comparison_p_value=comparison_p_value,
                      significant=significant, notes=notes))

m = pd.read_csv(f"{BASE}\\final_results_summary.csv")
for _, r in m.iterrows():
    is_classification = "Logistic Regression" in r["model_name"] and "classification" in r["model_name"]
    add("Regression model comparison" if not is_classification else "Classification (primary)",
        r["model_name"], "MAE" if not is_classification else "accuracy", r["MAE"],
        "pearson_r" if not is_classification else "AUC", r["pearson_r"],
        "permutation_p" if is_classification else "wilcoxon_p_vs_baseline",
        r["p_value"] if is_classification else r["wilcoxon_p_vs_baseline"],
        r["significant_at_0.05"], r["notes"])

# Classification comparison (classifier vs XGBoost) 
c = pd.read_csv(f"{BASE}\\classification_model_comparison.csv")
for _, r in c.iterrows():
    if r["classifier"] == "Logistic Regression":
        continue  # already added above with full notes
    add("Classification (primary)", r["classifier"], "accuracy", r["accuracy"], "AUC", r["AUC"],
        "permutation_p", r["permutation_p"], r["permutation_p"] < 0.05,
        f"sensitivity={r['sensitivity']}, specificity={r['specificity']}, n_features={r['n_features']}")

# Original data augmentation: Ridge/Lasso/LR (augmentation_comparison.csv) 
a1 = pd.read_csv(f"{BASE}\\augmentation_comparison.csv")
for _, r in a1.iterrows():
    if r["method"] == "none":
        continue  # baselines already captured above
    is_lr = r["model"] == "Logistic Regression"
    add("Data augmentation (Ridge/Lasso/LR)", f"{r['model']} + {r['method']}",
        "accuracy" if is_lr else "MAE", r["accuracy_or_MAE"],
        "AUC" if is_lr else "pearson_r", r["AUC_or_r"],
        "wilcoxon_p_vs_no_augmentation", r["wilcoxon_p_vs_no_augmentation"],
        r["wilcoxon_p_vs_no_augmentation"] < 0.05 if pd.notna(r["wilcoxon_p_vs_no_augmentation"]) else False,
        r["notes"])

# GPR data augmentation 
a2 = pd.read_csv(f"{BASE}\\gpr_augmentation_comparison.csv")
for _, r in a2.iterrows():
    add("Data augmentation (GPR)", f"GPR ({r['kernel']}) + {r['method']}",
        "MAE", r["mae"], "pearson_r", r["pearson_r"],
        "wilcoxon_p_vs_baseline", r["wilcoxon_p_vs_baseline"], r["confirmed_improvement"],
        f"baseline_mae={r['baseline_mae']}; bonferroni_alpha={r['bonferroni_alpha']}; "
        f"bonferroni_significant={r['bonferroni_significant']}")

# gplearn data augmentation 
gplearn_noise = dict(mae=14.1460, r=0.2813, wp=0.1689, ci="[-4.9061, -0.0192]")
gplearn_smote = dict(mae=14.5646, r=0.2654, wp=0.5647, ci="[-5.3869, 1.6970]")

try:
    gmm_df = pd.read_csv(f"{BASE}\\gplearn_augmentation_smote_gmm_comparison.csv")
    gmm_row = gmm_df[gmm_df["method"] == "gmm"].iloc[0]
    add("Data augmentation (gplearn)", "gplearn (nested top-3) + gmm", "MAE", gmm_row["mae"],
        "pearson_r", gmm_row["pearson_r"], "wilcoxon_p_vs_baseline", gmm_row["wilcoxon_p_vs_baseline"],
        gmm_row["confirmed_improvement"],
        f"baseline_mae=16.4464; bootstrap 95% CI [{gmm_row['bootstrap_ci_lower']}, {gmm_row['bootstrap_ci_upper']}]")
except FileNotFoundError:
    add("Data augmentation (gplearn)", "gplearn (nested top-3) + gmm", "MAE", "PENDING",
        "pearson_r", "PENDING", "wilcoxon_p_vs_baseline", "PENDING", "PENDING",
        "Re-run this script once gplearn_augmentation_smote_gmm_comparison.csv exists to fill in this row.")

# Logistic regression improvement attempts
lr = pd.read_csv(f"{BASE}\\logistic_regression_improvement_attempts.csv")
for _, r in lr.iterrows():
    add("LR improvement attempts", r["attempt"], "accuracy", r["accuracy"], "AUC", r["AUC"],
        "permutation_p" if pd.notna(r["permutation_p"]) else "wilcoxon_p_vs_row2b",
        r["permutation_p"] if pd.notna(r["permutation_p"]) else r["wilcoxon_p_vs_row2b"],
        False, r["verdict"])

# Group interaction model 
add("Group interaction model", "Likelihood-ratio test (complete-case, N=18)",
    "LR_statistic", 8.5898, "df", 3, "chi2_p_value", 0.0353, True,
    "Joint test significant, but no individual interaction term significant independently.")
add("Group interaction model", "Marginal term: group:peak_roll_walkingincline (complete-case, N=18)",
    "coefficient", 0.543, "95%_CI", "[-0.064, 1.150]", "p_value", 0.074, False, "Closest individual term to significance.")
add("Group interaction model", "Marginal term: group:peak_roll_walkingincline (imputed, N=29, 20 MICE)",
    "coefficient", 0.316, "95%_CI", "[-0.215, 0.847]", "p_value", 0.243, False,
    "Imputation weakens rather than strengthens the term -- not a missingness artefact.")

out = pd.DataFrame(rows)
out.to_csv(f"{BASE}\\final_all_results.csv", index=False)
print(f"Saved -> {BASE}\\final_all_results.csv")
print(f"Rows: {len(out)}")
print(out["category"].value_counts().to_string())


