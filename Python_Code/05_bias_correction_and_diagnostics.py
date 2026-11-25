import numpy as np
import pandas as pd
from scipy import stats

IN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\per_participant_all_runs.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\bias_correction_results.csv"

df = pd.read_csv(IN_CSV)
actual = df["actual_age"].values.astype(float)
n = len(df)

models = {
    "Ridge": df["ridge_pred"].values.astype(float),
    "Lasso": df["lasso_pred"].values.astype(float),
    "Linear": df["linear_pred"].values.astype(float),
    "gplearn": df["gplearn_pred"].values.astype(float),
    "Baseline": df["baseline_pred"].values.astype(float)}

rows = []
for name, pred in models.items():
    signed_error = pred - actual
    mae = float(np.mean(np.abs(signed_error)))
    rmse = float(np.sqrt(np.mean(signed_error ** 2)))
    r_before, p_before = stats.pearsonr(actual, pred)
    bias_r_before, bias_p_before = stats.pearsonr(actual, signed_error)

    row = {
        "model": name,
        "mae_before": round(mae, 4), "rmse_before": round(rmse, 4),
        "r_before": round(r_before, 4),
        "bias_corr_before": round(bias_r_before, 4), "bias_corr_p_before": round(bias_p_before, 4)}

    if name != "Baseline":
        corrected = np.empty(n)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            slope, intercept, _, _, _ = stats.linregress(actual[others], pred[others])
            corrected[i] = pred[i] - (slope * actual[i] + intercept - actual[i])

        signed_error_after = corrected - actual
        mae_after = float(np.mean(np.abs(signed_error_after)))
        rmse_after = float(np.sqrt(np.mean(signed_error_after ** 2)))
        r_after, _ = stats.pearsonr(actual, corrected)
        bias_r_after, bias_p_after = stats.pearsonr(actual, signed_error_after)

        row.update({
            "mae_after": round(mae_after, 4), "rmse_after": round(rmse_after, 4),
            "r_after": round(r_after, 4),
            "bias_corr_after": round(bias_r_after, 4), "bias_corr_p_after": round(bias_p_after, 4)})
    else:
        row.update({
            "mae_after": None, "rmse_after": None, "r_after": None,
            "bias_corr_after": None, "bias_corr_p_after": None})
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}\n")
pd.set_option("display.width", 200)
print(table.to_string(index=False))

import pandas as pd
import numpy as np

IN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\per_participant_all_runs.csv"
GPLEARN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_predictions.csv"
OUT_RATIO = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\rmse_mae_ratio.csv"
OUT_OUTLIERS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\gplearn_outliers.csv"

df = pd.read_csv(IN_CSV)
actual = df["actual_age"].values.astype(float)

models = {
    "Ridge": df["ridge_pred"].values.astype(float),
    "Lasso": df["lasso_pred"].values.astype(float),
    "Linear": df["linear_pred"].values.astype(float),
    "gplearn": df["gplearn_pred"].values.astype(float),
    "Baseline": df["baseline_pred"].values.astype(float)}

rows = []
for name, pred in models.items():
    err = np.abs(pred - actual)
    mae = float(err.mean())
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    rows.append({"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                 "RMSE_over_MAE": round(rmse / mae, 4)})

ratio_table = pd.DataFrame(rows).sort_values("RMSE_over_MAE", ascending=False)
ratio_table.to_csv(OUT_RATIO, index=False)
print(f"Saved -> {OUT_RATIO}\n")
print(ratio_table.to_string(index=False))

# gplearn outliers
gplearn_mae = np.abs(models["gplearn"] - actual).mean()
threshold = 2 * gplearn_mae

g_detail = pd.read_csv(GPLEARN_CSV)  # has selected_features per fold
merged = df[["participant_id", "actual_age", "gplearn_pred", "gplearn_error", "group"]].merge(
    g_detail[["participant", "selected_features"]],
    left_on="participant_id", right_on="participant", how="left").drop(columns="participant")

outliers = merged[merged["gplearn_error"] > threshold].sort_values("gplearn_error", ascending=False)
outliers.to_csv(OUT_OUTLIERS, index=False)

print(f"\ngplearn MAE = {gplearn_mae:.4f}, outlier threshold (2x MAE) = {threshold:.4f}")
print(f"Saved -> {OUT_OUTLIERS}  ({len(outliers)} outlier participants)\n")
print(outliers.to_string(index=False))


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\per_participant_all_runs.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"

df = pd.read_csv(IN_CSV)
actual = df["actual_age"].values.astype(float)
group = df["group"].values

model_cols = {"ridge": "Ridge", "lasso": "Lasso", "linear": "Linear", "gplearn": "gplearn"}

for col, label in model_cols.items():
    pred = df[f"{col}_pred"].values.astype(float)
    mean_vals = (actual + pred) / 2
    diff_vals = pred - actual
    mean_diff = diff_vals.mean()
    sd_diff = diff_vals.std(ddof=1)
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    young_mask = group == 1
    old_mask = group == 2
    ax.scatter(mean_vals[young_mask], diff_vals[young_mask], color="#1f77b4", label="Young (group=1)", s=55, alpha=0.8)
    ax.scatter(mean_vals[old_mask], diff_vals[old_mask], color="#d62728", label="Older (group=2)", s=55, alpha=0.8)

    ax.axhline(mean_diff, color="black", linestyle="-", linewidth=1.2, label=f"Mean diff = {mean_diff:.2f}")
    ax.axhline(upper, color="gray", linestyle="--", linewidth=1, label=f"+1.96 SD = {upper:.2f}")
    ax.axhline(lower, color="gray", linestyle="--", linewidth=1, label=f"-1.96 SD = {lower:.2f}")

    ax.set_xlabel("Mean of actual and predicted age (years)")
    ax.set_ylabel("Predicted - Actual age (years)")
    ax.set_title(f"Bland-Altman: {label} (nested LOOCV)")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_path = f"{PLOTS}\\bland_altman_{col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}  (mean_diff={mean_diff:.2f}, SD={sd_diff:.2f}, limits=[{lower:.2f}, {upper:.2f}])")


