from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.power import FTestAnovaPower


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "final_run_may15_2026" / "tables"

ALPHA = 0.05
TARGET_POWER = 0.80
CONFIDENCE_LEVEL = 0.95
N_BOOT = 3000
SEED = 42


def _paired_auc_delta_ci(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delta = float(roc_auc_score(y_true, p_b) - roc_auc_score(y_true, p_a))

    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        da = roc_auc_score(yb, p_a[idx])
        db = roc_auc_score(yb, p_b[idx])
        draws.append(float(db - da))

    boot = np.asarray(draws, dtype=float)
    if boot.size == 0:
        return {
            "delta_auc": delta,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "boot_mean": np.nan,
            "boot_sd": np.nan,
            "n_boot_used": 0,
        }

    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))
    return {
        "delta_auc": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "boot_mean": float(np.mean(boot)),
        "boot_sd": float(np.std(boot, ddof=1)),
        "n_boot_used": int(boot.size),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    pred54 = pd.read_csv(TABLES / "test_predictions_threshold_54.csv")
    pred56 = pd.read_csv(TABLES / "test_predictions_threshold_56.csv")
    del54 = pd.read_csv(TABLES / "rq3_delong_modelA_vs_modelB_threshold_54.csv").iloc[0]
    del56 = pd.read_csv(TABLES / "rq3_delong_modelA_vs_modelB_threshold_56.csv").iloc[0]
    anova = pd.read_csv(TABLES / "rq4_anova_final.csv").iloc[0]
    chi = pd.read_csv(TABLES / "rq4_cluster_high54_chi_square.csv").iloc[0]
    clustered = pd.read_csv(TABLES / "rq4_clustered_dataset_final.csv")

    y54 = pred54["y_true"].to_numpy(dtype=int)
    y56 = pred56["y_true"].to_numpy(dtype=int)
    p54_a = pred54["prob_ModelA_Logistic"].to_numpy(dtype=float)
    p54_b = pred54["prob_ModelB_Logistic"].to_numpy(dtype=float)
    p56_a = pred56["prob_ModelA_Logistic"].to_numpy(dtype=float)
    p56_b = pred56["prob_ModelB_Logistic"].to_numpy(dtype=float)

    ci54 = _paired_auc_delta_ci(y54, p54_a, p54_b, N_BOOT, SEED)
    ci56 = _paired_auc_delta_ci(y56, p56_a, p56_b, N_BOOT, SEED + 1)

    # Effect sizes for RQ4
    eta_sq = float(anova["eta_squared"])
    cohen_f = float(np.sqrt(eta_sq / (1.0 - eta_sq)))

    # Cramer's V for cluster x high54 association
    ct = pd.crosstab(clustered["behavior_cluster_final"], clustered["FWB_high_54"])
    n_obs = int(ct.to_numpy().sum())
    r, c = ct.shape
    k = min(r - 1, c - 1)
    cramers_v = float(np.sqrt(float(chi["chi2"]) / (n_obs * k))) if k > 0 else np.nan

    # Sample-size/power calculations
    z_alpha = stats.norm.ppf(1 - ALPHA / 2)
    z_beta = stats.norm.ppf(TARGET_POWER)
    z54 = float(del54["z_stat"])
    z56 = float(del56["z_stat"])
    n_rq3_req_54 = float(len(pred54) * ((z_alpha + z_beta) / z54) ** 2)
    n_rq3_req_56 = float(len(pred56) * ((z_alpha + z_beta) / z56) ** 2)

    # EPV-based logistic adequacy at threshold 54 (Peduzzi rule-of-thumb)
    n_features_model_b = 61
    required_events = 10 * n_features_model_b
    prevalence_54 = float(clustered["FWB_high_54"].mean())
    min_class_rate_54 = min(prevalence_54, 1.0 - prevalence_54)
    n_epv_required = float(required_events / min_class_rate_54)

    # ANOVA required sample size with observed effect size and power target
    anova_power = FTestAnovaPower()
    n_rq4_required = float(
        anova_power.solve_power(effect_size=cohen_f, k_groups=3, alpha=ALPHA, power=TARGET_POWER)
    )

    infer_rows = [
        {
            "analysis_block": "RQ3",
            "metric": "AUC delta (ModelB_Logistic - ModelA_Logistic) threshold 54",
            "estimate": ci54["delta_auc"],
            "ci_low": ci54["ci_low"],
            "ci_high": ci54["ci_high"],
            "method": "Paired bootstrap (percentile CI)",
            "n_boot": ci54["n_boot_used"],
            "alpha": ALPHA,
        },
        {
            "analysis_block": "RQ3",
            "metric": "AUC delta (ModelB_Logistic - ModelA_Logistic) threshold 56",
            "estimate": ci56["delta_auc"],
            "ci_low": ci56["ci_low"],
            "ci_high": ci56["ci_high"],
            "method": "Paired bootstrap (percentile CI)",
            "n_boot": ci56["n_boot_used"],
            "alpha": ALPHA,
        },
        {
            "analysis_block": "RQ4",
            "metric": "ANOVA eta-squared",
            "estimate": eta_sq,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "method": "ANOVA effect size",
            "n_boot": np.nan,
            "alpha": ALPHA,
        },
        {
            "analysis_block": "RQ4",
            "metric": "ANOVA Cohen's f",
            "estimate": cohen_f,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "method": "Converted from eta-squared",
            "n_boot": np.nan,
            "alpha": ALPHA,
        },
        {
            "analysis_block": "RQ4",
            "metric": "Chi-square Cramer's V (cluster vs high54)",
            "estimate": cramers_v,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "method": "Association effect size",
            "n_boot": np.nan,
            "alpha": ALPHA,
        },
    ]
    infer_df = pd.DataFrame(infer_rows)
    infer_df.to_csv(TABLES / "inference_effectsize_ci_summary.csv", index=False)

    power_rows = [
        {
            "research_block": "Global settings",
            "method": "Inference policy",
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "confidence_level": CONFIDENCE_LEVEL,
            "required_n": np.nan,
            "observed_n": len(clustered),
            "adequacy": "Configured",
            "notes": "Alpha and power defaults used for sample-size justification.",
        },
        {
            "research_block": "RQ1 classification",
            "method": "EPV>=10 for logistic model (61 predictors)",
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "confidence_level": CONFIDENCE_LEVEL,
            "required_n": n_epv_required,
            "observed_n": len(clustered),
            "adequacy": "Adequate" if len(clustered) >= n_epv_required else "Not adequate",
            "notes": f"Required events={required_events}; min class rate={min_class_rate_54:.4f}.",
        },
        {
            "research_block": "RQ3 nested AUC lift (threshold54)",
            "method": "Normal-approx sample size from observed DeLong z",
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "confidence_level": CONFIDENCE_LEVEL,
            "required_n": n_rq3_req_54,
            "observed_n": len(pred54),
            "adequacy": "Adequate" if len(pred54) >= n_rq3_req_54 else "Not adequate",
            "notes": f"z_observed={z54:.4f}.",
        },
        {
            "research_block": "RQ3 nested AUC lift (threshold56)",
            "method": "Normal-approx sample size from observed DeLong z",
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "confidence_level": CONFIDENCE_LEVEL,
            "required_n": n_rq3_req_56,
            "observed_n": len(pred56),
            "adequacy": "Adequate" if len(pred56) >= n_rq3_req_56 else "Not adequate",
            "notes": f"z_observed={z56:.4f}.",
        },
        {
            "research_block": "RQ4 clustering",
            "method": "ANOVA power analysis using observed Cohen's f (k=3)",
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "confidence_level": CONFIDENCE_LEVEL,
            "required_n": n_rq4_required,
            "observed_n": len(clustered),
            "adequacy": "Adequate" if len(clustered) >= n_rq4_required else "Not adequate",
            "notes": f"Cohen's f={cohen_f:.4f}.",
        },
    ]
    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(TABLES / "sample_size_power_summary.csv", index=False)

    print("saved", TABLES / "inference_effectsize_ci_summary.csv")
    print("saved", TABLES / "sample_size_power_summary.csv")


if __name__ == "__main__":
    main()
