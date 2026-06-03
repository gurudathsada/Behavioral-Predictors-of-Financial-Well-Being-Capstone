from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "outputs" / "final_run_may15_2026"
TABLES = OUT_BASE / "tables"
DOCS = ROOT / "docs" / "final_report"
DATA = ROOT / "audit"


def p4(v) -> float:
    return round(float(v), 4)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)

    summary = json.loads((DATA / "final_dataset_summary.json").read_text())
    m54 = pd.read_csv(TABLES / "rq1_rq3_metrics_threshold_54_unweighted.csv").set_index("model")
    m56 = pd.read_csv(TABLES / "rq1_rq3_metrics_threshold_56_unweighted.csv").set_index("model")
    weighted = pd.read_csv(TABLES / "weighted_descriptive_summary_final.csv").set_index("metric")
    del54 = pd.read_csv(TABLES / "rq3_delong_modelA_vs_modelB_threshold_54.csv").iloc[0]
    del56 = pd.read_csv(TABLES / "rq3_delong_modelA_vs_modelB_threshold_56.csv").iloc[0]
    shap = pd.read_csv(TABLES / "rq2_shap_importance_full.csv")
    anova = pd.read_csv(TABLES / "rq4_anova_final.csv").iloc[0]
    tukey = pd.read_csv(TABLES / "rq4_tukey_final.csv")
    cluster = pd.read_csv(TABLES / "rq4_cluster_profile_final.csv")
    cv = pd.read_csv(TABLES / "rq1_rq3_repeated_cv_threshold54.csv")
    ablation = pd.read_csv(TABLES / "rq1_overlap_ablation_threshold54.csv")
    subgroup_gap = pd.read_csv(TABLES / "rq1_subgroup_gap_summary_threshold54.csv")
    infer = pd.read_csv(TABLES / "inference_effectsize_ci_summary.csv")
    power = pd.read_csv(TABLES / "sample_size_power_summary.csv")

    # Fixed archetype naming for final narrative.
    cluster_sorted = cluster.sort_values("fwb_mean").copy()
    ordered_clusters = cluster_sorted["behavior_cluster_final"].tolist()
    cluster_name_map = {}
    if len(ordered_clusters) >= 3:
        cluster_name_map = {
            int(ordered_clusters[0]): "Reactive-Strained",
            int(ordered_clusters[1]): "Transitional Planner",
            int(ordered_clusters[2]): "Stable Planner",
        }
    cluster["archetype_label_final"] = cluster["behavior_cluster_final"].map(cluster_name_map).fillna(cluster["archetype_label"])

    cv_best = cv.sort_values("cv_auc_mean", ascending=False).iloc[0]
    ab_delta = ablation.loc[ablation["model_variant"] == "Delta_Full_minus_Ablated"].iloc[0]
    top_shap = shap.head(10)

    def infer_est(metric_name: str, col: str = "estimate") -> float:
        row = infer.loc[infer["metric"] == metric_name]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][col])

    def power_req(block_name: str) -> float:
        row = power.loc[power["research_block"] == block_name]
        if row.empty:
            return float("nan")
        return float(row.iloc[0]["required_n"])

    records = [
        ("dataset_raw_rows", summary["raw_shape"]["rows"]),
        ("dataset_raw_cols", summary["raw_shape"]["cols"]),
        ("dataset_final_rows", summary["final_shape"]["rows"]),
        ("dataset_final_cols", summary["final_shape"]["cols"]),
        ("dataset_retained_source_vars", summary["retained_variable_count"]),
        ("dataset_derived_vars", 6),
        ("rq1_auc_baseline_54", p4(m54.loc["Baseline_Logistic", "auc_roc"])),
        ("rq1_auc_modelA_54", p4(m54.loc["ModelA_Logistic", "auc_roc"])),
        ("rq1_auc_modelB_xgb_54", p4(m54.loc["ModelB_XGBoost", "auc_roc"])),
        ("rq1_logloss_modelB_xgb_54", p4(m54.loc["ModelB_XGBoost", "log_loss"])),
        ("rq1_f1_modelB_xgb_54", p4(m54.loc["ModelB_XGBoost", "f1"])),
        ("rq1_auc_modelB_xgb_56", p4(m56.loc["ModelB_XGBoost", "auc_roc"])),
        ("rq3_delong_z_54", p4(del54["z_stat"])),
        ("rq3_delong_p_54", float(del54["p_value"])),
        ("rq3_delong_z_56", p4(del56["z_stat"])),
        ("rq3_delong_p_56", float(del56["p_value"])),
        ("rq2_top_shap_1_feature", str(top_shap.iloc[0]["feature"])),
        ("rq2_top_shap_1_value", p4(top_shap.iloc[0]["mean_abs_shap"])),
        ("rq2_top_shap_2_feature", str(top_shap.iloc[1]["feature"])),
        ("rq2_top_shap_2_value", p4(top_shap.iloc[1]["mean_abs_shap"])),
        ("rq4_best_k", int(anova["best_k"])),
        ("rq4_anova_f", p4(anova["anova_f"])),
        ("rq4_anova_p", float(anova["anova_p"])),
        ("rq4_eta_squared", p4(anova["eta_squared"])),
        ("weighted_mean_fwb", p4(weighted.loc["mean_FWBscore", "weighted"])),
        ("weighted_high54_rate", p4(weighted.loc["high54_rate", "weighted"])),
        ("weighted_high56_rate", p4(weighted.loc["high56_rate", "weighted"])),
        ("cv_best_model", str(cv_best["model"])),
        ("cv_best_auc_mean", p4(cv_best["cv_auc_mean"])),
        ("cv_best_auc_sd", p4(cv_best["cv_auc_std"])),
        ("ablation_auc_drop", p4(ab_delta["auc_roc"])),
        (
            "subgroup_max_recall_gap",
            p4(subgroup_gap[subgroup_gap["metric"] == "recall"]["gap_max_minus_min"].max()),
        ),
        ("rq3_auc_delta_54", p4(infer_est("AUC delta (ModelB_Logistic - ModelA_Logistic) threshold 54"))),
        ("rq3_auc_delta_54_ci_low", p4(infer_est("AUC delta (ModelB_Logistic - ModelA_Logistic) threshold 54", "ci_low"))),
        ("rq3_auc_delta_54_ci_high", p4(infer_est("AUC delta (ModelB_Logistic - ModelA_Logistic) threshold 54", "ci_high"))),
        ("rq3_auc_delta_56", p4(infer_est("AUC delta (ModelB_Logistic - ModelA_Logistic) threshold 56"))),
        ("rq3_auc_delta_56_ci_low", p4(infer_est("AUC delta (ModelB_Logistic - ModelA_Logistic) threshold 56", "ci_low"))),
        ("rq3_auc_delta_56_ci_high", p4(infer_est("AUC delta (ModelB_Logistic - ModelA_Logistic) threshold 56", "ci_high"))),
        ("rq4_cohen_f", p4(infer_est("ANOVA Cohen's f"))),
        ("rq4_cramers_v", p4(infer_est("Chi-square Cramer's V (cluster vs high54)"))),
        ("inference_alpha", 0.05),
        ("inference_target_power", 0.80),
        ("inference_confidence_level", 0.95),
        ("power_required_n_rq1_epv10", p4(power_req("RQ1 classification"))),
        ("power_required_n_rq3_threshold54", p4(power_req("RQ3 nested AUC lift (threshold54)"))),
        ("power_required_n_rq3_threshold56", p4(power_req("RQ3 nested AUC lift (threshold56)"))),
        ("power_required_n_rq4_anova", p4(power_req("RQ4 clustering"))),
    ]
    ledger = pd.DataFrame(records, columns=["metric_key", "metric_value"])
    ledger.to_csv(TABLES / "canonical_numbers_ledger.csv", index=False)
    ledger.to_json(TABLES / "canonical_numbers_ledger.json", orient="records", indent=2)

    cluster_out = cluster[
        ["behavior_cluster_final", "archetype_label_final", "n", "fwb_mean", "high54_rate", "high56_rate"]
    ].copy()
    cluster_out.to_csv(TABLES / "canonical_cluster_labels.csv", index=False)

    # Human-readable summary.
    md_lines = [
        "# Canonical Numbers Ledger (Final Freeze)",
        "",
        "This ledger is the single source of truth for final report/deck numeric consistency.",
        "",
        "## Core Metrics",
    ]
    for _, r in ledger.iterrows():
        md_lines.append(f"- `{r['metric_key']}`: `{r['metric_value']}`")
    md_lines += ["", "## Final Cluster Labels", ""]
    for _, r in cluster_out.sort_values("fwb_mean").iterrows():
        md_lines.append(
            f"- Cluster `{int(r['behavior_cluster_final'])}` -> `{r['archetype_label_final']}` "
            f"(N={int(r['n'])}, mean_FWB={p4(r['fwb_mean'])}, high54={p4(r['high54_rate'])})"
        )
    (DOCS / "00_Canonical_Numbers_Ledger.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("saved", TABLES / "canonical_numbers_ledger.csv")
    print("saved", TABLES / "canonical_cluster_labels.csv")
    print("saved", DOCS / "00_Canonical_Numbers_Ledger.md")


if __name__ == "__main__":
    main()
