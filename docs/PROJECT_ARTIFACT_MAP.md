# Project Artifact Map

## Data

- Raw CFPB data: `data/raw/NFWBS_PUF_2016_data.csv`
- Codebook: `data/raw/cfpb_nfwbs-puf-codebook.pdf`
- User guide: `data/raw/cfpb_nfwbs-puf-user-guide.pdf`
- Final modeling dataset: `data/processed/final_dataset_capstone_v2.csv`

## Audit Evidence

- Official source verification: `audit/official_source_verification.csv`
- Variable selection: `audit/variable_selection_audit.csv`
- Missingness report: `audit/variable_missingness_quality_report.csv`
- Imputation log: `audit/imputation_log.csv`
- Row filtering log: `audit/row_filter_log.csv`

## Final Analysis Outputs

- Canonical numbers ledger: `outputs/final_run_may15_2026/tables/canonical_numbers_ledger.csv`
- Sample-size/power summary: `outputs/final_run_may15_2026/tables/sample_size_power_summary.csv`
- Inference/effect-size summary: `outputs/final_run_may15_2026/tables/inference_effectsize_ci_summary.csv`
- Classification metrics: `outputs/final_run_may15_2026/tables/rq1_rq3_metrics_threshold_54_unweighted.csv`
- SHAP importance: `outputs/final_run_may15_2026/tables/rq2_shap_importance_full.csv`
- Cluster profiles: `outputs/final_run_may15_2026/tables/rq4_cluster_profile_final.csv`
