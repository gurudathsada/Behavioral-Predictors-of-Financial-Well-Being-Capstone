# Behavioral Predictors of Financial Well-Being

QM640 Data Analytics Capstone project by Gurudath Sadanandan.

Repository URL: https://github.com/gurudathsada/Behavioral-Predictors-of-Financial-Well-Being-Capstone

## Project Overview

This project studies whether behavioral, hardship, social-background, psychological-value, and demographic-knowledge variables can predict financial well-being among U.S. adults using the CFPB National Financial Well-Being Survey public-use dataset. The final analysis combines supervised classification, model comparison, SHAP interpretability, clustering, and statistical inference.

## Final Report

The final report PDF is submitted directly through the academic portal and is intentionally not stored in this repository. This repository contains the reproducible data, code, audit logs, tables, and figures supporting that submission.

## Repository Structure

- `data/raw/` - official CFPB public-use source files.
- `data/processed/` - cleaned and derived datasets used for analysis.
- `audit/` - source verification, missingness, imputation, row-filtering, and variable-selection logs.
- `notebooks/` - notebook views for EDA, modeling, clustering, and SHAP interpretation.
- `src/` - reusable project code.
- `scripts/final_capstone/` - final analysis, validation, inference, ledger, and report-generation scripts.
- `outputs/final_run_may15_2026/tables/` - final metric, hypothesis-test, power, SHAP, clustering, and canonical-ledger tables.
- `outputs/final_run_may15_2026/figures/` - final figures used in reporting.
- `outputs/final_run_may15_2026/logs/` - run summaries and execution notes.

## Data Source

Consumer Financial Protection Bureau. National Financial Well-Being Survey public-use file, codebook, and user guide.

Official source page: https://www.consumerfinance.gov/data-research/financial-well-being-survey-data/

Files used:

- `NFWBS_PUF_2016_data.csv`
- `cfpb_nfwbs-puf-codebook.pdf`
- `cfpb_nfwbs-puf-user-guide.pdf`

## Reproducibility

1. Install dependencies: `pip install -r requirements.txt`
2. Confirm CFPB source files are present in `data/raw/`.
3. Review data lineage and cleaning decisions in `audit/`.
4. Run or inspect notebooks in `notebooks/`.
5. Full final pipeline scripts are in `scripts/final_capstone/`.
6. Final numbers used in the report are in `outputs/final_run_may15_2026/tables/canonical_numbers_ledger.csv`.

## Key Final Artifacts

- Canonical metrics ledger: `outputs/final_run_may15_2026/tables/canonical_numbers_ledger.csv`
- Sample-size and power summary: `outputs/final_run_may15_2026/tables/sample_size_power_summary.csv`
- Inference effect-size and confidence interval summary: `outputs/final_run_may15_2026/tables/inference_effectsize_ci_summary.csv`
- RQ1/RQ3 model performance: `outputs/final_run_may15_2026/tables/rq1_rq3_metrics_threshold_54_unweighted.csv`
- RQ2 SHAP importance: `outputs/final_run_may15_2026/tables/rq2_shap_importance_full.csv`
- RQ4 cluster profiles: `outputs/final_run_may15_2026/tables/rq4_cluster_profile_final.csv`
