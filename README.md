# Behavioral Predictors of Financial Well-Being (QM640 Interim)

This repository contains interim-stage capstone artifacts based on the CFPB National Financial Well-Being Survey.

## Official data source
- CFPB survey page: https://www.consumerfinance.gov/data-research/financial-well-being-survey-data/

## Repository structure
- `data/raw/` -> official source files (`NFWBS_PUF_2016_data.csv`, codebook, user guide)
- `data/processed/` -> cleaned and modeling datasets
- `audit/` -> variable selection, missingness, imputation, row-level filtering, summary logs
- `notebooks/` -> EDA/modeling/clustering/SHAP notebooks
- `src/` -> reusable code
- `outputs/figures/` -> figures used in report and analysis
- `outputs/tables/` -> model and statistical tables
- `docs/` -> interim report and synopsis

## How to run
1. `pip install -r requirements.txt`
2. `python run_analysis.py`
