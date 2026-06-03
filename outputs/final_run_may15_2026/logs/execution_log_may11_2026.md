# Execution Log (May 11, 2026)

## Pipeline Steps Executed
1. Loaded final cleaned dataset (`final_dataset_capstone_v2.csv`).
2. Built feature sets for baseline, Model A, Model B, and clustering.
3. Tuned Random Forest and XGBoost hyperparameters at threshold 54 using randomized CV.
4. Trained and evaluated models at thresholds 54 and 56 (unweighted and weighted metrics).
5. Computed DeLong tests for Model A vs Model B logistic probabilities.
6. Generated ROC, confusion matrix, and calibration figures.
7. Ran full SHAP interpretability on final Model B XGBoost for threshold 54.
8. Ran k-means and hierarchical clustering diagnostics; selected best k.
9. Computed ANOVA, Tukey HSD, chi-square, and controlled OLS for cluster validity.
10. Generated final narrative summary and artifact map.

## Feature Set Snapshot
- Baseline: ['agecat', 'PPINCIMP', 'PPEDUC']
- Model A: ['agecat', 'PPINCIMP', 'PPEDUC', 'EMPLOY', 'PPGENDER', 'PPETHM', 'LMscore', 'KHscore', 'FSscore']
- Model B feature count: 63
- Cluster feature count: 32