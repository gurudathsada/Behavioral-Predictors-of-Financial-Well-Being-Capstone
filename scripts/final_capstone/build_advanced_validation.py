from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

try:
    from xgboost import XGBClassifier

    XGB_OK = True
except Exception:
    XGB_OK = False


ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = ROOT / "outputs" / "final_run_may15_2026"
TABLES = OUT_BASE / "tables"
FIGS = OUT_BASE / "figures"
DATA = ROOT / "data" / "processed"
AUDIT = ROOT / "audit"


def cls_metrics(y_true: pd.Series, proba: np.ndarray) -> dict:
    y_hat = (proba >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def safe_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    if len(pd.Series(y_true).unique()) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA / "final_dataset_capstone_v2.csv")
    rationale = pd.read_csv(AUDIT / "final_dataset_variable_rationale.csv")

    y = df["FWB_high_54"].astype(int)

    baseline_vars = [c for c in ["agecat", "PPINCIMP", "PPEDUC"] if c in df.columns]
    model_a_vars = [c for c in ["agecat", "PPINCIMP", "PPEDUC", "EMPLOY", "LMscore", "KHscore", "FSscore"] if c in df.columns]

    behavior_vars = rationale[rationale["group"].isin(
        [
            "behavioral",
            "stress_hardship",
            "major_life_events",
            "family_social_background",
            "psychological_values",
        ]
    )]["variable"].tolist()
    behavior_vars = [c for c in behavior_vars if c in df.columns]

    model_b_vars = list(dict.fromkeys(model_a_vars + behavior_vars + [c for c in ["HARDSHIP_TOTAL"] if c in df.columns]))

    # ------------------------------------------------------------
    # Repeated CV uncertainty (5-fold stratified)
    # ------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "auc": "roc_auc",
        "f1": "f1",
        "accuracy": "accuracy",
        "neg_log_loss": "neg_log_loss",
    }

    model_specs = {
        "Baseline_Logistic": (
            LogisticRegression(max_iter=3000),
            baseline_vars,
        ),
        "ModelA_Logistic": (
            LogisticRegression(max_iter=3000),
            model_a_vars,
        ),
        "ModelB_Logistic": (
            LogisticRegression(max_iter=3000),
            model_b_vars,
        ),
        "ModelB_RandomForest": (
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            model_b_vars,
        ),
    }

    if XGB_OK:
        model_specs["ModelB_XGBoost"] = (
            XGBClassifier(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            model_b_vars,
        )

    cv_rows = []
    for name, (model, feats) in model_specs.items():
        X = df[feats].astype(float)
        cv_out = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        cv_rows.append(
            {
                "model": name,
                "cv_auc_mean": float(np.mean(cv_out["test_auc"])),
                "cv_auc_std": float(np.std(cv_out["test_auc"], ddof=1)),
                "cv_f1_mean": float(np.mean(cv_out["test_f1"])),
                "cv_f1_std": float(np.std(cv_out["test_f1"], ddof=1)),
                "cv_accuracy_mean": float(np.mean(cv_out["test_accuracy"])),
                "cv_accuracy_std": float(np.std(cv_out["test_accuracy"], ddof=1)),
                "cv_log_loss_mean": float(np.mean(-cv_out["test_neg_log_loss"])),
                "cv_log_loss_std": float(np.std(-cv_out["test_neg_log_loss"], ddof=1)),
                "n_features": len(feats),
            }
        )

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_auc_mean", ascending=False)
    cv_df.to_csv(TABLES / "rq1_rq3_repeated_cv_threshold54.csv", index=False)

    # plot CV AUC means +- std
    plt.figure(figsize=(8, 4.8))
    order = cv_df.sort_values("cv_auc_mean", ascending=True)
    plt.errorbar(order["cv_auc_mean"], order["model"], xerr=order["cv_auc_std"], fmt="o", capsize=4)
    plt.xlabel("AUC (mean ± SD, 5-fold CV)")
    plt.title("RQ1/RQ3 Repeated CV Stability at Threshold 54")
    plt.tight_layout()
    plt.savefig(FIGS / "rq1_cv_auc_errorbars_threshold54.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Ablation test to address proximal-outcome leakage concerns
    # ------------------------------------------------------------
    proximal_vars = [c for c in [
        "DISTRESS",
        "ABSORBSHOCK",
        "VOLATILITY",
        "ENDSMEET",
        "COVERCOSTS",
        "MATHARDSHIP_1",
        "MATHARDSHIP_2",
        "MATHARDSHIP_3",
        "MATHARDSHIP_4",
        "MATHARDSHIP_5",
        "MATHARDSHIP_6",
        "HARDSHIP_TOTAL",
    ] if c in model_b_vars]

    ablated_vars = [c for c in model_b_vars if c not in set(proximal_vars)]

    X_full = df[model_b_vars].astype(float)
    X_ab = df[ablated_vars].astype(float)
    Xf_tr, Xf_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.25, random_state=42, stratify=y)
    Xa_tr, Xa_te, _, _ = train_test_split(X_ab, y, test_size=0.25, random_state=42, stratify=y)

    if XGB_OK:
        mdl_full = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        mdl_ab = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        mdl_full = RandomForestClassifier(n_estimators=450, random_state=42, n_jobs=-1)
        mdl_ab = RandomForestClassifier(n_estimators=450, random_state=42, n_jobs=-1)

    mdl_full.fit(Xf_tr, y_tr)
    mdl_ab.fit(Xa_tr, y_tr)
    p_full = mdl_full.predict_proba(Xf_te)[:, 1]
    p_ab = mdl_ab.predict_proba(Xa_te)[:, 1]

    m_full = cls_metrics(y_te, p_full)
    m_ab = cls_metrics(y_te, p_ab)

    ab_df = pd.DataFrame(
        [
            {
                "model_variant": "ModelB_Full",
                "n_features": len(model_b_vars),
                **m_full,
            },
            {
                "model_variant": "ModelB_Ablated_No_Proximal_StressHardship",
                "n_features": len(ablated_vars),
                **m_ab,
            },
            {
                "model_variant": "Delta_Full_minus_Ablated",
                "n_features": len(model_b_vars) - len(ablated_vars),
                "auc_roc": m_full["auc_roc"] - m_ab["auc_roc"],
                "log_loss": m_full["log_loss"] - m_ab["log_loss"],
                "accuracy": m_full["accuracy"] - m_ab["accuracy"],
                "precision": m_full["precision"] - m_ab["precision"],
                "recall": m_full["recall"] - m_ab["recall"],
                "f1": m_full["f1"] - m_ab["f1"],
            },
        ]
    )
    ab_df.to_csv(TABLES / "rq1_overlap_ablation_threshold54.csv", index=False)

    pd.DataFrame({"removed_proximal_variables": proximal_vars}).to_csv(
        TABLES / "rq1_overlap_ablation_removed_variables.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Subgroup performance using saved holdout predictions
    # ------------------------------------------------------------
    pred = pd.read_csv(TABLES / "test_predictions_threshold_54.csv")
    key = pred["index"].astype(int).values
    hold = df.iloc[key].copy().reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    hold["y_true"] = pred["y_true"].astype(int)
    hold["proba"] = pred["prob_ModelB_XGBoost"].astype(float)
    hold["y_hat"] = (hold["proba"] >= 0.5).astype(int)

    # Compact subgroup bins for readability and fair counts
    hold["age_group_3"] = pd.cut(
        hold["agecat"],
        bins=[0, 2, 5, 8],
        labels=["Young(18-34)", "Mid(35-64)", "Older(65+)"],
        right=False,
    )
    hold["income_group_3"] = pd.cut(
        hold["PPINCIMP"],
        bins=[0, 7, 14, 20],
        labels=["Low", "Middle", "High"],
        right=False,
    )
    hold["education_group_3"] = pd.cut(
        hold["PPEDUC"],
        bins=[0, 3, 5, 7],
        labels=["Lower", "Middle", "Higher"],
        right=False,
    )

    subgroup_rows = []
    for dim in ["age_group_3", "income_group_3", "education_group_3"]:
        for g, sub in hold.groupby(dim, dropna=False):
            if pd.isna(g):
                continue
            y_t = sub["y_true"].astype(int)
            p = sub["proba"].astype(float).values
            y_h = sub["y_hat"].astype(int).values
            tn = int(((y_t == 0) & (y_h == 0)).sum())
            fp = int(((y_t == 0) & (y_h == 1)).sum())
            fn = int(((y_t == 1) & (y_h == 0)).sum())
            tp = int(((y_t == 1) & (y_h == 1)).sum())
            subgroup_rows.append(
                {
                    "dimension": dim,
                    "group": str(g),
                    "n": len(sub),
                    "positive_rate": float(y_t.mean()),
                    "auc_roc": safe_auc(y_t, p),
                    "accuracy": float(accuracy_score(y_t, y_h)),
                    "precision": float(precision_score(y_t, y_h, zero_division=0)),
                    "recall": float(recall_score(y_t, y_h, zero_division=0)),
                    "f1": float(f1_score(y_t, y_h, zero_division=0)),
                    "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else np.nan,
                    "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan,
                }
            )

    sub_df = pd.DataFrame(subgroup_rows)
    sub_df.to_csv(TABLES / "rq1_subgroup_performance_threshold54.csv", index=False)

    gap_rows = []
    for dim, dsub in sub_df.groupby("dimension"):
        for metric in ["auc_roc", "recall", "precision", "f1", "false_negative_rate", "false_positive_rate"]:
            vals = dsub[metric].dropna()
            if len(vals) == 0:
                continue
            gap_rows.append(
                {
                    "dimension": dim,
                    "metric": metric,
                    "max": float(vals.max()),
                    "min": float(vals.min()),
                    "gap_max_minus_min": float(vals.max() - vals.min()),
                }
            )
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(TABLES / "rq1_subgroup_gap_summary_threshold54.csv", index=False)

    # subgroup recall plot
    plot_df = sub_df[sub_df["dimension"].isin(["age_group_3", "income_group_3", "education_group_3"])].copy()
    plt.figure(figsize=(9, 4.8))
    for dim, dsub in plot_df.groupby("dimension"):
        x = [f"{dim}:{g}" for g in dsub["group"]]
        yv = dsub["recall"].values
        plt.plot(x, yv, marker="o", label=dim)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Recall")
    plt.title("Subgroup Recall Comparison (ModelB_XGBoost, Threshold 54)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "rq1_subgroup_recall_threshold54.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Save short summary
    # ------------------------------------------------------------
    summary = {
        "cv_best_model": cv_df.iloc[0]["model"],
        "cv_best_auc_mean": float(cv_df.iloc[0]["cv_auc_mean"]),
        "ablation_full_auc": float(m_full["auc_roc"]),
        "ablation_ablated_auc": float(m_ab["auc_roc"]),
        "ablation_auc_drop": float(m_full["auc_roc"] - m_ab["auc_roc"]),
        "subgroup_dimensions": sorted(sub_df["dimension"].unique().tolist()),
    }
    (TABLES / "rq1_advanced_validation_summary_v4.json").write_text(json.dumps(summary, indent=2))
    print("advanced validation complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
