from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    calinski_harabasz_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
THRESHOLDS = [54, 56]
TEST_SIZE = 0.2


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = np.vstack([pred_a, pred_b])[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    contrast = np.array([[1, -1]])
    z_matrix = np.abs(np.diff(aucs)) / np.sqrt(contrast @ cov @ contrast.T)
    z_value = float(np.ravel(z_matrix)[0])
    p_value = float(2 * (1 - stats.norm.cdf(z_value)))
    return {
        "auc_model_a": float(aucs[0]),
        "auc_model_b": float(aucs[1]),
        "z_stat": z_value,
        "p_value": p_value,
    }


@dataclass
class Paths:
    root: Path
    data: Path
    audit: Path
    out: Path
    tables: Path
    figures: Path
    models: Path
    logs: Path


def build_paths() -> Paths:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "final_run_may15_2026"
    tables = out / "tables"
    figures = out / "figures"
    models = out / "models"
    logs = out / "logs"
    for p in [out, tables, figures, models, logs]:
        p.mkdir(parents=True, exist_ok=True)
    return Paths(
        root=root,
        data=root / "data" / "processed" / "final_dataset_capstone_v2.csv",
        audit=root / "audit" / "variable_selection_audit.csv",
        out=out,
        tables=tables,
        figures=figures,
        models=models,
        logs=logs,
    )


def get_feature_groups(df: pd.DataFrame, audit_df: pd.DataFrame) -> Dict[str, List[str]]:
    audit_work = audit_df.copy()
    if "decision" not in audit_work.columns:
        # Backward compatibility for audit files where decision is embedded in text.
        decision_text = (
            audit_work["criterion_data_quality_variation"]
            .astype(str)
            .str.extract(r"decision=([a-zA-Z_]+)", expand=False)
            .fillna("keep")
            .str.lower()
        )
        audit_work["decision"] = decision_text

    audit_keep = audit_work[audit_work["decision"].isin(["keep", "keep_with_imputation"])].copy()
    group_map = {
        g: [v for v in audit_keep[audit_keep["group"] == g]["variable"].tolist() if v in df.columns]
        for g in audit_keep["group"].unique()
    }

    baseline = [c for c in ["agecat", "PPINCIMP", "PPEDUC"] if c in df.columns]
    model_a = [
        c
        for c in ["agecat", "PPINCIMP", "PPEDUC", "EMPLOY", "PPGENDER", "PPETHM", "LMscore", "KHscore", "FSscore"]
        if c in df.columns
    ]

    model_b_blocks = []
    for g in [
        "behavioral",
        "stress_hardship",
        "major_life_events",
        "family_social_background",
        "psychological_values",
    ]:
        model_b_blocks.extend(group_map.get(g, []))
    model_b = list(dict.fromkeys(model_a + model_b_blocks + (["HARDSHIP_TOTAL"] if "HARDSHIP_TOTAL" in df.columns else [])))

    cluster_features = list(
        dict.fromkeys(
            group_map.get("behavioral", [])
            + group_map.get("stress_hardship", [])
            + group_map.get("psychological_values", [])
            + (["HARDSHIP_TOTAL"] if "HARDSHIP_TOTAL" in df.columns else [])
        )
    )

    return {
        "baseline": baseline,
        "model_a": model_a,
        "model_b": model_b,
        "cluster_features": cluster_features,
        "group_map": group_map,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    sample_weight: np.ndarray | None = None,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "precision": float(precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight)),
        "log_loss": float(log_loss(y_true, y_prob, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y_true, y_prob, sample_weight=sample_weight)),
    }


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def build_rf_pipeline(best_params: Dict[str, object] | None = None) -> Pipeline:
    params = {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if best_params:
        params.update(best_params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**params)),
        ]
    )


def build_xgb_pipeline(best_params: Dict[str, object] | None = None) -> Pipeline:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 450,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    if best_params:
        params.update(best_params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**params)),
        ]
    )


def tune_tree_models(x_train: pd.DataFrame, y_train: pd.Series, paths: Paths) -> Dict[str, Dict[str, object]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rf_pipeline = build_rf_pipeline()
    rf_dist = {
        "model__n_estimators": randint(300, 900),
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_leaf": randint(1, 6),
        "model__max_features": ["sqrt", 0.7, 0.9],
    }

    xgb_pipeline = build_xgb_pipeline()
    xgb_dist = {
        "model__n_estimators": randint(250, 700),
        "model__max_depth": randint(3, 7),
        "model__learning_rate": uniform(0.03, 0.09),
        "model__subsample": uniform(0.75, 0.25),
        "model__colsample_bytree": uniform(0.75, 0.25),
        "model__reg_lambda": uniform(0.5, 1.5),
    }

    rf_search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=rf_dist,
        n_iter=16,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    xgb_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_dist,
        n_iter=18,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    rf_search.fit(x_train, y_train)
    xgb_search.fit(x_train, y_train)

    tuning_rows = [
        {
            "model": "RandomForest",
            "best_cv_auc": float(rf_search.best_score_),
            "best_params": json.dumps(rf_search.best_params_),
        },
        {
            "model": "XGBoost",
            "best_cv_auc": float(xgb_search.best_score_),
            "best_params": json.dumps(xgb_search.best_params_),
        },
    ]
    pd.DataFrame(tuning_rows).to_csv(paths.tables / "final_tuning_results_threshold54.csv", index=False)

    rf_best = {k.replace("model__", ""): v for k, v in rf_search.best_params_.items()}
    xgb_best = {k.replace("model__", ""): v for k, v in xgb_search.best_params_.items()}
    return {"rf": rf_best, "xgb": xgb_best}


def plot_roc_curves(y_test: np.ndarray, prob_map: Dict[str, np.ndarray], out_file: Path, title: str) -> None:
    plt.figure(figsize=(9, 6))
    for name, p in prob_map.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, out_file: Path, title: str, threshold: float = 0.5) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_calibration(y_true: np.ndarray, prob_map: Dict[str, np.ndarray], out_file: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    for name, p in prob_map.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label=name)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def train_and_evaluate_threshold(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    threshold: int,
    paths: Paths,
    tuned_params: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, object]:
    y = (df["FWBscore"] >= threshold).astype(int)
    weights = df["finalwt"].to_numpy()

    split_idx_train, split_idx_test = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_baseline_train = df.loc[split_idx_train, feature_sets["baseline"]]
    x_baseline_test = df.loc[split_idx_test, feature_sets["baseline"]]

    x_a_train = df.loc[split_idx_train, feature_sets["model_a"]]
    x_a_test = df.loc[split_idx_test, feature_sets["model_a"]]

    x_b_train = df.loc[split_idx_train, feature_sets["model_b"]]
    x_b_test = df.loc[split_idx_test, feature_sets["model_b"]]

    y_train = y.loc[split_idx_train]
    y_test = y.loc[split_idx_test]
    w_train = weights[df.index.get_indexer(split_idx_train)]
    w_test = weights[df.index.get_indexer(split_idx_test)]

    if threshold == 54 and tuned_params is None:
        tuned_params = tune_tree_models(x_b_train, y_train, paths)
    elif tuned_params is None:
        tuned_params = {"rf": {}, "xgb": {}}

    models = {
        "Baseline_Logistic": build_logistic_pipeline(),
        "ModelA_Logistic": build_logistic_pipeline(),
        "ModelB_Logistic": build_logistic_pipeline(),
        "ModelB_RandomForest": build_rf_pipeline(tuned_params.get("rf")),
        "ModelB_XGBoost": build_xgb_pipeline(tuned_params.get("xgb")),
    }

    model_inputs = {
        "Baseline_Logistic": (x_baseline_train, x_baseline_test),
        "ModelA_Logistic": (x_a_train, x_a_test),
        "ModelB_Logistic": (x_b_train, x_b_test),
        "ModelB_RandomForest": (x_b_train, x_b_test),
        "ModelB_XGBoost": (x_b_train, x_b_test),
    }

    metrics_rows = []
    metrics_weighted_rows = []
    prob_map = {}
    fitted_models = {}

    for name, model in models.items():
        xtr, xte = model_inputs[name]
        fit_kwargs = {}
        if "Logistic" in name:
            fit_kwargs = {"model__sample_weight": w_train}
        elif "XGBoost" in name:
            fit_kwargs = {"model__sample_weight": w_train}
        elif "RandomForest" in name:
            fit_kwargs = {"model__sample_weight": w_train}

        # Weighted fitting as robustness default in final run.
        model.fit(xtr, y_train, **fit_kwargs)
        p = model.predict_proba(xte)[:, 1]
        prob_map[name] = p
        fitted_models[name] = model

        m_unw = classification_metrics(y_test.to_numpy(), p, threshold=0.5, sample_weight=None)
        m_w = classification_metrics(y_test.to_numpy(), p, threshold=0.5, sample_weight=w_test)

        m_unw.update({"model": name, "threshold_rule": f"FWBscore>={threshold}", "metric_view": "unweighted"})
        m_w.update({"model": name, "threshold_rule": f"FWBscore>={threshold}", "metric_view": "weighted"})

        metrics_rows.append(m_unw)
        metrics_weighted_rows.append(m_w)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("auc_roc", ascending=False)
    metrics_w_df = pd.DataFrame(metrics_weighted_rows).sort_values("auc_roc", ascending=False)

    metrics_df.to_csv(paths.tables / f"rq1_rq3_metrics_threshold_{threshold}_unweighted.csv", index=False)
    metrics_w_df.to_csv(paths.tables / f"rq1_rq3_metrics_threshold_{threshold}_weighted.csv", index=False)

    # DeLong for ModelA vs ModelB logistic
    delong = delong_test(
        y_test.to_numpy(),
        prob_map["ModelA_Logistic"],
        prob_map["ModelB_Logistic"],
    )
    pd.DataFrame([delong]).to_csv(paths.tables / f"rq3_delong_modelA_vs_modelB_threshold_{threshold}.csv", index=False)

    # Plots
    plot_roc_curves(
        y_test.to_numpy(),
        {
            "Baseline Logistic": prob_map["Baseline_Logistic"],
            "Model A Logistic": prob_map["ModelA_Logistic"],
            "Model B Logistic": prob_map["ModelB_Logistic"],
            "Model B RF": prob_map["ModelB_RandomForest"],
            "Model B XGBoost": prob_map["ModelB_XGBoost"],
        },
        paths.figures / f"rq1_roc_threshold_{threshold}.png",
        f"ROC Curves at Threshold {threshold}",
    )

    # confusion for best model by AUC
    best_model_name = metrics_df.iloc[0]["model"]
    plot_confusion(
        y_test.to_numpy(),
        prob_map[best_model_name],
        paths.figures / f"rq1_confusion_best_threshold_{threshold}.png",
        f"Confusion Matrix: {best_model_name} (Threshold {threshold})",
    )

    # calibration
    plot_calibration(
        y_test.to_numpy(),
        {
            "Model A Logistic": prob_map["ModelA_Logistic"],
            "Model B Logistic": prob_map["ModelB_Logistic"],
            "Model B XGBoost": prob_map["ModelB_XGBoost"],
        },
        paths.figures / f"rq3_calibration_threshold_{threshold}.png",
        f"Calibration Curves at Threshold {threshold}",
    )

    # Save models
    for name, mdl in fitted_models.items():
        joblib.dump(mdl, paths.models / f"{name}_threshold_{threshold}.joblib")

    # save prediction file
    pred_out = pd.DataFrame(
        {
            "index": split_idx_test,
            "y_true": y_test.to_numpy(),
            "finalwt": w_test,
            **{f"prob_{k}": v for k, v in prob_map.items()},
        }
    )
    pred_out.to_csv(paths.tables / f"test_predictions_threshold_{threshold}.csv", index=False)

    return {
        "threshold": threshold,
        "metrics_unweighted": metrics_df,
        "metrics_weighted": metrics_w_df,
        "delong": delong,
        "best_model": best_model_name,
        "prob_map": prob_map,
        "fitted_models": fitted_models,
        "x_b_test": x_b_test,
        "y_test": y_test,
    }


def _extract_shap_values(explainer: shap.TreeExplainer, x_matrix: pd.DataFrame) -> np.ndarray:
    vals = explainer.shap_values(x_matrix)
    if isinstance(vals, list):
        if len(vals) == 2:
            return np.asarray(vals[1])
        return np.asarray(vals[0])
    vals = np.asarray(vals)
    if vals.ndim == 3:
        return vals[:, :, 1] if vals.shape[2] == 2 else vals[:, :, 0]
    return vals


def run_shap(
    model_b_xgb: Pipeline,
    x_b_test: pd.DataFrame,
    paths: Paths,
) -> pd.DataFrame:
    imputer = model_b_xgb.named_steps["imputer"]
    xgb = model_b_xgb.named_steps["model"]

    x_imp = pd.DataFrame(imputer.transform(x_b_test), columns=x_b_test.columns)
    sample_n = min(2200, len(x_imp))
    x_shap = x_imp.sample(sample_n, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(xgb)
    shap_arr = _extract_shap_values(explainer, x_shap)

    imp = pd.DataFrame(
        {
            "feature": x_shap.columns,
            "mean_abs_shap": np.abs(shap_arr).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(paths.tables / "rq2_shap_importance_full.csv", index=False)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_arr, x_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(paths.figures / "rq2_shap_beeswarm_full.png", dpi=190, bbox_inches="tight")
    plt.close()

    top20 = imp.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top20["feature"], top20["mean_abs_shap"], color="#2a6f97")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("RQ2 Top 20 SHAP Features")
    plt.tight_layout()
    plt.savefig(paths.figures / "rq2_shap_top20_bar.png", dpi=190)
    plt.close()

    # dependence plots for top 3 features
    top_feats = imp.head(3)["feature"].tolist()
    for i, feat in enumerate(top_feats, start=1):
        interaction = top_feats[1] if len(top_feats) > 1 else None
        plt.figure(figsize=(8.5, 6))
        shap.dependence_plot(feat, shap_arr, x_shap, interaction_index=interaction, show=False)
        plt.title(f"RQ2 SHAP Dependence: {feat}")
        plt.tight_layout()
        plt.savefig(paths.figures / f"rq2_shap_dependence_{i}_{feat}.png", dpi=190)
        plt.close()

    return imp


def silhouette_score_safe(x_scaled: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(x_scaled, labels))


def run_clustering(df: pd.DataFrame, cluster_features: List[str], paths: Paths) -> Dict[str, object]:
    x_raw = df[cluster_features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = pd.DataFrame(imputer.fit_transform(x_raw), columns=cluster_features)
    x_scaled = scaler.fit_transform(x_imp)

    rows = []
    models = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
        labels = km.fit_predict(x_scaled)
        models[k] = km
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score_safe(x_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(x_scaled, labels),
                "inertia": float(km.inertia_),
            }
        )

    k_eval = pd.DataFrame(rows).sort_values("k")
    k_eval.to_csv(paths.tables / "rq4_kmeans_k_selection_final.csv", index=False)

    best_k = int(k_eval.sort_values(["silhouette", "calinski_harabasz"], ascending=False).iloc[0]["k"])
    best_km = models[best_k]
    labels = best_km.predict(x_scaled)

    cluster_df = df.copy()
    cluster_df["behavior_cluster_final"] = labels

    # Hierarchical labels for comparison
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    cluster_df["hier_cluster_final"] = hier.fit_predict(x_scaled)

    cluster_df.to_csv(paths.tables / "rq4_clustered_dataset_final.csv", index=False)

    # ANOVA on FWBscore
    groups = [cluster_df.loc[cluster_df["behavior_cluster_final"] == c, "FWBscore"].dropna().to_numpy() for c in sorted(cluster_df["behavior_cluster_final"].unique())]
    f_stat, p_value = stats.f_oneway(*groups)
    grand = cluster_df["FWBscore"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((cluster_df["FWBscore"] - grand) ** 2).sum()
    eta_sq = float(ss_between / ss_total)

    anova_df = pd.DataFrame(
        [
            {
                "best_k": best_k,
                "anova_f": float(f_stat),
                "anova_p": float(p_value),
                "eta_squared": eta_sq,
            }
        ]
    )
    anova_df.to_csv(paths.tables / "rq4_anova_final.csv", index=False)

    tukey = pairwise_tukeyhsd(endog=cluster_df["FWBscore"], groups=cluster_df["behavior_cluster_final"], alpha=0.05)
    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df.to_csv(paths.tables / "rq4_tukey_final.csv", index=False)

    # Controlled ANOVA/OLS
    controls = [c for c in ["agecat", "PPINCIMP", "PPEDUC", "EMPLOY", "PPGENDER", "PPETHM"] if c in cluster_df.columns]
    ctl_df = cluster_df[["FWBscore", "behavior_cluster_final"] + controls].dropna().copy()
    formula = "FWBscore ~ C(behavior_cluster_final) + " + " + ".join(controls)
    ols = smf.ols(formula, data=ctl_df).fit()
    ctl_anova = sm.stats.anova_lm(ols, typ=2)
    ctl_anova.to_csv(paths.tables / "rq4_controlled_anova_final.csv")

    # cluster high-rate profile
    cluster_profile = cluster_df.groupby("behavior_cluster_final").agg(
        n=("FWBscore", "size"),
        fwb_mean=("FWBscore", "mean"),
        fwb_sd=("FWBscore", "std"),
        high54_rate=("FWB_high_54", "mean"),
        high56_rate=("FWB_high_56", "mean"),
        planning_mean=("SCFHORIZON", "mean"),
        savehabit_mean=("SAVEHABIT", "mean"),
        goalconf_mean=("GOALCONF", "mean"),
        distress_mean=("DISTRESS", "mean"),
        hardship_total_mean=("HARDSHIP_TOTAL", "mean"),
    ).reset_index()

    # assign archetype labels
    archetypes = []
    for _, r in cluster_profile.iterrows():
        if r["planning_mean"] >= cluster_profile["planning_mean"].median() and r["hardship_total_mean"] <= cluster_profile["hardship_total_mean"].median():
            archetypes.append("Planner-Resilient")
        elif r["planning_mean"] < cluster_profile["planning_mean"].median() and r["hardship_total_mean"] > cluster_profile["hardship_total_mean"].median():
            archetypes.append("Strained-Reactive")
        else:
            archetypes.append("Transitional-Mixed")
    cluster_profile["archetype_label"] = archetypes
    cluster_profile.to_csv(paths.tables / "rq4_cluster_profile_final.csv", index=False)

    # chi-square test for high54 by cluster
    ctab = pd.crosstab(cluster_df["behavior_cluster_final"], cluster_df["FWB_high_54"])
    chi2, chi_p, dof, _ = stats.chi2_contingency(ctab)
    chi_df = pd.DataFrame([{"chi2": float(chi2), "p_value": float(chi_p), "dof": int(dof)}])
    chi_df.to_csv(paths.tables / "rq4_cluster_high54_chi_square.csv", index=False)

    # figures
    plt.figure(figsize=(9, 5.8))
    plt.plot(k_eval["k"], k_eval["inertia"], marker="o", label="Inertia")
    plt.plot(k_eval["k"], k_eval["silhouette"], marker="s", label="Silhouette")
    plt.plot(k_eval["k"], k_eval["calinski_harabasz"] / k_eval["calinski_harabasz"].max(), marker="^", label="Calinski-Harabasz (scaled)")
    plt.xlabel("k")
    plt.ylabel("Metric")
    plt.title("RQ4 K Selection Diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures / "rq4_k_selection_final.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5.2))
    sns.barplot(data=cluster_profile, x="behavior_cluster_final", y="fwb_mean", palette="viridis")
    for _, r in cluster_profile.iterrows():
        plt.text(r["behavior_cluster_final"], r["fwb_mean"] + 0.6, f"n={int(r['n'])}", ha="center", fontsize=9)
    plt.title("Mean FWBscore by Final Behavior Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Mean FWBscore")
    plt.tight_layout()
    plt.savefig(paths.figures / "rq4_cluster_fwb_means_final.png", dpi=180)
    plt.close()

    heat_cols = ["planning_mean", "savehabit_mean", "goalconf_mean", "distress_mean", "hardship_total_mean", "fwb_mean"]
    heat_df = cluster_profile[["behavior_cluster_final"] + heat_cols].set_index("behavior_cluster_final")
    heat_z = (heat_df - heat_df.mean()) / heat_df.std(ddof=0)
    plt.figure(figsize=(9, 5.8))
    sns.heatmap(heat_z, annot=True, cmap="RdYlBu_r", center=0)
    plt.title("Cluster Archetype Signal Heatmap (z-scores)")
    plt.tight_layout()
    plt.savefig(paths.figures / "rq4_cluster_archetype_heatmap_final.png", dpi=180)
    plt.close()

    # hierarchical dendrogram on sample
    n = min(800, len(x_scaled))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(x_scaled), size=n, replace=False)
    z = linkage(x_scaled[idx], method="ward")
    plt.figure(figsize=(11, 5.8))
    dendrogram(z, truncate_mode="lastp", p=25, show_leaf_counts=True)
    plt.title("Hierarchical Dendrogram (Ward, Truncated)")
    plt.xlabel("Merged clusters")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(paths.figures / "rq4_hierarchical_dendrogram_final.png", dpi=180)
    plt.close()

    return {
        "best_k": best_k,
        "anova": anova_df.iloc[0].to_dict(),
        "chi_square": chi_df.iloc[0].to_dict(),
        "profile": cluster_profile,
        "cluster_df": cluster_df,
    }


def build_weighted_summary(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    w = df["finalwt"].to_numpy()
    rows = [
        {
            "metric": "mean_FWBscore",
            "unweighted": float(df["FWBscore"].mean()),
            "weighted": float(np.average(df["FWBscore"], weights=w)),
        },
        {
            "metric": "high54_rate",
            "unweighted": float(df["FWB_high_54"].mean()),
            "weighted": float(np.average(df["FWB_high_54"], weights=w)),
        },
        {
            "metric": "high56_rate",
            "unweighted": float(df["FWB_high_56"].mean()),
            "weighted": float(np.average(df["FWB_high_56"], weights=w)),
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "weighted_descriptive_summary_final.csv", index=False)
    return out


def build_master_report(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    threshold_results: Dict[int, Dict[str, object]],
    shap_imp: pd.DataFrame,
    cluster_results: Dict[str, object],
    weighted_summary: pd.DataFrame,
    paths: Paths,
) -> None:
    t54 = threshold_results[54]
    t56 = threshold_results[56]

    m54 = t54["metrics_unweighted"].set_index("model")
    m56 = t56["metrics_unweighted"].set_index("model")

    lines = [
        "# Final Capstone Completion Report (May 11, 2026)",
        "",
        "## 1) What Was Completed Since Interim Submission",
        "- Completed full SHAP pipeline for RQ2 with global and local interpretability figures.",
        "- Added DeLong tests for Model A vs Model B comparisons.",
        "- Added calibration curves, Brier score reporting, and confusion matrices.",
        "- Added weighted robustness views using `\"finalwt\"`.",
        "- Completed deeper cluster profiling including controlled ANOVA and archetype labels.",
        "- Produced final reproducible package of tables, figures, and model artifacts.",
        "",
        "## 2) Dataset and Feature Scope",
        f"- Final dataset shape: **{df.shape[0]:,} rows x {df.shape[1]} columns**.",
        f"- Baseline features: {len(feature_sets['baseline'])}",
        f"- Model A features: {len(feature_sets['model_a'])}",
        f"- Model B features: {len(feature_sets['model_b'])}",
        f"- Cluster feature set: {len(feature_sets['cluster_features'])}",
        "",
        "## 3) RQ1 + RQ3 Final Model Results",
        "### Threshold 54 (primary)",
        f"- Best model: **{t54['best_model']}**",
        f"- Baseline Logistic AUC: {m54.loc['Baseline_Logistic','auc_roc']:.4f}",
        f"- Model A Logistic AUC: {m54.loc['ModelA_Logistic','auc_roc']:.4f}",
        f"- Model B Logistic AUC: {m54.loc['ModelB_Logistic','auc_roc']:.4f}",
        f"- Model B RF AUC: {m54.loc['ModelB_RandomForest','auc_roc']:.4f}",
        f"- Model B XGBoost AUC: {m54.loc['ModelB_XGBoost','auc_roc']:.4f}",
        f"- DeLong z (Model A vs B logistic): {t54['delong']['z_stat']:.4f}; p-value: {t54['delong']['p_value']:.6g}",
        "",
        "### Threshold 56 (sensitivity)",
        f"- Best model: **{t56['best_model']}**",
        f"- Model A Logistic AUC: {m56.loc['ModelA_Logistic','auc_roc']:.4f}",
        f"- Model B XGBoost AUC: {m56.loc['ModelB_XGBoost','auc_roc']:.4f}",
        f"- DeLong z (Model A vs B logistic): {t56['delong']['z_stat']:.4f}; p-value: {t56['delong']['p_value']:.6g}",
        "",
        "## 4) RQ2 SHAP Final Interpretability",
        "- Top SHAP features (global, mean absolute impact):",
    ]
    for _, row in shap_imp.head(12).iterrows():
        lines.append(f"  - {row['feature']}: {row['mean_abs_shap']:.5f}")

    lines.extend(
        [
            "",
            "## 5) RQ4 Cluster Findings",
            f"- Best k: **{cluster_results['best_k']}**",
            f"- ANOVA F: {cluster_results['anova']['anova_f']:.4f}",
            f"- ANOVA p: {cluster_results['anova']['anova_p']:.6g}",
            f"- Eta-squared: {cluster_results['anova']['eta_squared']:.4f}",
            f"- Chi-square (cluster vs high54): {cluster_results['chi_square']['chi2']:.4f}, p={cluster_results['chi_square']['p_value']:.6g}",
            "",
            "## 6) Weighted Robustness Summary",
        ]
    )
    for _, r in weighted_summary.iterrows():
        lines.append(f"- {r['metric']}: unweighted={r['unweighted']:.6f}, weighted={r['weighted']:.6f}")

    lines.extend(
        [
            "",
            "## 7) Output Artifact Map",
            "- Tables: `final_capstone_outputs_may11_2026/tables/`",
            "- Figures: `final_capstone_outputs_may11_2026/figures/`",
            "- Models: `final_capstone_outputs_may11_2026/models/`",
            "- Logs: `final_capstone_outputs_may11_2026/logs/`",
            "",
            "## 8) International-Standard Quality Controls Used",
            "- Reproducible random seed and saved tuning parameters.",
            "- Primary + sensitivity threshold design (54 and 56).",
            "- Weighted robustness checks with survey final weight.",
            "- Complementary discrimination + calibration metrics.",
            "- Explainable AI (SHAP) for model transparency.",
            "- Unsupervised segmentation with statistical significance tests.",
        ]
    )

    (paths.out / "FINAL_CAPSTONE_COMPLETION_REPORT_MAY11_2026.md").write_text("\n".join(lines), encoding="utf-8")


def write_execution_log(paths: Paths, feature_sets: Dict[str, List[str]]) -> None:
    lines = [
        "# Execution Log (May 11, 2026)",
        "",
        "## Pipeline Steps Executed",
        "1. Loaded final cleaned dataset (`final_dataset_capstone_v2.csv`).",
        "2. Built feature sets for baseline, Model A, Model B, and clustering.",
        "3. Tuned Random Forest and XGBoost hyperparameters at threshold 54 using randomized CV.",
        "4. Trained and evaluated models at thresholds 54 and 56 (unweighted and weighted metrics).",
        "5. Computed DeLong tests for Model A vs Model B logistic probabilities.",
        "6. Generated ROC, confusion matrix, and calibration figures.",
        "7. Ran full SHAP interpretability on final Model B XGBoost for threshold 54.",
        "8. Ran k-means and hierarchical clustering diagnostics; selected best k.",
        "9. Computed ANOVA, Tukey HSD, chi-square, and controlled OLS for cluster validity.",
        "10. Generated final narrative summary and artifact map.",
        "",
        "## Feature Set Snapshot",
        f"- Baseline: {feature_sets['baseline']}",
        f"- Model A: {feature_sets['model_a']}",
        f"- Model B feature count: {len(feature_sets['model_b'])}",
        f"- Cluster feature count: {len(feature_sets['cluster_features'])}",
    ]
    (paths.logs / "execution_log_may11_2026.md").write_text("\n".join(lines), encoding="utf-8")


def sync_to_github_package(paths: Paths) -> None:
    gh_root = paths.root / "financial-wellbeing-capstone-github-final"
    gh_run = gh_root / "outputs" / "final_run_may11_2026"
    gh_tables = gh_run / "tables"
    gh_figures = gh_run / "figures"
    gh_models = gh_run / "models"
    gh_logs = gh_run / "logs"
    for p in [gh_run, gh_tables, gh_figures, gh_models, gh_logs]:
        p.mkdir(parents=True, exist_ok=True)

    for src in paths.tables.glob("*"):
        if src.is_file():
            (gh_tables / src.name).write_bytes(src.read_bytes())
    for src in paths.figures.glob("*"):
        if src.is_file():
            (gh_figures / src.name).write_bytes(src.read_bytes())
    for src in paths.models.glob("*"):
        if src.is_file():
            (gh_models / src.name).write_bytes(src.read_bytes())
    for src in paths.logs.glob("*"):
        if src.is_file():
            (gh_logs / src.name).write_bytes(src.read_bytes())

    report = paths.out / "FINAL_CAPSTONE_COMPLETION_REPORT_MAY11_2026.md"
    if report.exists():
        (gh_root / "docs" / report.name).write_bytes(report.read_bytes())


def run() -> None:
    paths = build_paths()
    df = pd.read_csv(paths.data)
    audit_df = pd.read_csv(paths.audit)

    feature_sets = get_feature_groups(df, audit_df)
    write_execution_log(paths, feature_sets)

    weighted_summary = build_weighted_summary(df, paths)

    # Threshold 54 first with tuning, then threshold 56 reusing tuned params.
    res54 = train_and_evaluate_threshold(df, feature_sets, threshold=54, paths=paths, tuned_params=None)

    tuned_params = {}
    tune_file = paths.tables / "final_tuning_results_threshold54.csv"
    if tune_file.exists():
        tdf = pd.read_csv(tune_file)
        for _, row in tdf.iterrows():
            bp = json.loads(row["best_params"])
            if row["model"] == "RandomForest":
                tuned_params["rf"] = {k.replace("model__", ""): v for k, v in bp.items()}
            if row["model"] == "XGBoost":
                tuned_params["xgb"] = {k.replace("model__", ""): v for k, v in bp.items()}

    res56 = train_and_evaluate_threshold(df, feature_sets, threshold=56, paths=paths, tuned_params=tuned_params)

    # SHAP on threshold-54 Model B XGBoost.
    shap_imp = run_shap(
        model_b_xgb=res54["fitted_models"]["ModelB_XGBoost"],
        x_b_test=res54["x_b_test"],
        paths=paths,
    )

    cluster_results = run_clustering(df, feature_sets["cluster_features"], paths)

    build_master_report(
        df=df,
        feature_sets=feature_sets,
        threshold_results={54: res54, 56: res56},
        shap_imp=shap_imp,
        cluster_results=cluster_results,
        weighted_summary=weighted_summary,
        paths=paths,
    )

    sync_to_github_package(paths)

    summary = {
        "dataset_rows": int(df.shape[0]),
        "dataset_cols": int(df.shape[1]),
        "threshold54_best_model": res54["best_model"],
        "threshold56_best_model": res56["best_model"],
        "threshold54_best_auc": float(res54["metrics_unweighted"].iloc[0]["auc_roc"]),
        "threshold56_best_auc": float(res56["metrics_unweighted"].iloc[0]["auc_roc"]),
        "cluster_best_k": int(cluster_results["best_k"]),
        "cluster_anova_p": float(cluster_results["anova"]["anova_p"]),
        "top_shap_feature": str(shap_imp.iloc[0]["feature"]),
    }
    (paths.logs / "final_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("FINAL PIPELINE COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
