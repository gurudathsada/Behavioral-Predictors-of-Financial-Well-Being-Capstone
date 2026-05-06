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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_THRESHOLD = 54

DEMOGRAPHIC_FEATURES = ["agecat", "PPINCIMP", "PPEDUC", "EMPLOY", "PPHHSIZE"]
KNOWLEDGE_FEATURES = ["LMscore", "KHscore"]
BEHAVIORAL_FEATURES = [
    "ACT1_1",
    "ACT1_2",
    "FINGOALS",
    "PROPPLAN_1",
    "PROPPLAN_2",
    "PROPPLAN_3",
    "PROPPLAN_4",
    "MANAGE1_1",
    "MANAGE1_2",
    "MANAGE1_3",
    "MANAGE1_4",
    "SAVEHABIT",
    "GOALCONF",
    "DISTRESS",
    "SELFCONTROL_1",
    "SELFCONTROL_2",
    "SELFCONTROL_3",
    "HARDSHIP_TOTAL",
]

HARDSHIP_ITEMS = [
    "MATHARDSHIP_1",
    "MATHARDSHIP_2",
    "MATHARDSHIP_3",
    "MATHARDSHIP_4",
    "MATHARDSHIP_5",
    "MATHARDSHIP_6",
]

VALID_RANGES = {
    "FWBscore": (0, 100),
    "FSscore": (0, 100),
    "LMscore": (0, 3),
    "KHscore": (-5, 5),
    "ACT1_1": (1, 5),
    "ACT1_2": (1, 5),
    "FINGOALS": (0, 1),
    "PROPPLAN_1": (1, 5),
    "PROPPLAN_2": (1, 5),
    "PROPPLAN_3": (1, 5),
    "PROPPLAN_4": (1, 5),
    "MANAGE1_1": (1, 5),
    "MANAGE1_2": (1, 5),
    "MANAGE1_3": (1, 5),
    "MANAGE1_4": (1, 5),
    "SAVEHABIT": (1, 6),
    "GOALCONF": (1, 4),
    "DISTRESS": (1, 5),
    "SELFCONTROL_1": (1, 4),
    "SELFCONTROL_2": (1, 4),
    "SELFCONTROL_3": (1, 4),
    "MATHARDSHIP_1": (1, 3),
    "MATHARDSHIP_2": (1, 3),
    "MATHARDSHIP_3": (1, 3),
    "MATHARDSHIP_4": (1, 3),
    "MATHARDSHIP_5": (1, 3),
    "MATHARDSHIP_6": (1, 3),
    "agecat": (1, 8),
    "PPEDUC": (1, 5),
    "PPINCIMP": (1, 9),
    "EMPLOY": (1, 8),
    "PPHHSIZE": (1, 5),
}


@dataclass
class ProjectPaths:
    project_root: Path
    data_raw: Path
    data_processed: Path
    outputs_figures: Path
    outputs_tables: Path
    outputs_models: Path
    reports: Path


def build_paths(project_root: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=project_root,
        data_raw=project_root / "data" / "raw",
        data_processed=project_root / "data" / "processed",
        outputs_figures=project_root / "outputs" / "figures",
        outputs_tables=project_root / "outputs" / "tables",
        outputs_models=project_root / "outputs" / "models",
        reports=project_root / "reports",
    )


def _ensure_dirs(paths: ProjectPaths) -> None:
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    paths.outputs_figures.mkdir(parents=True, exist_ok=True)
    paths.outputs_tables.mkdir(parents=True, exist_ok=True)
    paths.outputs_models.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)


def _enforce_range(series: pd.Series, low: float | None, high: float | None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if low is not None:
        out = out.where(out >= low)
    if high is not None:
        out = out.where(out <= high)
    return out


def load_and_prepare_data(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.data_raw / "NFWBS_PUF_2016_data.csv")

    required_columns = (
        ["FWBscore", "FSscore"]
        + DEMOGRAPHIC_FEATURES
        + KNOWLEDGE_FEATURES
        + [c for c in BEHAVIORAL_FEATURES if c != "HARDSHIP_TOTAL"]
        + HARDSHIP_ITEMS
    )
    required_columns = sorted(set(required_columns))

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in source file: {missing}")

    use_df = df[required_columns].copy()

    for col, (low, high) in VALID_RANGES.items():
        if col in use_df.columns:
            use_df[col] = _enforce_range(use_df[col], low, high)

    hardship_bin = use_df[HARDSHIP_ITEMS].apply(
        lambda col: col.map(lambda x: np.nan if pd.isna(x) else (1 if x >= 2 else 0))
    )
    use_df["HARDSHIP_TOTAL"] = hardship_bin.sum(axis=1, min_count=1)

    use_df = use_df[use_df["FWBscore"].notna()].copy()
    use_df["FWB_high"] = (use_df["FWBscore"] > TARGET_THRESHOLD).astype(int)

    # Keep a 3-level category aligned to CFPB distribution ranges.
    use_df["FWBcat"] = pd.cut(
        use_df["FWBscore"],
        bins=[-np.inf, 44, 61, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    )

    use_df.to_csv(paths.data_processed / "analysis_dataset.csv", index=False)
    return use_df


def save_data_quality_table(df: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for col in sorted(df.columns):
        if col in {"FWB_high", "FWBcat"}:
            continue
        rows.append(
            {
                "variable": col,
                "missing_n": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().mean() * 100),
                "min": float(df[col].min(skipna=True)),
                "max": float(df[col].max(skipna=True)),
            }
        )
    quality = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
    quality.to_csv(paths.outputs_tables / "data_quality_summary.csv", index=False)
    return quality


def build_modeling_matrices(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["FWB_high"].astype(int)

    x_demo = df[DEMOGRAPHIC_FEATURES].copy()
    x_model_a = df[DEMOGRAPHIC_FEATURES + KNOWLEDGE_FEATURES].copy()
    x_model_b = df[DEMOGRAPHIC_FEATURES + KNOWLEDGE_FEATURES + BEHAVIORAL_FEATURES].copy()

    return x_demo, y, x_model_a, x_model_b, df[BEHAVIORAL_FEATURES].copy()


def metric_frame(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
                "log_loss": log_loss(y_true, y_proba),
                "auc_roc": roc_auc_score(y_true, y_proba),
            }
        ]
    )


def fit_logistic_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def fit_ensemble_models(x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 12],
        "model__min_samples_leaf": [1, 4],
        "model__max_features": ["sqrt", 0.7],
    }

    xgb_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    xgb_grid = {
        "model__n_estimators": [250, 450],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    rf_search = GridSearchCV(
        rf_pipeline,
        rf_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    xgb_search = GridSearchCV(
        xgb_pipeline,
        xgb_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    rf_search.fit(x_train, y_train)
    xgb_search.fit(x_train, y_train)

    return {"RandomForest": rf_search, "XGBoost": xgb_search}


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


def save_roc_plot(
    y_test: pd.Series,
    probs: Dict[str, np.ndarray],
    paths: ProjectPaths,
    filename: str,
) -> None:
    plt.figure(figsize=(10, 7))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc_val = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RQ1 ROC Curves: Demographic Baseline vs Ensemble Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(paths.outputs_figures / filename, dpi=160)
    plt.close()


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, paths: ProjectPaths, filename: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(paths.outputs_figures / filename, dpi=160)
    plt.close()


def run_rq1_and_rq3(
    x_demo: pd.DataFrame,
    y: pd.Series,
    x_model_a: pd.DataFrame,
    x_model_b: pd.DataFrame,
    paths: ProjectPaths,
) -> Dict[str, object]:
    x_train_demo, x_test_demo, y_train, y_test = train_test_split(
        x_demo,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_train_a = x_model_a.loc[x_train_demo.index]
    x_test_a = x_model_a.loc[x_test_demo.index]
    x_train_b = x_model_b.loc[x_train_demo.index]
    x_test_b = x_model_b.loc[x_test_demo.index]

    # Baseline for RQ1: demographics only logistic regression.
    demo_model = fit_logistic_model(x_train_demo, y_train)
    demo_probs = demo_model.predict_proba(x_test_demo)[:, 1]
    demo_pred = (demo_probs >= 0.5).astype(int)
    demo_metrics = metric_frame(y_test, demo_pred, demo_probs, "Logistic (Demographic baseline)")

    # Ensemble models on expanded feature set.
    ensemble_searches = fit_ensemble_models(x_train_b, y_train)

    ensemble_metrics: List[pd.DataFrame] = []
    ensemble_probabilities: Dict[str, np.ndarray] = {}
    best_estimators: Dict[str, Pipeline] = {}

    for label, search in ensemble_searches.items():
        best = search.best_estimator_
        probs = best.predict_proba(x_test_b)[:, 1]
        pred = (probs >= 0.5).astype(int)
        ensemble_metrics.append(metric_frame(y_test, pred, probs, label))
        ensemble_probabilities[label] = probs
        best_estimators[label] = best

    rq1_metrics = pd.concat([demo_metrics] + ensemble_metrics, ignore_index=True)
    rq1_metrics.to_csv(paths.outputs_tables / "rq1_model_performance.csv", index=False)

    cv_rows = []
    for label, search in ensemble_searches.items():
        cv_rows.append(
            {
                "model": label,
                "best_cv_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_),
            }
        )
    pd.DataFrame(cv_rows).to_csv(paths.outputs_tables / "rq1_cv_best_params.csv", index=False)

    save_roc_plot(
        y_test,
        {
            "Demographic Logistic": demo_probs,
            "RandomForest": ensemble_probabilities["RandomForest"],
            "XGBoost": ensemble_probabilities["XGBoost"],
        },
        paths,
        "rq1_roc_curves.png",
    )

    best_ensemble_name = rq1_metrics.loc[
        rq1_metrics["model"].isin(["RandomForest", "XGBoost"]), "auc_roc"
    ].idxmax()
    best_ensemble_label = rq1_metrics.loc[best_ensemble_name, "model"]
    best_ensemble_estimator = best_estimators[best_ensemble_label]
    best_ensemble_probs = ensemble_probabilities[best_ensemble_label]
    best_ensemble_pred = (best_ensemble_probs >= 0.5).astype(int)

    save_confusion_matrix(
        y_test,
        best_ensemble_pred,
        paths,
        "rq1_best_model_confusion_matrix.png",
        f"Best Ensemble Confusion Matrix ({best_ensemble_label})",
    )

    # RQ3 nested logistic comparison.
    model_a = fit_logistic_model(x_train_a, y_train)
    model_b = fit_logistic_model(x_train_b, y_train)

    a_probs = model_a.predict_proba(x_test_a)[:, 1]
    b_probs = model_b.predict_proba(x_test_b)[:, 1]
    a_pred = (a_probs >= 0.5).astype(int)
    b_pred = (b_probs >= 0.5).astype(int)

    rq3_metrics = pd.concat(
        [
            metric_frame(y_test, a_pred, a_probs, "Model A: Demographics + Knowledge"),
            metric_frame(y_test, b_pred, b_probs, "Model B: Model A + Behavioral"),
        ],
        ignore_index=True,
    )

    delong = delong_test(y_test.to_numpy(), a_probs, b_probs)
    delong_row = pd.DataFrame([delong])

    rq3_metrics.to_csv(paths.outputs_tables / "rq3_nested_model_metrics.csv", index=False)
    delong_row.to_csv(paths.outputs_tables / "rq3_delong_test.csv", index=False)

    joblib.dump(best_ensemble_estimator, paths.outputs_models / "best_ensemble_model.joblib")
    joblib.dump(model_a, paths.outputs_models / "rq3_model_a_logistic.joblib")
    joblib.dump(model_b, paths.outputs_models / "rq3_model_b_logistic.joblib")

    return {
        "x_train_b": x_train_b,
        "x_test_b": x_test_b,
        "y_test": y_test,
        "best_ensemble_label": best_ensemble_label,
        "best_ensemble_estimator": best_ensemble_estimator,
        "best_ensemble_probs": best_ensemble_probs,
        "rq1_metrics": rq1_metrics,
        "rq3_metrics": rq3_metrics,
        "rq3_delong": delong,
    }


def _extract_shap_values(explainer: shap.TreeExplainer, x_matrix: pd.DataFrame) -> np.ndarray:
    shap_values = explainer.shap_values(x_matrix)
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        # Expected shape for multiclass/binary depending backend.
        return shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
    return shap_values


def run_rq2_shap(
    best_model: Pipeline,
    x_test: pd.DataFrame,
    paths: ProjectPaths,
) -> pd.DataFrame:
    imputer: SimpleImputer = best_model.named_steps["imputer"]
    estimator = best_model.named_steps["model"]
    x_test_imp = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns)

    # SHAP on a manageable sample for speed and reproducibility.
    sample_n = min(2000, len(x_test_imp))
    x_shap = x_test_imp.sample(sample_n, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(estimator)
    shap_array = _extract_shap_values(explainer, x_shap)

    shap_importance = pd.DataFrame(
        {
            "feature": x_shap.columns,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(paths.outputs_tables / "rq2_shap_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_array, x_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(paths.outputs_figures / "rq2_shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    top15 = shap_importance.head(15).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top15["feature"], top15["mean_abs_shap"], color="#2a6f97")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("RQ2 Top 15 SHAP Features")
    plt.tight_layout()
    plt.savefig(paths.outputs_figures / "rq2_shap_top15_bar.png", dpi=180)
    plt.close()

    if {"PROPPLAN_4", "GOALCONF"}.issubset(set(x_shap.columns)):
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            "PROPPLAN_4",
            shap_array,
            x_shap,
            interaction_index="GOALCONF",
            show=False,
        )
        plt.title("RQ2 SHAP Dependence: PROPPLAN_4 vs GOALCONF")
        plt.tight_layout()
        plt.savefig(paths.outputs_figures / "rq2_shap_dependence_propplan4_goalconf.png", dpi=180)
        plt.close()

    return shap_importance


def run_rq4_clustering(
    df: pd.DataFrame,
    behavioral_only: pd.DataFrame,
    paths: ProjectPaths,
) -> Dict[str, object]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_behavior = pd.DataFrame(imputer.fit_transform(behavioral_only), columns=behavioral_only.columns)
    x_scaled = scaler.fit_transform(x_behavior)

    k_rows = []
    inertia_vals = []
    k_models: Dict[int, KMeans] = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(x_scaled)
        k_models[k] = km
        inertia_vals.append((k, km.inertia_))
        sil = silhouette_score_safe(x_scaled, labels)
        ch = calinski_harabasz_score(x_scaled, labels)
        k_rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "inertia": km.inertia_})

    k_eval = pd.DataFrame(k_rows)
    k_eval.to_csv(paths.outputs_tables / "rq4_kmeans_k_selection.csv", index=False)

    best_k = int(k_eval.sort_values(["silhouette", "calinski_harabasz"], ascending=False).iloc[0]["k"])
    best_kmeans = k_models[best_k]
    kmeans_labels = best_kmeans.predict(x_scaled)

    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hier_labels = hier.fit_predict(x_scaled)

    cluster_df = df.copy()
    cluster_df["kmeans_cluster"] = kmeans_labels
    cluster_df["hier_cluster"] = hier_labels
    cluster_df.to_csv(paths.data_processed / "clustered_analysis_dataset.csv", index=False)

    profile = cluster_df.groupby("kmeans_cluster")[BEHAVIORAL_FEATURES + ["FWBscore"]].mean().round(3)
    profile.to_csv(paths.outputs_tables / "rq4_cluster_profiles.csv")

    # ANOVA and effect size.
    groups = [
        cluster_df.loc[cluster_df["kmeans_cluster"] == c, "FWBscore"].dropna().to_numpy()
        for c in sorted(cluster_df["kmeans_cluster"].unique())
    ]
    f_stat, p_value = stats.f_oneway(*groups)
    grand_mean = cluster_df["FWBscore"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((cluster_df["FWBscore"] - grand_mean) ** 2).sum()
    eta_sq = float(ss_between / ss_total)

    anova_table = pd.DataFrame(
        [
            {
                "f_stat": float(f_stat),
                "p_value": float(p_value),
                "eta_squared": eta_sq,
                "best_k": best_k,
            }
        ]
    )
    anova_table.to_csv(paths.outputs_tables / "rq4_cluster_anova.csv", index=False)

    tukey = pairwise_tukeyhsd(
        endog=cluster_df["FWBscore"],
        groups=cluster_df["kmeans_cluster"],
        alpha=0.05,
    )
    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df.to_csv(paths.outputs_tables / "rq4_cluster_tukey_hsd.csv", index=False)

    control_df = cluster_df[["FWBscore", "kmeans_cluster"] + DEMOGRAPHIC_FEATURES].dropna().copy()
    model = smf.ols(
        "FWBscore ~ C(kmeans_cluster) + agecat + PPINCIMP + PPEDUC + EMPLOY + PPHHSIZE",
        data=control_df,
    ).fit()
    control_anova = sm.stats.anova_lm(model, typ=2)
    control_anova.to_csv(paths.outputs_tables / "rq4_controlled_anova.csv")

    # K selection figure.
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(k_eval["k"], k_eval["inertia"], marker="o", color="#ef476f", label="Inertia (Elbow)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia", color="#ef476f")
    ax1.tick_params(axis="y", labelcolor="#ef476f")

    ax2 = ax1.twinx()
    ax2.plot(k_eval["k"], k_eval["silhouette"], marker="s", color="#118ab2", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="#118ab2")
    ax2.tick_params(axis="y", labelcolor="#118ab2")

    plt.title("RQ4 K-Means Selection Metrics (k=3..6)")
    fig.tight_layout()
    plt.savefig(paths.outputs_figures / "rq4_kmeans_selection_metrics.png", dpi=170)
    plt.close()

    # Cluster profile heatmap.
    plt.figure(figsize=(12, 7))
    sns.heatmap(profile, cmap="RdYlBu_r", center=profile.values.mean())
    plt.title("RQ4 Behavioral Cluster Profiles (K-Means)")
    plt.tight_layout()
    plt.savefig(paths.outputs_figures / "rq4_cluster_profile_heatmap.png", dpi=170)
    plt.close()

    # Ward dendrogram on a sample for readability.
    dendro_sample = min(800, len(x_scaled))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(x_scaled), size=dendro_sample, replace=False)
    z = linkage(x_scaled[idx], method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(z, truncate_mode="lastp", p=25, show_leaf_counts=True)
    plt.title("RQ4 Hierarchical Clustering Dendrogram (Ward, Truncated)")
    plt.xlabel("Merged Clusters")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(paths.outputs_figures / "rq4_hierarchical_dendrogram.png", dpi=170)
    plt.close()

    return {
        "best_k": best_k,
        "k_eval": k_eval,
        "cluster_profile": profile,
        "anova": anova_table,
        "controlled_anova": control_anova,
    }


def silhouette_score_safe(x_scaled: np.ndarray, labels: np.ndarray) -> float:
    # Guard in case a candidate k collapses into one label (unlikely but possible).
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(x_scaled, labels))


def build_executive_summary(
    df: pd.DataFrame,
    rq1_metrics: pd.DataFrame,
    rq3_metrics: pd.DataFrame,
    rq3_delong: Dict[str, float],
    shap_importance: pd.DataFrame,
    rq4_results: Dict[str, object],
    paths: ProjectPaths,
) -> None:
    best_model_row = rq1_metrics.sort_values("auc_roc", ascending=False).iloc[0]
    top_shap = shap_importance.head(10)

    lines = [
        "# Capstone Results Summary",
        "",
        "## Dataset",
        f"- Records analyzed: {len(df):,}",
        f"- Features in behavioral model: {len(DEMOGRAPHIC_FEATURES + KNOWLEDGE_FEATURES + BEHAVIORAL_FEATURES)}",
        f"- Target threshold: FWBscore > {TARGET_THRESHOLD}",
        f"- High well-being prevalence: {df['FWB_high'].mean() * 100:.2f}%",
        "",
        "## RQ1 Classification",
        f"- Best model: {best_model_row['model']}",
        f"- Best model AUC-ROC: {best_model_row['auc_roc']:.4f}",
        f"- Best model log-loss: {best_model_row['log_loss']:.4f}",
        "",
        "## RQ3 Nested Model Test",
        f"- Model A AUC: {rq3_delong['auc_model_a']:.4f}",
        f"- Model B AUC: {rq3_delong['auc_model_b']:.4f}",
        f"- DeLong z: {rq3_delong['z_stat']:.4f}",
        f"- DeLong p-value: {rq3_delong['p_value']:.6f}",
        f"- Model A log-loss: {rq3_metrics.loc[rq3_metrics['model'].str.contains('Model A'),'log_loss'].iloc[0]:.4f}",
        f"- Model B log-loss: {rq3_metrics.loc[rq3_metrics['model'].str.contains('Model B'),'log_loss'].iloc[0]:.4f}",
        "",
        "## RQ2 Top SHAP Features",
    ]
    for _, row in top_shap.iterrows():
        lines.append(f"- {row['feature']}: {row['mean_abs_shap']:.5f}")

    lines.extend(
        [
            "",
            "## RQ4 Cluster Findings",
            f"- Selected k (K-Means): {rq4_results['best_k']}",
            f"- ANOVA p-value across clusters (FWBscore): {rq4_results['anova']['p_value'].iloc[0]:.6f}",
            f"- Eta-squared: {rq4_results['anova']['eta_squared'].iloc[0]:.4f}",
            "",
            "## Output Artifacts",
            "- Tables: outputs/tables/",
            "- Figures: outputs/figures/",
            "- Models: outputs/models/",
        ]
    )

    (paths.reports / "capstone_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_full_analysis(project_root: Path | str) -> Dict[str, object]:
    project_root = Path(project_root).resolve()
    paths = build_paths(project_root)
    _ensure_dirs(paths)

    df = load_and_prepare_data(paths)
    save_data_quality_table(df, paths)

    x_demo, y, x_model_a, x_model_b, behavior_only = build_modeling_matrices(df)

    rq1_rq3 = run_rq1_and_rq3(x_demo, y, x_model_a, x_model_b, paths)
    shap_importance = run_rq2_shap(
        rq1_rq3["best_ensemble_estimator"],
        rq1_rq3["x_test_b"],
        paths,
    )
    rq4_results = run_rq4_clustering(df, behavior_only, paths)

    build_executive_summary(
        df,
        rq1_rq3["rq1_metrics"],
        rq1_rq3["rq3_metrics"],
        rq1_rq3["rq3_delong"],
        shap_importance,
        rq4_results,
        paths,
    )

    return {
        "rows": len(df),
        "high_class_rate": float(df["FWB_high"].mean()),
        "best_model": rq1_rq3["best_ensemble_label"],
        "rq1_metrics": rq1_rq3["rq1_metrics"],
        "rq3_metrics": rq1_rq3["rq3_metrics"],
        "rq3_delong": rq1_rq3["rq3_delong"],
        "top_shap": shap_importance.head(10),
        "best_k": int(rq4_results["best_k"]),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = run_full_analysis(root)
    print("Analysis complete")
    print("Best model:", out["best_model"])
    print("Best k:", out["best_k"])
