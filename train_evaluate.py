"""
CodeNova AI/ML Internship - Task 3: Prediction Model
Binary classification: predict whether a breast mass is MALIGNANT from
Breast Cancer Wisconsin (Diagnostic) features (public UCI dataset, bundled
with scikit-learn -> no download, no paid tools).

Run:  python train_evaluate.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path(__file__).parent
(OUT / "charts").mkdir(exist_ok=True)
(OUT / "models").mkdir(exist_ok=True)

# ------------------------------------------------------------------ 1. DATA
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# sklearn encodes 0 = malignant, 1 = benign. Flip so POSITIVE class = malignant
# (the clinically important class we must not miss).
df["malignant"] = 1 - df["target"]
df = df.drop(columns="target")
print(f"Shape: {df.shape} | missing values: {int(df.isna().sum().sum())} | "
      f"duplicates: {int(df.duplicated().sum())}")
print("Class balance:\n", df["malignant"].value_counts(normalize=True).round(3))

X, y = df.drop(columns="malignant"), df["malignant"]

# ------------------------------------------------- 2. FEATURE ENGINEERING
# Simple, label-free ratio features (computed row-wise -> no leakage).
X = X.copy()
X["area_to_perimeter"] = X["mean area"] / X["mean perimeter"]
X["worst_to_mean_radius"] = X["worst radius"] / X["mean radius"]
X["concavity_x_points"] = X["mean concavity"] * X["mean concave points"]

# Split FIRST; every fitted step (scaler, bins, encoders) lives in a Pipeline
# and is fitted on training folds only -> zero leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# The dataset has no native categorical columns, so we create categorical
# features by quantile-binning 3 key columns (tiny / small / large ...) and
# one-hot encoding them. Bin edges are learned from training data only.
BIN_COLS = ["mean radius", "mean texture", "mean concavity"]
NUM_COLS = [c for c in X.columns]


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", KBinsDiscretizer(n_bins=4, encode="onehot-dense",
                                 strategy="quantile"), BIN_COLS),
    ])


def make_pipe(model):
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


# ------------------------------------------------------ 3. MODEL COMPARISON
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
candidates = {
    "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_rows = []
for name, model in candidates.items():
    res = cross_validate(make_pipe(model), X_train, y_train, cv=cv, scoring=scoring)
    row = {"model": name}
    for s in scoring:
        row[f"{s}_mean"] = res[f"test_{s}"].mean()
        row[f"{s}_std"] = res[f"test_{s}"].std()
    cv_rows.append(row)
cv_df = pd.DataFrame(cv_rows).set_index("model")
print("\n5-fold CV (training set only):\n",
      cv_df[[f"{s}_mean" for s in scoring]].round(4))

# ----------------------------------------------- 4. HYPERPARAMETER TUNING
grids = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "Random Forest": {"model__max_depth": [None, 5, 10],
                      "model__min_samples_leaf": [1, 3]},
    "Gradient Boosting": {"model__n_estimators": [100, 200],
                          "model__learning_rate": [0.05, 0.1],
                          "model__max_depth": [2, 3]},
}
tuned = {}
for name, grid in grids.items():
    gs = GridSearchCV(make_pipe(candidates[name]), grid, cv=cv,
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs
    print(f"Tuned {name}: best CV ROC-AUC={gs.best_score_:.4f} {gs.best_params_}")

# Pick the final model by CV ROC-AUC (test set is NOT used for selection).
best_name = max(tuned, key=lambda n: tuned[n].best_score_)
best_model = tuned[best_name].best_estimator_
print(f"\nSelected model: {best_name}")

# --------------------------------------------------- 5. HELD-OUT TEST EVAL
baseline = make_pipe(candidates["Baseline (majority class)"]).fit(X_train, y_train)
results = {}
for name, est in [("Baseline (majority class)", baseline)] + \
                 [(n, g.best_estimator_) for n, g in tuned.items()]:
    pred = est.predict(X_test)
    proba = est.predict_proba(X_test)[:, 1]
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
res_df = pd.DataFrame(results).T.round(4)
print("\nHeld-out TEST results:\n", res_df)
res_df.to_csv(OUT / "test_results.csv")
cv_df.round(4).to_csv(OUT / "cv_results.csv")

pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]
print("\nClassification report (selected model):\n",
      classification_report(y_test, pred, target_names=["benign", "malignant"]))

# --------------------------------------------------------------- 6. CHARTS
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})

# 6a model comparison (CV ROC-AUC with error bars + test)
fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(results)
cv_means = [cv_df.loc[n, "roc_auc_mean"] if n.startswith("Baseline") else
            tuned[n].best_score_ for n in names]
test_auc = [results[n]["roc_auc"] for n in names]
x = np.arange(len(names)); w = .38
ax.bar(x - w/2, cv_means, w, label="5-fold CV ROC-AUC (train)")
ax.bar(x + w/2, test_auc, w, label="Held-out test ROC-AUC")
ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
ax.set_ylim(0.4, 1.02); ax.set_ylabel("ROC-AUC")
ax.set_title("Model comparison vs. baseline"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "charts/01_model_comparison.png"); plt.close(fig)

# 6b ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for n, est in [(k, g.best_estimator_) for k, g in tuned.items()] + [("Baseline", baseline)]:
    RocCurveDisplay.from_estimator(est, X_test, y_test, name=n, ax=ax)
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_title("ROC curves (held-out test set)")
fig.tight_layout(); fig.savefig(OUT / "charts/02_roc_curves.png"); plt.close(fig)

# 6c confusion matrix
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["benign", "malignant"], cmap="Purples", ax=ax)
ax.grid(False); ax.set_title(f"Confusion matrix - {best_name}")
fig.tight_layout(); fig.savefig(OUT / "charts/03_confusion_matrix.png"); plt.close(fig)

# 6d feature importance (permutation importance on test set - model agnostic)
from sklearn.inspection import permutation_importance
feat_names = list(X.columns)
perm = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                              n_repeats=20, random_state=SEED)
imp = pd.Series(perm.importances_mean, index=feat_names).sort_values().tail(10)
err = pd.Series(perm.importances_std, index=feat_names)[imp.index]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(imp.index, imp.values, xerr=err.values, color="#7c3aed")
ax.set_xlabel("Drop in test ROC-AUC when feature is shuffled")
ax.set_title(f"Top 10 permutation importances - {best_name}")
fig.tight_layout(); fig.savefig(OUT / "charts/04_feature_importance.png"); plt.close(fig)

# 6e probability distribution + threshold (limitation analysis)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(proba[y_test == 0], bins=20, alpha=.7, label="benign (true)")
ax.hist(proba[y_test == 1], bins=20, alpha=.7, label="malignant (true)")
ax.axvline(.5, color="k", ls="--", label="default threshold 0.5")
ax.set_xlabel("Predicted P(malignant)"); ax.set_ylabel("Patients")
ax.set_title("Score separation on test set"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "charts/05_score_distribution.png"); plt.close(fig)

# ------------------------------------------- 7. FAILURE CASE / THRESHOLD
wrong = X_test.assign(true=y_test, pred=pred, p_malignant=proba.round(3))
wrong = wrong[wrong["true"] != wrong["pred"]][["true", "pred", "p_malignant",
                                                  "mean radius", "mean concavity"]]
print("\nMisclassified test samples:\n", wrong)
wrong.to_csv(OUT / "misclassified_test_samples.csv")

thr_rows = []
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (proba >= t).astype(int)
    thr_rows.append({"threshold": t,
                     "precision": precision_score(y_test, p, zero_division=0),
                     "recall": recall_score(y_test, p),
                     "false_negatives": int(((p == 0) & (y_test == 1)).sum()),
                     "false_positives": int(((p == 1) & (y_test == 0)).sum())})
thr_df = pd.DataFrame(thr_rows).round(4)
print("\nThreshold trade-off:\n", thr_df)
thr_df.to_csv(OUT / "threshold_tradeoff.csv", index=False)

# ----------------------------------------------------------------- 8. SAVE
joblib.dump(best_model, OUT / "models/best_model.joblib")
with open(OUT / "models/summary.json", "w") as f:
    json.dump({"selected_model": best_name,
               "best_params": tuned[best_name].best_params_,
               "test_metrics": results[best_name]}, f, indent=2, default=str)
print("\nSaved model + charts.")
