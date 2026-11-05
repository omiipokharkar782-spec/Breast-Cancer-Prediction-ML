"""Load the saved pipeline and score a few held-out-style samples."""
import joblib, pandas as pd
from sklearn.datasets import load_breast_cancer

model = joblib.load("models/best_model.joblib")
X = load_breast_cancer(as_frame=True).data
X["area_to_perimeter"] = X["mean area"] / X["mean perimeter"]
X["worst_to_mean_radius"] = X["worst radius"] / X["mean radius"]
X["concavity_x_points"] = X["mean concavity"] * X["mean concave points"]
for i in [0, 10, 100, 300]:
    p = model.predict_proba(X.iloc[[i]])[0, 1]
    print(f"Sample {i}: P(malignant)={p:.3f} -> {'MALIGNANT' if p >= .5 else 'benign'}")
