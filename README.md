# Breast Cancer Malignancy Prediction — CodeNova AI/ML Internship (Task 3)

Binary classification model that predicts whether a breast mass is **malignant** from
cell-nucleus measurements, with baseline comparison, cross-validation, tuning,
threshold analysis and an honest limitations section.

## Dataset
**Breast Cancer Wisconsin (Diagnostic)** — public UCI dataset, 569 samples, 30 numeric
features, no missing values, no duplicates. Loaded via `sklearn.datasets.load_breast_cancer`
(bundled, no download). Positive class = **malignant** (37.3% of data).

## Methodology
1. **Split first**: 80/20 stratified train/test (`random_state=42`). Test set (114 rows) is touched only for final reporting.
2. **Feature engineering**: `area_to_perimeter`, `worst_to_mean_radius`, `concavity_x_points` (row-wise ratios/products, no label information).
3. **Categorical encoding**: the dataset has no native categorical columns, so `mean radius`, `mean texture`, `mean concavity` are quantile-binned into 4 groups and one-hot encoded (bin edges learned on training data only).
4. **Scaling**: `StandardScaler` on all numeric features.
5. **Leakage control**: scaler, binning and encoder are inside a scikit-learn `Pipeline`, so they are re-fitted on each CV training fold and never see validation/test data.
6. **Models**: Majority-class baseline, Logistic Regression, Random Forest, Gradient Boosting.
7. **Validation**: 5-fold stratified CV on the training set; `GridSearchCV` (scoring = ROC-AUC) for hyperparameters.
8. **Model selection** by CV ROC-AUC only (not by test score).

## Results

### 5-fold CV on training set (default hyperparameters)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (majority) | 0.626 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.976 | 0.982 | 0.953 | 0.967 | 0.996 |
| Random Forest | 0.963 | 0.959 | 0.941 | 0.950 | 0.989 |
| Gradient Boosting | 0.965 | 0.959 | 0.947 | 0.953 | 0.991 |

(Full table incl. std-devs: `cv_results.csv`.)

### Held-out test set (tuned models)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (majority) | 0.632 | 0.000 | 0.000 | 0.000 | 0.500 |
| **Logistic Regression (selected)** | **0.974** | **0.976** | **0.952** | **0.964** | **0.997** |
| Random Forest | 0.974 | 1.000 | 0.929 | 0.963 | 0.996 |
| Gradient Boosting | 0.965 | 1.000 | 0.905 | 0.950 | 0.998 |

**Gain over baseline:** accuracy +34 points (0.632 → 0.974) and F1 0 → 0.964. The baseline never predicts malignant, which is why accuracy alone is misleading on imbalanced data.

Best params — LR: `C=1`; RF: `max_depth=5, min_samples_leaf=1`; GB: `lr=0.1, max_depth=2, n_estimators=100`.

### Charts (`charts/`)
| File | Content |
|---|---|
| `01_model_comparison.png` | CV vs. test ROC-AUC for every model and the baseline |
| `02_roc_curves.png` | ROC curves, test set |
| `03_confusion_matrix.png` | Confusion matrix of selected model |
| `04_feature_importance.png` | Permutation importance (top 10) |
| `05_score_distribution.png` | Predicted-probability separation by true class |

## Limitations & failure analysis
- **Small test set (114 rows, 42 malignant).** One extra mistake changes recall by ~2.4 points. Differences between LR, RF and GB on the test set (e.g. 0.905 vs 0.952 recall) are **not statistically reliable**; CV differences between models are small.
- **Failure cases:** 3 of 114 test samples were misclassified (`misclassified_test_samples.csv`): 1 false positive (P=0.54, borderline) and 2 false negatives (P=0.06 and 0.21). The false negatives look "benign-like" on mean radius and concavity, so the model is confidently wrong on atypical tumors.
- **Cost asymmetry:** a missed malignancy (false negative) is far worse than a false alarm. `threshold_tradeoff.csv` shows lowering the threshold to 0.2 cuts false negatives from 2 to 1 at no extra false positives on this test set; a real system would choose the threshold with clinicians, ideally on a larger validation set.
- **Single source, single population.** All data come from one study/imaging pipeline; performance on other hospitals, scanners or demographics is unknown. No external validation was done.
- **Features are pre-extracted measurements** from fine-needle aspirate images, so the model depends on correct upstream segmentation. Many features are highly correlated, which makes individual importance scores unstable (see wide error bars in chart 04).
- **Not a medical device.** This is an educational project and must not be used for diagnosis.
- **Engineered categorical bins** add little here (a linear model on scaled numerics already performs near ceiling); they are included to demonstrate the encoding step correctly inside a leakage-safe pipeline.

## Reproduce
```bash
pip install -r requirements.txt
python train_evaluate.py     # trains, tunes, evaluates, writes charts/ models/ *.csv
python predict_demo.py       # loads saved pipeline and scores sample rows
```
Runs fully locally; no paid tools, APIs or hosting.

## Project structure
```
train_evaluate.py          main pipeline
predict_demo.py            inference example (samples shown may come from the training split; illustrative only)
models/best_model.joblib   saved full pipeline (preprocessing + model)
charts/                    evaluation figures
*.csv                      CV results, test results, threshold trade-off, misclassified rows
```
