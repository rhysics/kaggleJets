# ROCKET (random convolutional kernels) + XGBoost, ported from
# tt-jet-tagging-codas-hep-2026/rocket_xgb_tagger.py, trained on the
# accept-reject class-balanced, augmented jet images.
#
# Pipeline: training images -> accept-reject on total pT (class-balance the
# spectrum) -> augment (pT/position smear) -> canonicalize -> 2D-ROCKET
# featurize -> XGBoost -> evaluate on validation.
#
# The ROCKET transform is frozen/random; its only *fitted* piece is the
# per-kernel ppv bias, calibrated on a subsample of the (augmented) training
# images. XGBoost does all the actual learning on top of the fixed features.

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from utils.jet_utils import load_images, model_metrics, save_benchmark_info
from utils.jet_plotting_utils import plot_confusion_matrix, plot_roc_curve
from utils.new_data import (
    build_image_array,
    total_pt_per_image,
    accept_reject_mask,
    plot_accept_reject_pt,
)
from utils.rocket import RocketTransform2D

model_name = "rocket_xgb_accept_reject_augmented"

seed = 42
n_copies, pt_smear, pos_smear = 5, 0.1, 0.1
n_kernels, n_ppv_biases = 2000, 3
n_estimators, max_depth, learning_rate = 400, 5, 0.05

# Compare the total-pT spectrum before/after accept-reject
plot_accept_reject_pt(output_path="output/accept_reject_pt.png")

(
    X_train_images, y_train, _train_ids,
    X_val_images, y_val, _val_ids,
    X_test_images, test_ids,
) = load_images()

# Accept-reject on total pT so QCD/TT share the same spectrum before augmenting
rng = np.random.default_rng(seed)
total_pt = total_pt_per_image(X_train_images)
keep = accept_reject_mask(total_pt, y_train, bins=50, rng=rng)
X_train_images, y_train = X_train_images[keep], y_train[keep]

# Train: augmented (clean + smeared copies). Val/test: clean only.
# normalize=False keeps the (max-normalized) energy scale, which ROCKET's
# max-pooling feature exploits -- unlike the DNN/BDT scripts, which L1-normalize.
X_train_img, tr_idx = build_image_array(
    X_train_images, augment=True, n_copies=n_copies,
    pt_smear=pt_smear, pos_smear=pos_smear, seed=seed, normalize=False,
)
y_train = y_train[tr_idx].astype(int)

X_val_img, val_idx = build_image_array(
    X_val_images, augment=False, n_copies=0, pt_smear=0.0, pos_smear=0.0,
    seed=seed, normalize=False,
)
y_val = y_val[val_idx].astype(int)

X_test_img, _ = build_image_array(
    X_test_images, augment=False, n_copies=0, pt_smear=0.0, pos_smear=0.0,
    seed=seed, normalize=False,
)

print(f"Train images: {X_train_img.shape} (class balance {y_train.mean():.2f} top)")
print(f"Val images:   {X_val_img.shape} (class balance {y_val.mean():.2f} top)")

# 2D-ROCKET featurize: calibrate ppv biases on a random training subsample
# (quantiles are stable from a few thousand jets), then featurize every image.
rocket = RocketTransform2D(n_kernels=n_kernels, n_ppv_biases=n_ppv_biases, seed=seed)
fit_n = min(1500, len(X_train_img))
fit_idx = rng.choice(len(X_train_img), size=fit_n, replace=False)
rocket.fit(X_train_img[fit_idx])

F_train = rocket.transform(X_train_img)
F_val = rocket.transform(X_val_img)
F_test = rocket.transform(X_test_img)

print(f"ROCKET features: {F_train.shape[1]} per jet")

# scale_pos_weight = n_neg / n_pos handles the QCD:top class imbalance.
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
params = dict(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=0.8,
    colsample_bytree=0.3,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    tree_method="hist",
    random_state=seed,
    n_jobs=-1,
)

# prefer GPU, fall back to CPU on OOM/no-GPU
try:
    model = xgb.XGBClassifier(device="cuda", **params)
    model.fit(F_train, y_train)
except Exception:
    model = xgb.XGBClassifier(device="cpu", **params)
    model.fit(F_train, y_train)

# Make predictions
y_pred_proba = model.predict_proba(F_val)[:, 1]
discrete_pred = np.where(y_pred_proba > 0.5, 1, 0)
print(f"Validation AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")

# Plot confusion matrix
plot_confusion_matrix(y_val, discrete_pred)
val_metrics = model_metrics(y_val, discrete_pred)
plot_roc_curve(y_val, y_pred_proba)

save_benchmark_info(model_name, val_metrics, "output")

test_predictions = model.predict_proba(F_test)[:, 1]
solution = pd.DataFrame({'id': test_ids, 'label': test_predictions})
solution.to_csv('submission_rocket_augmented.csv', index=False)
