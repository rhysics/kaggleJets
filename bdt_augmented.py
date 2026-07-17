# https://www.kaggle.com/code/livhelen/01-benchmark-boosted-decision-tree

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.jet_utils import load_images, model_metrics, save_benchmark_info
from utils.jet_plotting_utils import plot_confusion_matrix, plot_roc_curve
from utils.new_data import build_image_array, generate_new_train_accept_reject, plot_accept_reject_pt

model_name = "bdt_200_estimators_max_depth_10_learning_rate_0.01_accept_reject_augmented"

# Compare the total-pT spectrum before/after accept-reject
plot_accept_reject_pt(output_path="output/accept_reject_pt.png")

# Generate the accept-reject augmented training set (saves X_train_ar.npy / y_train_ar.npy)
generate_new_train_accept_reject(n_copies=50)
X_train = np.load("X_train_ar.npy").reshape(-1, 30 * 30)
y_train = np.load("y_train_ar.npy")

print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")

# Validation and test sets: canonicalized images only (no augmentation), matching the training pipeline
_X_train_images, _y_train, _train_ids, X_val_images, y_val, val_ids, X_test_images, test_ids = load_images()

X_val, val_idx = build_image_array(
    X_val_images, augment=False, n_copies=0, pt_smear=0.0, pos_smear=0.0, seed=42
)
X_val = X_val.reshape(-1, 30 * 30)
y_val = y_val[val_idx]

X_test, _test_idx = build_image_array(
    X_test_images, augment=False, n_copies=0, pt_smear=0.0, pos_smear=0.0, seed=42
)
X_test = X_test.reshape(-1, 30 * 30)

# Initialize and train model
model = xgb.XGBClassifier(
    n_estimators=200,  # Number of boosting rounds
    max_depth=10,      # Maximum tree depth
    learning_rate=1e-2, # Step size shrinkage
    objective='binary:logistic',  # Binary classification
    random_state=42
)

# Train the model
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=True)

# Make predictions

# this gives us probabilities for both categories - we only want for ttbar, so we select one column
# with a binary classification, the probability for one category implies the other
y_pred = model.predict_proba(X_val)[:, 1]

# to test accuracy and confusion matrix, we need labels 0 and 1, so we set that based on a threshold
discrete_pred = np.where(y_pred > 0.5, 1, 0)
# Calculate accuracy
accuracy = accuracy_score(y_val, discrete_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Plot confusion matrix
plot_confusion_matrix(y_val, discrete_pred)
val_metrics = model_metrics(y_val, discrete_pred)
plot_roc_curve(y_val, y_pred)

# Feature importance is over raw pixels here (not named cluster features), so
# just show the top 20 -- all 900 would be unreadable.
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': [f"pixel_{i}" for i in range(len(importance))],
    'importance': importance
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance (top 20 pixels)')
plt.tight_layout()
plt.show()

save_benchmark_info(model_name, val_metrics, "output")

test_predictions = model.predict_proba(X_test)[:, 1]
solution = pd.DataFrame({'id': test_ids, 'label': test_predictions})
solution.to_csv('submission_augmented.csv', index=False)
