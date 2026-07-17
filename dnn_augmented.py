import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from utils.jet_utils import load_images, save_benchmark_info, model_metrics
from utils.jet_plotting_utils import plot_training_history, plot_confusion_matrix, plot_roc_curve
from utils.new_data import build_image_array, generate_new_train_accept_reject, plot_accept_reject_pt

model_name = "dnn_64_32_32_16_accept_reject_augmented"

# Compare the total-pT spectrum before/after accept-reject
plot_accept_reject_pt(output_path="output/accept_reject_pt.png")

# Generate the accept-reject augmented training set (saves X_train_ar.npy / y_train_ar.npy)
generate_new_train_accept_reject(n_copies=50)
X_train = np.load("X_train_ar.npy").reshape(-1, 30 * 30)
y_train = np.load("y_train_ar.npy")

print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")

# Validation set: canonicalized images only (no augmentation), matching the training pipeline
_X_train_images, _y_train, _train_ids, X_val_images, y_val, val_ids, _X_test_images, _test_ids = load_images()
X_val, val_idx = build_image_array(
    X_val_images, augment=False, n_copies=0, pt_smear=0.0, pos_smear=0.0, seed=42
)
X_val = X_val.reshape(-1, 30 * 30)
y_val = y_val[val_idx]


def build_dnn_model(input_dim):
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=(input_dim,)),
        # the number in these layers refers to the number of neurons in that layer, you can change this to whatever you want!
        # but it is conventional to stick to powers of 2 because of resource optimisation
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),

        # Hidden layers
        # you can always change the number of layers
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        # Output layer
        # the final layer must have one as the output neuron since we have a binary classification
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    return model

# Create and compile model
model = build_dnn_model(X_train.shape[1])
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=500, # how many times do we go through the full dataset
    batch_size=2**9, # how many data samples do we consider before updating our model
    validation_data=(X_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
    ]
)

# Plot training history
plot_training_history(history)

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Make predictions
y_pred = (model.predict(X_val) > 0.5).astype(int)

# Plot confusion matrix
plot_confusion_matrix(y_val, y_pred)
val_metrics = model_metrics(y_val, y_pred)
plot_roc_curve(y_val, model.predict(X_val))

save_benchmark_info(model_name, val_metrics, "output")
