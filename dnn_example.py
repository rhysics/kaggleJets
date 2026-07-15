import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from utils.jet_utils import load_processed_data
from utils.jet_plotting_utils import plot_training_history, plot_confusion_matrix, plot_roc_curve

X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids= load_processed_data()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)



def build_dnn_model(input_dim):
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=(input_dim,)),
        # the number in these layers refers to the number of neurons in that layer, you can change this to whatever you want!
        # but it is conventional to stick to powers of 2 because of resource optimisation
        keras.layers.Dense(64, activation='relu'),
        # keras.layers.Dropout(0.2),
        
        # Hidden layers
        # you can always change the number of layers 
        keras.layers.Dense(32, activation='relu'),
        # keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
                # Output layer
        # the final layer must have one as the output neuron since we have a binary classification 
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Create and compile model
model = build_dnn_model(X_train.shape[1])
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100, # how many times do we go through the full dataset
    batch_size=32, # how many data samples do we consider before updating our model 
    validation_split=0.2, # this uses another validation split, you can set this to 0 and train on the full data set and test on our validation set if you'd like 
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# Plot training history
plot_training_history(history)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = (model.predict(X_val) > 0.5).astype(int)

# Plot confusion matrix
plot_confusion_matrix(y_val, y_pred)

plot_roc_curve(y_val, model.predict(X_val))