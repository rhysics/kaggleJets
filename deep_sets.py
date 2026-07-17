import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.jet_dataloader import get_dataloaders
from utils.jet_plotting_utils import plot_confusion_matrix, plot_training_history, plot_roc_curve
from utils.jet_utils import load_images, save_benchmark_info, model_metrics

class DeepSets(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(DeepSets, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x shape: [batch_size, max_clusters, input_dim]
        
        # Apply phi network to each cluster
        phi_output = self.phi(x)  # [batch_size, max_clusters, hidden_dim]
        
        # Mean pooling over clusters
        pooled = phi_output.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Apply rho network
        output = self.rho(pooled)  # [batch_size, 1]
        
        return output.squeeze(-1)  # [batch_size]

# Set data path and parameters
batch_size = 32
R = 0.4  # Jet radius parameter
pt_min = 0.1  # Minimum pT threshold

# Get dataloaders
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=batch_size,
    R=R,
    pt_min=pt_min
)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# Initialize model and training components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepSets().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for clusters, labels in train_loader:
        clusters, labels = clusters.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(clusters)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Calculate accuracy
        preds = (outputs > 0).float()
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for clusters, labels in val_loader:
            clusters, labels = clusters.to(device), labels.to(device)
            outputs = model(clusters)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            preds = (outputs > 0).float()
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()
    
    # Calculate epoch metrics
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_acc = 100. * correct / total
    val_acc = 100. * val_correct / val_total
    
    # Append to history
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(train_acc)
    history['val_accuracy'].append(val_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

plot_training_history(history, metrics=['loss', 'accuracy'])



# Load best model and evaluate on test set
# model.load_state_dict(torch.load('best_model.pt'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for clusters, labels in val_loader:  # Changed 'masks' to 'mask' to match dataset output
        clusters, labels = clusters.to(device), labels.to(device)  # Move mask to device too
        outputs = torch.sigmoid(model(clusters))  # Pass both clusters and mask to model
        
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds_discrete = np.where(np.array(all_preds) > 0.5, 1, 0)
# Calculate accuracy
accuracy = np.mean(np.array(all_preds_discrete) == np.array(all_labels))
print(f'Test Accuracy: {accuracy:.4f}')

# Plot confusion matrix
plot_confusion_matrix(all_labels, all_preds_discrete)
val_metrics = model_metrics(all_labels, all_preds_discrete)
save_benchmark_info("deep_sets", val_metrics, "output")
plot_roc_curve(all_labels, all_preds)