import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from utils.jet_utils import load_images, save_benchmark_info, model_metrics
from utils.jet_gnn_utils import create_graph_data
from utils.jet_plotting_utils import (plot_confusion_matrix,
                                     plot_training_history,
                                     plot_roc_curve)

model_name = "default_gnn"
# Load data
X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids = load_images()

# Convert to graph format - this might take a bit of time to run 
X_train_graphs = create_graph_data(X_train, y_train, max_nodes=900, consider_all_nodes=True)
X_val_graphs = create_graph_data(X_val, y_val, max_nodes=900)
X_test_graphs = create_graph_data(X_test, max_nodes=900)


class GNN(nn.Module):
    def __init__(self, num_features):
        super(GNN, self).__init__()
        # Graph convolution layers
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        
        # Dense layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return torch.sigmoid(x)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNN(num_features=1).to(device)  # 4 features: pt, eta, phi, charge
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()



train_loader = DataLoader(X_train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(X_val_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(X_test_graphs, batch_size=32)



def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.to(device))
        loss = criterion(out, data.y.view(-1, 1))
        pred = (out > 0.5).float()
        correct += int((pred == data.y.view(-1, 1)).sum())
    return correct / len(loader.dataset), loss


history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}


# Training loop
best_acc = 0
for epoch in range(70):
    loss = train()
    train_acc = test(train_loader)[0]
    val_acc, val_loss = test(val_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')


    # Append to history
    history['loss'].append(loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(train_acc)
    history['val_accuracy'].append(val_acc)

    # Load best model
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate on test set
model.eval()
y_true = []
y_pred = []

with torch.no_grad(): 
    for data in val_loader:
        out = model(data.to(device))
        y_true.extend(data.to('cpu').y.numpy())
        y_pred.extend(out.to('cpu').numpy())

y_pred = np.array([x[0] for x in y_pred])
pred_discrete = np.where(y_pred > 0.5, 1, 0)
# Plot confusion matrix
plot_confusion_matrix(y_true, pred_discrete)
val_metrics = model_metrics(y_true, pred_discrete)
save_benchmark_info(model_name, val_metrics, "output")
plot_roc_curve(y_true, y_pred)
