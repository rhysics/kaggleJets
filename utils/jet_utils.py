import os
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import roc_curve, auc

def preprocess_jet_images(jet_images, target_size=(32, 32)):
    """
    Preprocess jet images for CNN
    """
    processed_images = {}
    for key, images in jet_images.items():
        # Resize if needed
        if images.shape[1:] != target_size:
            # Add resizing logic here if needed
            pass
        # Normalize
        processed_images[key] = images / np.max(images)
    return processed_images

def load_processed_data(path_prefix='/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data'):
    """
    Load processed data with unique IDs.
    
    Returns:
        tuple: (X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test, test_ids)
    """

    # Load training data
    X_train = pd.read_csv(f'{path_prefix}/train/features/cluster_features.csv')
    y_train = np.load(f'{path_prefix}/train/labels/labels.npy')
    train_ids = np.load(f'{path_prefix}/train/ids/ids.npy')

    # Load validation data
    X_val = pd.read_csv(f'{path_prefix}/val/features/cluster_features.csv')
    y_val = np.load(f'{path_prefix}/val/labels/labels.npy')
    val_ids = np.load(f'{path_prefix}/val/ids/ids.npy')

    # Load test data
    X_test = pd.read_csv(f'{path_prefix}/test/features/cluster_features.csv')
    test_ids = np.load(f'{path_prefix}/test/ids/ids.npy')

    return X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids

def load_images():
    """
    Load jet images and labels with unique IDs.
    
    Returns:
        tuple: (X_train_images, y_train, train_ids, X_val_images, y_val, val_ids, X_test_images, y_test, test_ids)
    """
    # Load training data
    y_train = np.load('/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data/train/labels/labels.npy')
    train_ids = np.load('/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data/train/ids/ids.npy')
    with h5py.File('/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data/train/images/jet_images.h5', 'r') as f:
            X_train_images = np.expand_dims(f['images'][:], axis=-1)
    
    # Load validation data
    y_val = np.load('/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data/val/labels/labels.npy')
    val_ids = np.load('/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data/val/ids/ids.npy')
    with h5py.File('/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data/val/images/jet_images.h5', 'r') as f:
            X_val_images = np.expand_dims(f['images'][:], axis=-1)
    
    # Load test data
    test_ids = np.load('/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data/test/ids/ids.npy')
    with h5py.File('/Users/rhystaus/Documents/Profesional/codas/kaggleJets/data/test/images/jet_images.h5', 'r') as f:
            X_test_images = np.expand_dims(f['images'][:], axis=-1)
    
    
    return X_train_images, y_train, train_ids, X_val_images, y_val, val_ids, X_test_images, test_ids

def anti_kt_clustering(image, R=0.4, pt_min=0.1):
    """
    Perform anti-kt clustering on a jet image.
    
    Args:
        image (numpy.ndarray): 2D or 3D array representing the jet image (if 3D, first channel is used)
        R (float): Jet radius parameter
        pt_min (float): Minimum pT threshold for particles
        
    Returns:
        list: List of clusters with their properties
    """
    # Handle 3D images (with channel dimension)
    if len(image.shape) == 3:
        image = image[..., 0]  # Take first channel
    
    # Get non-zero pixels (particles)
    y, x = np.where(image > pt_min)
    pts = image[y, x]
    
    if len(pts) == 0:
        return []
    
    # Convert pixel coordinates to eta-phi space
    # Assuming the image is centered at (15, 15) with 0.1 units per pixel
    eta = (y - 15) * 0.1
    phi = (x - 15) * 0.1
    
    # Create particle list with coordinates and pT
    particles = np.column_stack((eta, phi, pts))
    
    # Calculate distance matrix in eta-phi space
    coords = particles[:, :2]
    dist_matrix = squareform(pdist(coords))
    
    # Anti-kt distance measure
    pt_matrix = np.outer(1/pts, 1/pts)
    anti_kt_dist = dist_matrix**2 * pt_matrix
    
    # Clustering
    n_particles = len(particles)
    clusters = []
    used = np.zeros(n_particles, dtype=bool)
    
    while not all(used):
        # Find minimum distance
        valid_dist = anti_kt_dist.copy()
        valid_dist[used] = np.inf
        valid_dist[:, used] = np.inf
        min_dist = np.min(valid_dist)
        
        if min_dist > R**2:
            # Start new cluster
            idx = np.where(~used)[0][0]
            clusters.append([particles[idx]])
            used[idx] = True
        else:
            # Merge clusters
            i, j = np.where(valid_dist == min_dist)
            i, j = i[0], j[0]
            
            # Find clusters containing i and j
            cluster_i = next((c for c in clusters if any(p[2] == particles[i][2] for p in c)), None)
            cluster_j = next((c for c in clusters if any(p[2] == particles[j][2] for p in c)), None)
            
            if cluster_i is None and cluster_j is None:
                # Create new cluster
                clusters.append([particles[i], particles[j]])
            elif cluster_i is None:
                cluster_j.append(particles[i])
            elif cluster_j is None:
                cluster_i.append(particles[j])
            else:
                # Merge clusters
                cluster_i.extend(cluster_j)
                clusters.remove(cluster_j)
            
            used[i] = True
            used[j] = True
    
    return clusters

def extract_cluster_features(clusters):
    """
    Extract features from clusters.
    
    Args:
        clusters (list): List of clusters from anti-kt clustering
        
    Returns:
        dict: Dictionary of cluster features
    """
    features = {
        'n_clusters': len(clusters),
        'max_cluster_pt': 0.0,
        'mean_cluster_pt': 0.0,
        'std_cluster_pt': 0.0,
        'max_cluster_size': 0,
        'mean_cluster_size': 0.0,
        'std_cluster_size': 0.0,
        'total_pt': 0.0,
        'max_cluster_eta': 0.0,
        'max_cluster_phi': 0.0,
        'mean_cluster_eta': 0.0,
        'mean_cluster_phi': 0.0,
        'cluster_pt_ratio': 0.0,  # Ratio of highest to second highest cluster pT
        'cluster_size_ratio': 0.0  # Ratio of largest to second largest cluster size
    }
    
    if not clusters:
        return features
    
    cluster_pts = []
    cluster_sizes = []
    cluster_etas = []
    cluster_phis = []
    
    for cluster in clusters:
        cluster = np.array(cluster)
        pt = np.sum(cluster[:, 2])
        size = len(cluster)
        eta = np.mean(cluster[:, 0])  # eta is first column
        phi = np.mean(cluster[:, 1])  # phi is second column
        
        cluster_pts.append(pt)
        cluster_sizes.append(size)
        cluster_etas.append(eta)
        cluster_phis.append(phi)
    
    # Sort cluster properties
    cluster_pts.sort(reverse=True)
    cluster_sizes.sort(reverse=True)
    
    # Calculate additional features
    pt_ratio = cluster_pts[0] / cluster_pts[1] if len(cluster_pts) > 1 else 1.0
    size_ratio = cluster_sizes[0] / cluster_sizes[1] if len(cluster_sizes) > 1 else 1.0
    
    features.update({
        'max_cluster_pt': np.max(cluster_pts),
        'mean_cluster_pt': np.mean(cluster_pts),
        'std_cluster_pt': np.std(cluster_pts),
        'max_cluster_size': np.max(cluster_sizes),
        'mean_cluster_size': np.mean(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'total_pt': np.sum(cluster_pts),
        'max_cluster_eta': np.max(np.abs(cluster_etas)),
        'max_cluster_phi': np.max(np.abs(cluster_phis)),
        'mean_cluster_eta': np.mean(np.abs(cluster_etas)),
        'mean_cluster_phi': np.mean(np.abs(cluster_phis)),
        'cluster_pt_ratio': pt_ratio,
        'cluster_size_ratio': size_ratio
    })
    
    return features


def get_auc_score(y_true, y_pred_proba): 
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    return roc_auc

