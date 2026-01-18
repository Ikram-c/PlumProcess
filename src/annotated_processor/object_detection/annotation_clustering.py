"""
Autoencoder-Optimized Annotation Clustering Module

This module provides lazy unsupervised clustering algorithms enhanced with autoencoder-based
feature learning to group annotation classes by the features that separate them the most.
Includes deep feature extraction, multiple autoencoder architectures, and advanced clustering.

Key Features:
- Adaptive Random Forest parameters that scale with the number of discovered clusters
- Multiple autoencoder architectures for different data patterns
- Automatic feature importance analysis with ensemble methods
- Comprehensive visualization and reporting

Random Forest Adaptive Parameters:
- n_estimators: max(50, min(200, n_classes * 15)) - scales with cluster count
- max_depth: max(3, min(15, log₂(n_classes) + 3)) - prevents overfitting
- min_samples_split/leaf: scale with samples per class ratio
- max_features: √(total_features) for optimal feature selection
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')

# PyTorch imports for autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Clustering and ML imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances


@dataclass
class ClusteringResult:
    """Container for clustering results and analysis"""
    algorithm: str
    cluster_labels: np.ndarray
    feature_names: List[str]
    feature_importance: Dict[str, float]
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    n_clusters: int
    cluster_centers: Optional[np.ndarray] = None
    feature_data: Optional[pd.DataFrame] = None
    encoded_features: Optional[np.ndarray] = None
    reconstruction_error: Optional[float] = None


class AutoencoderNet(nn.Module):
    """Deep Autoencoder for feature learning"""

    def __init__(self, input_dim: int, encoding_dim: int = None, architecture: str = 'deep'):
        super(AutoencoderNet, self).__init__()

        if encoding_dim is None:
            encoding_dim = max(2, input_dim // 4)  # Default to quarter of input size

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.architecture = architecture

        if architecture == 'shallow':
            # Simple single-layer autoencoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, input_dim),
                nn.Sigmoid()
            )

        elif architecture == 'deep':
            # Multi-layer autoencoder with bottleneck
            hidden_dim1 = max(encoding_dim * 2, input_dim // 2)
            hidden_dim2 = max(encoding_dim * 1.5, input_dim // 3)

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim2, encoding_dim),
                nn.ReLU()
            )

            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim2, hidden_dim1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim1, input_dim),
                nn.Sigmoid()
            )

        elif architecture == 'sparse':
            # Sparse autoencoder with L1 regularization
            hidden_dim = max(encoding_dim * 2, input_dim // 2)

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, encoding_dim),
                nn.ReLU()
            )

            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        elif architecture == 'variational':
            # Variational autoencoder
            hidden_dim = max(encoding_dim * 2, input_dim // 2)

            self.encoder_common = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

            self.mu_layer = nn.Linear(hidden_dim, encoding_dim)
            self.logvar_layer = nn.Linear(hidden_dim, encoding_dim)

            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

    def encode(self, x):
        if self.architecture == 'variational':
            h = self.encoder_common(x)
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            return mu, logvar
        else:
            return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.architecture == 'variational':
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar
        else:
            encoded = self.encode(x)
            decoded = self.decode(encoded)
            return decoded, encoded


class AnnotationFeatureExtractor:
    """Extract features from COCO-style annotations"""

    def __init__(self):
        self.feature_names = []

    def extract_bbox_features(self, annotations: List[Dict]) -> pd.DataFrame:
        """Extract bounding box related features"""
        features = []

        for ann in tqdm(annotations, desc="Extracting bbox features"):
            if 'bbox' not in ann:
                continue

            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox

            feature_dict = {
                'bbox_width': w,
                'bbox_height': h,
                'bbox_area': w * h,
                'bbox_aspect_ratio': w / h if h > 0 else 0,
                'bbox_x_center': x + w / 2,
                'bbox_y_center': y + h / 2,
                'bbox_perimeter': 2 * (w + h),
                'bbox_diagonal': np.sqrt(w ** 2 + h ** 2),
                'bbox_compactness': (4 * np.pi * w * h) / ((2 * (w + h)) ** 2) if w + h > 0 else 0,
                'bbox_elongation': max(w, h) / min(w, h) if min(w, h) > 0 else 1,
                'bbox_rectangularity': (w * h) / ((w + h) / 2) ** 2 if w + h > 0 else 0,
                'category_id': ann.get('category_id', 0),
                'annotation_id': ann.get('id', 0),
                'image_id': ann.get('image_id', 0)
            }

            # Add segmentation features if available
            if 'segmentation' in ann and ann['segmentation']:
                try:
                    if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                        seg = ann['segmentation'][0]
                        if len(seg) >= 6:  # At least 3 points
                            seg_array = np.array(seg).reshape(-1, 2)
                            feature_dict['segmentation_points'] = len(seg_array)
                            feature_dict['segmentation_area'] = self._polygon_area(seg_array)
                            feature_dict['bbox_to_seg_ratio'] = feature_dict['bbox_area'] / feature_dict[
                                'segmentation_area'] if feature_dict['segmentation_area'] > 0 else 1
                            feature_dict['segmentation_perimeter'] = self._polygon_perimeter(seg_array)
                            feature_dict['seg_circularity'] = (4 * np.pi * feature_dict['segmentation_area']) / (
                                        feature_dict['segmentation_perimeter'] ** 2) if feature_dict[
                                                                                            'segmentation_perimeter'] > 0 else 0
                        else:
                            feature_dict.update({
                                'segmentation_points': 0, 'segmentation_area': 0, 'bbox_to_seg_ratio': 1,
                                'segmentation_perimeter': 0, 'seg_circularity': 0
                            })
                    else:
                        feature_dict.update({
                            'segmentation_points': 0, 'segmentation_area': 0, 'bbox_to_seg_ratio': 1,
                            'segmentation_perimeter': 0, 'seg_circularity': 0
                        })
                except:
                    feature_dict.update({
                        'segmentation_points': 0, 'segmentation_area': 0, 'bbox_to_seg_ratio': 1,
                        'segmentation_perimeter': 0, 'seg_circularity': 0
                    })
            else:
                feature_dict.update({
                    'segmentation_points': 0, 'segmentation_area': 0, 'bbox_to_seg_ratio': 1,
                    'segmentation_perimeter': 0, 'seg_circularity': 0
                })

            features.append(feature_dict)

        df = pd.DataFrame(features)
        self.feature_names = [col for col in df.columns if col not in ['category_id', 'annotation_id', 'image_id']]
        return df

    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(vertices) < 3:
            return 0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))

    def _polygon_perimeter(self, vertices: np.ndarray) -> float:
        """Calculate polygon perimeter"""
        if len(vertices) < 2:
            return 0
        perimeter = 0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            perimeter += np.sqrt((vertices[j][0] - vertices[i][0]) ** 2 + (vertices[j][1] - vertices[i][1]) ** 2)
        return perimeter

    def extract_spatial_features(self, annotations: List[Dict], image_info: Dict[int, Dict]) -> pd.DataFrame:
        """Extract spatial distribution features"""
        features = []

        for ann in tqdm(annotations, desc="Extracting spatial features"):
            if 'bbox' not in ann or ann['image_id'] not in image_info:
                continue

            bbox = ann['bbox']
            img_info = image_info[ann['image_id']]
            img_width = img_info.get('width', 1)
            img_height = img_info.get('height', 1)

            x, y, w, h = bbox

            feature_dict = {
                'relative_x': x / img_width,
                'relative_y': y / img_height,
                'relative_width': w / img_width,
                'relative_height': h / img_height,
                'relative_area': (w * h) / (img_width * img_height),
                'distance_from_center': np.sqrt(
                    ((x + w / 2) - img_width / 2) ** 2 + ((y + h / 2) - img_height / 2) ** 2) / np.sqrt(
                    img_width ** 2 + img_height ** 2),
                'edge_proximity': min(x, y, img_width - (x + w), img_height - (y + h)) / min(img_width, img_height),
                'corner_distance_tl': np.sqrt((x) ** 2 + (y) ** 2) / np.sqrt(img_width ** 2 + img_height ** 2),
                'corner_distance_br': np.sqrt((img_width - (x + w)) ** 2 + (img_height - (y + h)) ** 2) / np.sqrt(
                    img_width ** 2 + img_height ** 2),
                'horizontal_position': (x + w / 2) / img_width,  # 0=left, 1=right
                'vertical_position': (y + h / 2) / img_height,  # 0=top, 1=bottom
                'category_id': ann.get('category_id', 0),
                'annotation_id': ann.get('id', 0),
                'image_id': ann.get('image_id', 0)
            }

            features.append(feature_dict)

        df = pd.DataFrame(features)
        spatial_features = [col for col in df.columns if col not in ['category_id', 'annotation_id', 'image_id']]
        self.feature_names.extend(spatial_features)
        return df


class AutoencoderTrainer:
    """Trainer class for autoencoder models"""

    def __init__(self, input_dim: int, encoding_dim: int = None, architecture: str = 'deep'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoencoderNet(input_dim, encoding_dim, architecture).to(self.device)
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, X: np.ndarray, epochs: int = 100, batch_size: int = 64,
              learning_rate: float = 0.001, validation_split: float = 0.2) -> Dict[str, List[float]]:
        """Train the autoencoder"""

        # Normalize data
        X_scaled = self.scaler.fit_transform(X)

        # Split into train/validation
        n_val = int(len(X_scaled) * validation_split)
        indices = np.random.permutation(len(X_scaled))
        val_indices, train_indices = indices[:n_val], indices[n_val:]

        X_train, X_val = X_scaled[train_indices], X_scaled[val_indices]

        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training history
        history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': []}

        # Training loop
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training autoencoder"):
            train_loss = 0.0
            train_recon_loss = 0.0

            for batch_data, batch_target in train_loader:
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()

                if self.model.architecture == 'variational':
                    recon_batch, mu, logvar = self.model(batch_data)
                    recon_loss = F.mse_loss(recon_batch, batch_target, reduction='sum')
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.1 * kld_loss  # Beta-VAE with beta=0.1
                    train_recon_loss += recon_loss.item()
                elif self.model.architecture == 'sparse':
                    recon_batch, encoded = self.model(batch_data)
                    recon_loss = F.mse_loss(recon_batch, batch_target, reduction='sum')
                    sparsity_loss = torch.mean(torch.abs(encoded))  # L1 regularization
                    loss = recon_loss + 0.01 * sparsity_loss
                    train_recon_loss += recon_loss.item()
                else:
                    recon_batch, encoded = self.model(batch_data)
                    loss = F.mse_loss(recon_batch, batch_target, reduction='sum')
                    train_recon_loss += loss.item()

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0

            with torch.no_grad():
                for batch_data, batch_target in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_target = batch_target.to(self.device)

                    if self.model.architecture == 'variational':
                        recon_batch, mu, logvar = self.model(batch_data)
                        recon_loss = F.mse_loss(recon_batch, batch_target, reduction='sum')
                        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + 0.1 * kld_loss
                        val_recon_loss += recon_loss.item()
                    elif self.model.architecture == 'sparse':
                        recon_batch, encoded = self.model(batch_data)
                        recon_loss = F.mse_loss(recon_batch, batch_target, reduction='sum')
                        sparsity_loss = torch.mean(torch.abs(encoded))
                        loss = recon_loss + 0.01 * sparsity_loss
                        val_recon_loss += recon_loss.item()
                    else:
                        recon_batch, encoded = self.model(batch_data)
                        loss = F.mse_loss(recon_batch, batch_target, reduction='sum')
                        val_recon_loss += loss.item()

                    val_loss += loss.item()

            # Record history
            history['train_loss'].append(train_loss / len(train_loader.dataset))
            history['val_loss'].append(val_loss / len(val_loader.dataset))
            history['train_recon'].append(train_recon_loss / len(train_loader.dataset))
            history['val_recon'].append(val_recon_loss / len(val_loader.dataset))

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if epoch > 20 and len(history['val_loss']) > 10:
                recent_val_loss = history['val_loss'][-10:]
                if all(recent_val_loss[i] >= recent_val_loss[i + 1] for i in range(len(recent_val_loss) - 1)):
                    print(f"Early stopping at epoch {epoch}")
                    break

            self.model.train()

        self.trained = True
        return history

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input data to latent space"""
        if not self.trained:
            raise ValueError("Model must be trained before encoding")

        X_scaled = self.scaler.transform(X)
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            if self.model.architecture == 'variational':
                mu, logvar = self.model.encode(X_tensor)
                encoded = mu  # Use mean for deterministic encoding
            else:
                encoded = self.model.encode(X_tensor)

            return encoded.cpu().numpy()

    def reconstruct(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Reconstruct input data and return reconstruction error"""
        if not self.trained:
            raise ValueError("Model must be trained before reconstruction")

        X_scaled = self.scaler.transform(X)
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            if self.model.architecture == 'variational':
                recon, mu, logvar = self.model(X_tensor)
            else:
                recon, encoded = self.model(X_tensor)

            recon_numpy = recon.cpu().numpy()

            # Calculate reconstruction error
            mse_error = np.mean((X_scaled - recon_numpy) ** 2)

            return self.scaler.inverse_transform(recon_numpy), mse_error


class EnhancedClusteringAnalyzer:
    """Enhanced clustering analyzer with autoencoder optimization"""

    def __init__(self, feature_data: pd.DataFrame, use_autoencoder: bool = True,
                 autoencoder_config: Dict[str, Any] = None):
        self.feature_data = feature_data.copy()
        self.use_autoencoder = use_autoencoder
        self.autoencoder_config = autoencoder_config or {}
        self.scaler = StandardScaler()
        self.results = {}
        self.best_result = None
        self.autoencoder = None

        # Prepare features for clustering
        self.feature_columns = [col for col in feature_data.columns
                                if col not in ['category_id', 'annotation_id', 'image_id']]
        self.X = self.feature_data[self.feature_columns].fillna(0)
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Train autoencoder if requested
        if self.use_autoencoder:
            self._train_autoencoder()

    def _train_autoencoder(self):
        """Train autoencoder for feature learning"""
        print("Training autoencoder for feature learning...")

        # Default autoencoder configuration
        config = {
            'encoding_dim': max(2, len(self.feature_columns) // 3),
            'architecture': 'deep',
            'epochs': 150,
            'batch_size': min(64, len(self.X_scaled) // 4),
            'learning_rate': 0.001
        }
        config.update(self.autoencoder_config)

        self.autoencoder = AutoencoderTrainer(
            input_dim=len(self.feature_columns),
            encoding_dim=config['encoding_dim'],
            architecture=config['architecture']
        )

        # Train autoencoder
        history = self.autoencoder.train(
            self.X_scaled,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )

        # Get encoded features
        self.X_encoded = self.autoencoder.encode(self.X_scaled)
        _, self.reconstruction_error = self.autoencoder.reconstruct(self.X_scaled)

        print(f"Autoencoder trained. Reconstruction error: {self.reconstruction_error:.6f}")
        print(f"Feature dimensionality reduced from {len(self.feature_columns)} to {self.X_encoded.shape[1]}")

    def find_optimal_clusters(self, max_clusters: int = 10, algorithms: List[str] = None,
                              compare_with_raw: bool = True) -> Dict[str, ClusteringResult]:
        """Find optimal number of clusters using autoencoder features"""

        if algorithms is None:
            algorithms = ['kmeans', 'gaussian_mixture', 'hierarchical']

        print("Finding optimal clustering solutions...")

        # Use encoded features if autoencoder was trained
        X_for_clustering = self.X_encoded if self.use_autoencoder else self.X_scaled

        for algorithm in algorithms:
            print(f"\nTesting {algorithm} clustering on {'encoded' if self.use_autoencoder else 'raw'} features...")
            best_score = -1
            best_n_clusters = 2

            # Test different numbers of clusters
            for n_clusters in tqdm(range(2, max_clusters + 1), desc=f"{algorithm} optimization"):
                try:
                    labels = self._cluster_with_algorithm(algorithm, n_clusters, X_for_clustering)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X_for_clustering, labels)
                        if score > best_score:
                            best_score = score
                            best_n_clusters = n_clusters
                except:
                    continue

            # Generate final clustering with best parameters
            final_labels = self._cluster_with_algorithm(algorithm, best_n_clusters, X_for_clustering)
            result = self._analyze_clustering_result(algorithm, final_labels, best_n_clusters, X_for_clustering)

            suffix = "_encoded" if self.use_autoencoder else "_raw"
            self.results[algorithm + suffix] = result

        # Also try DBSCAN
        if 'dbscan' in algorithms or algorithms is None:
            print(f"\nTesting DBSCAN clustering on {'encoded' if self.use_autoencoder else 'raw'} features...")
            eps_values = np.linspace(0.1, 2.0, 10)
            best_score = -1
            best_labels = None

            for eps in tqdm(eps_values, desc="DBSCAN optimization"):
                try:
                    labels = self._cluster_with_algorithm('dbscan', eps_param=eps, X=X_for_clustering)
                    if len(np.unique(labels)) > 1 and -1 not in labels:
                        score = silhouette_score(X_for_clustering, labels)
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                except:
                    continue

            if best_labels is not None:
                result = self._analyze_clustering_result('dbscan', best_labels,
                                                         len(np.unique(best_labels)), X_for_clustering)
                suffix = "_encoded" if self.use_autoencoder else "_raw"
                self.results['dbscan' + suffix] = result

        # Compare with raw features if using autoencoder
        if self.use_autoencoder and compare_with_raw:
            print("\nComparing with raw feature clustering...")
            for algorithm in ['kmeans']:  # Just test one algorithm for comparison
                best_score = -1
                best_n_clusters = 2

                for n_clusters in range(2, max_clusters + 1):
                    try:
                        labels = self._cluster_with_algorithm(algorithm, n_clusters, self.X_scaled)
                        if len(np.unique(labels)) > 1:
                            score = silhouette_score(self.X_scaled, labels)
                            if score > best_score:
                                best_score = score
                                best_n_clusters = n_clusters
                    except:
                        continue

                final_labels = self._cluster_with_algorithm(algorithm, best_n_clusters, self.X_scaled)
                result = self._analyze_clustering_result(algorithm, final_labels, best_n_clusters, self.X_scaled)
                self.results[algorithm + "_raw_comparison"] = result

        # Find best overall result
        self.best_result = max(self.results.values(), key=lambda x: x.silhouette_score)
        print(
            f"\nBest clustering: {self.best_result.algorithm} with silhouette score: {self.best_result.silhouette_score:.3f}")

        return self.results

    def _cluster_with_algorithm(self, algorithm: str, n_clusters: int = None,
                                X: np.ndarray = None, eps_param: float = 0.5) -> np.ndarray:
        """Apply clustering algorithm"""

        if X is None:
            X = self.X_encoded if self.use_autoencoder else self.X_scaled

        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'gaussian_mixture':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        elif algorithm == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == 'dbscan':
            clusterer = DBSCAN(eps=eps_param, min_samples=5)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return clusterer.fit_predict(X)

    def _analyze_clustering_result(self, algorithm: str, labels: np.ndarray,
                                   n_clusters: int, X: np.ndarray) -> ClusteringResult:
        """Analyze clustering results and find most important features"""

        # Calculate clustering metrics
        sil_score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
        ch_score = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0
        db_score = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else float('inf')

        # Find feature importance
        if self.use_autoencoder and X.shape[1] == self.X_encoded.shape[1]:
            # For encoded features, use autoencoder weights to map back to original features
            feature_importance = self._calculate_encoded_feature_importance(labels)
        else:
            # For raw features, use standard methods
            feature_importance = self._calculate_feature_importance(labels, X)

        # Calculate cluster centers
        cluster_centers = None
        if algorithm in ['kmeans', 'gaussian_mixture']:
            unique_labels = np.unique(labels)
            cluster_centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

        # Create feature dataframe with cluster labels
        feature_df = self.feature_data.copy()
        feature_df['cluster_label'] = labels

        return ClusteringResult(
            algorithm=algorithm,
            cluster_labels=labels,
            feature_names=self.feature_columns,
            feature_importance=feature_importance,
            silhouette_score=sil_score,
            calinski_harabasz_score=ch_score,
            davies_bouldin_score=db_score,
            n_clusters=n_clusters,
            cluster_centers=cluster_centers,
            feature_data=feature_df,
            encoded_features=self.X_encoded if self.use_autoencoder else None,
            reconstruction_error=self.reconstruction_error if self.use_autoencoder else None
        )

    def _calculate_encoded_feature_importance(self, labels: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for encoded features by mapping back to original space"""

        # Get encoder weights (first layer weights for mapping back)
        encoder_weights = None
        if hasattr(self.autoencoder.model, 'encoder'):
            first_layer = list(self.autoencoder.model.encoder.children())[0]
            if isinstance(first_layer, nn.Linear):
                encoder_weights = first_layer.weight.data.cpu().numpy()

        # Calculate importance in encoded space first
        encoded_importance = self._calculate_feature_importance(labels, self.X_encoded,
                                                                feature_names=[f'encoded_{i}' for i in
                                                                               range(self.X_encoded.shape[1])])

        # Map back to original feature space using encoder weights
        final_importance = {}
        if encoder_weights is not None:
            # Weight the original features by their contribution to encoded features
            for i, feature_name in enumerate(self.feature_columns):
                importance_sum = 0
                for j, encoded_importance_val in encoded_importance.items():
                    if encoded_importance_val > 0:
                        encoded_idx = int(j.split('_')[1]) if '_' in j else 0
                        if encoded_idx < encoder_weights.shape[0]:
                            importance_sum += abs(encoder_weights[encoded_idx, i]) * encoded_importance_val
                final_importance[feature_name] = importance_sum
        else:
            # Fallback: use variance in clusters for original features
            final_importance = self._calculate_feature_importance(labels, self.X_scaled)

        return final_importance

    def _calculate_feature_importance(self, labels: np.ndarray, X: np.ndarray,
                                      feature_names: List[str] = None) -> Dict[str, float]:
        """Calculate feature importance using multiple methods"""

        if feature_names is None:
            feature_names = self.feature_columns

        importance_scores = {}

        # Method 1: Mutual Information
        try:
            mi_scores = mutual_info_classif(X, labels, random_state=42)
            for i, feature in enumerate(feature_names):
                importance_scores[f"{feature}_mutual_info"] = mi_scores[i]
        except:
            pass

        # Method 2: Random Forest Feature Importance
        try:
            n_classes = len(np.unique(labels))
            n_samples = len(labels)
            n_features = len(feature_names)

            # Adaptive parameters based on number of classes and data size
            n_estimators = max(50, min(200, n_classes * 15))  # Scale with classes, cap at 200
            max_depth = max(3, min(15, int(np.log2(n_classes)) + 3))  # Depth based on classes
            min_samples_split = max(2, min(20, n_samples // (n_classes * 10)))  # Scale with samples per class
            min_samples_leaf = max(1, min(10, n_samples // (n_classes * 20)))  # Ensure leaf purity
            max_features = min(n_features, max(1, int(np.sqrt(n_features))))  # Square root rule

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            rf.fit(X, labels)
            for i, feature in enumerate(feature_names):
                importance_scores[f"{feature}_rf_importance"] = rf.feature_importances_[i]
        except:
            pass

        # Method 3: Cluster variance analysis
        try:
            cluster_means = {}
            for label in np.unique(labels):
                cluster_means[label] = X[labels == label].mean(axis=0)

            cluster_mean_matrix = np.array(list(cluster_means.values()))
            feature_variances = np.var(cluster_mean_matrix, axis=0)

            for i, feature in enumerate(feature_names):
                importance_scores[f"{feature}_cluster_variance"] = feature_variances[i]
        except:
            pass

        # Aggregate importance scores
        final_importance = {}
        for feature in feature_names:
            feature_scores = [v for k, v in importance_scores.items() if k.startswith(feature)]
            if feature_scores:
                final_importance[feature] = np.mean(feature_scores)
            else:
                final_importance[feature] = 0.0

        return final_importance

    def plot_clustering_results(self, result: ClusteringResult = None, save_path: str = None):
        """Plot comprehensive clustering results"""

        if result is None:
            result = self.best_result

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Enhanced Clustering Analysis: {result.algorithm.upper()}', fontsize=16)

        # Determine which features to use for visualization
        X_viz = result.encoded_features if result.encoded_features is not None else self.X_scaled

        # 1. t-SNE plot of clusters
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_viz) // 4))
        X_tsne = tsne.fit_transform(X_viz)

        scatter = axes[0, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=result.cluster_labels, cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('t-SNE Visualization of Clusters')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[0, 0])

        # 2. Feature importance
        importance_items = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)[:12]
        features, scores = zip(*importance_items) if importance_items else ([], [])

        axes[0, 1].barh(range(len(features)), scores)
        axes[0, 1].set_yticks(range(len(features)))
        axes[0, 1].set_yticklabels([f.replace('_', ' ') for f in features], fontsize=8)
        axes[0, 1].set_title('Top Features Separating Clusters')
        axes[0, 1].set_xlabel('Importance Score')

        # 3. Cluster size distribution
        unique_labels, counts = np.unique(result.cluster_labels, return_counts=True)
        axes[0, 2].bar(unique_labels, counts, color='skyblue', alpha=0.7)
        axes[0, 2].set_title('Cluster Size Distribution')
        axes[0, 2].set_xlabel('Cluster Label')
        axes[0, 2].set_ylabel('Number of Annotations')

        # 4. Autoencoder reconstruction (if available)
        if self.use_autoencoder and hasattr(self, 'autoencoder'):
            # Show reconstruction error by cluster
            cluster_errors = []
            cluster_labels_unique = sorted(np.unique(result.cluster_labels))

            for cluster_id in cluster_labels_unique:
                cluster_mask = result.cluster_labels == cluster_id
                cluster_data = self.X_scaled[cluster_mask]
                if len(cluster_data) > 0:
                    _, error = self.autoencoder.reconstruct(cluster_data)
                    cluster_errors.append(error)
                else:
                    cluster_errors.append(0)

            axes[1, 0].bar(cluster_labels_unique, cluster_errors, color='coral', alpha=0.7)
            axes[1, 0].set_title('Reconstruction Error by Cluster')
            axes[1, 0].set_xlabel('Cluster Label')
            axes[1, 0].set_ylabel('MSE Reconstruction Error')
        else:
            axes[1, 0].text(0.5, 0.5, 'No autoencoder used', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Autoencoder Analysis')

        # 5. Algorithm comparison
        if len(self.results) > 1:
            algorithms = list(self.results.keys())
            sil_scores = [self.results[alg].silhouette_score for alg in algorithms]

            colors = ['lightblue' if 'encoded' in alg else 'lightcoral' for alg in algorithms]
            bars = axes[1, 1].bar(range(len(algorithms)), sil_scores, color=colors, alpha=0.7)
            axes[1, 1].set_xticks(range(len(algorithms)))
            axes[1, 1].set_xticklabels([alg.replace('_', '\n') for alg in algorithms], fontsize=8, rotation=45)
            axes[1, 1].set_title('Silhouette Score Comparison')
            axes[1, 1].set_ylabel('Silhouette Score')

            # Add legend for encoded vs raw
            if any('encoded' in alg for alg in algorithms):
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='lightblue', label='Autoencoder Features'),
                                   Patch(facecolor='lightcoral', label='Raw Features')]
                axes[1, 1].legend(handles=legend_elements, loc='upper right')
        else:
            axes[1, 1].text(0.5, 0.5, 'Only one algorithm tested', ha='center', va='center',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Algorithm Comparison')

        # 6. Feature correlation with clusters (heatmap)
        if len(result.feature_importance) > 0:
            # Get top features for heatmap
            top_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
            top_feature_names = [f[0] for f in top_features]

            # Calculate mean feature values per cluster
            cluster_means = []
            for cluster_id in sorted(np.unique(result.cluster_labels)):
                cluster_mask = result.cluster_labels == cluster_id
                cluster_data = self.feature_data[cluster_mask]
                means = [cluster_data[feat].mean() if feat in cluster_data.columns else 0 for feat in top_feature_names]
                cluster_means.append(means)

            im = axes[1, 2].imshow(np.array(cluster_means).T, cmap='RdYlBu', aspect='auto')
            axes[1, 2].set_xticks(range(len(np.unique(result.cluster_labels))))
            axes[1, 2].set_xticklabels([f'C{i}' for i in sorted(np.unique(result.cluster_labels))])
            axes[1, 2].set_yticks(range(len(top_feature_names)))
            axes[1, 2].set_yticklabels([f.replace('_', ' ') for f in top_feature_names], fontsize=8)
            axes[1, 2].set_title('Feature Values by Cluster')
            plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced clustering visualization saved to: {save_path}")

        plt.show()

    def generate_report(self, save_path: str = None) -> str:
        """Generate comprehensive clustering report"""

        report = []
        report.append("=" * 90)
        report.append("AUTOENCODER-ENHANCED ANNOTATION CLUSTERING ANALYSIS REPORT")
        report.append("=" * 90)
        report.append(f"Total annotations analyzed: {len(self.feature_data)}")
        report.append(f"Number of original features: {len(self.feature_columns)}")
        report.append(f"Autoencoder used: {'Yes' if self.use_autoencoder else 'No'}")

        if self.use_autoencoder:
            report.append(f"Encoded feature dimensions: {self.X_encoded.shape[1]}")
            report.append(f"Reconstruction error: {self.reconstruction_error:.6f}")
            report.append(
                f"Dimensionality reduction: {len(self.feature_columns)} → {self.X_encoded.shape[1]} ({(1 - self.X_encoded.shape[1] / len(self.feature_columns)) * 100:.1f}% reduction)")

        report.append(f"Algorithms tested: {', '.join(set([k.split('_')[0] for k in self.results.keys()]))}")
        report.append("")

        # Best clustering result
        if self.best_result:
            report.append("BEST CLUSTERING RESULT")
            report.append("-" * 50)
            report.append(f"Algorithm: {self.best_result.algorithm.upper()}")
            report.append(f"Number of clusters: {self.best_result.n_clusters}")
            report.append(f"Silhouette Score: {self.best_result.silhouette_score:.4f}")
            report.append(f"Calinski-Harabasz Score: {self.best_result.calinski_harabasz_score:.4f}")
            report.append(f"Davies-Bouldin Score: {self.best_result.davies_bouldin_score:.4f}")
            report.append("")

            # Most important features
            report.append("TOP 10 FEATURES THAT SEPARATE CLUSTERS THE MOST")
            report.append("-" * 50)
            importance_items = sorted(self.best_result.feature_importance.items(), key=lambda x: x[1], reverse=True)[
                               :10]
            for i, (feature, score) in enumerate(importance_items, 1):
                report.append(f"{i:2d}. {feature:<30} {score:.6f}")

            # Add Random Forest parameter info
            n_classes = len(np.unique(self.best_result.cluster_labels))
            n_samples = len(self.feature_data)
            n_features = len(self.feature_columns)

            rf_n_estimators = max(50, min(200, n_classes * 15))
            rf_max_depth = max(3, min(15, int(np.log2(n_classes)) + 3))

            report.append("")
            report.append("RANDOM FOREST PARAMETERS USED FOR FEATURE IMPORTANCE")
            report.append("-" * 50)
            report.append(f"Number of classes detected: {n_classes}")
            report.append(f"Adaptive n_estimators: {rf_n_estimators} (15 × {n_classes} classes)")
            report.append(f"Adaptive max_depth: {rf_max_depth} (log₂({n_classes}) + 3)")
            report.append(f"min_samples_split: {max(2, min(20, n_samples // (n_classes * 10)))}")
            report.append(f"min_samples_leaf: {max(1, min(10, n_samples // (n_classes * 20)))}")
            report.append(f"max_features: {min(n_features, max(1, int(np.sqrt(n_features))))} (√{n_features})")
            report.append("")

            # Cluster characteristics
            report.append("DETAILED CLUSTER CHARACTERISTICS")
            report.append("-" * 50)
            for cluster_id in sorted(np.unique(self.best_result.cluster_labels)):
                cluster_data = self.best_result.feature_data[
                    self.best_result.feature_data['cluster_label'] == cluster_id]
                report.append(
                    f"Cluster {cluster_id}: {len(cluster_data)} annotations ({len(cluster_data) / len(self.feature_data) * 100:.1f}%)")

                # Top categories in this cluster
                if 'category_id' in cluster_data.columns:
                    top_categories = cluster_data['category_id'].value_counts().head(3)
                    report.append(f"  Top annotation categories: {dict(top_categories)}")

                # Most distinctive features for this cluster
                cluster_means = cluster_data[self.feature_columns].mean()
                overall_means = self.feature_data[self.feature_columns].mean()
                differences = ((cluster_means - overall_means) / overall_means.abs()).abs().sort_values(ascending=False)
                report.append(f"  Most distinctive features (relative difference):")
                for feat, diff in differences.head(3).items():
                    report.append(f"    {feat}: {diff:.3f}")
                report.append("")

        # Autoencoder vs Raw comparison
        if self.use_autoencoder:
            encoded_results = {k: v for k, v in self.results.items() if 'encoded' in k}
            raw_results = {k: v for k, v in self.results.items() if 'raw' in k}

            if encoded_results and raw_results:
                report.append("AUTOENCODER VS RAW FEATURES COMPARISON")
                report.append("-" * 50)

                best_encoded = max(encoded_results.values(), key=lambda x: x.silhouette_score)
                best_raw = max(raw_results.values(), key=lambda x: x.silhouette_score)

                report.append(
                    f"Best Autoencoder Result: {best_encoded.algorithm} (Silhouette: {best_encoded.silhouette_score:.4f})")
                report.append(
                    f"Best Raw Features Result: {best_raw.algorithm} (Silhouette: {best_raw.silhouette_score:.4f})")

                improvement = ((
                                           best_encoded.silhouette_score - best_raw.silhouette_score) / best_raw.silhouette_score) * 100
                report.append(f"Improvement with autoencoder: {improvement:+.1f}%")
                report.append("")

        # All results summary
        report.append("ALL ALGORITHM RESULTS SUMMARY")
        report.append("-" * 50)
        for alg_name, result in sorted(self.results.items(), key=lambda x: x[1].silhouette_score, reverse=True):
            feature_type = "Encoded" if "encoded" in alg_name else "Raw"
            clean_name = alg_name.replace("_encoded", "").replace("_raw", "").replace("_comparison", "")
            report.append(
                f"{clean_name.upper():<12} ({feature_type:<7}) | Clusters: {result.n_clusters:2d} | Silhouette: {result.silhouette_score:.4f}")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Enhanced report saved to: {save_path}")

        return report_text


def load_coco_annotations(json_path: str) -> Tuple[List[Dict], Dict[int, Dict]]:
    """Load COCO format annotations and image info"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    images = data.get('images', [])

    # Create image info lookup
    image_info = {img['id']: img for img in images}

    return annotations, image_info


def main_enhanced_clustering_analysis(json_path: str, output_dir: str = "enhanced_clustering_results",
                                      use_autoencoder: bool = True, autoencoder_config: Dict[str, Any] = None):
    """Main function to run complete autoencoder-enhanced clustering analysis"""

    os.makedirs(output_dir, exist_ok=True)

    print("Loading annotations...")
    annotations, image_info = load_coco_annotations(json_path)
    print(f"Loaded {len(annotations)} annotations from {len(image_info)} images")

    # Extract features
    print("Extracting enhanced features...")
    extractor = AnnotationFeatureExtractor()

    # Extract bbox features
    bbox_features = extractor.extract_bbox_features(annotations)

    # Extract spatial features
    spatial_features = extractor.extract_spatial_features(annotations, image_info)

    # Combine features
    combined_features = bbox_features.merge(
        spatial_features,
        on=['category_id', 'annotation_id', 'image_id'],
        how='inner'
    )

    print(f"Extracted {len(combined_features.columns) - 3} features for clustering")

    # Configure autoencoder
    if autoencoder_config is None:
        autoencoder_config = {
            'encoding_dim': max(3, (len(combined_features.columns) - 3) // 3),
            'architecture': 'deep',  # Options: 'shallow', 'deep', 'sparse', 'variational'
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        }

    # Run enhanced clustering analysis
    print("Starting autoencoder-enhanced clustering analysis...")
    analyzer = EnhancedClusteringAnalyzer(
        combined_features,
        use_autoencoder=use_autoencoder,
        autoencoder_config=autoencoder_config
    )

    results = analyzer.find_optimal_clusters(max_clusters=8, compare_with_raw=True)

    # Generate visualizations
    print("Generating enhanced visualizations...")
    viz_path = os.path.join(output_dir, "enhanced_clustering_visualization.png")
    analyzer.plot_clustering_results(save_path=viz_path)

    # Generate comprehensive report
    print("Generating enhanced report...")
    report_path = os.path.join(output_dir, "enhanced_clustering_report.txt")
    report_text = analyzer.generate_report(save_path=report_path)

    # Save detailed results
    print("Saving detailed results...")
    best_result = analyzer.best_result
    best_result.feature_data.to_csv(os.path.join(output_dir, "clustered_annotations_enhanced.csv"), index=False)

    # Save autoencoder model if used
    if use_autoencoder and analyzer.autoencoder:
        model_path = os.path.join(output_dir, "autoencoder_model.pth")
        torch.save(analyzer.autoencoder.model.state_dict(), model_path)
        print(f"Autoencoder model saved to: {model_path}")

    print(f"\nAutoencoder-enhanced clustering analysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"Best algorithm: {best_result.algorithm} with {best_result.n_clusters} clusters")
    print(f"Silhouette score: {best_result.silhouette_score:.4f}")

    if use_autoencoder:
        print(f"Autoencoder reconstruction error: {best_result.reconstruction_error:.6f}")
        print(f"Feature compression: {len(analyzer.feature_columns)} → {analyzer.X_encoded.shape[1]} dimensions")

    return analyzer, results


if __name__ == "__main__":
    # Example usage with different autoencoder configurations
    json_path = ""

    # Test different autoencoder architectures
    architectures = ['deep', 'sparse', 'variational']

    for arch in architectures:
        print(f"\n{'=' * 60}")
        print(f"Testing {arch.upper()} autoencoder architecture")
        print(f"{'=' * 60}")

        config = {
            'architecture': arch,
            'encoding_dim': 8,
            'epochs': 80,
            'batch_size': 32,
            'learning_rate': 0.001
        }

        output_dir = f"clustering_results_{arch}"

        analyzer, results = main_enhanced_clustering_analysis(
            json_path,
            output_dir=output_dir,
            autoencoder_config=config
        )

        # Print top separating features
        print(f"\nTop 5 features that separate clusters the most ({arch} autoencoder):")
        importance_items = sorted(analyzer.best_result.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (feature, score) in enumerate(importance_items, 1):
            print(f"{i}. {feature}: {score:.6f}")