from sklearn.cluster import KMeans
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time


class ChunkedNNClusteringClassifier:
    """Nearest Neighbor Classifier with K-means clustering"""
    
    def __init__(self, cluster_centers, cluster_labels):
        """
        Initialize the classifier with cluster centers
        
        Args:
            cluster_centers: Cluster center vectors (n_clusters, n_features)
            cluster_labels: Label for each cluster (n_clusters,)
        """
        self.cluster_centers = cluster_centers.astype(np.float32)  # Ensure float32
        self.cluster_labels = cluster_labels
        self.n_clusters = cluster_centers.shape[0]
        
        # Precompute squared norms of cluster centers for efficient distance computation
        self.cluster_norms = np.sum(self.cluster_centers ** 2, axis=1, keepdims=True)
    
    def _compute_distances_squared(self, test_chunk):
        """
        Compute squared Euclidean distances (do NOT take sqrt for speed).
        Uses the efficient formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b)
        
        Args:
            test_chunk: Test samples (n_chunk, n_features)
            
        Returns:
            Squared distance matrix (n_chunk, n_clusters)
        """
        # Compute squared norms of test chunk
        test_norms = np.sum(test_chunk ** 2, axis=1, keepdims=True)
        
        # Matrix multiplication: test @ cluster_centers.T (most time-critical operation)
        scalar_product = np.dot(test_chunk, self.cluster_centers.T)
        
        # Efficient distance formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b)
        distances_squared = test_norms + self.cluster_norms.T - 2 * scalar_product
        
        # Ensure non-negative (handle small numerical errors from floating point)
        return np.maximum(distances_squared, 0)
    
    def predict(self, test_data, chunk_size=5000):
        """
        Fast prediction using chunked processing without expensive sqrt operations.
        
        Args:
            test_data: Test images (n_test, n_features)
            chunk_size: Number of test samples per chunk (larger = faster due to less loop overhead)
            
        Returns:
            Predicted labels (n_test,)
        """
        n_test = test_data.shape[0]
        predictions = np.zeros(n_test, dtype=int)
        
        n_chunks = int(np.ceil(n_test / chunk_size))
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_test)
            
            chunk = test_data[start_idx:end_idx]
            
            # Compute squared distances (skip sqrt for speed)
            distances_squared = self._compute_distances_squared(chunk)
            
            # Find nearest cluster and assign label
            nearest_clusters = np.argmin(distances_squared, axis=1)
            predictions[start_idx:end_idx] = self.cluster_labels[nearest_clusters]
        
        return predictions


def evaluate_classifier(y_true, y_pred, class_names=None):
    """
    Evaluate classifier performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)
        
    Returns:
        Dictionary with metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    error_rate = 1 - accuracy
    
    print("=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nPer-class metrics:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'error_rate': error_rate,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Nearest Neighbor with Clustering Classifier')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix plot saved to {save_path}")
    
    plt.show()



def main():
    """Main execution"""
    
    print("=" * 60)
    print("NEAREST NEIGHBOR WITH K-MEANS CLUSTERING FOR MNIST")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST data...")
    data = sio.loadmat('data_all.mat')
    
    train_data = data['trainv'].astype(np.float32)
    train_labels = data['trainlab'].flatten().astype(int)
    test_data = data['testv'].astype(np.float32)
    test_labels = data['testlab'].flatten().astype(int)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of classes: {len(np.unique(train_labels))}")
    
    # Normalize data
    print("\nNormalizing data...")
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    
    # Convert to float32 for faster computation
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    
    # Perform K-means clustering on training data
    print("\nPerforming K-means clustering on training data...")
    n_clusters = 64
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=300)
    train_cluster_assignments = kmeans.fit_predict(train_data)
    cluster_centers = kmeans.cluster_centers_
    
    # Compute majority label for each cluster
    print("Computing cluster labels (majority vote)...")
    cluster_labels = np.zeros(n_clusters, dtype=int)
    for cluster_id in range(n_clusters):
        cluster_mask = train_cluster_assignments == cluster_id
        if cluster_mask.sum() > 0:
            cluster_labels[cluster_id] = np.bincount(train_labels[cluster_mask]).argmax()
    
    # Create and train classifier
    print("\nInitializing Nearest Neighbor Clustering Classifier...")
    classifier = ChunkedNNClusteringClassifier(cluster_centers, cluster_labels)
    
    # Make predictions - ultra-fast with optimized computation
    print("\nMaking predictions on test set...")
    start_time = time.time()
    # Use large chunk size (10000) - minimal loop overhead, maximum performance
    predictions = classifier.predict(test_data, chunk_size=10000)
    elapsed_time = time.time() - start_time
    print(f"Prediction time: {elapsed_time:.2f} seconds")
    print(f"Speed: {len(test_data) / elapsed_time:.0f} samples/second")
    
    # Evaluate
    print("\n")
    class_names = [str(i) for i in range(10)]
    metrics = evaluate_classifier(test_labels, predictions, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], 
                         save_path='nn_clustering_confusion_matrix.png')
    
    # Print summary statistics
    print("\nPer-class accuracy:")
    cm = metrics['confusion_matrix']
    for class_id in range(10):
        class_acc = cm[class_id, class_id] / cm[class_id, :].sum()
        print(f"  Digit {class_id}: {class_acc:.4f}")
    
    print(f"\nNumber of clusters: {n_clusters}")


if __name__ == '__main__':
    main()
