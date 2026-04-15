"""
Nearest Neighbor Classifier using Euclidean Distance on MNIST Dataset
Implements chunked processing for memory efficiency
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import cdist
import time

class ChunkedNNClassifier:
    """Nearest Neighbor Classifier with chunked test processing"""
    
    def __init__(self, train_data, train_labels):
        """
        Initialize the classifier with training data
        
        Args:
            train_data: Training images (n_samples, n_features)
            train_labels: Training labels (n_samples,)
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.n_train = train_data.shape[0]
        
        # Precompute squared norms of training data for efficient distance computation
        self.train_norms = np.sum(self.train_data ** 2, axis=1, keepdims=True)
    
    def _compute_distances_vectorized(self, test_chunk):
        """
        Compute Euclidean distances between test chunk and all training samples
        Uses vectorized operations for efficiency.
        
        Args:
            test_chunk: Test samples (n_chunk, n_features)
            
        Returns:
            Distance matrix (n_chunk, n_train)
        """
        # Compute squared norms of test chunk
        test_norms = np.sum(test_chunk ** 2, axis=1, keepdims=True)
        
        # Compute cross product: test @ train.T
        scalar_product = np.dot(test_chunk, self.train_data.T)
        
        # Efficient distance formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b)
        distances_squared = test_norms + self.train_norms.T - 2 * scalar_product
        
        # Ensure non-negative (handle numerical errors)
        distances_squared = np.maximum(distances_squared, 0)
        
        # Return Euclidean distances
        return distances_squared
    
    def predict_batch_chunked(self, test_data, chunk_size=1000, verbose=True):
        """
        Predict labels for test data using chunked processing with vectorized distance computation
        
        Args:
            test_data: Test images (n_test, n_features)
            chunk_size: Number of test samples to process at once
            verbose: Print progress information
            
        Returns:
            Predicted labels (n_test,)
        """
        n_test = test_data.shape[0]
        predictions = np.zeros(n_test, dtype=int)
        
        n_chunks = int(np.ceil(n_test / chunk_size))
        
        if verbose:
            print(f"Processing {n_test} test samples in {n_chunks} chunks (size={chunk_size})")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_test)
            
            chunk = test_data[start_idx:end_idx]
            
            # Compute distances using vectorized operations
            distances = self._compute_distances_vectorized(chunk)
            
            # Find nearest neighbor for each sample in chunk
            nearest_neighbor = np.argmin(distances, axis=1)
            predictions[start_idx:end_idx] = self.train_labels[nearest_neighbor]
            
            if verbose and (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
                print(f"  Processed chunk {chunk_idx + 1}/{n_chunks} "
                      f"({min(end_idx, n_test)}/{n_test} samples)")
        
        return predictions
    
    def predict_with_neighbors(self, test_data, chunk_size=1000):
        """
        Predict labels and return nearest neighbor indices and distances
        
        Args:
            test_data: Test images (n_test, n_features)
            chunk_size: Number of test samples to process at once
            
        Returns:
            Tuple of (predictions, neighbor_indices, neighbor_distances)
        """
        n_test = test_data.shape[0]
        predictions = np.zeros(n_test, dtype=int)
        neighbor_indices = np.zeros(n_test, dtype=int)
        neighbor_distances = np.zeros(n_test, dtype=float)
        
        n_chunks = int(np.ceil(n_test / chunk_size))
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_test)
            
            chunk = test_data[start_idx:end_idx]
            
            # Compute distances
            distances = self._compute_distances_vectorized(chunk)
            
            # Find nearest neighbor
            nearest_indices = np.argmin(distances, axis=1)
            nearest_dists = np.min(distances, axis=1)
            
            predictions[start_idx:end_idx] = self.train_labels[nearest_indices]
            neighbor_indices[start_idx:end_idx] = nearest_indices
            neighbor_distances[start_idx:end_idx] = nearest_dists
        
        return predictions, neighbor_indices, neighbor_distances


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
    plt.title('Confusion Matrix - Nearest Neighbor Classifier')
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


def visualize_single_prediction(test_image, test_label, pred_label, neighbor_image, 
                               neighbor_label, distance, index_test):
    """
    Visualize a single test image, prediction, and its nearest neighbor
    
    Args:
        test_image: Test image vector (784,)
        test_label: True label of test image
        pred_label: Predicted label
        neighbor_image: Nearest neighbor image vector (784,)
        neighbor_label: Label of nearest neighbor
        distance: Euclidean distance to neighbor
        index_test: Index of test sample
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Test image
    test_matrix = test_image.reshape((28, 28))
    axes[0].imshow(test_matrix, cmap='gray')
    match_color = 'green' if test_label == pred_label else 'red'
    axes[0].set_title(f'Test Image {index_test}\nTrue: {test_label}, Pred: {pred_label}', 
                     color=match_color, fontweight='bold')
    axes[0].axis('off')
    
    # Nearest neighbor
    neighbor_matrix = neighbor_image.reshape((28, 28))
    axes[1].imshow(neighbor_matrix, cmap='gray')
    axes[1].set_title(f'Nearest Neighbor\nLabel: {neighbor_label}\nDistance: {distance:.2f}', 
                     fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_misclassifications(test_data, test_labels, predictions, neighbor_indices, 
                                neighbor_distances, train_data, train_labels, 
                                num_samples=10, save_path=None):
    """
    Visualize misclassified samples with their nearest neighbors
    
    Args:
        test_data: Test images
        test_labels: True test labels
        predictions: Predicted labels
        neighbor_indices: Indices of nearest neighbors
        neighbor_distances: Distances to nearest neighbors
        train_data: Training data
        train_labels: Training labels
        num_samples: Number of misclassifications to show
        save_path: Path to save figure
    """
    # Find misclassified indices
    misclassified_mask = test_labels != predictions
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassifications found!")
        return
    
    # Select samples to visualize
    n_show = min(num_samples, len(misclassified_indices))
    selected_indices = misclassified_indices[:n_show]
    
    # Create grid
    n_cols = min(3, n_show)
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, test_idx in enumerate(selected_indices):
        row = i // n_cols
        
        # Test image
        col_test = (i % n_cols) * 2
        test_image = test_data[test_idx].reshape((28, 28))
        axes[row, col_test].imshow(test_image, cmap='gray')
        true_label = test_labels[test_idx]
        pred_label = predictions[test_idx]
        axes[row, col_test].set_title(
            f'Test {test_idx}\nTrue: {true_label}, Pred: {pred_label}',
            color='red', fontweight='bold'
        )
        axes[row, col_test].axis('off')
        
        # Nearest neighbor
        col_neighbor = col_test + 1
        neighbor_idx = neighbor_indices[test_idx]
        neighbor_image = train_data[neighbor_idx].reshape((28, 28))
        axes[row, col_neighbor].imshow(neighbor_image, cmap='gray')
        neighbor_label = train_labels[neighbor_idx]
        distance = neighbor_distances[test_idx]
        axes[row, col_neighbor].set_title(
            f'Neighbor {neighbor_idx}\nLabel: {neighbor_label}\nDist: {distance:.2f}',
            fontweight='bold'
        )
        axes[row, col_neighbor].axis('off')
    
    # Hide remaining subplots
    for j in range(i + 1, n_rows * n_cols * 2):
        row = j // (n_cols * 2)
        col = j % (n_cols * 2)
        if row < n_rows:
            axes[row, col].axis('off')
    
    plt.suptitle(f'Misclassified Samples (showing {n_show} of {len(misclassified_indices)})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Misclassification plot saved to {save_path}")
    
    plt.show()


def visualize_correct_classifications(test_data, test_labels, predictions, neighbor_indices, 
                                     neighbor_distances, train_data, train_labels, 
                                     num_samples=10, save_path=None):
    """
    Visualize correctly classified samples with their nearest neighbors
    
    Args:
        test_data: Test images
        test_labels: True test labels
        predictions: Predicted labels
        neighbor_indices: Indices of nearest neighbors
        neighbor_distances: Distances to nearest neighbors
        train_data: Training data
        train_labels: Training labels
        num_samples: Number of correct classifications to show
        save_path: Path to save figure
    """
    # Find correctly classified indices
    correct_mask = test_labels == predictions
    correct_indices = np.where(correct_mask)[0]
    
    if len(correct_indices) == 0:
        print("No correct classifications found!")
        return
    
    # Select samples to visualize
    n_show = min(num_samples, len(correct_indices))
    selected_indices = correct_indices[:n_show]
    
    # Create grid
    n_cols = min(3, n_show)
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, test_idx in enumerate(selected_indices):
        row = i // n_cols
        
        # Test image
        col_test = (i % n_cols) * 2
        test_image = test_data[test_idx].reshape((28, 28))
        axes[row, col_test].imshow(test_image, cmap='gray')
        label = test_labels[test_idx]
        axes[row, col_test].set_title(
            f'Test {test_idx}\nLabel: {label}',
            color='green', fontweight='bold'
        )
        axes[row, col_test].axis('off')
        
        # Nearest neighbor
        col_neighbor = col_test + 1
        neighbor_idx = neighbor_indices[test_idx]
        neighbor_image = train_data[neighbor_idx].reshape((28, 28))
        axes[row, col_neighbor].imshow(neighbor_image, cmap='gray')
        neighbor_label = train_labels[neighbor_idx]
        distance = neighbor_distances[test_idx]
        axes[row, col_neighbor].set_title(
            f'Neighbor {neighbor_idx}\nLabel: {neighbor_label}\nDist: {distance:.2f}',
            fontweight='bold'
        )
        axes[row, col_neighbor].axis('off')
    
    # Hide remaining subplots
    for j in range(i + 1, n_rows * n_cols * 2):
        row = j // (n_cols * 2)
        col = j % (n_cols * 2)
        if row < n_rows:
            axes[row, col].axis('off')
    
    plt.suptitle(f'Correctly Classified Samples (showing {n_show} of {len(correct_indices)})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Correct classification plot saved to {save_path}")
    
    plt.show()


def main():
    """Main execution"""
    
    print("=" * 60)
    print("NEAREST NEIGHBOR CLASSIFIER FOR MNIST")
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
    
    # Normalize data (optional but recommended for Euclidean distance)
    print("\nNormalizing data...")
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    
    # Create and train classifier
    print("\nInitializing Nearest Neighbor Classifier...")
    classifier = ChunkedNNClassifier(train_data, train_labels)
    
    # Make predictions with chunking (and get neighbor information)
    print("\nMaking predictions on test set (with chunking)...")
    start_time = time.time()
    predictions, neighbor_indices, neighbor_distances = classifier.predict_with_neighbors(
        test_data, chunk_size=1000
    )
    elapsed_time = time.time() - start_time
    print(f"Prediction time: {elapsed_time:.2f} seconds")
    
    # Evaluate
    print("\n")
    class_names = [str(i) for i in range(10)]
    metrics = evaluate_classifier(test_labels, predictions, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], 
                         save_path='nn_confusion_matrix.png')
    
    # Print summary statistics
    print("\nPer-class accuracy:")
    cm = metrics['confusion_matrix']
    for class_id in range(10):
        class_acc = cm[class_id, class_id] / cm[class_id, :].sum()
        print(f"  Digit {class_id}: {class_acc:.4f}")
    
    # Visualize misclassifications
    print("\n" + "="*60)
    print("VISUALIZING MISCLASSIFICATIONS")
    print("="*60)
    visualize_misclassifications(test_data, test_labels, predictions, neighbor_indices, 
                                neighbor_distances, train_data, train_labels, 
                                num_samples=9, save_path='misclassified_samples.png')
    
    # Visualize correct classifications
    print("\n" + "="*60)
    print("VISUALIZING CORRECT CLASSIFICATIONS")
    print("="*60)
    visualize_correct_classifications(test_data, test_labels, predictions, neighbor_indices, 
                                     neighbor_distances, train_data, train_labels, 
                                     num_samples=9, save_path='correct_samples.png')


if __name__ == '__main__':
    main()
