"""
K-Nearest Neighbors (k-NN) Classifier for MNIST
Pure k-NN without clustering, k=7
"""

import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time


class KNNClassifier:
    """K-Nearest Neighbors Classifier"""
    
    def __init__(self, train_data, train_labels, k=7):
        """
        Initialize the k-NN classifier with training data
        
        Args:
            train_data: Training images (n_train, n_features)
            train_labels: Training labels (n_train,)
            k: Number of nearest neighbors to consider
        """
        self.train_data = train_data.astype(np.float32)
        self.train_labels = train_labels
        self.n_train = train_data.shape[0]
        self.k = k
        
        # Precompute squared norms of training samples for efficient distance computation
        self.train_norms = np.sum(self.train_data ** 2, axis=1, keepdims=True)
    
    def _compute_distances_squared(self, test_chunk):
        """
        Compute squared Euclidean distances to all training samples.
        Uses: ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b)
        
        Args:
            test_chunk: Test samples (n_chunk, n_features)
            
        Returns:
            Squared distance matrix (n_chunk, n_train)
        """
        test_norms = np.sum(test_chunk ** 2, axis=1, keepdims=True)
        scalar_product = np.dot(test_chunk, self.train_data.T)
        distances_squared = test_norms + self.train_norms.T - 2 * scalar_product
        return distances_squared
    
    def predict(self, test_data, chunk_size=100):
        """
        Predict using k-NN: majority vote among k nearest neighbors.
        
        Args:
            test_data: Test images (n_test, n_features)
            chunk_size: Samples per chunk (smaller chunks use less memory)
            
        Returns:
            Predicted labels (n_test,)
        """
        n_test = test_data.shape[0]
        predictions = np.zeros(n_test, dtype=int)
        
        n_chunks = int(np.ceil(n_test / chunk_size))
        
        print(f"Processing {n_test} test samples in {n_chunks} chunks (k={self.k})")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_test)
            
            chunk = test_data[start_idx:end_idx]
            distances_squared = self._compute_distances_squared(chunk)
            
            # Find k nearest training samples for each test sample
            # argsort returns indices sorted by distance (ascending)
            nearest_k_indices = np.argsort(distances_squared, axis=1)[:, :self.k]
            nearest_k_labels = self.train_labels[nearest_k_indices]
            
            # Majority vote among k neighbors
            for i, labels in enumerate(nearest_k_labels):
                # np.bincount finds frequency of each label
                predictions[start_idx + i] = np.bincount(labels).argmax()
            
            if (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
                print(f"  Processed chunk {chunk_idx + 1}/{n_chunks} ({end_idx}/{n_test} samples)")
        
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


def plot_confusion_matrix(cm, title, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
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
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def main():
    """Main execution"""
    
    print("=" * 60)
    print("CONFUSION MATRIX - K-NEAREST NEIGHBORS CLASSIFIER (k=7)")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST data...")
    data = sio.loadmat('data_all.mat')
    
    train_data = data['trainv'].astype(np.float32) / 255.0
    train_labels = data['trainlab'].flatten().astype(int)
    test_data = data['testv'].astype(np.float32) / 255.0
    test_labels = data['testlab'].flatten().astype(int)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of classes: {len(np.unique(train_labels))}")
    
    # Create and test k-NN classifier
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS (k=7)")
    print("=" * 60)
    
    classifier = KNNClassifier(train_data, train_labels, k=7)
    
    print("\nMaking predictions on test set...")
    start_time = time.time()
    predictions = classifier.predict(test_data, chunk_size=100)
    elapsed_time = time.time() - start_time
    
    print(f"\nPrediction time: {elapsed_time:.2f} seconds")
    print(f"Speed: {len(test_data) / elapsed_time:.0f} samples/second")
    
    # Evaluate
    print("\n")
    class_names = [str(i) for i in range(10)]
    metrics = evaluate_classifier(test_labels, predictions, class_names)
    
    plot_confusion_matrix(metrics['confusion_matrix'], 
                         'Confusion Matrix - K-Nearest Neighbors Classifier (k=7)', 
                         save_path='knn_k7_confusion_matrix.png')
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    cm = metrics['confusion_matrix']
    for class_id in range(10):
        class_acc = cm[class_id, class_id] / cm[class_id, :].sum()
        print(f"  Digit {class_id}: {class_acc:.4f}")


if __name__ == '__main__':
    main()
