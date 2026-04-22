import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time


class NNClassifier:
    def __init__(self, train_data, train_labels):
        self.templates = train_data.astype(np.float32)
        self.labels = train_labels
        # precompute norms once to speed up distance calculations later
        self.norms = np.sum(self.templates ** 2, axis=1, keepdims=True)

    def _sq_distances(self, chunk):
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a.b), avoids computing sqrt
        chunk_norms = np.sum(chunk ** 2, axis=1, keepdims=True)
        dot = np.dot(chunk, self.templates.T)
        return np.maximum(chunk_norms + self.norms.T - 2 * dot, 0)

    def predict(self, test_data, chunk_size=1000):
        n = test_data.shape[0]
        preds = np.zeros(n, dtype=int)
        nn_idx = np.zeros(n, dtype=int)
        nn_dist = np.zeros(n)

        for i in range(0, n, chunk_size):
            chunk = test_data[i:i + chunk_size]
            dists = self._sq_distances(chunk)
            nearest = np.argmin(dists, axis=1)
            preds[i:i + len(chunk)] = self.labels[nearest]
            nn_idx[i:i + len(chunk)] = nearest
            nn_dist[i:i + len(chunk)] = dists[np.arange(len(chunk)), nearest]

        return preds, nn_idx, nn_dist


def show_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix — Nearest Neighbor Classifier')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.show()


def show_samples(test_data, test_labels, preds, nn_idx, nn_dist, train_data, n=9, correct=True):
    mask = test_labels == preds if correct else test_labels != preds
    indices = np.where(mask)[0][:n]
    label_color = 'green' if correct else 'red'

    n_cols = min(3, len(indices))
    n_rows = int(np.ceil(len(indices) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        row, col = i // n_cols, (i % n_cols) * 2

        axes[row, col].imshow(test_data[idx].reshape(28, 28), cmap='gray')
        axes[row, col].set_title(
            f'Test {idx}\nTrue: {test_labels[idx]}, Pred: {preds[idx]}',
            color=label_color, fontweight='bold'
        )
        axes[row, col].axis('off')

        axes[row, col + 1].imshow(train_data[nn_idx[idx]].reshape(28, 28), cmap='gray')
        axes[row, col + 1].set_title(
            f'Neighbor {nn_idx[idx]}\nLabel: {train_labels[nn_idx[idx]]}\nDist: {nn_dist[idx]:.2f}',
            fontweight='bold'
        )
        axes[row, col + 1].axis('off')

    # hide unused subplots
    for j in range(len(indices), n_rows * n_cols):
        r, c = j // n_cols, (j % n_cols) * 2
        if r < n_rows:
            axes[r, c].axis('off')
            axes[r, c + 1].axis('off')

    plt.tight_layout()
    plt.show()


data = sio.loadmat('data_all.mat')

train_data   = data['trainv'].astype(np.float32) / 255.0
train_labels = data['trainlab'].flatten().astype(int)
test_data    = data['testv'].astype(np.float32) / 255.0
test_labels  = data['testlab'].flatten().astype(int)

clf = NNClassifier(train_data, train_labels)

print("\nrunning NN on test set...")
t0 = time.time()
preds, nn_idx, nn_dist = clf.predict(test_data)
print(f"done in {time.time() - t0:.1f}s")

cm = confusion_matrix(test_labels, preds)
acc = np.diag(cm).sum() / cm.sum()
print(f"\naccuracy:   {acc:.4f}")
print(f"error rate: {1 - acc:.4f}")
print("\nconfusion matrix:")
print(cm)

show_confusion_matrix(cm)
show_samples(test_data, test_labels, preds, nn_idx, nn_dist, train_data, correct=False)
show_samples(test_data, test_labels, preds, nn_idx, nn_dist, train_data, correct=True)