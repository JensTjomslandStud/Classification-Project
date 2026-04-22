from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

K = 7
M = 64 


class KNNClassifier:
    def __init__(self, templates, labels, k=K):
        self.templates = templates.astype(np.float32)
        self.labels = labels
        self.k = k
        # precompute norms once to speed up distance calculations later
        self.norms = np.sum(self.templates ** 2, axis=1, keepdims=True)

    def _sq_distances(self, chunk):
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a.b), avoids computing sqrt
        chunk_norms = np.sum(chunk ** 2, axis=1, keepdims=True)
        dot = np.dot(chunk, self.templates.T)
        return np.maximum(chunk_norms + self.norms.T - 2 * dot, 0)

    def predict(self, test_data, chunk_size=5000):
        n = test_data.shape[0]
        preds = np.zeros(n, dtype=int)

        for i in range(0, n, chunk_size):
            chunk = test_data[i:i + chunk_size]
            dists = self._sq_distances(chunk)

            # grab k nearest for each sample, majority vote decides label
            knn_idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]
            knn_labels = self.labels[knn_idx]
            for j in range(len(chunk)):
                preds[i + j] = np.bincount(knn_labels[j]).argmax()

        return preds


def build_templates(train_data, train_labels, m=M):
    classes = np.unique(train_labels)
    centers, center_labels = [], []

    for cls in classes:
        km = KMeans(n_clusters=m, random_state=42, n_init=3, max_iter=300)
        km.fit(train_data[train_labels == cls])
        centers.append(km.cluster_centers_)
        center_labels.extend([cls] * m)

    template_center = np.vstack(centers).astype(np.float32)
    template_label =np.array(center_labels, dtype=int)
    return template_center, template_label


def show_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix — KNN (K={K}), {M} clusters/class')
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


data = sio.loadmat('data_all.mat')

train_data   = data['trainv'].astype(np.float32) / 255.0
train_labels = data['trainlab'].flatten().astype(int)
test_data    = data['testv'].astype(np.float32) / 255.0
test_labels  = data['testlab'].flatten().astype(int)

print(f"\nbuilding {M} templates per class via k-means...")
clustering_time = time.time()
templates, t_labels = build_templates(train_data, train_labels)
print(f"Clustering done in {time.time() - clustering_time:.2f}s")
print(f"total templates: {templates.shape[0]}")


clf = KNNClassifier(templates, t_labels, k=K)

print("\nrunning KNN on test set...")
prediction_time = time.time()
preds = clf.predict(test_data)
print(f"Prediction done in {time.time() - prediction_time:.2f}s")


cm = confusion_matrix(test_labels, preds)
acc = np.diag(cm).sum() / cm.sum()
print(f"\naccuracy:   {acc:.4f}")

show_confusion_matrix(cm)

