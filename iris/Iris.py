import torch
import numpy as np

# Load class data files
class_1 = np.loadtxt('class_1', delimiter=',')
class_2 = np.loadtxt('class_2', delimiter=',')
class_3 = np.loadtxt('class_3', delimiter=',')

# Extract first 30 samples from each class for training
class_1_training = class_1[:30]
class_2_training = class_2[:30]
class_3_training = class_3[:30]

# Extract last 20 samples from each class for testing
class_1_test = class_1[30:]
class_2_test = class_2[30:]
class_3_test = class_3[30:]

# Combine all training and test features
X_train = np.vstack([class_1_training, class_2_training, class_3_training])
X_test = np.vstack([class_1_test, class_2_test, class_3_test])

# Select wanted features
X_train = X_train[:, [2]]
X_test = X_test[:, [2]]

# Hardcode labels
t_train = np.vstack([
    np.concatenate([[1]*30, [0]*30, [0]*30]),
    np.concatenate([[0]*30, [1]*30, [0]*30]),
    np.concatenate([[0]*30, [0]*30, [1]*30]),
])
t_test = np.vstack([
    np.concatenate([[1]*20, [0]*20, [0]*20]),
    np.concatenate([[0]*20, [1]*20, [0]*20]),
    np.concatenate([[0]*20, [0]*20, [1]*20]),
])

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
t_train = torch.tensor(t_train).float()
t_test = torch.tensor(t_test).float()

# Normalize the features
mean = X_train.mean(dim=0)
std = X_train.std(dim=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Define the model
model = torch.nn.Linear(in_features = 1, out_features =3, bias = True)

def MSE(g, t):
    return 0.5 * torch.sum((g - t) ** 2)

# Train the model with manual gradient descent
sigmoid = torch.nn.Sigmoid()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

N = 1000
for i in range(N):
    # Forward pass
    z = model(X_train).t()
    g = sigmoid(z)
    loss = MSE(g, t_train) 
    optimizer.zero_grad()   # Clear old gradients
    loss.backward()         # Compute gradients for all params
    optimizer.step()        # Update weights using those gradients

    # Print the loss every 100 epochs
    if (i+1) % 100 == 0:
        print(f'Iteration [{i+1}/{N}], Loss: {loss.item():.4f}')


# Create confusion matrices using final parameters
print("\n" + "CONFUSION MATRIX - TRAINING SET")
# Get predictions on training set
z_train = model(X_train).t()
g_train = sigmoid(z_train)

# Convert predictions to class labels (argmax)
predictions_train = torch.argmax(g_train, dim=0).numpy() + 1
actual_train = torch.argmax(t_train, dim=0).numpy() + 1

# Create confusion matrix for training
confusion_matrix_train = np.zeros((3, 3), dtype=int)
for i in range(len(predictions_train)):
    confusion_matrix_train[actual_train[i] - 1, predictions_train[i] - 1] += 1
print("        Class 1  Class 2  Class 3")
for i in range(3):
    print(f"Class {i+1}  {confusion_matrix_train[i, 0]:5d}   {confusion_matrix_train[i, 1]:5d}   {confusion_matrix_train[i, 2]:5d}")

# Calculate training accuracy
accuracy_train = np.trace(confusion_matrix_train) / np.sum(confusion_matrix_train)
print(f"\nTraining Accuracy: {accuracy_train:.4f} ({100*accuracy_train:.2f}%)")

print("\n" + "CONFUSION MATRIX - TEST SET")

# Get predictions on test set
z_test = model(X_test).t()
g_test = sigmoid(z_test)

# Convert predictions to class labels (argmax)
predictions = torch.argmax(g_test, dim=0).numpy() + 1  # +1 to match class numbers 1, 2, 3
actual = torch.argmax(t_test, dim=0).numpy() + 1

# Create confusion matrix
confusion_matrix = np.zeros((3, 3), dtype=int)
for i in range(len(predictions)):
    confusion_matrix[actual[i] - 1, predictions[i] - 1] += 1

print("        Class 1  Class 2  Class 3")
for i in range(3):
    print(f"Class {i+1}  {confusion_matrix[i, 0]:5d}   {confusion_matrix[i, 1]:5d}   {confusion_matrix[i, 2]:5d}")

# Calculate accuracy
accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
print(f"\nTest Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")

# Create histograms for each feature and class
"""
print("\n" + "="*50)
print("FEATURE HISTOGRAMS")
print("="*50)

feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
class_names = ['Class 1', 'Class 2', 'Class 3']
class_data = [class_1, class_2, class_3]
colors = ['red', 'blue', 'green']

# Create a figure with subplots (4 features x 3 classes)
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Feature Distributions by Class', fontsize=16)

for feature_idx in range(4):
    for class_idx in range(3):
        ax = axes[feature_idx, class_idx]
        ax.hist(class_data[class_idx][:, feature_idx], bins=10, color=colors[class_idx], alpha=0.7, edgecolor='black')
        ax.set_title(f'{feature_names[feature_idx]} - {class_names[class_idx]}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('feature_histograms2.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nHistogram plot saved as 'feature_histograms.png'")
"""

# Analyze feature overlap across classes
"""
print("\n" + "="*70)
print("FEATURE OVERLAP ANALYSIS")
print("="*70)

feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
class_names = ['Class 1', 'Class 2', 'Class 3']
class_data = [class_1, class_2, class_3]

overlap_scores = []

for feature_idx in range(4):
    print(f"\n{feature_names[feature_idx]}:")
    print("-" * 60)
    
    # Get statistics for each class
    stats = []
    for class_idx in range(3):
        feature_values = class_data[class_idx][:, feature_idx]
        mean = np.mean(feature_values)
        std = np.std(feature_values)
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        stats.append({'mean': mean, 'std': std, 'min': min_val, 'max': max_val})
        print(f"{class_names[class_idx]:10s}: mean={mean:7.3f}, std={std:7.3f}, range=[{min_val:7.3f}, {max_val:7.3f}]")
    
    # Calculate pairwise overlap between classes
    total_overlap = 0
    overlap_count = 0
    
    for i in range(3):
        for j in range(i+1, 3):
            # Calculate intersection of ranges
            overlap_start = max(stats[i]['min'], stats[j]['min'])
            overlap_end = min(stats[i]['max'], stats[j]['max'])
            
            if overlap_end > overlap_start:
                # Calculate overlap as percentage of the smaller range
                range_i = stats[i]['max'] - stats[i]['min']
                range_j = stats[j]['max'] - stats[j]['min']
                smaller_range = min(range_i, range_j)
                overlap_pct = 100 * (overlap_end - overlap_start) / smaller_range if smaller_range > 0 else 0
                
                total_overlap += overlap_pct
                overlap_count += 1
                print(f"  {class_names[i]} vs {class_names[j]}: overlap={overlap_pct:.1f}%")
            else:
                print(f"  {class_names[i]} vs {class_names[j]}: overlap=0.0%")
    
    avg_overlap = total_overlap / overlap_count if overlap_count > 0 else 0
    overlap_scores.append((feature_names[feature_idx], avg_overlap))
    print(f"  Average Overlap: {avg_overlap:.1f}%")

# Rank features by overlap
print("\n" + "="*70)
print("FEATURE RANKING (by overlap - higher = more overlap = less discriminative)")
print("="*70)
overlap_scores.sort(key=lambda x: x[1], reverse=True)

for rank, (feature, overlap) in enumerate(overlap_scores, 1):
    print(f"{rank}. {feature:12s}: {overlap:6.1f}% overlap")
"""