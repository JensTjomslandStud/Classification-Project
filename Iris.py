# Import the required libraries
import torch
import numpy as np
import pandas as pd

# Load class data files
class_1 = np.loadtxt('class_1', delimiter=',')
class_2 = np.loadtxt('class_2', delimiter=',')
class_3 = np.loadtxt('class_3', delimiter=',')

# Extract first 30 samples from each class for training
class_1_training = class_1[:30]
class_2_training = class_2[:30]
class_3_training = class_3[:30]

# Extract last 20 samples from each class for testing
class_1_test = class_1[31:]
class_2_test = class_2[31:]
class_3_test = class_3[31:]

# Combine all training and test features
X_train = np.vstack([class_1_training, class_2_training, class_3_training])
X_test = np.vstack([class_1_test, class_2_test, class_3_test])

# Hardcode labels - shape (3, num_samples)
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
model = torch.nn.Linear(in_features = 4, out_features =3, bias = False)

def MSE(g, t):
    return 0.5 * torch.sum((g - t) ** 2)

# Train the model with manual gradient descent
sigmoid = torch.nn.Sigmoid()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    z = model(X_train).t()
    g = sigmoid(z)
    loss = MSE(g, t_train) 
    optimizer.zero_grad()   # Clear old gradients
    loss.backward()         # Compute gradients for all params
    optimizer.step()        # Update weights using those gradients

    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')