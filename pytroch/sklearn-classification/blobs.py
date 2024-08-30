import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_FEATURES = 2
NUM_CLASSES = 4

X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5)

X_blob, y_blob = torch.from_numpy(X_blob).type(torch.float), torch.from_numpy(y_blob).type(torch.LongTensor)
print(X_blob[:5], y_blob[:5])

X_btr, X_bte, y_btr, y_bte = train_test_split(X_blob, y_blob, test_size=0.2)

# plt.figure(figsize=(10,7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob)

def accuracy(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

class BlobModel(nn.Module):
    def __init__(self, input_features, out_features, hidden=8):
        super().__init__()
        self.lstack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=out_features),
        )

    def forward(self, x):
        return self.lstack(x)
    
model_b = BlobModel(input_features=NUM_FEATURES, out_features=NUM_CLASSES, hidden=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_b.parameters(), lr=0.1)

X_btr, y_btr = X_btr.to(device), y_btr.to(device)
X_bte, y_bte = X_bte.to(device), y_bte.to(device)

epochs = 100

for i in range(epochs):
    model_b.train()

    y_logits = model_b(X_btr)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = criterion(y_logits, y_btr)
    acc = accuracy(y_true=y_btr, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_b.eval()
    with torch.inference_mode():
        test_logits = model_b(X_bte)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = criterion(test_logits, y_bte)
        test_acc = accuracy(y_true=y_bte, y_pred=test_pred)
    
    if i % 10 == 0:
        print(f"Epoch: {i} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%") 

model_b.eval()
with torch.inference_mode():
    y_logits = model_b(X_bte)

y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = y_pred_probs.argmax(dim=1)

print(f"Predictions: {y_preds[:10]}\nLabels: {y_bte[:10]}")
print(f"Test accuracy: {accuracy(y_true=y_bte, y_pred=y_preds)}%")


