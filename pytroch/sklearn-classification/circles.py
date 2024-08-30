import pandas as pd
import numpy as np

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else "cpu"

X, y = make_circles(1000, noise=0.03)
df = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
df.label.value_counts()

print(X.shape, y.shape)

# plt.scatter(x=X[:, 0], y=X[:, 1], c=y)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.out(self.relu(self.fc2(self.relu(self.fc1(x)))))

model = Model().to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr=0.1)

def accuracy(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

Xtr, ytr = Xtr.to(device), ytr.to(device)
Xte, yte = Xte.to(device), yte.to(device)

epochs = 1000

for i in range(epochs):
    model.train()

    y_logits = model(Xtr).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> pred probabilites -> pred labels

    loss = criterion(y_logits, ytr)
    acc = accuracy(y_pred, ytr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(Xte).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = criterion(test_logits, yte)
        test_acc = accuracy(test_pred, yte)

    if i % 100 == 0:
        print(f"Epoch: {i} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

model.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model(Xte))).squeeze()
print(y_preds[:10], yte[:10])
