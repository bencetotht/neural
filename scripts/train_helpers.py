import torch

def train(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, accuracy_fn= accuracy):
    train_loss, train_acc = 0, 0

    model.to(device)
    model.train()
    
    # iterating through batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = criterion(y_pred, y) # prediciton, target
        train_loss += loss
        train_acc += accuracy_fn(y_pred.argmax(dim=1), y) # argmax: logits -> pred labels

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f'Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}')

def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device, accuracy_fn = accuracy):
    test_loss, test_acc = 0, 0
    
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += criterion(test_pred, y)
            test_acc += accuracy_fn(test_pred.argmax(dim=1), y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}')

def accuracy(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100