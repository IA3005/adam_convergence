import torch
import random
import numpy as np

def fix_seed(seed=0):
  # Set seed for PyTorch
  torch.manual_seed(seed)

  # Set seed for CUDA (if using GPUs)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

  # Set seed for Python's random module
  random.seed(seed)

  # Set seed for NumPy
  np.random.seed(seed)

  # Ensure deterministic behavior for PyTorch operations
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def seed_worker(worker_id, seed=0):
    # Set seed for Python and NumPy in each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    
def train_classifier(num_epochs, batch_size, criterion, optimizer, model, dataloader,
                     device, seed):
    avg_loss_list = []
    avg_accuracy_list = []
    model.train()
    for epoch in (range(num_epochs)):
        epoch_average_loss = 0.0
        correct = 0.0
        total = 0
        fix_seed(seed)
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            y_pred = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_average_loss += loss.item() * batch_size / len(dataloader.dataset)
            total += y.size(0)
            correct += (y_pred == y).sum().item()
        epoch_average_accuracy = 100* correct / total
        avg_loss_list.append(epoch_average_loss)
        avg_accuracy_list.append(epoch_average_accuracy)
        if ((epoch+1)%1 == 0):
                print('Epoch [{}/{}], Loss_error: {:.4f} --- Accuracy: {:.4f}'
                      .format(epoch+1, num_epochs,  epoch_average_loss, epoch_average_accuracy))
    return avg_loss_list, avg_accuracy_list


def validate_classifier(criterion, model, testloader, batch_size, device, seed):
    correct = 0
    total = 0
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        fix_seed(seed)
        for data in testloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            y_pred = torch.argmax(outputs, dim=1)
            total += y.size(0)
            correct += (y_pred == y).sum().item()
            val_loss += criterion(outputs, y).item() * y.size(0)

    # Calculate the validation accuracy and validation loss
    val_accuracy = 100 * correct / total
    val_loss /= len(testloader.dataset)
    return val_loss, val_accuracy
