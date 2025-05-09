import os
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')


# Helper libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import pickle

from utils.helpers import fix_seed , train_classifier, validate_classifier
from torch_optim import MypseudoSGD, MyAdam

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    


if __main__:
    
    criterion = nn.CrossEntropyLoss()
    
    h_pseudos = [0.01, 0.05, 0.001,0.0005,0.0001,0.00005]
    epsilons = [1e-3,1e-5 ,1e-8, 1e-10]
    
    #the mini_batch size
    batch_size = 64
    
    #the number of epochs
    num_epochs = 25
    
    #fix random seed
    seed = 0
    seed_worker_ = lambda worker_id : seed_worker(worker_id, seed=seed)
    
    #device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #path for saving models and results
    path = f"./cifar10_models_batch_size={batch_size}_num_epochs={num_epochs}_seed={seed}/"
    if os.path.exists(path) == False:
      os.mkdir(path)
      
    #load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2,
                                              worker_init_fn = seed_worker_)
                                            
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2,
                                         worker_init_fn = seed_worker_)
                                     
    res = {}
    # Pseudo SGD optimizers
    for h in h_pseudos:
        for eps  in epsilons:
            model_name= f"pseudoSGD_h={h}_eps={eps}"
            fix_seed(seed)
            model = Net()
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            MypseudoSGD_ = MypseudoSGD(model.parameters(), lr= h, eps=eps)
            start_time = time.time()
            avg_loss_list, avg_accuracy_list = train_classifier(num_epochs, batch_size, criterion, MypseudoSGD_, model, trainloader, device, seed)
            duration = time.time() - start_time        
            torch.save(model.state_dict(),path+"/model_"+model_name+".pth")
            model.eval()
            train_loss, train_acc = validate_classifier(criterion, model, trainloader, batch_size, device, seed)
            test_loss, test_acc = validate_classifier(criterion, model, testloader, batch_size, device, seed)
            res[model_name] = {"duration": duration,
                               "avg_loss_list":avg_loss_list,
                               "avg_accuracy_list":avg_accuracy_list,
                               "train_loss":train_loss,"train_acc":train_acc,
                               "test_loss":test_loss, "test_acc":test_acc }
            
    for h in h_pseudos:
        for eps  in epsilons:
            model_name= f"MyAdam_h={h}_eps={eps}"
            fix_seed(seed)
            model = Net()
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            MyAdam_ = MyAdam(model.parameters(), lr= h, eps=eps)
            start_time = time.time()
            avg_loss_list, avg_accuracy_list = train_classifier(num_epochs, batch_size, criterion, MyAdam_, model, trainloader, device, seed)
            duration = time.time() - start_time        
            torch.save(model.state_dict(),path+"/model_"+model_name+".pth")
            model.eval()
            train_loss, train_acc = validate_classifier(criterion, model, trainloader, batch_size, device, seed)
            test_loss, test_acc = validate_classifier(criterion, model, testloader, batch_size, device, seed)
            res[model_name] = {"duration": duration,
                               "avg_loss_list":avg_loss_list,
                               "avg_accuracy_list":avg_accuracy_list,
                               "train_loss":train_loss,"train_acc":train_acc,
                               "test_loss":test_loss, "test_acc":test_acc }
    
    #save results 
    file = open(path+f"results.pkl", "rb")
    pickle.dump(res, file)
    file.close()