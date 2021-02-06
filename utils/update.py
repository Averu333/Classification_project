from torchvision import transforms
import torch
from torch import nn
import numpy as np
import time
import datetime
import math

def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    train_print_freq=None,
                    use_wandb=False):
    '''
    Trains the model for one epoch.
    Args:
        model (torch model): model to train
        optimizer (torch optimizer): Optimizer to figure out gradient decent.
        data_loader (torch Dataloader): Data loader containing the training data.
        device (str): Device to perform the training on. Often 'cpu' or 'cuda:0'.
        epoch (int): The current epoch. Used for verbdose.
        train_prin_freq (int or None): After how many batches to print the verbdose. If None no printing.
    Returns:
        model (torch model): Model that has been trained for one epoch.
    '''
    if train_print_freq <= 0:
        train_print_freq = None
    model.train()
    loss_function = nn.CrossEntropyLoss()
    total_batches = len(data_loader)
    t0 = time.time()
    for batch_num, data in enumerate(data_loader):
        # Format the data and set it to correct device
        images = torch.stack([(i[0]) for i in data])
        images = images.to(device=device)
        targets = torch.stack([torch.tensor(i[1]) for i in data])
        targets = targets.to(device=device)
        
        # prediction and loss calculation
        predictions = model(images)
        loss = loss_function(predictions, targets)
        
        # if loss nan exit
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss))
            exit(1)
        
        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print verbdose if needed
        if train_print_freq:
            if batch_num % train_print_freq == 0:
                eta = ((time.time() - t0) / (batch_num+1)) * (total_batches-batch_num-1)
                eta = int(eta)
                eta = str(datetime.timedelta(seconds=eta))
                print("Epoch {}, batch {}/{}, loss {}, learning_rate {}, eta {}".format(epoch,
                                                                                batch_num,
                                                                                total_batches,
                                                                                loss,
                                                                                optimizer.param_groups[0]["lr"],
                                                                                eta)) 
    
def validate_model(model, data_loader, device):
    '''
    A function to calculate the validation loss of model.
    The validation loss is used in hyperparameter search.
    Args:
        model (torch model): model to validate
        data_loader (torch dataloader): dataloader containing the validation set
        device (str): device to use
    Retrurns:
        val_loss (float): validation_loss value of the model
    '''
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_batches = len(data_loader)
    loss = 0
    for batch_num, data in enumerate(data_loader):
        # Format the data and set it to correct device
        images = torch.stack([(i[0]) for i in data])
        images = images.to(device=device)
        targets = torch.stack([torch.tensor(i[1]) for i in data])
        targets = targets.to(device=device)
        
        # prediction and loss calculation
        with torch.no_grad():
            predictions = model(images)
        loss += loss_function(predictions, targets).item()
    
    #Return validation loss   
    return loss / len(data_loader)