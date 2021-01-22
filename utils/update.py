from torchvision import transforms
import torch
from torch import nn
import numpy as np

def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    train_print_freq=None):
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
    for batch_num, data in enumerate(data_loader):
        # Format the data and set it to correct device
        images = torch.stack([(i[0]) for i in data])
        images = images.to(device=device)
        targets = torch.stack([torch.tensor(i[1]) for i in data])
        targets = targets.to(device=device)
        
        # prediction and loss calculation
        predictions = model(images)
        loss = loss_function(predictions, targets)
        
        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print verbdose if needed
        if train_print_freq:
            if batch_num % train_print_freq == 0:
                print("Epoch {}, batch {}/{}, loss {}, learning_rate {}".format(epoch,
                                                                                batch_num,
                                                                                total_batches,
                                                                                loss,
                                                                                optimizer.param_groups[0]["lr"]))
        
    return model
    