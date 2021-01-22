import numpy as np
import torch
import torchvision
import wandb

#Local functions
from utils.options import args_parser
from utils.models import get_classifier_model
from utils.myutils import save_weights, mycollate
from utils.update import train_one_epoch
from utils.evaluation import evaluate_model

if __name__ == "__main__":
    #Setting up argparser and hyperparameter optimization
    args = args_parser()
    if args.use_wandb: wandb.init()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.use_wandb: wandb.config.update(args)
    
    #Setting up training and test datasets
    dataset = torchvision.datasets.CIFAR10(root='./data/train',
                                           train=True,
                                           download=True)
    
    dataset_test = torchvision.datasets.CIFAR10(root='./data/test',
                                                train=False,
                                                download=True)
    
    #Setting up dataloaders for datasets
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=mycollate)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=mycollate)
    
    #Get model and set to device
    model = get_classifier_model(args.model_name, num_classes=10, use_pretrained=True)
    model.to(args.device)
    
    #Create the optimizer and 
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(params,
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    
    for epoch in range(args.num_epochs):
        #train one epoch
        train_one_epoch(model,
                        optimizer,
                        data_loader,
                        args.device,
                        epoch,
                        train_print_freq=args.train_print_freq)
        #Step lr scheduler
        lr_scheduler.step()
        #Evaluate results to get evaluation score and a data_log that logs all classification results
        eval_score, data_log = evaluate_model(model,
                                              data_loader_test,
                                              args.device)
        print("Epoch {}, evaluation score {}".format(epoch, eval_score))
        #Check if network is best and save weights
        save_weights(weights_folder='./weights',
                     model_weights=model.state_dict(),
                     eval_score=eval_score,
                     use_wandb=args.use_wandb)
        #Log results and let the hyperparameter optimizer take care of early stops
        if args.use_wandb: wandb.log({'eval_score': eval_score, 'data_log': data_log})
            