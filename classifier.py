import numpy as np
import torch
import torchvision
import wandb
from torchvision import transforms

#Local functions
from utils.options import args_parser
from utils.models import get_classifier_model
from utils.myutils import save_weights, mycollate
from utils.update import train_one_epoch, validate_model
from utils.evaluation import evaluate_model
from utils.augmentation import generate_augmentation, Transfrom_using_aug
from utils.datasets import MyDataset

def istrue(input):
    if input == 1:
        return True
    else:
        return False
    
if __name__ == "__main__":
    #Setting up argparser and hyperparameter optimization
    args = args_parser()
    if args.use_wandb: wandb.init()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.use_wandb: wandb.config.update(args)
    
    #Generate augment used to transform training images
    augment = generate_augmentation(istrue(args.aug_pad),
                                istrue(args.aug_affine),
                                istrue(args.aug_ch_suffle),
                                istrue(args.aug_dropout),
                                istrue(args.aug_AGN),
                                istrue(args.aug_fliplr),
                                istrue(args.aug_flipud),
                                args.aug_percent)
    
    #Generate transform function using augmentation
    trans_train = Transfrom_using_aug(augment)
    trans_val = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    
    #Setting up dataset
    dataset = torchvision.datasets.CIFAR10(root='./data/train',
                                           train=True,
                                           download=True)

    if not args.train_full:
        #Splitting up dataset to test and validation set
        #Note: k-fold cross validation could also be implemented
        #However it requires k times more computation which is
        #unesessary for this small example project
        indices = np.arange(len(dataset))
        np.random.seed(1758935)
        np.random.shuffle(indices)
        split_index = int(len(dataset) * 0.8)
        dataset_train = torch.utils.data.Subset(dataset, indices[:split_index])
        dataset_val = torch.utils.data.Subset(dataset, indices[split_index:])
    else:
        dataset_train = dataset
        
    #Add transformations with use of my own dataset object
    dataset_train = MyDataset(dataset_train, trans_train)
    if not args.train_full:
        dataset_val = MyDataset(dataset_val, trans_val)
    
    #Put subsets to dataloaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=mycollate)
    if not args.train_full:
        data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    collate_fn=mycollate)
    
    #Get model and set to device
    torch.hub.set_dir(args.weights_folder)
    model = get_classifier_model(args.model_name,
                                 num_classes=10,
                                 use_pretrained=True,
                                 train_last_layer_only=True)
    model.to(args.device)
    
    #Create the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(params,
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=args.weight_decay)
    
    #Create lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.1,
                                                              patience=2,
                                                              threshold=0.01)
    
    for epoch in range(args.num_epochs):
        #train one epoch
        train_one_epoch(model,
                        optimizer,
                        data_loader_train,
                        args.device,
                        epoch,
                        train_print_freq=args.train_print_freq,
                        use_wandb=args.use_wandb)
        
        if not args.train_full:
            val_loss = validate_model(model,
                                    data_loader_val,
                                    args.device)
            
            #Step lr scheduler
            lr_scheduler.step(val_loss)
            
            print("Epoch {}, validation loss {}".format(epoch, val_loss))
            #Log results and let the hyperparameter optimizer take care of early stops
            if args.use_wandb: wandb.log({'val_loss': val_loss})
    
    #Save fully trained network        
    if args.train_full:        
        save_weights(weights_folder=args.weights_folder,
                    model_weights=model.state_dict(),
                    model_name=args.model_name)