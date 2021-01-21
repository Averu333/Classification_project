import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    #Network arguments
    parser.add_argument('--device', type=int, default=-1, help='If device is -1 uses cpu, else uses cuda of device argument')
    parser.add_argument('--model_name', type=str, default='resnet', help='What model to use. Options: resnet')
    
    #Training arguments
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs that the model will train.")
    parser.add_argument('--train_print_freq', type=int, default=100, help="Printing frequency while training, number of batches before each print.")   
    
    #Hyperparameters
    parser.add_argument('--use_wandb', action='store_true', help="Use wandb hyperparameter optimization.")    
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size for network testing and training.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.000005, help="Optimizer weight decay.")
    
    args = parser.parse_args()
    return args