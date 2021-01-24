import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    #Network arguments
    parser.add_argument('--gpu', type=int, default=0, help='If device is -1 uses cpu, else uses cuda of device argument')
    parser.add_argument('--model_name', type=str, default='resnet', help='What model to use. Options: resnet')
    
    #Training arguments
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs that the model will train.")
    parser.add_argument('--train_print_freq', type=int, default=100, help="Printing frequency while training, number of batches before each print.")   
    
    #Hyperparameters
    parser.add_argument('--use_wandb', action='store_true', help="Use wandb hyperparameter optimization.")    
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for network testing and training.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.000005, help="Optimizer weight decay.")
    
    #Augmentation parameters
    parser.add_argument('--aug_pad', type=int, default=0, help="0 or 1. With 1 augments padding to training images.")
    parser.add_argument('--aug_affine', type=int, default=0, help="0 or 1. With 1 aguments affine rotation to training images.")
    parser.add_argument('--aug_ch_suffle', type=int, default=0, help="0 or 1. With 1 augments channel suffle to training images.")
    parser.add_argument('--aug_dropout', type=int, default=0, help="0 or 1. With 1 augments dropout to training images.")
    parser.add_argument('--aug_AGN', type=int, default=0, help="0 or 1. With 1 augments AdditiveGaussianNoise to training images.")
    parser.add_argument('--aug_fliplr', type=int, default=0, help="0 or 1. With 1 augments left-right flip to training images.")
    parser.add_argument('--aug_flipud', type=int, default=0, help="0 or 1. With 1 augments up-down flip to training images.")
    parser.add_argument('--aug_percent', type=float, default=0.0, help="A value between 0 and 1. Value determines how often chosen augmenters are applied.")
    
    args = parser.parse_args()
    return args