import torch
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Local functions
from utils.options import args_parser
from utils.evaluation import evaluate_model
from utils.myutils import mycollate
from utils.models import get_classifier_model

if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #Create transformer
    trans_test = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    #Load test dataset
    dataset_test = torchvision.datasets.CIFAR10(root='./data/train',
                                        train=False,
                                        transform=trans_test,
                                        download=True)
    
    #Create dataloader
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=mycollate)
    
    #Setup model
    model = get_classifier_model(args.model_name,
                                num_classes=10,
                                use_pretrained=False,
                                train_last_layer_only=True)
    
    model.load_state_dict(torch.load(args.testmodel_path))
    model.to(args.device)
    
    score, predictions, ground_truth, class_names = evaluate_model(model, data_loader_test, args.device)
    print("Evaluation score {}".format(score))
    
    #Create and save confusion matrix
    conf_mat = confusion_matrix(ground_truth, predictions, normalize='true')
    df_cm = pd.DataFrame(conf_mat, index = class_names, columns = class_names)
    conf_mat_fig = sn.heatmap(df_cm, annot=True, xticklabels=class_names, yticklabels=class_names, cmap="Blues", cbar=False) #"Blues", 
    conf_mat_fig.figure.savefig("./confusion_matrix.png")
    print('end')
    