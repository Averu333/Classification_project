import torch
import torchvision
import PIL
import numpy as np
from torchvision import transforms
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from utils.myutils import mycollate, save_weights
from utils.models import get_classifier_model, mymodel
from utils.augmentation import generate_augmentation, Transfrom_using_aug
from utils.datasets import MyDataset


def test_evaluation():
    device = 'cpu'
    model_path = './weights/1.0_'
    torch.hub.set_dir('./weights')
    model = get_classifier_model('resnet', num_classes=10, use_pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    
    dataset_test = torchvision.datasets.CIFAR10(root='./data/test',
                                                train=False,
                                                download=False)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=10,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=mycollate)
    
    evaluate_model(model, data_loader_test, device)

def test_save_weights():
    weights_folder = './weights'
    model_weights = torch.load('./weights/1.0_r_')
    eval_score = 1.0
    model_name = 'resnet'
    
    save_weights(weights_folder,
                 model_weights,
                 eval_score,
                 model_name,
                 use_wandb=False)

def test_generate_augmentation():
    aug_pad = False
    aug_affine = False
    aug_ch_suffle = False
    aug_dropout = False
    aug_AGN = False
    aug_fliplr = False
    aug_flipud = True
    aug_percent = 0.0
    return generate_augmentation(aug_pad,
                                aug_affine,
                                aug_ch_suffle,
                                aug_dropout,
                                aug_AGN,
                                aug_fliplr,
                                aug_flipud,
                                aug_percent)
    
def test_transform_using_aug():
    augment = test_generate_augmentation()
    trans = Transfrom_using_aug(augment)
    dataset = torchvision.datasets.CIFAR10(root='./data/train',
                                           train=True,
                                           transform=trans,
                                           download=True)
    image, target = dataset.__getitem__(0)
    # image = PIL.Image.fromarray(np.uint8(image))
    # image.save('./testimage.jpg')
    print('end')

def test_mymodel():
    model = mymodel(10)
    image = torch.zeros((2, 3, 224,224))
    prediction = model(image)
    print('end')
 
def test_mydataset():
    dataset = torchvision.datasets.CIFAR10(root='./data/train',
                                        train=True,
                                        download=True)   
    trans_val = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    dataset_val = MyDataset(dataset, trans_val)
    print(dataset_val.__getitem__(0))
    print('test_mydataset end')

def test_confmatrix():
    data = np.random.rand(4, 4)
    class_names = ['A', 'B', 'C', 'D']
    df_cm = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10,7))
    conf_mat_fig = sn.heatmap(df_cm, annot=True, xticklabels=class_names, yticklabels=class_names, cmap="Blues", cbar=False)
    plt.xlabel('Prediction', fontsize=18)
    ax.xaxis.set_label_position('top') 
    plt.ylabel('Actual', fontsize=18)
    ax.xaxis.tick_top()
    conf_mat_fig.figure.savefig("./conf_mat_test.png")

if __name__ == "__main__":
    # test_evaluation()
    # test_save_weights()
    # test_generate_augmentation()
    # test_transform_using_aug()
    # test_mymodel()
    # test_mydataset()
    test_confmatrix()