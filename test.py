import torch
import torchvision
import PIL
import numpy as np

from utils.myutils import mycollate, save_weights
from utils.models import get_classifier_model
from utils.evaluation import evaluate_model
from utils.augmentation import generate_augmentation, Transfrom_using_aug

def test_evaluation():
    device = 'cpu'
    model_path = './weights/1.0_'
    
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
    aug_pad = True
    aug_affine = True
    aug_ch_suffle = False
    aug_dropout = False
    aug_AGN = False
    aug_fliplr = False
    aug_flipud = False
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

if __name__ == "__main__":
    # test_evaluation()
    # test_save_weights()
    # test_generate_augmentation()
    test_transform_using_aug()