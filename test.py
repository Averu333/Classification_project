import torch
import torchvision

from utils.myutils import mycollate, save_weights
from utils.models import get_classifier_model
from utils.evaluation import evaluate_model

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

if __name__ == "__main__":
    # test_evaluation()
    test_save_weights()