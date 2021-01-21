import torch
import torchvision
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(model, data_loader_test, device):
    '''
    Evaluation function. Calculates accuracy for the model using testset.
    Args:
        model (torch model): network model
        data_loader_test (torch Dataloader): testsets dataloader
        device (str): device that calculates the results, commonly 'cpu' or 'cuda:0'
    Returns:
        score (float): A score representing accuracy on testset. Multiple lower scores over few epochs might
                       mean overfitting.
        data_log (numpy array): num_classes x num_classes confusion matrix. Tells the positive results compared to
                                false negatives.
    '''
    model.eval()
    model.to(device=device)
    data_log = np.zeros((10,10))
    for batch_num, data in enumerate(data_loader_test):
        # Usualy data formating is done in custom dataset object
        # However this time I'm using the dataset from torchvision
        # so I have to change the data format here
        images = torch.stack([transforms.ToTensor()(i[0]).squeeze(0) for i in data])
        images = images.to(device=device)
        targets = torch.stack([torch.tensor(i[1]) for i in data])
        targets = targets.to(device=device)
        
        with torch.no_grad():
            prediction = model(images)
        #Choose max prediction to represent class and compare to target    
        data_log += confusion_matrix(prediction, targets)
    
    score = np.sum(np.diagonal(data_log))/np.sum(data_log)
    # return score, data_log    
    return 0, 0


def mycollate(batch):
    return batch

if __name__ == "__main__":
    device = 'cpu'
    model_path = '/workspaces/cvproject/weights/1_'
    
    model = get_classifier_model('resnet', num_classes=10, use_pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(PATH))
    
    dataset_test = torchvision.datasets.CIFAR10(root='/workspaces/cvproject/data/test',
                                                train=False,
                                                download=False)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=10,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=mycollate)
    
    evaluate_model(model, data_loader_test, device)