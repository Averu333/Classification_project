import torch
import torchvision
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision import transforms

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
    predictions, ground_truth = [], []
    correct = 0

    for batch_num, data in enumerate(data_loader_test):
        #Correct input data format and set device
        images = torch.stack([i[0]for i in data])
        images = images.to(device=device)
        targets = np.array([i[1] for i in data])
        
        with torch.no_grad():
            prediction = model(images)
        #Choose max prediction to represent class
        prediction = prediction.detach().cpu().numpy()
        prediction = np.argmax(prediction, axis=1)
        
        #Add preditcions and ground truths to the result list
        predictions += list(prediction)
        ground_truth += list(targets)
    
    #Compare gt and pred lists and calculate number
    #of correct predictions
    for i in range(len(predictions)):
        if predictions[i] == ground_truth[i]:
            correct += 1
    score = correct / len(ground_truth)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return score, predictions, ground_truth, class_names
