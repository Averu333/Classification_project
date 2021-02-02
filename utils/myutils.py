import os
import torch
import wandb

def save_weights(weights_folder, model_weights, model_name):
    '''
    An helper function that saves model weights if its evaluation score is higher than
    the best models in weights_folder. The save name is score_wandb.run.name.
    Args:
        weights_folder (str): function looks this folder for best model and replaces the best model if needed.
        model_weights (dict): model weights that will be saved if score is better than current best.
        model_name (str): model_name
    Returns:
        Nothing
    '''
    model_save_name = model_name + '_full_training'
    model_save_path = os.path.join(weights_folder, model_save_name)
    torch.save(model_weights, model_save_path)
        
def mycollate(batch):
    return batch

def string_is_num(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
