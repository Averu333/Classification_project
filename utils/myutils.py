import os
import torch
import wandb

def save_weights(weights_folder, model_weights, eval_score, model_name, use_wandb=False):
    '''
    An helper function that saves model weights if its evaluation score is higher than
    the best models in weights_folder. The save name is score_wandb.run.name.
    Args:
        weights_folder (str): function looks this folder for best model and replaces the best model if needed.
        model_weights (dict): model weights that will be saved if score is better than current best.
        eval_score (float): the score for model.
        use_wandb (bool): A bool to determine if wandb is in use. If in use the save name will be score_wandb.run.name
                            if false the save name will be score_.
    Returns:
        Nothing
    '''
    #Create filename and savepath if new weights are saved
    if use_wandb:
        model_save_name = '{}_{}_{}'.format(str(eval_score), model_name[0], wandb.run.name)
    else:
        model_save_name = '{}_{}_'.format(str(eval_score), model_name[0])
    model_save_path = os.path.join(weights_folder, model_save_name)
    
    #List all current networks and check if current score is better
    #The networks score is saved to the name
    no_saved_networks = True
    network_names = os.listdir(weights_folder)
    for name in network_names:
        name_num_part = name.split('_')[0]
        if string_is_num(name_num_part):
            name_num_part = float(name_num_part)
            no_saved_networks = False
            if name_num_part < eval_score:
                os.remove(os.path.join(weights_folder, name))
                torch.save(model_weights, model_save_path)
    
    #If no weights in folder save this as first
    if no_saved_networks:
        torch.save(model_weights, model_save_path)
        
def mycollate(batch):
    return batch

def string_is_num(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
