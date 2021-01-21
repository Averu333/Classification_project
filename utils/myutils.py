import os
import torch
import wandb

def save_weights(weights_folder, model_weights, eval_score, use_wandb=False):
    #Create filename and savepath if new weights are saved
    if use_wandb:
        model_save_name = str(eval_score) + '_' + wandb.run.name
    else:
        model_save_name = str(eval_score) + '_'
    model_save_path = os.path.join(weights_folder, model_save_name)
    
    #List all current networks and check if current score is better
    #The networks score is saved to the name
    network_names = os.listdir(weights_folder)
    for name in network_names:
        if int(name.split('_')[0]) < eval_score:
            os.remove(os.path.join(weights_folder, name))
            torch.save(model_weights, model_save_path)
    
    #If no weights in folder save this as first
    if not network_names:
        torch.save(model_weights, model_save_path)