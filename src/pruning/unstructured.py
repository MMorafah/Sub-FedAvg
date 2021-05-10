import numpy as np
from scipy import linalg
from scipy.spatial import distance

import torch
import torch.nn as nn 

def make_init_mask(model, is_print=False):
    '''
    Makes the initial pruning mask for the given model. For example, for LeNet-5 architecture it return a list of
    5 arrays, each array is the same size of each layer's weights and with all 1 entries. We do not prune bias
    
    :param model: a pytorch model 
    :return mask: a list of pruning masks 
    '''
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 

    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            if is_print:
                print(f'step {step}, shape: {mask[step].shape}')
            step = step + 1
            
    return mask

def fake_prune(percent, model, mask):
    '''
    This function derives the new pruning mask, it put 0 for the weights under the given percentile
    
    :param percent: pruning percent 
    :param model: a pytorch model 
    :param mask: the pruning mask 
    
    :return mask: updated pruning mask
    '''
    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():
        
        # We do not prune bias term
        if 'weight' in name:
            weight_dev = param.device
            
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor * mask[step])] # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)
            
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply mask
            mask[step] = new_mask
            step += 1

    return mask

def real_prune(model, mask):
    '''
    This function applies the derived mask. It zeros the weights needed to be pruned based on the updated mask
    
    :param model: a pytorch model 
    :param mask: pruning mask 
    
    :return state_dict: updated (pruned) model state_dict
    '''
    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            weight_dev = param.device

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * mask[step]).to(weight_dev)
            step += 1
            
    return model.state_dict()

def print_pruning(model, is_print = False):
    '''
    This function prints the pruning percentage and status of a given model 
    
    :param model: a pytorch model 
    
    :return pruning percentage, number of remaining weights: 
    '''
    nonzero = 0
    total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
    if is_print: 
        print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total},'
              f'Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:3.2f}% pruned)')
    return 100 * (total-nonzero) / total, nonzero

def dist_masks(m1, m2): 
    '''
    Calculates hamming distance of two pruning masks. It averages the hamming distance of all layers and returns it
    
    :param m1: pruning mask 1 
    :param m2: pruning mask 2 
    
    :return average hamming distance of two pruning masks: 
    '''
    temp_dist = []
    for step in range(len(m1)): 
        #1 - float(m1[step].reshape([-1]) == m2[step].reshape([-1])) / len(m2[step].reshape([-1]))
        temp_dist.append(distance.hamming(m1[step].reshape([-1]), m2[step].reshape([-1])))
    dist = np.mean(temp_dist)
    return dist