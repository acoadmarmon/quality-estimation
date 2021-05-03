import math
import time
import random

# Pytorch packages
import torch
import torch.optim  as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm_notebook

import sklearn.metrics as metrics

def qe_train(qe_model, source_model, target_model, dataloader, optimizer, criterion, scheduler = None):
    '''
    source_model: translates target --> source. Can get the source embedding
    target_model: translates source --> target. Can get the target embedding
    '''
    
    qe_model.train()
    source_model.eval()
    target_model.eval()
    #model = model.cuda()

    # Record total loss
    total_loss = 0.0

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii = True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        optimizer.zero_grad()
        source = data.src.transpose(1,0)
        target = data.trg.transpose(1,0)
        z = data.z

        source_emb = source_model.encode(target)
        target_emb = target_model.encode(source)
        emb = torch.cat((source_emb,target_emb),dim=-1)
        emb = torch.flatten(emb,start_dim=1)
        output = qe_model(emb).view(len(z),)
        # if batch_idx == 0:
        #     print(output.shape)
        #     print(z.shape)
        #     print('preds:', output[0:5])
        #     print('label:', z[0:5])
        loss = criterion(output, z)
        loss.backward()
        optimizer.step()

        total_loss += loss
        progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx+1), loss.item()))
    
    return total_loss, total_loss / len(dataloader)

def qe_evaluate(qe_model, source_model, target_model, dataloader, criterion):
    '''
    source_model: translates target --> source. Can get the source embedding
    target_model: translates source --> target. Can get the target embedding
    '''
    
    qe_model.eval()
    source_model.eval()
    target_model.eval()
    #model = model.cuda()

    # Record total loss
    total_loss = 0.0

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii = True)

    # Mini-batch training
    ys = []
    yhats = []
    for batch_idx, data in enumerate(progress_bar):
        source = data.src.transpose(1,0)
        target = data.trg.transpose(1,0)
        z = data.z

        source_emb = source_model.encode(target)
        target_emb = target_model.encode(source)
        emb = torch.cat((source_emb,target_emb),dim=-1)
        emb = torch.flatten(emb,start_dim=1)
        output = qe_model(emb).view(len(z),)
        loss = criterion(output, z)
        if batch_idx == 0:
            print('preds:', output[0:5])
            print('label:', z[0:5])
        total_loss += loss
        progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx+1), loss.item()))
        ys += z.cpu().numpy().tolist()
        yhats += output.detach().cpu().numpy().tolist()
    
    r2 = metrics.r2_score(ys, yhats)
    mae = metrics.mean_absolute_error(ys, yhats)
    return total_loss, total_loss / len(dataloader), r2, mae