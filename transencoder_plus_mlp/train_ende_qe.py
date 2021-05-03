import numpy as np
import csv
import torch

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

import sys
sys.path.append('/data/rgur/courses/cs_7643_deep_learning/hw4/')

# Just run this block. Please do not modify the following code.
import math
import time

# Pytorch package
import torch
import torch.nn as nn
import torch.optim as optim

# Torchtest package
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, Example, TabularDataset, interleave_keys, LabelField

# Tqdm progress bar
from tqdm import tqdm_notebook, tqdm

# Code provide to you for training and evaluation
#from hw4_code.utils import train, evaluate, set_seed_nb, unit_test_values
from utils import qe_train, qe_evaluate

import importlib

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

# You don't need to modify any code in this block

# Define the maximum length of the sentence. Shorter sentences will be padded to that length and longer sentences will be croped. Given that the average length of the sentence in the corpus is around 13, we can set it to 20
MAX_LEN = 20

# Define the source and target language
SRC = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            fix_length = MAX_LEN,
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            fix_length = MAX_LEN,
            lower = True)

Z = LabelField(dtype=torch.float, batch_first=True, use_vocab=False)

train_data = TabularDataset(path='/data/rgur/courses/cs_7643_deep_learning/project/mlqe/data/en-de/train.ende.df.short.tsv',format='TSV',fields={'original':('src',SRC),
'translation':('trg',TRG),
'z_mean':('z',Z)})

val_data = TabularDataset(path='/data/rgur/courses/cs_7643_deep_learning/project/mlqe/data/en-de/dev.ende.df.short.tsv',format='TSV',fields={'original':('src',SRC),
'translation':('trg',TRG),
'z_mean':('z',Z)})

# Define Batchsize
BATCH_SIZE = 128

# Build the vocabulary associated with each language
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

# Get the padding index to be ignored later in loss calculation
PAD_IDX = TRG.vocab.stoi['<pad>']

# Get data-loaders using BucketIterator
train_loader = BucketIterator( #Defines an iterator that batches examples of similar lengths together.
    train_data,
    batch_size = BATCH_SIZE, device = device, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))

val_loader = BucketIterator( #Defines an iterator that batches examples of similar lengths together.
    val_data,
    batch_size = BATCH_SIZE, device = device, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))

# Get the input and the output sizes for model
input_size = len(SRC.vocab)
output_size = len(TRG.vocab)

class QEModel(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size=256*20, p=0):
        super(QEModel, self).__init__()
        self.lin1 = nn.Linear(256*20,1028)
        self.lin2 = nn.Linear(1028,1)
        self.p = p
    
    def forward(self, x):
        x = nn.functional.relu( self.lin1(x) )
        x = nn.functional.dropout(x,p=self.p)
        return self.lin2(x)

#importlib.reload(hw4_code.models.Transformer)
from hw4_code.models.Transformer import TransformerTranslator
source_model = TransformerTranslator(output_size, input_size, device, max_length = MAX_LEN).to(device)
target_model = TransformerTranslator(input_size, output_size, device, max_length = MAX_LEN).to(device)

source_model.load_state_dict(torch.load('/data/rgur/courses/cs_7643_deep_learning/project/de_en.pt'))
target_model.load_state_dict(torch.load('/data/rgur/courses/cs_7643_deep_learning/project/en_de.pt'))

for learning_rate in [.05, .01, .001, .0001]:
    for P in [.2, .3, .5]:
        print('\nlearning_rate', learning_rate)
        print('P', P, '\n')
        # Hyperparameters
        EPOCHS = 10

        # Model
        qe_model = QEModel(p=P).to(device)

        # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        optimizer = torch.optim.Adam(qe_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        criterion = nn.MSELoss()

        for epoch_idx in range(EPOCHS):
            print("-----------------------------------")
            print("Epoch %d" % (epoch_idx+1), flush=True)
            print("-----------------------------------")
            
            train_loss, avg_train_loss = qe_train(qe_model, source_model, target_model, train_loader, optimizer, criterion, scheduler = None)
            train_loss = train_loss.item()
            scheduler.step(train_loss)

            #val_loss, avg_val_loss = evaluate(trans_model, val_loader, criterion)

            avg_train_loss = avg_train_loss.item()
            val_loss, avg_val_loss, r2, mae = qe_evaluate(qe_model, source_model, target_model, val_loader, criterion)
            #avg_val_loss = 0
            avg_val_loss = avg_val_loss.item()

            print("Training Loss: %.4f. Validation RMSE: %.4f. Validation R2: %.4f. Validation MAE %.4f" % (np.sqrt(avg_train_loss), np.sqrt(avg_val_loss), r2, mae))