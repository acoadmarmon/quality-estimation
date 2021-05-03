import numpy as np
import csv
import torch
import pandas as pd

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
from hw4_code.utils import train, evaluate, set_seed_nb, unit_test_values

import glob
import csv

from hw4_code.models.Transformer import TransformerTranslator
# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)
from utils import qe_train, qe_evaluate

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

folders = ['en-de', 'en-zh', 'et-en', 'ne-en', 'ro-en', 'si-en']
#folders = ['si-en']

for folder_name in folders:
    train_data_path = [file for file in glob.glob("mlqe/data/%s/train*.tsv" %folder_name)][0]
    val_data_path = [file for file in glob.glob("mlqe/data/%s/dev*.tsv" %folder_name)][0]
    test_data_path = [file for file in glob.glob("mlqe/data/%s/test*.tsv" %folder_name)][0] 

    for path in (train_data_path, val_data_path, test_data_path): #clean data
        print(path)
        df = pd.read_csv(path, delimiter='\t', error_bad_lines=False)
        df.replace('', np.nan, inplace=True)
        df.dropna(inplace=True)
        #df.to_csv(path, index=False, sep='\t', quoting=csv.QUOTE_NONE)
        df.to_csv(path, index=False, sep='\t')

for folder_name in folders:
    print('folder_name', folder_name, flush=True)
    lang_A = folder_name.split('-')[0]
    lang_B = folder_name.split('-')[1]

    MAX_LEN = 20
    # Define the source and target language
    SRC = Field(tokenize = "spacy",
                tokenizer_language="xx_ent_wiki_sm",
                init_token = '<sos>',
                eos_token = '<eos>',
                fix_length = MAX_LEN,
                lower = True)

    TRG = Field(tokenize = "spacy",
                tokenizer_language="xx_ent_wiki_sm",
                #tokenizer_language="de",
                init_token = '<sos>',
                eos_token = '<eos>',
                fix_length = MAX_LEN,
                lower = True)

    Z = LabelField(dtype=torch.float, batch_first=True, use_vocab=False)

    train_data_path = [file for file in glob.glob("mlqe/data/%s/train*.tsv" %folder_name)][0]
    val_data_path = [file for file in glob.glob("mlqe/data/%s/dev*.tsv" %folder_name)][0]
    test_data_path = [file for file in glob.glob("mlqe/data/%s/test*.tsv" %folder_name)][0] 

    train_data = TabularDataset(path=train_data_path,format='TSV',fields={'original':('src',SRC),
    'translation':('trg',TRG),
    'z_mean':('z',Z)})

    val_data = TabularDataset(path=val_data_path,format='TSV',fields={'original':('src',SRC),
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

    # Hyperparameters
    learning_rate = .001
    EPOCHS = 500

    # Model
    trans_model = TransformerTranslator(input_size, output_size, device, max_length = MAX_LEN).to(device)

    # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = torch.optim.Adam(trans_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_avg_val_loss = np.inf
    for epoch_idx in range(EPOCHS):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx+1))
        print("-----------------------------------")
        
        train_loss, avg_train_loss = train(trans_model, train_loader, optimizer, criterion)
        scheduler.step(train_loss)

        val_loss, avg_val_loss = evaluate(trans_model, val_loader, criterion)
        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item()
        print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
        print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " % (np.exp(avg_train_loss), np.exp(avg_val_loss)))

        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            torch.save(trans_model.state_dict(), '%s_%s.pt' %(lang_A, lang_B))
            print('Best Model Saved')

    # train second
    train_data = TabularDataset(path=train_data_path,format='TSV',fields={'translation':('src',SRC),
    'original':('trg',TRG),
    'z_mean':('z',Z)})

    val_data = TabularDataset(path=val_data_path,format='TSV',fields={'translation':('src',SRC),
    'original':('trg',TRG),
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

    # Hyperparameters
    learning_rate = .001
    EPOCHS = 500

    # Model
    trans_model = TransformerTranslator(input_size, output_size, device, max_length = MAX_LEN).to(device)

    # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = torch.optim.Adam(trans_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_avg_val_loss = np.inf
    for epoch_idx in range(EPOCHS):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx+1))
        print("-----------------------------------")
        
        train_loss, avg_train_loss = train(trans_model, train_loader, optimizer, criterion)
        scheduler.step(train_loss)

        val_loss, avg_val_loss = evaluate(trans_model, val_loader, criterion)
        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item()
        print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
        print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " % (np.exp(avg_train_loss), np.exp(avg_val_loss)))

        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            torch.save(trans_model.state_dict(), '%s_%s.pt' %(lang_B, lang_A))
            print('Best Model Saved')

    train_data_path = [file for file in glob.glob("mlqe/data/%s/train*.tsv" %folder_name)][0]
    val_data_path = [file for file in glob.glob("mlqe/data/%s/dev*.tsv" %folder_name)][0]
    test_data_path = [file for file in glob.glob("mlqe/data/%s/test*.tsv" %folder_name)][0] 

    #train qe model
    SRC = Field(tokenize = "spacy",
                tokenizer_language="xx_ent_wiki_sm",
                init_token = '<sos>',
                eos_token = '<eos>',
                fix_length = MAX_LEN,
                lower = True)

    TRG = Field(tokenize = "spacy",
                tokenizer_language="xx_ent_wiki_sm",
                #tokenizer_language="de",
                init_token = '<sos>',
                eos_token = '<eos>',
                fix_length = MAX_LEN,
                lower = True)

    Z = LabelField(dtype=torch.float, batch_first=True, use_vocab=False)

    train_data = TabularDataset(path=train_data_path,format='TSV',fields={'original':('src',SRC),
    'translation':('trg',TRG),
    'z_mean':('z',Z)})

    val_data = TabularDataset(path=val_data_path,format='TSV',fields={'original':('src',SRC),
    'translation':('trg',TRG),
    'z_mean':('z',Z)})

    test_data = TabularDataset(path=test_data_path,format='TSV',fields={'original':('src',SRC),
    'translation':('trg',TRG),
    'z_mean':('z',Z)})

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

    test_loader = BucketIterator( #Defines an iterator that batches examples of similar lengths together.
        test_data,
        batch_size = BATCH_SIZE, device = device, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))

    # Get the input and the output sizes for model
    input_size = len(SRC.vocab)
    output_size = len(TRG.vocab)

    source_model = TransformerTranslator(output_size, input_size, device, max_length = MAX_LEN).to(device)
    target_model = TransformerTranslator(input_size, output_size, device, max_length = MAX_LEN).to(device)

    source_model.load_state_dict(torch.load('%s_%s.pt' %(lang_B, lang_A)))
    target_model.load_state_dict(torch.load('%s_%s.pt' %(lang_A, lang_B)))

    EPOCHS = 2000
    P = 0.2
    learning_rate = .01

    # Model
    qe_model = QEModel(p=P).to(device)

    # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = torch.optim.Adam(qe_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.MSELoss()

    #train qe
    best_loss = np.inf
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
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(qe_model.state_dict(), '%s_%s_qe.pt' %(lang_A, lang_B))

        print("Training Loss: %.4f. Validation RMSE: %.4f. Validation R2: %.4f. Validation MAE %.4f" % (np.sqrt(avg_train_loss), np.sqrt(avg_val_loss), r2, mae))

    #predict on test set
    qe_model.load_state_dict(torch.load('%s_%s_qe.pt' %(lang_A, lang_B)))
    val_loss, avg_val_loss, r2, mae = qe_evaluate(qe_model, source_model, target_model, val_loader, criterion)
    with open('%s_%s.txt' %(lang_A, lang_B), 'w') as filehandle:
        for listitem in (val_loss, avg_val_loss, r2, mae):
            filehandle.write('%s\n' % listitem)
