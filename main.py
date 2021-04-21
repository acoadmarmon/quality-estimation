import data_loader.en_de_data_loader as dl
import model.contrastive_model as cm

import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import datetime
import copy
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_embeddings, test_embeddings = dl.get_train_test_embeddings()

train_dataset = dl.EnDeDataset(train_embeddings[0][0], train_embeddings[0][1], train_embeddings[1])
test_dataset = dl.EnDeDataset(test_embeddings[0][0], test_embeddings[0][1], test_embeddings[1])
dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}

model = cm.ContrastiveModel().to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
optim = AdamW(model.parameters(), lr=1e-4)

num_epochs = 10
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = np.inf
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        
        loader = train_loader if phase == 'train' else test_loader
        for batch in loader:
            optim.zero_grad()

            original_input_ids = batch['original_input_ids'].to(device)
            original_attention_mask = batch['original_attention_mask'].to(device)

            translation_input_ids = batch['translation_input_ids'].to(device)
            translation_attention_mask = batch['translation_attention_mask'].to(device)

            qualities = batch['quality'].to(device)
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(original_input_ids=original_input_ids, original_attention_mask=original_attention_mask, translation_input_ids=translation_input_ids, 
                                translation_attention_mask=translation_attention_mask)
                loss = torch.nn.MSELoss()(outputs[:, 0], qualities)

                if phase == 'train':
                    loss.backward()
                    optim.step()
            
            running_loss += loss.item() * qualities.size(0)
        
        epoch_loss = running_loss / dataset_sizes[phase]

        print('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

torch.save(best_model_wts, 'best_model_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
