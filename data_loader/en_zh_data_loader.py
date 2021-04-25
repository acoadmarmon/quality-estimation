from pathlib import Path
import pandas as pd
import os
from transformers import DistilBertTokenizerFast
import torch

def read_en_de_split(data_dir, train=True):
    data = None
    if train:
        data = pd.read_csv(os.path.join(data_dir, 'train.enzh.df.short.tsv'), sep='\t')
    else:
        data = pd.read_csv(os.path.join(data_dir, 'dev.enzh.df.short.tsv'), sep='\t')
    #data.astype({'mean': 'float32'})
    texts = (list(data['original']), list(data['translation']))
    # print(data['mean'])
    # for i in list(data['mean']):
    #     if i[0] == '[':
    #         print(list(data['mean']).index(i))
    qualities = list(data['z_mean'].astype('float32'))
    return texts, qualities

def get_train_test_embeddings():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    de_tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-chinese')
    #de_tokenizer = bert-base-german-dbmdz-cased

    train_texts, train_qualities = read_en_de_split('./data/en-zh')
    test_texts, test_qualities = read_en_de_split('./data/en-zh', train=False)

    print(max([len(i) for i in train_texts[0]]), max([len(i) for i in train_texts[1]]))
    train_encodings = ((tokenizer(train_texts[0], max_length=201, truncation=True, padding='max_length'), de_tokenizer(train_texts[1], max_length=201, truncation=True, padding='max_length')), train_qualities)
    test_encodings = ((tokenizer(test_texts[0], max_length=201, truncation=True, padding='max_length'), de_tokenizer(test_texts[1], max_length=201, truncation=True, padding='max_length')), test_qualities)
    return train_encodings, test_encodings


class EnZhDataset(torch.utils.data.Dataset):
    def __init__(self, original_encodings, translation_encodings, qualities):
        self.original_encodings = original_encodings
        self.translation_encodings = translation_encodings
        self.qualities = qualities

    def __getitem__(self, idx):
        item = {'original_' + key: torch.tensor(val[idx]) for key, val in self.original_encodings.items()}
        item.update({'translation_' + key: torch.tensor(val[idx]) for key, val in self.translation_encodings.items()})
        item['quality'] = torch.tensor(self.qualities[idx])
        return item

    def __len__(self):
        return len(self.qualities)