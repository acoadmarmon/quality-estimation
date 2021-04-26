from transformers import XLMRobertaModel
import torch
from torch import nn


class MonoTransQuest(nn.Module):

    def __init__(self, config):
        super(MonoTransQuest, self).__init__()
        self.model = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        x = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return torch.sigmoid(x)