import torch
from transformers import DistilBertModel, BertModel

class ContrastiveModel(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ContrastiveModel, self).__init__()
        self.original_transformer = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.translation_transformer = BertModel.from_pretrained('bert-base-chinese')
        self.original_linear_1 = torch.nn.Linear(201*768, 256)
        self.translation_linear_1 = torch.nn.Linear(201*768, 256)
        self.original_norm = torch.nn.BatchNorm1d(256)
        self.translation_norm = torch.nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout(p=0.3)

        self.final_linear_1 = torch.nn.Linear(512, 256)
        self.final_linear_2 = torch.nn.Linear(256, 1)

    def forward(self, original_input_ids=None, original_attention_mask=None, translation_input_ids=None, translation_attention_mask=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        original_attention = self.original_transformer(input_ids=original_input_ids, attention_mask=original_attention_mask)['last_hidden_state']
        translation_attention = self.translation_transformer(input_ids=translation_input_ids, attention_mask=translation_attention_mask)['last_hidden_state']

        original_encoded = torch.nn.ReLU()(self.original_norm(self.original_linear_1(torch.flatten(original_attention, start_dim=1))))
        translation_encoded = torch.nn.ReLU()(self.translation_norm(self.translation_linear_1(torch.flatten(translation_attention, start_dim=1))))

        y_pred = self.final_linear_2(torch.nn.ReLU()(self.final_linear_1(torch.cat([original_encoded, translation_encoded], 1))))
        return y_pred