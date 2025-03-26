import torch
import torch.nn as nn
import torch.nn.init as nn_init
from transformers import AutoModel

class UnicodeClassifier_v2(nn.Module):
    def __init__(self, model_name, num_classes, freeze_encoder = True, dropout_rate = 0.3):
        super(UnicodeClassifier_v2, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(self.encoder.config.hidden_size, 512)
        self.ac1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.ac2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)  # Large Unicode classification head

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)  # gamma to 1
                m.bias.data.zero_()     # beta to 0
            elif isinstance(m, nn.Linear):
                nn_init.xavier_uniform_(m.weight)  # Xavier initialization
                if m.bias is not None:
                    m.bias.data.zero_()  # bias to 0

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.ac1(self.fc1(outputs.last_hidden_state[:, 0, :]))
        outputs = self.dropout1(outputs)
        outputs = self.ac2(self.fc2(outputs))
        outputs = self.dropout2(outputs)
        logits = self.fc(outputs)  # CLS token output
        return logits