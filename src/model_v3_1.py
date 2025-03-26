import torch
import torch.nn as nn
import torch.nn.init as nn_init
from transformers import AutoModel

class UnicodeClassifier_v3_1(nn.Module):
    def __init__(self, model_name, num_classes, freeze_encoder=True, dropout_rate=0.3):
        super(UnicodeClassifier_v3_1, self).__init__()
        
        # Load transformer model
        self.encoder = AutoModel.from_pretrained(model_name)

        # LayerNorm for stable training
        self.norm = nn.LayerNorm(self.encoder.config.hidden_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self.encoder.config.hidden_size, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Normalize activations
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(256, num_classes)  # Final classification head

        # Option to freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        for param in self.encoder.encoder.layer[-3:].parameters():
            param.requires_grad = True

        self._initialize_weights()

    def _initialize_weights(self):
        """ Initialize weights for linear layers and batch norm layers """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn_init.xavier_uniform_(m.weight)  # Xavier/Glorot initialization
                if m.bias is not None:
                    nn_init.zeros_(m.bias)  # Initialize bias to zero
            elif isinstance(m, nn.BatchNorm1d):
                nn_init.ones_(m.weight)  # Set gamma to 1
                nn_init.zeros_(m.bias)   # Set beta to 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Transformer output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.norm(outputs.last_hidden_state[:, 0, :])  # CLS token with LayerNorm

        # Fully connected layers with normalization, activation, and dropout
        x = self.fc1(pooled_output)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout2(x)

        logits = self.fc(x)  # Final output

        return logits
