import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pos_encoding = torch.zeros(max_seq_len, d_model)
        
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, n_features, n_classes, d_model=512, n_heads=8, n_encoder_layers=6, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        self.encoder = nn.Linear(n_features, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True),
            num_layers=n_encoder_layers
        )
        
        self.decoder = nn.Linear(d_model, n_classes)
    
    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.decoder(output)
        
        return output
