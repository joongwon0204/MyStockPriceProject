from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class StockPredictor(nn.Module):
    def __init__(self, input_size=6, conv_size=32, rnn_size=64, kernels=(1,3,5,9,19)):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv1d(input_size, conv_size, k, padding=k//2) for k in kernels
        ])
        self.mix1 = nn.Conv1d(conv_size * len(kernels), conv_size, 1)
        self.drop1 = nn.Dropout(0.3)

        self.convs2 = nn.ModuleList([
            nn.Conv1d(conv_size, conv_size, k, padding=k//2) for k in kernels
        ])
        self.mix2 = nn.Conv1d(conv_size * len(kernels), conv_size, 1)
        self.drop2 = nn.Dropout(0.3)

        self.rnn = nn.GRU(input_size=conv_size, hidden_size=rnn_size, batch_first=True)
        self.fc = nn.Linear(rnn_size, 1)

    def forward(self, x): # (B, C, L)
        z = torch.cat([torch.relu(c(x)) for c in self.convs1], dim=1)
        z = torch.relu(self.mix1(z)) # (B, conv_size, L)
        z = self.drop1(z)
        
        z = torch.cat([torch.relu(c(z)) for c in self.convs2], dim=1)
        z = torch.relu(self.mix2(z))  # (B, conv_size, L)
        z = self.drop2(z)

        z = z.permute(0, 2, 1)        # (B, L, conv_size)
        out, _ = self.rnn(z)
        last = out[:, -1, :]          # (B, rnn_size)
        logits = self.fc(last).squeeze(-1)  # (B,)
        return logits