import torch.nn as nn
from CFG import CFG
import torch
class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        if x.dim() == 3:
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            out = self.dropout(out)
            return out
        elif x.dim() == 2:
            out, _ = self.lstm(x)
            out = self.fc(out[-1, :])
            out = self.dropout(out)
            return out

