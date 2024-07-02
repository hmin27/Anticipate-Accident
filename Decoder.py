import torch
from torch import nn

from Model import Attention


class DSA_LSTM(nn.Module):
  def __init__(self, device):
    super(DSA_LSTM, self).__init__()
    self.encoder_dim = 1000

    self.device = device
    self.dsa = Attention(self.encoder_dim).to(self.device)
    self.lstm = nn.LSTM(input_size = self.encoder_dim,
                        hidden_size = self.encoder_dim,
                        batch_first = True).to(self.device)
                        # batch_first = [batch, time_step, input]

    self.prediction = nn.Linear(self.encoder_dim, 1).to(self.device)
    self.sigmoid = nn.Sigmoid()

  def forward(self, features):
    h0 = torch.randn(1, features.size(0), self.encoder_dim).to(self.device)
    c0 = torch.randn(1, features.size(0), self.encoder_dim).to(self.device)

    phi = self.dsa(features, h0)
    out, _ = self.lstm(phi, (h0, c0))
    out = self.prediction(out)
    out = self.sigmoid(out)
    
    return out