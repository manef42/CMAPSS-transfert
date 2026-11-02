"""Only LSTM/LSTM-DANN. No domain-head execution on supervised inference."""
import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class RULModel(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, num_layers=2, dropout=.2, head_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.)
        def head():
            return nn.Sequential(nn.Linear(hidden_size, head_size), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(head_size, 1))
        self.rul_head, self.domain_head = head(), head()

    def encode(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]

    def predict_rul(self, x):
        return self.rul_head(self.encode(x)).squeeze(-1)

    def forward(self, x):
        return self.predict_rul(x)

    def forward_dann(self, x, grl_coefficient):
        z = self.encode(x)
        return self.rul_head(z).squeeze(-1), self.domain_head(
            GradientReversal.apply(z, grl_coefficient)).squeeze(-1)

    def reset_domain_head(self):
        for m in self.domain_head.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
