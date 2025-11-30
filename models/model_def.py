"""Complex sequence predictor combining 1D convolution, LSTM, and a simple attention mechanism.
"""

import torch
from torch import nn
from typing import Optional

class ComplexSeqPredictor(nn.Module):
    """
    Compact but expressive architecture for short time-series forecasting.

    Structure:
      - 1D convolutional front-end to learn local temporal patterns
      - LSTM encoder to capture longer context
      - simple attention pooling over LSTM outputs
      - MLP head for final regression output
    """

    def __init__(self,
                 input_len: int = 8,
                 conv_channels: int = 32,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 1,
                 attn_dim: int = 32,
                 fc_hidden: int = 64,
                 out_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        # convolutional front-end (expects input shaped (batch, seq_len, 1) or (batch, seq_len))
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=input_len)  # keep sequence length for LSTM
        )

        # LSTM encoder (batch_first=True => input shape (batch, seq_len, features))
        self.lstm = nn.LSTM(input_size=conv_channels,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=False)

        # simple attention projection
        self.attn_proj = nn.Linear(lstm_hidden, attn_dim)
        self.attn_score = nn.Linear(attn_dim, 1, bias=False)

        # MLP regression head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, out_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
          x: float tensor of shape (batch, seq_len) or (batch, seq_len, 1)
          mask: optional mask (batch, seq_len) where 1 indicates valid timesteps

        Returns:
          Tensor of shape (batch, out_dim)
        """
        # ensure shape (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # conv expects (batch, channels=1, seq_len)
        x_conv = x.permute(0, 2, 1)
        conv_out = self.conv(x_conv)  # (batch, conv_channels, seq_len)

        # LSTM expects (batch, seq_len, features)
        lstm_in = conv_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)  # (batch, seq_len, lstm_hidden)

        # attention pooling
        proj = torch.tanh(self.attn_proj(lstm_out))    # (batch, seq_len, attn_dim)
        scores = self.attn_score(proj).squeeze(-1)     # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        pooled = (weights * lstm_out).sum(dim=1)  # (batch, lstm_hidden)

        return self.head(pooled)
