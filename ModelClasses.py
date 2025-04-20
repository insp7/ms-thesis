import torch
import torch.nn as nn

class MultiTaskEEGTransformer(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 conv_out_channels=256, a
                 transformer_dim=128, 
                 num_heads=4, 
                 num_layers=2, 
                 dropout=0.1):
        super(MultiTaskEEGTransformer, self).__init__()

        # EEG-specific preprocessing (same as EEGViT_raw)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv_out_channels,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(conv_out_channels, affine=False)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_out_channels,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Shared head
        self.shared_head = nn.Sequential(
            nn.Linear(conv_out_channels, 128),
            nn.ReLU()
        )

        # Task-specific heads
        self.amplitude_head = nn.Linear(128, 1)
        self.angle_head = nn.Linear(128, 1)

    def forward(self, x):
        print(x.shape)
        # Input shape: [B, 1, 129, 500]
        x = self.conv1(x)                      # → [B, 256, 129, 14]
        x = self.batchnorm1(x)                 # → [B, 256, 129, 14]
        x = x.squeeze(-1).permute(0, 2, 1)     # → [B, 129, 256] (sequence length = 129, embedding dim = 256)

        x = self.transformer(x)                # → [B, 129, 256]
        pooled = x.mean(dim=1)                 # Mean pooling across sequence → [B, 256]

        features = self.shared_head(pooled)    # → [B, 128]
        amplitude = self.amplitude_head(features)
        angle = self.angle_head(features)

        return amplitude, angle