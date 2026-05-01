"""
SHAR Models:
  - Phase 3: CNN Baseline (ResNet50 on mean-pooled frames)
  - Phase 4: CNN-LSTM (ResNet50 feature extractor + LSTM)
  - Phase 5: Lightweight CNN-GRU variant
"""
import torch
import torch.nn as nn
from torchvision import models

# ─── Phase 3: CNN Baseline ────────────────────────────────
class CNNBaseline(nn.Module):
    """Frame-level ResNet50 with temporal mean pooling."""
    def __init__(self, num_classes=21, dropout=0.5, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # (B*T, 2048, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)
        feats = feats.view(B, T, -1).mean(dim=1)                   # (B, 2048) — temporal mean
        return self.classifier(feats)

# ─── Phase 4: CNN-LSTM ────────────────────────────────────
class CNNLSTM(nn.Module):
    """ResNet50 CNN encoder + LSTM temporal model — main model."""
    def __init__(self, num_classes=21, hidden_size=256, num_layers=2,
                 dropout=0.5, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        # Freeze early layers
        for param in list(self.cnn.parameters())[:6*4]:
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        lstm_out = hidden_size * 2  # bidirectional
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # CNN feature extraction
        x = x.view(B * T, C, H, W).float() # Force input to float32
        
        # Use modern amp.autocast
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=True):
            feats = self.cnn(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)
        
        # CRITICAL: Convert to float32 for LSTM
        feats = feats.view(B, T, -1).to(torch.float32)
        
        # LSTM
        lstm_out, _ = self.lstm(feats)                   # (B, T, hidden*2)
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)   # (B, hidden*2)
        return self.classifier(context)

# ─── Phase 5: CNN-GRU (lighter) ──────────────────────────
class CNNGRU(nn.Module):
    """MobileNetV3 + GRU — faster training, lighter memory."""
    def __init__(self, num_classes=21, hidden_size=256, num_layers=2,
                 dropout=0.4, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 576  # MobileNetV3-small output channels

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.pool(self.cnn(x)).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)
        gru_out, _ = self.gru(feats)
        out = gru_out[:, -1, :]  # last time step
        return self.classifier(out)

def get_model(model_name="cnn_lstm", num_classes=21, **kwargs):
    models_map = {
        "cnn_baseline": CNNBaseline,
        "cnn_lstm": CNNLSTM,
        "cnn_gru": CNNGRU,
    }
    if model_name not in models_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_map.keys())}")
    return models_map[model_name](num_classes=num_classes, **kwargs)
