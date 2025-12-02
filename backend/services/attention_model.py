from pathlib import Path
from typing import Any, Dict, Tuple

import timm
import torch
import torch.nn as nn
from torchvision import transforms


class LightAttentionModel(nn.Module):
    """
    Modelo baseado em backbone CNN + LSTM + classifier linear.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        hidden_size: int = 64,
        img_size: int = 224,
    ):
        super().__init__()

        self.cnn = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0,
            global_pool="",
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            features = self.cnn(dummy)
            self.feature_size = features.shape[1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch_size * seq_len, *x.shape[2:])
        features = self.cnn(x)
        features = self.pool(features).squeeze(-1).squeeze(-1)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]

        return self.classifier(lstm_out)


def create_image_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_attention_model(
    config: Dict[str, Any],
    weights_path: Path,
    device: torch.device,
) -> Tuple[LightAttentionModel, Any]:
    """
    Constrói e carrega o modelo de atenção com os pesos treinados.
    """
    model = LightAttentionModel(
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        hidden_size=config["hidden_lstm"],
        img_size=config["img_size"],
    )
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = create_image_transform(config["img_size"])
    return model, transform

