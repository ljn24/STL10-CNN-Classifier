import torch.nn as nn


def _make_classifier(in_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=0.25),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(256, num_classes),
    )


class BaselineModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def conv_block(in_ch, out_ch, dropout_p):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=dropout_p),
            )

        self.features = nn.Sequential(
            conv_block(3, 64, 0.02),
            conv_block(64, 128, 0.05),
            conv_block(128, 256, 0.10),
            conv_block(256, 512, 0.15),
        )
        self.classifier = _make_classifier(512, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class ResidualSEBlock(nn.Module):
    """Conv-BN-ReLU x2 + optional residual shortcut + optional SE attention."""

    def __init__(self, in_ch, out_ch, dropout_p,
                 use_residual: bool = True, use_se: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if use_residual:
            self.shortcut = (
                nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
                if in_ch != out_ch else nn.Identity()
            )
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.convs(x)
        if self.use_residual:
            out = out + self.shortcut(x)
        return self.dropout(self.pool(self.se(self.act(out))))


class AdvancedModel(nn.Module):
    def __init__(self, num_classes=10,
                 use_residual: bool = True, use_se: bool = True):
        super().__init__()
        self.features = nn.Sequential(
            ResidualSEBlock(3, 64, 0.02, use_residual, use_se),
            ResidualSEBlock(64, 128, 0.05, use_residual, use_se),
            ResidualSEBlock(128, 256, 0.10, use_residual, use_se),
            ResidualSEBlock(256, 512, 0.15, use_residual, use_se),
        )
        self.classifier = _make_classifier(512, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg: dict) -> nn.Module:
    model_type = cfg["model_type"]
    if model_type == "baseline":
        return BaselineModel()
    if model_type == "advanced":
        model_cfg = cfg.get("model", {})
        return AdvancedModel(
            use_residual=model_cfg.get("use_residual", True),
            use_se=model_cfg.get("use_se", True),
        )
    raise ValueError(f"Unknown model_type: {model_type}")
