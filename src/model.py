import torch.nn as nn


class STL10LiteVGG(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int, dropout_p: float) -> nn.Sequential:
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
            conv_block(3, 32, 0.05),
            conv_block(32, 64, 0.10),
            conv_block(64, 128, 0.15),
            conv_block(128, 256, 0.20),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
