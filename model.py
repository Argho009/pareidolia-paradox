import torch.nn as nn
import timm


class PareidoliaModel(nn.Module):
    def __init__(self, model_name="tf_efficientnetv2_s", num_classes=2, drop_rate=0.3):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=3,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        return self.model(x)
