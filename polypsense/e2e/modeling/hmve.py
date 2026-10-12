import torch
import torch.nn as nn
import torchvision.models as models


class SFEv3(nn.Module):
    def __init__(self, backbone_arch, backbone_weights):
        super().__init__()

        self.backbone_arch = backbone_arch
        self.backbone_weights = backbone_weights

        resnet_dict = {
            "resnet18": lambda: models.resnet18(weights=self.backbone_weights),
            "resnet50": lambda: models.resnet50(weights=self.backbone_weights),
        }

        if self.backbone_arch not in resnet_dict:
            raise ValueError(
                f"Invalid '{self.backbone_arch}' backbone. Use one of: {list(resnet_dict.keys())}"
            )

        self.backbone = resnet_dict[self.backbone_arch]()
        self.backbone.fc = nn.Identity()  # remove fc

    def forward(self, x):
        if not x.dim() == 4:
            raise ValueError(
                f"Input tensor must have 4 dimensions [b, c, h, w], got input with shape {x.shape}"
            )

        x = self.backbone(x)  # [b, d]

        return x  # [b, d]


class MVEv3(nn.Module):
    def __init__(
        self,
        sfe,
        d_in,
        d_model,
        d_feedforward,
        dropout,
        n_heads,
        n_layers,
        use_feature_norm,
        use_encoder_norm,
    ):
        super().__init__()

        self.sfe = sfe

        self.adapter = nn.Linear(d_in, d_model)
        self.feature_norm = nn.LayerNorm(d_model) if use_feature_norm else nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # [1, 1, d]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        encoder_norm = nn.LayerNorm(d_model) if use_encoder_norm else None
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=encoder_norm,
        )

    def forward(self, x):
        b, s, c, h, w = x.shape

        x = x.reshape(b * s, c, h, w)  # [b*s, c, h, w]
        x = self.sfe(x)  # [b*s, d_in]

        x = x.reshape(b, s, -1)  # [b, s, d_in]
        x = self.adapter(x)  # [b, s, d_model]
        x = self.feature_norm(x)  # [b, s, d_model]

        x = self._prepend_cls_token(x)  # [b, 1+s, d_model]
        x = self.encoder(x)  # [b, 1+s, d_model]

        return x  # [b, 1+s, d_model]

    def _prepend_cls_token(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [b, 1, d]
        return torch.cat((cls_tokens, x), dim=1)  # [b, 1+s, d]


