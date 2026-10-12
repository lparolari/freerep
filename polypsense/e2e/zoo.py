import logging

import lightning as L
import torch

from polypsense.e2e.modeling.dinov2 import DinoV2
from polypsense.e2e.modeling.dinov3 import DinoV3
from polypsense.e2e.modeling.endofm import get_endofm
from polypsense.e2e.modeling.endofmlv import get_endofmlv
from polypsense.e2e.modeling.endovit import get_endovit
from polypsense.e2e.modeling.hmve import MVEv3 as MVE
from polypsense.e2e.modeling.hmve import SFEv3 as SFE
from polypsense.e2e.modeling.surgenet import SurgenetBackbone
from polypsense.e2e.modeling.vjepa2 import VJEPA2


class MveEncoder(L.LightningModule):
    def __init__(
        self,
        *,
        backbone_arch=None,
        backbone_weights=None,
        d_in=None,
        d_model=None,
        d_feedforward=None,
        n_heads=None,
        n_layers=None,
        dropout=None,
        use_feature_norm=None,
        use_encoder_norm=None,
    ):
        super().__init__()

        sfe = SFE(
            backbone_arch=backbone_arch,
            backbone_weights=backbone_weights,
        )

        self.mve = MVE(
            sfe=sfe,
            d_in=d_in,
            d_model=d_model,
            d_feedforward=d_feedforward,
            dropout=dropout,
            n_heads=n_heads,
            n_layers=n_layers,
            use_feature_norm=use_feature_norm,
            use_encoder_norm=use_encoder_norm,
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.mve(x)


class EndofmEncoder(L.LightningModule):
    def __init__(self, *, pretrained_weights=None):
        super().__init__()

        self.endofm = get_endofm(ckpt_path=pretrained_weights)

        self.save_hyperparameters()

    def forward(self, x):
        return self.endofm(x)


class EnfofmlvEncoder(L.LightningModule):
    def __init__(self, *, pretrained_weights=None):
        super().__init__()

        self.endofmlv = get_endofmlv(ckpt_path=pretrained_weights)

        self.save_hyperparameters()

    def forward(self, x):
        return self.endofmlv(x)


class EndovitEncoder(L.LightningModule):
    def __init__(self, *, pretrained_weights=None):
        super().__init__()

        self.endovit = get_endovit(ckpt_path=pretrained_weights)

        self.save_hyperparameters()

    def forward(self, x):
        return self.endovit(x)


class SurgenetEncoder(L.LightningModule):
    def __init__(self, *, backbone_arch="caformer_s18", pretrained_weights=None):
        super().__init__()

        self.surgenet = SurgenetBackbone(
            backbone_arch=backbone_arch,
            pretrained_weights=pretrained_weights,
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.surgenet(x)


class Dinov2Encoder(L.LightningModule):
    def __init__(self, *, pretrained_weights=None):
        super().__init__()

        self.dinov2 = DinoV2(pretrained_weights=pretrained_weights)

        self.save_hyperparameters()

    def forward(self, x):
        return self.dinov2(x)


class Dinov3Encoder(L.LightningModule):
    def __init__(self, *, pretrained_weights=None):
        super().__init__()

        self.dinov3 = DinoV3(pretrained_weights=pretrained_weights)

        self.save_hyperparameters()

    def forward(self, x):
        return self.dinov3(x)


class VJEPA2Encoder(L.LightningModule):
    def __init__(self, *, pretrained_weights=None):
        super().__init__()

        self.vjepa2 = VJEPA2(pretrained_weights=pretrained_weights)

        self.save_hyperparameters()

    def forward(self, x):
        return self.vjepa2(x)


class EncoderFactory:
    encoder_class = {
        "mve": MveEncoder,
        "endofm": EndofmEncoder,
        "endofmlv": EnfofmlvEncoder,
        "endovit": EndovitEncoder,
        "surgenet": SurgenetEncoder,
        "dinov2": Dinov2Encoder,
        "dinov3": Dinov3Encoder,
        "vjepa2": VJEPA2Encoder,
    }

    def __init__(self, encoder_type):
        if encoder_type not in self.encoder_class:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.encoder_type = encoder_type

    def from_config(self, encoder_cfg):
        if self.encoder_type not in self.encoder_class:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

        return self.encoder_class[self.encoder_type](**encoder_cfg)

    def from_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        # we might want to fallback to "state_dict" if "encoder_state_dict" is
        # not present, but this is not required right now
        encoder = self.from_config(checkpoint["encoder_hyper_parameters"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        return encoder

    def build(self, encoder_cfg_or_ckpt):
        encoder_cfg, ckpt_path = self._validate_encoder_cfg_or_ckpt(encoder_cfg_or_ckpt)

        if encoder_cfg is not None:
            return self.from_config(encoder_cfg)

        if ckpt_path is not None:
            return self.from_checkpoint(ckpt_path)

    def _validate_encoder_cfg_or_ckpt(self, encoder_cfg_or_ckpt):
        if isinstance(encoder_cfg_or_ckpt, dict):
            return encoder_cfg_or_ckpt, None

        if isinstance(encoder_cfg_or_ckpt, str):
            if encoder_cfg_or_ckpt.endswith(".ckpt"):
                return None, encoder_cfg_or_ckpt
            else:
                logging.info(
                    f"The given checkpoint path is not an expected checkpoint (*.ckpt): {encoder_cfg_or_ckpt}. "
                    f"Continuing with pretrained weights."
                )
                return {"pretrained_weights": encoder_cfg_or_ckpt}, None

        raise ValueError(f"Expected dict or str, but got {type(encoder_cfg_or_ckpt)}")


def create_encoder(encoder_type, encoder_cfg_or_ckpt):
    return EncoderFactory(encoder_type).build(encoder_cfg_or_ckpt)
