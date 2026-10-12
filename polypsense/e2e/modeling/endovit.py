import logging
from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn


class EndoViT(VisionTransformer):
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.reshape(b * s, c, h, w)

        x = self.forward_features(x)  # [b * s, 1 + p, d]

        x = x.reshape(b, s, x.size(1), x.size(2))  # [b, s, 1 + p, d]

        # pooling over s
        x = x.mean(dim=1)  # [b, 1 + p, d]

        return x


def get_endovit(ckpt_path=None):
    model = EndoViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path, weights_only=False)
        state_dict = state_dict["model"]

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            logging.warning(f"EndoViT model loaded")
            logging.warning(f"  Missing keys: {missing_keys}")
            logging.warning(f"  Unexpected keys: {unexpected_keys}")
        else:
            logging.info("EndoViT model loaded successfully")

    return model
