import logging
from functools import partial

import torch
import torch.nn as nn

from polypsense.e2e.modeling.endofmlv import EndoFMLV


EndoFM = EndoFMLV  # the two models share the same architecture


def get_endofm(ckpt_path=None):
    model = EndoFMLV(
        img_size=224,
        num_classes=0,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=8,  # the only change with respect to EndoFM-LV!
        attention_type="divided_space_time",
    )

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # The paper is opaque about which weights to load, and the code as well
        # does not clarify it because it allows to use both teacher or student.
        # However, all the defaults point to the teacher and the bash script
        # does make any reference to student.
        # Moreover, in common practice (i.e. DINO) the teacher weights are used:
        # https://github.com/facebookresearch/dino/issues/44

        teacher_state_dict = checkpoint["teacher"]
        teacher_state_dict = {
            x[len("backbone.") :]: y
            for x, y in teacher_state_dict.items()
            if x.startswith("backbone.")
        }

        missing_keys, unexpected_keys = model.load_state_dict(teacher_state_dict)

        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            logging.warning(f"EndoFM model loaded")
            logging.warning(f"  Missing keys: {missing_keys}")
            logging.warning(f"  Unexpected keys: {unexpected_keys}")
        else:
            logging.info("EndoFM model loaded successfully")

    return model
