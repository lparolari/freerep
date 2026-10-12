import torch
from transformers import DINOv3ViTModel


class DinoV3(torch.nn.Module):
    """
    Implement DinoV3 model.

    Shapes for ViT small are as follows:
    - 1 cls token
    - 4 register tokens
    - 196 patch tokens (14x14)
     => total 201 tokens
    - d = 384
    """

    def __init__(
        self,
        pretrained_weights,
    ):
        super().__init__()

        self.model = DINOv3ViTModel.from_pretrained(pretrained_weights)
        self.model.train()

    def forward(self, x):
        # x [b, t, c, h, w]
        b, t, c, h, w = x.shape
        x = x.flatten(0, 1)  # [b * t, c, h, w]
        x = self.model(x).last_hidden_state  # [b * t, 1 + r + p, d]
        x = x.view(b, t, -1, x.shape[-1])  # [b, t, 1 + r + p, d]
        x = remove_registers(x)  # [b, t, 1 + p, d]
        return x  # [b, t, 1 + p, d]


def remove_registers(x):
    cls_token = x[..., :1, :]  # [..., 1, d]
    patch_tokens = x[..., 1 + 4 :, :]  # [..., p, d]
    # p = the number of patch tokens (e.g. 196 for ViT small)
    return torch.cat((cls_token, patch_tokens), dim=-2)  # [..., 1 + p, d]
