import torch
from transformers import Dinov2Model


class DinoV2(torch.nn.Module):
    def __init__(
        self,
        pretrained_weights,
    ):
        super().__init__()

        self.model = Dinov2Model.from_pretrained(pretrained_weights)
        self.model.train()

    def forward(self, x):
        # x [b, t, c, h, w]
        b, t, c, h, w = x.shape
        x = x.flatten(0, 1)  # [b * t, c, h, w]
        x = self.model(x).last_hidden_state  # [b * t, 1 + p, d]
        x = x.view(b, t, -1, x.shape[-1])  # [b, t, 1 + p, d]
        return x  # [b, t, 1 + p, d]
