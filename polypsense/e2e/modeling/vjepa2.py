import torch
from transformers import VJEPA2Model


class VJEPA2(torch.nn.Module):
    def __init__(
        self,
        pretrained_weights,
    ):
        super().__init__()

        self.model = VJEPA2Model.from_pretrained(pretrained_weights)
        self.model.train()

    def forward(self, x):
        # x [b, t, c, h, w]
        x = self.model(x).last_hidden_state  # [b, p, d]
        return x
