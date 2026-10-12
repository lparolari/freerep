import torch
import torch.nn as nn

from polypsense.downstream.pooler import get_pooler
from polypsense.e2e.metric import accuracy, get_map_score
from polypsense.e2e.zoo import create_encoder
from polypsense.e2e.trainer import Trainer
from polypsense.upstream.tpc.criterion import TPCLoss


class TPCTrainer(Trainer):
    def __init__(
        self,
        encoder_type=None,
        encoder_cfg=None,
        d_model=None,
        d_proj=None,
        pooling=None,
        loss_temperature=None,
        opt_lr=None,
        opt_weight_decay=None,
    ):
        super().__init__()

        self.opt_lr = opt_lr
        self.opt_weight_decay = opt_weight_decay
        self.encoder_type = encoder_type
        self.encoder_cfg = encoder_cfg

        self.encoder = create_encoder(encoder_type, encoder_cfg)

        self.pooler = get_pooler(pooling, in_dim=d_proj)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

        self.criterion = TPCLoss(
            loss_temperature,
            return_logits=True,
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x, tgt = batch  # [b, s, c, h, w]
        y = tgt["identity_id"]  # [b]

        h = self(x)  # [b, s, d]
        z = self.projector(h)  # [b, s, p]

        z_pool = self.pooler(z)  # [b, d]

        loss, logits = self.criterion(z_pool, y)

        self.log(f"train_loss", loss, prog_bar=True)
        top1, top5 = accuracy(logits, y, topk=(1, 5))
        self.log(f"train_acc_top1", top1)
        self.log(f"train_acc_top5", top5)
        preds = y[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
        map = get_map_score(y, preds, y)
        self.log(f"train_map", map)

        return loss
    
    def eval_step(self, batch, stage=None):
        x, tgt = batch  # [b, s, c, h, w]
        y = tgt["identity_id"]  # [b]

        h = self(x)  # [b, s, d]
        z = self.projector(h)  # [b, s, p]

        z_pool = self.pooler(z)  # [b, d]

        loss, logits = self.criterion(z_pool, y)

        if stage:
            self.log_eval(f"{stage}_loss", loss, prog_bar=True)
            top1, top5 = accuracy(logits, y, topk=(1, 5))
            self.log_eval(f"{stage}_acc_top1", top1)
            self.log_eval(f"{stage}_acc_top5", top5)
            preds = y[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
            map = get_map_score(y, preds, y)
            self.log_eval(f"{stage}_map", map)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_lr,
            weight_decay=self.opt_weight_decay,
        )
        return optimizer