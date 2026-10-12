import lightning as L
import torch
import torch.nn as nn

from polypsense.downstream.pooler import get_pooler
from polypsense.e2e.loss import MILNCELoss, YAwareLoss
from polypsense.e2e.metric import accuracy, get_map_score
from polypsense.e2e.zoo import create_encoder


class Trainer(L.LightningModule):
    def on_save_checkpoint(self, checkpoint):
        checkpoint["encoder_state_dict"] = self.encoder.state_dict()
        checkpoint["encoder_hyper_parameters"] = dict(self.encoder.hparams)

    def log_eval(self, *args, **kwargs):
        is_distributed = (self.trainer.num_nodes or 1) * (self.trainer.num_devices or 1) > 1
        self.log(*args, **kwargs, sync_dist=is_distributed)


class TemporalSelfSupervisedTrainer(Trainer):
    """
    Trains a generic encoder using temporal self-supervision with MIL-NCE loss.
    """

    def __init__(
        self,
        encoder_type=None,
        encoder_cfg=None,
        d_model=None,
        d_proj=None,
        loss_temperature=None,
        opt_lr=None,
        opt_weight_decay=None,
        k=None,
        fragment_pooling=None,
    ):
        super().__init__()

        self.opt_lr = opt_lr
        self.opt_weight_decay = opt_weight_decay
        self.k = k
        self.encoder_type = encoder_type
        self.encoder_cfg = encoder_cfg

        # The original plan was to pass factory instead of constraining the the
        # list of available encoder to the ones implemted in `get_encoder`.
        # However, the save_hyperparameters method does not support saving
        # functions. This means that if we pass a function we won't be able to
        # use `TemporalSelfSupervisedTrainer.load_from_checkpoint(...)` anymore.
        self.encoder = create_encoder(encoder_type, encoder_cfg)

        self.fragment_pooler = get_pooler(fragment_pooling, in_dim=d_proj)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

        self.criterion = MILNCELoss(loss_temperature, return_logits=True)

        self.save_hyperparameters()

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch["clip"]  # [b, s, c, h, w]
        b = x.size(0)

        h = self(x)  # [b, s, d]
        z = self.projector(h)  # [b, s, p]

        z_fragment = self.fragment_pooler(z).unsqueeze(1)  # [b, s', d]

        y_fragment = torch.arange(b // self.k).repeat_interleave(self.k)  # [b]
        y_fragment = y_fragment.to(self.device)

        loss, logits = self.criterion(z_fragment.reshape(b, -1), y_fragment)

        identity = batch["identity"]  # [b]

        self.log(f"train_loss", loss, prog_bar=True)

        top1_xffragment, top5_xffragment = accuracy(logits, identity, topk=(1, 5))
        self.log(f"train_acc_top1", top1_xffragment)
        self.log(f"train_acc_top5", top5_xffragment)
        preds = identity[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
        map = get_map_score(identity, preds, identity)
        self.log(f"train_map", map)

        return loss

    def eval_step(self, batch, stage=None):
        x = batch["clip"]  # [b, s, c, h, w]
        identity = batch["identity"]  # [b]

        h = self(x)  # [b, s, d]
        z = self.projector(h)  # [b, s, p]

        z_fragment = self.fragment_pooler(z)  # [b, d]

        loss, logits = self.criterion(z_fragment, identity)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            top1, top5 = accuracy(logits, identity, topk=(1, 5))
            top1, top5 = accuracy(logits, identity, topk=(1, 5))
            self.log(f"{stage}_acc_top1", top1)
            self.log(f"{stage}_acc_top5", top5)
            preds = identity[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
            map = get_map_score(identity, preds, identity)
            self.log(f"{stage}_map", map)

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


class DistributedTemporalSelfSupervisedTrainer(Trainer):
    """
    Same as `TemporalSelfSupervisedTrainer` but works with distributed training.
    """

    def __init__(
        self,
        encoder_type=None,
        encoder_cfg=None,
        d_model=None,
        d_proj=None,
        loss_temperature=None,
        opt_lr=None,
        opt_weight_decay=None,
        k=None,
        fragment_pooling=None,
    ):
        super().__init__()

        self.opt_lr = opt_lr
        self.opt_weight_decay = opt_weight_decay
        self.k = k
        self.encoder_type = encoder_type
        self.encoder_cfg = encoder_cfg

        # The original plan was to pass factory instead of constraining the the
        # list of available encoder to the ones implemted in `get_encoder`.
        # However, the save_hyperparameters method does not support saving
        # functions. This means that if we pass a function we won't be able to
        # use `TemporalSelfSupervisedTrainer.load_from_checkpoint(...)` anymore.
        self.encoder = create_encoder(encoder_type, encoder_cfg)

        self.fragment_pooler = get_pooler(fragment_pooling, in_dim=d_proj)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

        self.criterion = MILNCELoss(loss_temperature, return_logits=True)

        self.save_hyperparameters()

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch["clip"]  # [b, s, c, h, w]
        b = x.size(0)

        h = self(x)  # [b, s, d]
        z = self.projector(h)  # [b, s, p]

        z_fragment = self.fragment_pooler(z).unsqueeze(1)  # [b, s', d]

        y_fragment = torch.arange(b // self.k).repeat_interleave(self.k)  # [b]
        y_fragment = y_fragment.to(self.device)

        loss, logits = self.criterion(z_fragment.reshape(b, -1), y_fragment)

        identity = batch["identity"]  # [b]

        self.log(f"train_loss", loss, prog_bar=True)

        top1_xffragment, top5_xffragment = accuracy(logits, identity, topk=(1, 5))
        self.log(f"train_acc_top1", top1_xffragment)
        self.log(f"train_acc_top5", top5_xffragment)
        preds = identity[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
        map = get_map_score(identity, preds, identity)
        self.log(f"train_map", map)

        return loss

    def eval_step(self, batch, stage=None):
        x = batch["clip"]  # [b, s, c, h, w]
        identity = batch["identity"]  # [b]

        h = self(x)  # [b, s, d]
        z = self.projector(h)  # [b, s, p]

        z_fragment = self.fragment_pooler(z)  # [b, d]

        loss, logits = self.criterion(z_fragment, identity)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            top1, top5 = accuracy(logits, identity, topk=(1, 5))
            top1, top5 = accuracy(logits, identity, topk=(1, 5))
            self.log(f"{stage}_acc_top1", top1, sync_dist=True)
            self.log(f"{stage}_acc_top5", top5, sync_dist=True)
            preds = identity[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
            map = get_map_score(identity, preds, identity)
            self.log(f"{stage}_map", map, sync_dist=True)

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


class SupervisedTrainer(Trainer):
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

        self.criterion = YAwareLoss(
            loss_temperature,
            return_logits=True,
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.encoder(x)

    def step(self, batch, stage=None):
        x = batch["clip"]  # [b, s, c, h, w]
        y = batch["identity"]  # [b]

        h = self(x)  # [b, s, d]
        z = self.projector(h)  # [b, s, p]

        z_pool = self.pooler(z)  # [b, d]

        loss, logits = self.criterion(z_pool, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            top1, top5 = accuracy(logits, y, topk=(1, 5))
            self.log(f"{stage}_acc_top1", top1)
            self.log(f"{stage}_acc_top5", top5)
            preds = y[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
            map = get_map_score(y, preds, y)
            self.log(f"{stage}_map", map)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_lr,
            weight_decay=self.opt_weight_decay,
        )
        return optimizer


