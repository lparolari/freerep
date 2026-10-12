import logging
import os

import lightning as L
import torch

from polypsense.e2e.dm import End2EndDataModule
from polypsense.e2e.trainer import (
    DistributedTemporalSelfSupervisedTrainer,
    SupervisedTrainer,
    TemporalSelfSupervisedTrainer,
)


def train_v2(args):
    """
    This is a version of the train function that instanciates a trainer
    (LightningModule) to train generic encoders (backbones).
    """
    logging.basicConfig(level=args.engine_log_level)

    torch.set_float32_matmul_precision("high")

    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    trainer = get_trainer(args)

    engine = L.Trainer(
        logger=get_logger(args, None),
        callbacks=get_callbacks(args),
        max_epochs=args.max_epochs,
        accelerator="cuda",
        num_nodes=1,
        devices=1,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        fast_dev_run=args.engine_fast_dev_run,
        num_sanity_val_steps=args.engine_num_sanity_val_steps,
    )

    engine.test(trainer, dm)
    engine.fit(trainer, dm, ckpt_path=args.resume and args.ckpt_path)
    engine.test(trainer, dm, ckpt_path="best")


def train_ddp(args):
    logging.basicConfig(level=logging.INFO)

    torch.set_float32_matmul_precision("high")

    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    trainer = get_trainer(args)

    engine = L.Trainer(
        logger=get_logger(args, None),
        callbacks=get_callbacks(args),
        max_epochs=args.max_epochs,
        accelerator="cuda",
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        # ddp related
        num_nodes=args.engine_num_nodes,
        devices=args.engine_devices,
        precision=args.engine_precision,
        strategy=args.engine_strategy,
        use_distributed_sampler=False,
        # dev
        fast_dev_run=args.engine_fast_dev_run,
        num_sanity_val_steps=0,
    )

    # engine.test(trainer, dm)
    engine.fit(trainer, dm, ckpt_path=args.resume and args.ckpt_path)
    # engine.test(trainer, dm, ckpt_path="best")


def get_datamodule(args):
    default_eval_n_views = 2
    default_eval_batch_size = 16
    default_eval_fragment_length = 8

    return End2EndDataModule(
        dataset_root=args.dataset_root,
        im_size=args.im_size,
        fragment_length=args.fragment_length,
        fragment_stride=args.fragment_stride,
        fragment_drop_last=args.fragment_drop_last,
        fragment_padding_mode=args.fragment_padding_mode,
        bbox_scale_factor=args.bbox_scale_factor,
        bbox_scale_range=args.bbox_scale_range,
        min_tracklet_length=args.min_tracklet_length,
        sampler=args.sampler,
        sampler_ttb_tmin=args.sampler_ttb_tmin,
        sampler_ttb_tmax=args.sampler_ttb_tmax,
        aug_resize=args.aug_resize,
        aug_anchorcrop=args.aug_anchorcrop,
        aug_randomanchorcrop=args.aug_randomanchorcrop,
        aug_randomresizedcrop=args.aug_randomresizedcrop,
        aug_vflip=args.aug_vflip,
        aug_hflip=args.aug_hflip,
        aug_affine=args.aug_affine,
        aug_colorjitter=args.aug_colorjitter,
        aug_ioucrop=args.aug_ioucrop,
        aug_normalize=args.aug_normalize,
        aug_grayscale=args.aug_grayscale,
        train_subset_ratio=args.train_subset_ratio,
        do_upsample=args.do_upsample,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_views=args.n_views,
        seed=args.seed,
        bbox_diagonal_ratio_range=args.bbox_diagonal_ratio_range,
        eval_n_views=args.eval_n_views or default_eval_n_views,
        eval_batch_size=args.eval_batch_size or default_eval_batch_size,
        eval_fragment_length=args.eval_fragment_length or default_eval_fragment_length,
        eval_aug_resize=args.eval_aug_resize,
        eval_aug_anchorcrop=args.eval_aug_anchorcrop,
    )


def _wandb_project(args):
    return args.wandb_project if args.wandb_project else "freerep"


def get_logger(args, model):
    logger = L.pytorch.loggers.WandbLogger(
        entity=args.wandb_entity,
        project=_wandb_project(args),
        id=args.exp_id,
        name=args.exp_name,
        save_dir=os.path.join(os.getcwd(), "wandb_logs"),
        config=vars(args),
        allow_val_change=True,
        resume="allow",
    )
    # wandb.watch breaks the ability to save nn.Modules as hparams. This happens
    # because the watch method attach a non-pickble object to the module and
    # Pytorch Lightning removes non pickable objects from the hparams before
    # saving it. See https://github.com/wandb/wandb/issues/2588.
    return logger


def get_callbacks(args):
    ckpt_dirpath = (
        os.path.join(
            os.getcwd(),
            "wandb_logs",
            _wandb_project(args),
            args.exp_name,
            "checkpoints",
        )
        if args.exp_name
        else None
    )

    best_model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=ckpt_dirpath,
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{step}-{val_loss:.2f}-best",
        save_last="link",
        save_weights_only=True,
        enable_version_counter=False,
    )
    best_model_checkpoint.CHECKPOINT_NAME_LAST = "best"

    last_model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=ckpt_dirpath,
        monitor="step",
        mode="max",
        filename="{epoch}-{step}-last",
        save_last="link",
        enable_version_counter=False,
    )

    return [
        best_model_checkpoint,  # must be first in callback list
        last_model_checkpoint,
        L.pytorch.callbacks.LearningRateMonitor(),
    ]


def get_trainer(args):
    def compact_dict(d):
        return {k: v for k, v in d.items() if v is not None}

    encoder_ckpt = args.ckpt_path
    encoder_hparams = compact_dict(
        {
            "pretrained_weights": args.encoder_pretrained,
            "d_in": args.sfe_d_model,
            "backbone_arch": args.sfe_backbone_arch,
            "backbone_weights": args.sfe_backbone_weights,
            "d_model": args.mve_d_model,
            "d_feedforward": args.mve_d_feedforward,
            "n_heads": args.mve_n_heads,
            "n_layers": args.mve_n_layers,
            "dropout": args.mve_dropout,
            "use_feature_norm": args.mve_use_feature_norm,
            "use_encoder_norm": args.mve_use_encoder_norm,
        }
    )

    encoder_cfg = encoder_ckpt if encoder_ckpt else encoder_hparams

    if args.trainer == "supervised":
        return SupervisedTrainer(
            encoder_type=args.encoder_type,
            encoder_cfg=encoder_cfg,
            d_model=args.trainer_d_model,
            d_proj=args.trainer_d_proj,
            pooling=args.trainer_fragment_pooling,
            loss_temperature=args.loss_temperature,
            opt_lr=args.opt_lr,
            opt_weight_decay=args.opt_weight_decay,
        )

    if args.trainer == "temporalselfsupervised":
        return TemporalSelfSupervisedTrainer(
            encoder_type=args.encoder_type,
            encoder_cfg=encoder_cfg,
            d_model=args.trainer_d_model,
            d_proj=args.trainer_d_proj,
            loss_temperature=args.loss_temperature,
            opt_lr=args.opt_lr,
            opt_weight_decay=args.opt_weight_decay,
            k=args.n_views,
            fragment_pooling=args.trainer_fragment_pooling,
        )

    if args.trainer == "distributedtemporalselfsupervised":
        return DistributedTemporalSelfSupervisedTrainer(
            encoder_type=args.encoder_type,
            encoder_cfg=encoder_cfg,
            d_model=args.trainer_d_model,
            d_proj=args.trainer_d_proj,
            loss_temperature=args.loss_temperature,
            opt_lr=args.opt_lr,
            opt_weight_decay=args.opt_weight_decay,
            k=args.n_views,
            fragment_pooling=args.trainer_fragment_pooling,
        )

    raise ValueError(f"Unknown trainer type: {args.trainer}")
