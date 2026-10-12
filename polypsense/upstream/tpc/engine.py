import logging

import lightning as L
import torch

from polypsense.e2e.engine import get_logger, get_callbacks
from polypsense.upstream.tpc.trainer import TPCTrainer
from polypsense.upstream.tpc.dm import TPCDataModule


def train(args):
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
        num_sanity_val_steps=args.engine_num_sanity_val_steps,
        # data related
        check_val_every_n_epoch=10,  # dataset is very small but non-deterministic, we allow the model to train more before valiation
    )

    is_distrib = int(args.engine_num_nodes or 1) * int(args.engine_devices or 1) > 1
    do_test = not is_distrib

    if do_test:
        engine.test(trainer, dm)
    
    engine.fit(trainer, dm, ckpt_path=args.resume and args.ckpt_path)
    
    if do_test:
        engine.test(trainer, dm, ckpt_path="best")


def get_datamodule(args):
    default_eval_batch_size = 16
    default_eval_fragment_length = 8

    return TPCDataModule(
        train_images=args.train_images,
        train_annotations=args.train_annotations,
        val_images=args.val_images,
        val_annotations=args.val_annotations,
        test_images=args.test_images,
        test_annotations=args.test_annotations,
        batch_size=args.batch_size,
        im_size=args.im_size,
        fragment_length=args.fragment_length,
        fragment_stride=args.fragment_stride,
        sampler=args.sampler,
        sampling_strategy=args.sampling_strategy,
        aug_resize=args.aug_resize,
        aug_anchorcrop=args.aug_anchorcrop,
        aug_randomresizedcrop=args.aug_randomresizedcrop,
        aug_normalize=args.aug_normalize,
        bbox_scale_factor=args.bbox_scale_factor,
        num_workers=args.num_workers,
        seed=args.seed,
        eval_batch_size=default_eval_batch_size,
        eval_fragment_length=default_eval_fragment_length,
    )


def get_trainer(args):
    if args.trainer != "tpc":
        raise ValueError(f"Invalid trainer '{args.trainer}'")

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

    return TPCTrainer(
        encoder_type=args.encoder_type,
        encoder_cfg=encoder_cfg,
        d_model=args.trainer_d_model,
        d_proj=args.trainer_d_proj,
        pooling=args.trainer_pooling,
        loss_temperature=args.loss_temperature,
        opt_lr=args.opt_lr,
        opt_weight_decay=args.opt_weight_decay,
    )
