import logging

import lightning as L
import torch

from polypsense.e2e.zoo import create_encoder
from polypsense.reid.dm import ReidDataModule
from polypsense.reid.eval import (
    FewShotRetrievalEvaluator,
    ReidEvaluator,
    RetrievalEvaluator,
)
from polypsense.reid.eval.fewshotretrieval import FewShotConfig


def evaluate(args):
    logging.basicConfig(level=logging.INFO)

    torch.set_float32_matmul_precision("high")

    L.seed_everything(42, workers=True)

    dm = get_dm(args)
    evaluator = get_evaluator(args)

    engine = get_engine(args)

    stages = get_stages(args)

    if "fit" in stages:
        engine.fit(evaluator, dm)

    if "validate" in stages:
        engine.validate(evaluator, dm)

    if "test" in stages:
        engine.test(evaluator, dm)


def get_engine(args):
    if args.task in ["reid", "retrieval", "fewshotretrieval"]:
        return L.Trainer(
            accelerator="cpu",  # dm is in charge to extract features on GPU
            devices=1,
            logger=get_logger(args),
            log_every_n_steps=1,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
    raise ValueError(f"Unknown task: {args.task}")


def get_stages(args):
    if args.stages:
        return args.stages
    if args.task in ["reid", "retrieval", "fewshotretrieval"]:
        return ["test"]
    return []


def get_evaluator(args):
    if args.task == "reid":
        return ReidEvaluator(args.exp_name)
    if args.task == "retrieval":
        return RetrievalEvaluator(args.exp_name)
    if args.task == "fewshotretrieval":
        return FewShotRetrievalEvaluator(
            args.exp_name,
            "fewshot",
            FewShotConfig(
                enabled=True,
                k_shots=(1, 2, 3, 5, 10),
                n_repeats=50,
                gallery_mode=args.fsret_gallery_mode,
            )
        )
    raise ValueError(f"Unknown task: {args.task}")


def get_dm(args):
    return ReidDataModule(
        encoder=get_encoder(args),
        encoder_features_source=args.encoder_features_source,
        encoder_out_dim=args.encoder_out_dim,
        encoder_pooling=args.encoder_pooling,
        dataset_root=args.dataset_root,
        im_size=args.data_im_size,
        fragment_length=args.data_fragment_length,
        eval_aug_resize=args.data_aug_resize,
        eval_aug_anchorcrop=args.data_aug_anchorcrop,
        bbox_scale_factor=args.data_bbox_scale_factor,
        aug_normalize=args.data_aug_normalize,
        num_workers=args.num_workers,
    )


def get_logger(args):
    import os

    logger = L.pytorch.loggers.WandbLogger(
        entity=args.wandb_entity,
        project=args.wandb_project,
        id=args.exp_id,
        name=args.exp_name,
        save_dir=os.path.join(os.getcwd(), "wandb_logs"),
        allow_val_change=True,
        resume="allow",
    )
    logger.experiment.config.update(args, allow_val_change=True)
    return logger


def get_encoder(args):
    if args.encoder_ckpt:
        return create_encoder(args.encoder_type, args.encoder_ckpt)

    encoder_cfg = {"pretrained_weights": args.encoder_pretrained}

    return create_encoder(args.encoder_type, encoder_cfg)
