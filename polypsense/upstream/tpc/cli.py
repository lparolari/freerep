import argparse

from polypsense.upstream.tpc.engine import train


def main():
    parser = get_parser()
    args = parser.parse_args()

    train(args)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_images", type=str)
    parser.add_argument("--train_annotations", type=str)
    parser.add_argument("--val_images", type=str)
    parser.add_argument("--val_annotations", type=str)
    parser.add_argument("--test_images", type=str)
    parser.add_argument("--test_annotations", type=str)

    # training
    parser.add_argument("--mode", type=str, choices=["train"])
    parser.add_argument("--trainer", type=str, choices=["tpc"])  # fmt: skip
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # engine (lit trainer)
    parser.add_argument("--engine_num_nodes", type=int, default=1)
    parser.add_argument("--engine_devices", default="auto")
    parser.add_argument("--engine_strategy", type=str, default="auto", choices=["auto", "ddp", "ddp_find_unused_parameters_true"])  # fmt: skip
    parser.add_argument("--engine_precision", type=str)
    parser.add_argument("--engine_fast_dev_run", type=int)
    parser.add_argument("--engine_num_sanity_val_steps", type=int)
    parser.add_argument("--engine_log_level", type=int, default=20)  # 20: INFO, 10: DEBUG

    # datamodule
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--fragment_length", type=int)
    parser.add_argument("--fragment_stride", type=int)
    parser.add_argument("--bbox_scale_factor", type=float)
    parser.add_argument("--im_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--eval_fragment_length", type=int)
    parser.add_argument("--sampler", type=str, choices=["multipos", "distributedmultipos"])  # fmt: skip
    parser.add_argument("--sampling_strategy", type=str, choices=["extremes", "half"])

    # augmentation
    parser.add_argument("--aug_resize", action="store_true", default=None)
    parser.add_argument("--aug_anchorcrop", action="store_true", default=None)
    parser.add_argument("--aug_randomresizedcrop", action="store_true", default=None)
    parser.add_argument("--aug_normalize", action="store_true", default=None)

    # trainer
    parser.add_argument("--trainer_d_proj", type=int)
    parser.add_argument("--trainer_d_model", type=int)
    parser.add_argument("--trainer_pooling", type=str, choices=["cls"])  # fmt: skip

    # opt
    parser.add_argument("--opt_lr", type=float)
    parser.add_argument("--opt_weight_decay", type=float)
    # parser.add_argument("--optim_warmup_epochs", type=int)

    # loss
    parser.add_argument("--loss_temperature", type=float)

    # encoder
    parser.add_argument("--encoder_type", type=str, choices=["mve"])  # fmt: skip
    parser.add_argument("--encoder_pretrained", type=str)

    # encoder: sfe
    parser.add_argument("--sfe_d_model", type=int)
    parser.add_argument("--sfe_backbone_arch", type=str, choices=["resnet18", "resnet50"])  # fmt: skip
    parser.add_argument("--sfe_backbone_weights", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])  # fmt: skip

    # encoder: mve
    parser.add_argument("--mve_d_model", type=int)
    parser.add_argument("--mve_d_feedforward", type=int)
    parser.add_argument("--mve_n_heads", type=int)
    parser.add_argument("--mve_n_layers", type=int)
    parser.add_argument("--mve_dropout", type=float)
    parser.add_argument("--mve_use_feature_norm", action="store_true", default=None)
    parser.add_argument("--mve_use_encoder_norm", action="store_true", default=None)

    # experiment
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--exp_name", type=str)

    return parser


if __name__ == "__main__":
    main()
