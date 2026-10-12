import argparse

from polypsense.reid.engine import evaluate


def main():
    parser = get_parser()
    args = parser.parse_args()
    evaluate(args)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)

    parser.add_argument(
        "--task",
        type=str,
        choices=["reid", "retrieval", "fewshotretrieval"],
    )
    parser.add_argument("--stages", type=str, nargs="+", default=None)

    parser.add_argument("--dataset_root", type=str)

    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=[
            "mve",
            "endofm",
            "endofmlv",
            "endovit",
            "surgenet",
            "dinov2",
            "dinov3",
            "vjepa2",
        ],
    )
    parser.add_argument("--encoder_ckpt", type=str)
    parser.add_argument("--encoder_pretrained", type=str)
    parser.add_argument(
        "--encoder_features_source", type=str, choices=["encoder", "projector"]
    )
    parser.add_argument(
        "--encoder_pooling",
        type=str,
        choices=[
            "none",
            "cls",
            "mean",
            "patchmean",
            "mean-mean",  # e.g. dinov2
            "cls-mean",  # e.g. dinov3
            "cls-max",
        ],
    )
    parser.add_argument("--encoder_out_dim", type=int)

    parser.add_argument(
        "--fsret_gallery_mode",
        choices=["global", "same_video"],
        default="same_video",
    )

    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--data_fragment_length", type=int)
    parser.add_argument("--data_aug_resize", action="store_true")
    parser.add_argument("--data_aug_anchorcrop", action="store_true")
    parser.add_argument("--data_bbox_scale_factor", type=int)
    parser.add_argument("--data_im_size", type=int)
    parser.add_argument("--data_aug_normalize", action="store_true")

    return parser


if __name__ == "__main__":
    main()
