from typing import Optional

from polypsense.e2e.transforms import (
    affine,
    anchor_crop,
    color_jitter,
    compose,
    grayscale,
    hflip,
    normalize,
    random_anchor_crop,
    random_iou_crop,
    random_resized_crop,
    resize,
    to_tensor,
    vflip,
)


def get_fragment_transforms(
    *,
    im_size: int,
    aug_resize: Optional[bool] = None,
    aug_anchorcrop: Optional[bool] = None,
    aug_randomanchorcrop: Optional[bool] = None,
    aug_randomioucrop: Optional[bool] = None,
    aug_randomresizedcrop: Optional[bool] = None,
    aug_normalize: Optional[bool] = None,
    aug_vflip: Optional[bool] = None,
    aug_hflip: Optional[bool] = None,
    aug_affine: Optional[bool] = None,
    aug_colorjitter: Optional[bool] = None,
    aug_grayscale: Optional[bool] = None,
    bbox_scale_factor: Optional[float] = None,
    bbox_scale_range: Optional[tuple[float, float]] = None,
):
    resize_or_crop_augs = [
        bool(aug_resize),
        bool(aug_anchorcrop),
        bool(aug_randomanchorcrop),
        bool(aug_randomioucrop),
        bool(aug_randomresizedcrop)
    ]

    if not any(resize_or_crop_augs):
        raise ValueError(
            "Please specify a resize or crop augmentation between aug_resize, aug_anchorcrop, aug_randomanchorcrop, and aug_randomioucrop."
        )

    if sum(resize_or_crop_augs) > 1:
        raise ValueError(
            "Resize or crop augmentations are mutually exclusive. Please specify only one of aug_resize, aug_anchorcrop, aug_randomanchorcrop, and aug_randomioucrop."
        )

    if aug_anchorcrop and bbox_scale_factor is None:
        raise ValueError("Please specify bbox_scale_factor for anchor cropping.")

    if aug_randomanchorcrop and bbox_scale_range is None:
        raise ValueError("Please specify bbox_scale_range for random anchor cropping.")

    bbox_min_scale = bbox_scale_range and bbox_scale_range[0]
    bbox_max_scale = bbox_scale_range and bbox_scale_range[1]

    return compose(
        to_tensor()
        + resize(aug_resize, im_size)
        + anchor_crop(aug_anchorcrop, bbox_scale_factor, im_size)
        + random_anchor_crop(
            aug_randomanchorcrop, bbox_min_scale, bbox_max_scale, im_size
        )
        + random_iou_crop(aug_randomioucrop, im_size)
        + random_resized_crop(aug_randomresizedcrop, im_size)
        + hflip(aug_hflip)
        + vflip(aug_vflip)
        + affine(aug_affine)
        + color_jitter(aug_colorjitter)
        + grayscale(aug_grayscale)
        + normalize(aug_normalize)
    )
