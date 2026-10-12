import logging
import pathlib
import random
import re

import lightning as L
import torch
import torch.utils

from polypsense.dataset import InstanceDataset
from polypsense.dataset.repo.fragmenter import Fragmenter
from polypsense.dataset.repo.identifier import Identifier
from polypsense.dataset.repo.trackleter import Trackleter
from polypsense.dataset.repo.videoer import Videoer
from polypsense.e2e.data import FragmentIdentityDataset
from polypsense.e2e.data_utils import subsample, upsample
from polypsense.e2e.sampler import (
    DistributedTemporalTemperatureBagBatchSampler,
    MultiPosConBatchSampler,
    PartitionedTemporalTemperatureBagBatchSampler,
    TemporalKNNBagBatchSampler,
    TemporalTemperatureBagBatchSampler,
)
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
from polypsense.e2e.utils import select_instances_by_bboxes_diagonal_ratio


class End2EndDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root,
        *,
        im_size,
        fragment_length,
        fragment_stride,
        fragment_drop_last,
        fragment_padding_mode,
        bbox_scale_factor,
        bbox_scale_range,
        bbox_diagonal_ratio_range,
        min_tracklet_length,
        sampler,
        sampler_ttb_tmin,
        sampler_ttb_tmax,
        aug_resize,
        aug_anchorcrop,
        aug_randomanchorcrop,
        aug_randomresizedcrop,
        aug_vflip,
        aug_hflip,
        aug_affine,
        aug_colorjitter,
        aug_ioucrop,
        aug_normalize,
        aug_grayscale,
        train_subset_ratio,
        do_upsample,
        batch_size,
        num_workers,
        n_views,
        seed,
        eval_n_views,
        eval_batch_size,
        eval_fragment_length,
        eval_aug_resize,
        eval_aug_anchorcrop,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.im_size = im_size
        self.fragment_length = fragment_length
        self.fragment_stride = fragment_stride
        self.fragment_drop_last = fragment_drop_last
        self.fragment_padding_mode = fragment_padding_mode
        self.bbox_scale_factor = bbox_scale_factor
        self.bbox_scale_range = bbox_scale_range
        self.bbox_diagonal_ratio_range = bbox_diagonal_ratio_range
        self.min_tracklet_length = min_tracklet_length
        self.sampler = sampler
        self.sampler_ttb_tmin = sampler_ttb_tmin
        self.sampler_ttb_tmax = sampler_ttb_tmax
        self.aug_resize = aug_resize
        self.aug_anchorcrop = aug_anchorcrop
        self.aug_randomanchorcrop = aug_randomanchorcrop
        self.aug_randomresizedcrop = aug_randomresizedcrop
        self.aug_vflip = aug_vflip
        self.aug_hflip = aug_hflip
        self.aug_affine = aug_affine
        self.aug_colorjitter = aug_colorjitter
        self.aug_ioucrop = aug_ioucrop
        self.aug_normalize = aug_normalize
        self.aug_grayscale = aug_grayscale
        self.train_subset_ratio = train_subset_ratio
        self.do_upsample = do_upsample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_views = n_views
        self.seed = seed
        self.eval_n_views = eval_n_views
        self.eval_batch_size = eval_batch_size
        self.eval_fragment_length = eval_fragment_length
        self.eval_aug_resize = eval_aug_resize
        self.eval_aug_anchorcrop = eval_aug_anchorcrop

        self._rng = random.Random(seed)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._load_train_set()
            self.val_dataset = self._make_ds("val")

        if stage == "validate" or stage is None:
            self.val_dataset = self._make_ds("val")

        if stage == "test" or stage is None:
            self.test_dataset = self._make_ds("test")

    def train_dataloader(self):
        if isinstance(self.train_dataset, list):
            i = self.trainer.current_epoch % len(self.train_dataset)
            if len(self.train_dataset) != self.trainer.max_epochs:
                logging.warning(
                    f"Number of training splits ({len(self.train_dataset)}) does not match max_epochs ({self.trainer.max_epochs}). "
                    f"Using split {i} for epoch {self.trainer.current_epoch}."
                )
            return self._get_train_dataloader(self.train_dataset[i])

        return self._get_train_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._get_eval_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, ds):
        batch_sampler = self._get_train_sampler(ds)
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def _get_eval_dataloader(self, ds):
        batch_sampler = self._get_eval_sampler(ds)
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def _get_train_sampler(self, ds):
        if self.sampler is None or self.sampler == "multipos":
            return MultiPosConBatchSampler(
                ds.y,
                k=self.n_views,
                batch_size=self.batch_size,
                seed=self._rng.randint(0, 2**32 - 1),
            )
        if self.sampler == "temporalknnbag":
            return TemporalKNNBagBatchSampler(
                ds.v,
                ds.pv,
                k=self.n_views,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                seed=self._rng.randint(0, 2**32 - 1),
            )
        if self.sampler == "temporaltemperaturebag":
            return TemporalTemperatureBagBatchSampler(
                ds.v,
                ds.pv,
                k=self.n_views,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                c=self.trainer.current_epoch / self.trainer.max_epochs,
                t_min=self.sampler_ttb_tmin,
                t_max=self.sampler_ttb_tmax,
                seed=self._rng.randint(0, 2**32 - 1),
            )
        if self.sampler == "distributedtemporaltemperaturebag":
            return DistributedTemporalTemperatureBagBatchSampler(
                ds.v,
                ds.pv,
                k=self.n_views,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                c=self.trainer.current_epoch / self.trainer.max_epochs,
                t_min=self.sampler_ttb_tmin,
                t_max=self.sampler_ttb_tmax,
                seed=self.seed + self.trainer.current_epoch,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            )
        if self.sampler == "partitionedtemporaltemperaturebag":
            return PartitionedTemporalTemperatureBagBatchSampler(
                ds.v,
                ds.pv,
                ds.q,
                k=self.n_views,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                c=self.trainer.current_epoch / self.trainer.max_epochs,
                t_min=self.sampler_ttb_tmin,
                t_max=self.sampler_ttb_tmax,
                seed=self._rng.randint(0, 2**32 - 1),
            )
        raise ValueError(f"Unknown sampler: {self.sampler}")

    def _get_eval_sampler(self, ds):
        return MultiPosConBatchSampler(
            ds.y,
            k=self.eval_n_views,
            batch_size=self.eval_batch_size,
            seed=self.seed,
        )

    def _load_train_set(self):
        train_splits = self._scan_train_splits()

        if len(train_splits) == 1:
            return self._make_ds(train_splits[0])

        logging.info(f"Found {len(train_splits)} training instance files.")

        train_datasets = []

        for train_split in train_splits:
            train_dataset = self._make_ds(train_split)
            train_datasets.append(train_dataset)

            logging.debug(
                f"Loaded training split {train_split} with {len(train_dataset)} samples"
            )

        return train_datasets

    def _make_ds(self, split):
        instance_ds = self._get_instance_ds(split)

        trackleter = Trackleter(instance_ds)
        identifier = Identifier(instance_ds)
        videoer = Videoer(instance_ds)
        fragmenter = Fragmenter(
            trackleter,
            min_tracklet_length=self.min_tracklet_length,
            fragment_length=(
                self.fragment_length if self._is_training_split(split) else self.eval_fragment_length
            ),
            stride=self.fragment_stride,
            drop_last=self.fragment_drop_last,
            padding_mode=self.fragment_padding_mode,
        )

        ds = FragmentIdentityDataset(
            instance_ds,
            fragmenter,
            identifier,
            videoer,
            fragment_transforms=self._get_fragment_transforms(split),
            return_dict=True,
        )

        if self._is_training_split(split) and self.train_subset_ratio:
            ds = subsample(ds, ratio=self.train_subset_ratio, seed=self.seed)

        if self._is_training_split(split) and self.do_upsample:
            ds = upsample(ds, min_fragments_per_identity=self.n_views, seed=self.seed)

            # let's print a warning if upsampling is enabled but no data
            # augmentation is applied
            if not any(
                [
                    self.aug_vflip,
                    self.aug_hflip,
                    self.aug_affine,
                    self.aug_colorjitter,
                    self.aug_grayscale,
                    self.aug_ioucrop,
                    self.aug_randomanchorcrop,
                ]
            ):
                logging.warning(
                    "Upsampling is enabled but no data augmentation is applied. "
                    "This may lead to overfitting."
                )

        return ds

    def _get_instance_ds(self, split):
        dataset_root = pathlib.Path(self.dataset_root)
        img_folder = dataset_root / "images"
        ann_file = dataset_root / "annotations" / f"instances_{split}.json"

        instance_ds = InstanceDataset.from_instances(img_folder, ann_file)

        if self.bbox_diagonal_ratio_range:
            instance_ds = select_instances_by_bboxes_diagonal_ratio(
                instance_ds, *self.bbox_diagonal_ratio_range
            )

        return instance_ds

    def _get_fragment_transforms(self, split):
        if self._is_training_split(split):
            return compose(
                to_tensor()
                + resize(self.aug_resize, self.im_size)
                + anchor_crop(self.aug_anchorcrop, self.bbox_scale_factor, self.im_size)
                + random_anchor_crop(
                    self.aug_randomanchorcrop,
                    *(self.bbox_scale_range or (None, None)),
                    self.im_size,
                )
                + random_iou_crop(self.aug_ioucrop, self.im_size)
                + random_resized_crop(self.aug_randomresizedcrop, self.im_size)
                + hflip(self.aug_hflip)
                + vflip(self.aug_vflip)
                + affine(self.aug_affine)
                + grayscale(self.aug_grayscale)
                + color_jitter(self.aug_colorjitter)
                + normalize(self.aug_normalize)
            )

        return compose(
            to_tensor()
            + resize(self.eval_aug_resize, self.im_size)
            + anchor_crop(
                self.eval_aug_anchorcrop, self.bbox_scale_factor, self.im_size
            )
            + grayscale(self.aug_grayscale)
            + normalize(self.aug_normalize)
        )

    def _scan_train_splits(self):
        """
        Scans a folder for files matching 'instances_train.json' or
        'instances_train.N.json'.
        """
        target_dir = pathlib.Path(self.dataset_root) / "annotations"

        pattern = re.compile(r"^instances_(train(?:\.\d+)?)\.json$")

        split_names = []

        for file in target_dir.iterdir():
            match = pattern.match(file.name)
            if file.is_file() and match:
                split_names.append(match.group(1))

        return sorted(split_names)

    def _is_training_split(self, split):
        return "train" in split

def collate_fn(batch):
    return torch.utils.data.default_collate(batch)
