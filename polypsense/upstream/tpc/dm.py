import random

import lightning as L
import torch

from polypsense.dataset.fragment import FragmentDataset
from polypsense.dataset.instance import InstanceDataset
from polypsense.dataset.mapping import MappingDataset
from polypsense.dataset.multitask import MultiTaskDataset
from polypsense.dataset.repo.fragmenter import Fragmenter
from polypsense.dataset.repo.trackleter import Trackleter
from polypsense.downstream.data_util import get_fragment_transforms
from polypsense.e2e.sampler import (
    MultiPosConBatchSampler,
    DistributedMultiPosConBatchSampler,
)
from polypsense.upstream.tpc.data import TPCDataset


class TPCDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_images: str,
        train_annotations: str,
        val_images: str,
        val_annotations: str,
        test_images: str,
        test_annotations: str,
        *,
        batch_size: int,
        im_size: int,
        fragment_length: int,
        fragment_stride: int,
        sampler: str,
        sampling_strategy: str,
        aug_resize: bool,
        aug_anchorcrop: bool,
        aug_randomresizedcrop: bool,
        aug_normalize: bool,
        bbox_scale_factor: float,
        num_workers: int,
        seed: int,
        eval_batch_size: int,
        eval_fragment_length: int,
    ):
        super().__init__()
        self.train_images = train_images
        self.train_annotations = train_annotations
        self.val_images = val_images
        self.val_annotations = val_annotations
        self.test_images = test_images
        self.test_annotations = test_annotations
        self.batch_size = batch_size
        self.im_size = im_size
        self.fragment_length = fragment_length
        self.fragment_stride = fragment_stride
        self.sampler = sampler
        self.sampling_strategy = sampling_strategy
        self.aug_resize = aug_resize
        self.aug_anchorcrop = aug_anchorcrop
        self.aug_randomresizedcrop = aug_randomresizedcrop
        self.aug_normalize = aug_normalize
        self.bbox_scale_factor = bbox_scale_factor
        self.num_workers = num_workers
        self.seed = seed
        self.eval_batch_size = eval_batch_size
        self.eval_fragment_length = eval_fragment_length

        self._rng = random.Random(seed)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = None  # sic!
            self.val_dataset = self._make_eval_ds(
                self.val_images,
                self.val_annotations,
            )

        if stage == "test" or stage is None:
            self.test_dataset = self._make_eval_ds(
                self.test_images,
                self.test_annotations,
            )

    def train_dataloader(self):
        # since the dataset samples randomly two views from each tracklet, we
        # need to rebuild it every time the train_dataloader() function is
        # called (e.g. every epoch)
        ds = self._make_train_ds(
            self.train_images,
            self.train_annotations,
        )
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=self._get_train_sampler(ds),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=self._get_eval_sampler(self.val_dataset),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=self._get_eval_sampler(self.test_dataset),
        )

    def _get_train_sampler(self, ds):
        if self.sampler == "distributedmultipos":
            return DistributedMultiPosConBatchSampler(
                ds.t,
                k=2,
                batch_size=self.batch_size,
                seed=self._rng.randint(0, 2**32 - 1),
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            )
        return MultiPosConBatchSampler(
            ds.t,
            k=2,
            batch_size=self.batch_size,
            seed=self._rng.randint(0, 2**32 - 1),
        )

    def _get_eval_sampler(self, ds):
        return MultiPosConBatchSampler(
            ds.y,
            k=2,
            batch_size=self.eval_batch_size,
            seed=self.seed,
        )

    def _make_train_ds(self, root, ann_file):
        instance_ds = InstanceDataset.from_instances(root, ann_file)

        trackleter = Trackleter(
            instance_ds,
            separation_thresh=1,
            iou_thresh=0.1,
            identity_key="identity_id",
            pbar=True,
        )

        transforms = get_fragment_transforms(
            aug_resize=self.aug_resize,
            aug_anchorcrop=self.aug_anchorcrop,
            aug_randomresizedcrop=self.aug_randomresizedcrop,
            aug_normalize=self.aug_normalize,
            im_size=self.im_size,
            bbox_scale_factor=self.bbox_scale_factor,
            aug_vflip=False,
            aug_hflip=False,
            aug_affine=False,
            aug_colorjitter=False,
        )

        ds = TPCDataset(
            instance_ds,
            trackleter=trackleter,
            fragment_length=self.fragment_length,
            stride=self.fragment_stride,
            sampling_strategy=self.sampling_strategy,
            fragment_transforms=transforms,
            reduce_img_ann_keys=[
                ("id", "all"),
                ("file_name", "all"),
                ("width", "one"),
                ("height", "one"),
                ("frame_id", "all"),
                ("sequence_id", "one"),
                ("dataset_id", "one"),
            ],
            reduce_tgt_ann_keys=[
                ("id", "all"),
                ("image_id", "all"),
                ("bbox", "all"),
                ("identity_id", "one"),
                ("tracklet_id", "one"),  # added wrt eval, needed by sampler
            ],
        )

        return ds

    def _make_eval_ds(self, root, ann_file):
        instance_ds = InstanceDataset.from_instances(root, ann_file)

        trackleter = Trackleter(
            instance_ds,
            separation_thresh=1,
            iou_thresh=0.1,
            identity_key="identity_id",
            pbar=True,
        )

        fragmenter = Fragmenter(
            trackleter,
            min_tracklet_length=30,
            fragment_length=self.eval_fragment_length,
            stride=self.fragment_stride,
            drop_last=True,
            padding_mode=None,
        )

        fragment_ds = FragmentDataset(
            instance_ds,
            fragments=fragmenter.fragments(),
            reduce_img_ann_keys=[
                ("id", "all"),
                ("file_name", "all"),
                ("width", "one"),
                ("height", "one"),
                ("frame_id", "all"),
                ("sequence_id", "one"),
                ("dataset_id", "one"),
            ],
            reduce_tgt_ann_keys=[
                ("id", "all"),
                ("image_id", "all"),
                ("bbox", "all"),
                ("identity_id", "one"),
            ],
        )

        mapping_ds = MappingDataset(fragment_ds, mapping={"identity_id": "auto"})

        transforms = get_fragment_transforms(
            aug_anchorcrop=True,
            aug_normalize=self.aug_normalize,
            im_size=self.im_size,
            bbox_scale_factor=self.bbox_scale_factor,
        )

        multitask_ds = MultiTaskDataset(mapping_ds, transforms)

        # needed by sampler
        multitask_ds.y = torch.tensor(
            [
                multitask_ds.tgt_ann(multitask_ds.id(i))["identity_id"]
                for i in range(len(multitask_ds))
            ]
        )

        return multitask_ds


def collate_fn(batch):
    return torch.utils.data.default_collate(batch)
