import lightning as L
import torch

from polypsense.downstream.pooler import get_pooler
from polypsense.e2e.dm import End2EndDataModule
from polypsense.reid.data import ReidDataset, collate_fn, extract_features


class ReidDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        encoder,
        dataset_root,
        im_size,
        fragment_length,
        bbox_scale_factor,
        eval_aug_resize,
        eval_aug_anchorcrop,
        aug_normalize,
        encoder_features_source,
        encoder_out_dim,
        encoder_pooling,
        num_workers,
    ):
        super().__init__()
        self.encoder = Wrapper(
            encoder,
            num_features=encoder_out_dim,
            features_source=encoder_features_source,
            pooling=encoder_pooling,
        )
        self.dataset_root = dataset_root
        self.im_size = im_size
        self.fragment_length = fragment_length
        self.bbox_scale_factor = bbox_scale_factor
        self.eval_aug_resize = eval_aug_resize
        self.eval_aug_anchorcrop = eval_aug_anchorcrop
        self.aug_normalize = aug_normalize
        self.num_workers = num_workers

    def setup(self, stage=None):

        # altough overkill, using the datamodule to create the
        # FragmentIdentityDataset is the right approach beacause it already
        # handles all configurations properly
        dm = End2EndDataModule(
            dataset_root=self.dataset_root,
            im_size=self.im_size,
            fragment_length=None,
            fragment_stride=4,
            fragment_drop_last=True,
            fragment_padding_mode=None,
            bbox_scale_factor=self.bbox_scale_factor,
            bbox_scale_range=None,
            bbox_diagonal_ratio_range=None,
            min_tracklet_length=30,
            sampler=None,
            sampler_ttb_tmin=None,
            sampler_ttb_tmax=None,
            aug_resize=None,
            aug_anchorcrop=None,
            aug_randomanchorcrop=None,
            aug_randomresizedcrop=None,
            aug_vflip=None,
            aug_hflip=None,
            aug_affine=None,
            aug_colorjitter=None,
            aug_ioucrop=None,
            aug_normalize=self.aug_normalize,
            aug_grayscale=None,
            train_subset_ratio=None,
            do_upsample=None,
            batch_size=None,
            num_workers=12,
            n_views=None,
            seed=42,
            eval_n_views=None,
            eval_batch_size=None,
            eval_fragment_length=self.fragment_length,
            eval_aug_resize=self.eval_aug_resize,
            eval_aug_anchorcrop=self.eval_aug_anchorcrop,
        )

        dm.setup(stage)

        if stage == "validate" or stage is None:
            self.val_dataset = self._make_ds(dm.val_dataset)

        if stage == "test" or stage is None:
            self.test_dataset = self._make_ds(dm.test_dataset)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
        )

    def _make_ds(self, fragment_ds):
        return ReidDataset(extract_features(self.encoder, fragment_ds))


class Wrapper(torch.nn.Module):
    """
    Wraps an encoder providing a pooling layer to produce fixed-size feature vectors.
    """

    def __init__(self, encoder, *, num_features, features_source, pooling):
        super().__init__()

        if features_source is None:
            features_source = "encoder"

        if features_source not in ["projector", "encoder"]:
            raise ValueError(
                f"Unknown features source: {features_source}. "
                "Must be one of ['projector', 'encoder']."
            )

        self.encoder = encoder
        self.features_source = features_source

        self.pooler = get_pooler(pooling, in_dim=num_features)

    def forward(self, x):
        # x: [b, s, c, h, w]

        features = self.encoder(x)

        if self.features_source == "projector":
            features = self.encoder.projector(features)

        pooled = self.pooler(features)

        return pooled  # [b, *, d]
