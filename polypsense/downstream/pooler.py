from abc import ABC, abstractmethod

import torch

class Pooler(torch.nn.Module, ABC):
    def __init__(self, in_dim):
        super().__init__()
        self._in_dim = in_dim

    @property
    def in_dim(self):
        return self._in_dim

    @property
    @abstractmethod
    def out_dim(self):
        raise NotImplementedError()

    def forward(self, x):
        """
        Pools along second to last dimension (i.e. the token dimension).

        Args:
            x: A tensor of size [..., p, d]  where p is the number of tokens

        Returns:
            A tensor of size [..., d] or [..., p', d] where p' is the number of
            tokens after pooling
        """
        raise NotImplementedError()


class IdentityPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x  # [..., p, d]


class MeanPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x.mean(dim=-2)  # [..., d]


class MaxPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x.max(dim=-2).values  # [..., d]


class ClsPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x[..., 0, :]  # [..., d]


class PatchMeanPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x[..., 1:, :].mean(dim=-2)  # [..., d]


class PatchPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x[..., 1:, :]  # [..., p-1, d]


class SpaceTimePooler(Pooler):
    def __init__(self, in_dim, spatial_pooling, temporal_pooling):
        super().__init__(in_dim)
        self.spatial_pooler = get_pooler(spatial_pooling, in_dim)
        self.temporal_pooler = get_pooler(temporal_pooling, self.spatial_pooler.out_dim)

    @property
    def out_dim(self):
        return self.temporal_pooler.out_dim

    def forward(self, x):
        b, t, p, d = x.shape
        x = self.spatial_pooler(x)  # [b, t, d'] or [b, t, p', d']
        x = x.view(b, t, -1, d)  # ensure [b, t, p', d], p' might be 1
        p_1 = x.shape[-2]
        x = x.transpose(1, 2)  # [b, p', t, d]
        x = self.temporal_pooler(x)  # [b, p', d] or [b, p', t', d]
        x = x.view(b, p_1, -1, d)  # ensure [b, p', t', d], t' might be 1
        x = x.transpose(2, 1)  # [b, t', p', d]
        x = x.squeeze()  # remove any dimensions of size 1
        return x


def get_pooler(pooling: str, in_dim: int) -> Pooler:
    if pooling not in _get_choices():
        raise ValueError(
            f"Invalid pooling method. Choose from {_get_choices()} or do not set."
        )

    if pooling == "mean":
        return MeanPooler(in_dim)
    elif pooling == "max":
        return MaxPooler(in_dim)
    elif pooling == "cls":
        return ClsPooler(in_dim)
    elif pooling == "patchmean":
        return PatchMeanPooler(in_dim)
    elif pooling == "patch":
        return PatchPooler(in_dim)
    elif _is_space_time(pooling):
        spatial_pooling, temporal_pooling = _parse_space_time(pooling)
        return SpaceTimePooler(in_dim, spatial_pooling, temporal_pooling)
    else:
        return IdentityPooler(in_dim)


def _get_choices():
    space_only = [
        "none",  # deprecated in favor of "identity"
        "identity",
        "cls",
        "mean",
        "max",
        "patchmean",
        "patch",
    ]
    time_only = ["mean", "max"]
    space_time = [f"{s}-{t}" for s in space_only for t in time_only]
    joint_space_time = []
    return space_only + space_time + joint_space_time


def _is_space_time(pooling):
    return "-" in pooling


def _parse_space_time(pooling):
    return pooling.split("-")
