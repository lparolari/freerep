import random
from collections import defaultdict


from polypsense.e2e.data import FragmentIdentityDataset


def subsample(
    dataset: FragmentIdentityDataset, *, ratio, seed=None
) -> "SubsetFragmentIdentityDataset":
    """
    Subsample a FragmentIdentityDataset dataset by a given ratio.
    """
    n = len(dataset)
    n_subset = int(n * ratio)
    subset_indices = random.Random(seed).sample(list(range(n)), n_subset)
    subset_indices = sorted(subset_indices)
    return SubsetFragmentIdentityDataset(dataset, subset_indices)


def upsample(
    dataset: FragmentIdentityDataset,
    *,
    min_fragments_per_identity: int,
    seed: int = None,
) -> "SubsetFragmentIdentityDataset":
    """
    Upsample a FragmentIdentityDataset dataset to ensure each identity has at least min_k samples.
    """
    idx = upsample_indices_by_label(
        dataset.y, min_k=min_fragments_per_identity, seed=seed
    )
    return SubsetFragmentIdentityDataset(dataset, idx)


def upsample_indices_by_label(y, min_k=1, seed=None):
    """
    Return indices that upsample samples by label to ensure a minimum count per
    label.

    For each unique label in ``y``, this function ensures that at least
    ``min_k`` indices are returned. If a label occurs fewer than ``min_k``
    times, its indices are randomly repeated (with replacement) until the
    minimum is reached. Labels with at least ``min_k`` samples are left
    unchanged.

    Args:
        y (Sequence[Any]): Sequence of labels, one per sample.
        min_k (int, optional): Minimum number of samples >= 1 to ensure per
            label. Defaults to 1.
        seed (int, optional): Random seed for reproducible upsampling.
            If ``None``, the global random state is used.

    Returns:
        list[int]: List of indices into the original data representing the
        upsampled dataset. Indices may be repeated when upsampling is required.

    Example:
        >>> y = ["A", "A", "B"]
        >>> idx = upsample_indices_by_label(y, min_k=3, seed=42)
        >>> idx
        [0, 1, 1, 2, 2, 2]
    """
    rng = random.Random(seed) if seed is not None else random

    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y):
        label_to_indices[label].append(idx)

    upsampled_indices = []

    for label, idxs in label_to_indices.items():
        n = len(idxs)

        if n >= min_k:
            # Already enough → keep all original indices
            upsampled_indices.extend(idxs)
        else:
            # Upsample with replacement
            needed = min_k - n
            extra = [rng.choice(idxs) for _ in range(needed)]
            upsampled_indices.extend(idxs + extra)

    return upsampled_indices


class SubsetFragmentIdentityDataset:
    """
    Subset of a FragmentIdentityDataset dataset at specified indices.
    """

    def __init__(self, dataset, indices):
        if not isinstance(dataset, FragmentIdentityDataset):
            raise TypeError(
                f"Expected dataset to be of type FragmentIdentityDataset, got {type(dataset)}"
            )

        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.dataset[self.indices[i]] for i in idx]
        return self.dataset[self.indices[idx]]

    @property
    def x(self):
        return [self.dataset.x[i] for i in self.indices]

    @property
    def y(self):
        return self.dataset.y[self.indices]

    @property
    def v(self):
        return self.dataset.v[self.indices]
    
    @property
    def pv(self):
        return self.dataset.pv[self.indices]

    @property
    def pi(self):
        return self.dataset.pi[self.indices]

    @property
    def q(self):
        return self.dataset.q[self.indices]

    @property
    def ids(self):
        return [self.dataset.ids[i] for i in self.indices]

    @property
    def identity2label(self):
        return self.dataset.identity2label

    @property
    def label2identity(self):
        return self.dataset.label2identity

    @property
    def video2label(self):
        return self.dataset.video2label

    @property
    def label2video(self):
        return self.dataset.label2video

    @property
    def histology2label(self):
        return self.dataset.histology2label

    @property
    def label2histology(self):
        return self.dataset.label2histology