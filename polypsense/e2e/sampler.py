import math
import warnings

import torch


class MultiPosConBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, *, k=2, batch_size=8, seed=42):
        """
        Builds batches of `batch_size` items, where `k` items belong to the same
        class. A class may appear multiple time in a batch, but there must be at
        least two different classes in batch. Maximum number of different
        classes in batch is `m = batch_size // k` classes.
        """
        if batch_size % k != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by k ({k})")

        self.rng = torch.Generator().manual_seed(seed)
        self.labels = (
            labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
        )
        self.k = k
        self.batch_size = batch_size
        self.indices = self._build_indices()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def _build_indices(self):
        indices = []

        labels = self.labels
        labels_unique = labels.unique()
        label2idx = {c: torch.where(labels == c)[0] for c in labels_unique.tolist()}

        n = len(labels)
        b = self.batch_size
        k = self.k
        m = b // k

        while n > 0:
            # - sample m unique labels
            # - for each label, sample k data points
            # - remove sampled data points from available data points
            # - update n

            available = torch.tensor([len(v) >= k for v in label2idx.values()])
            availability = available.sum().item()  # number of available classes for this batch  # fmt: skip
            numerosity = torch.tensor([len(v) for v in label2idx.values()])

            # this potentially discards a lot of data...
            if availability < m:
                break

            weights = numerosity * available
            weights = weights / weights.sum()

            index_sampled = torch.multinomial(
                weights,
                m,
                replacement=False,
                generator=self.rng,
            )
            labels_sampled = labels_unique[index_sampled]

            # get k samples from each label, then remove samples from available,
            # update n
            batch = []

            for label in labels_sampled:
                idx = label2idx[label.item()]
                idx = idx[torch.randperm(len(idx), generator=self.rng)]
                idx1 = idx[:k]
                idx2 = idx[k:]

                # remove idx from available
                label2idx[label.item()] = idx2

                batch.extend(idx1.tolist())

                n -= len(idx1)

            if len(batch) == self.batch_size:
                indices.append(batch)

        # pretty print the error if no sampling can be performed, otherwise the
        # error raised later is hardly intelligible
        if len(indices) == 0:
            raise ValueError(
                f"Not enough data to build batches of size {self.batch_size} "
                f"with {m} classes and {k} samples per class. We suggest "
                f"seting the batch size to {self.k * availability}."
            )

        indices = torch.tensor(indices)
        # shuffle indices inside each batch to avoid having consecutive positive
        # samples
        indices = indices[:, torch.randperm(b, generator=self.rng)]

        return indices


class TemporalKNNBagBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Construct a batch from a set of bags. Each bag contains an anchor and `k -
    1` samples that are temporally close to it (i.e. the k nearest neighbors in
    time). Thus, a batch contains `m = batch_size / k` bags.
    """

    def __init__(
        self,
        y,
        p,
        k: int = 5,
        batch_size: int = 60,
        drop_last: bool = True,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            y: Tensor of shape [n]. Bags are built within each label.
            p: Tensor of shape [n]. Temporal position of each sample within its label.
            k: Number of samples per bag (including the anchor).
            batch_size: Number of samples per batch. Must be divisible by k.
            drop_last: Whether to drop the last incomplete batch.
            shuffle: Whether to shuffle bags order and samples within each batch.
            seed: Random seed for shuffling.
        """
        if batch_size % k != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by k ({k})")
        if k <= 0:
            raise ValueError("k must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.y = torch.as_tensor(y)
        self.p = torch.as_tensor(p, dtype=torch.float32)

        if self.y.ndim != 1 or self.p.ndim != 1:
            raise ValueError("y and p must be 1D tensors (shape [n])")
        if self.y.numel() != self.p.numel():
            raise ValueError("y and p must have the same length")

        m = batch_size // k  # number of bags per batch

        if m <= 1:
            raise ValueError(
                f"Could not build a batch with >= 2 bags with the given batch_size={batch_size} and k={k} values."
            )

        num_classes = self.y.unique().numel()
        if m > num_classes:
            warnings.warn(
                f"Requested {m} bags per batch, but the dataset contains only "
                f"{num_classes} unique classes. Some batches may contain multiple "
                f"bags from the same class.",
                category=UserWarning,
            )

        self.n = self.y.numel()
        self.k = k
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self._g = torch.Generator(device="cpu")
        self._g.manual_seed(seed)

        self._indices = self._precompute_indices()

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

    def _precompute_indices(self):

        # boolean matrix indicating whether fragments belong to the same class or not
        y_match = torch.eq(self.y.view(-1, 1), self.y.view(1, -1))  # [n, n]

        min_num_of_samples_per_class = y_match.sum(dim=-1).min().item()

        if min_num_of_samples_per_class < self.k:
            raise AssertionError(
                f"Each class must contain at least k={self.k} samples; "
                f"found a class with only {min_num_of_samples_per_class} sample(s)."
            )

        # float matrix with pairwise temporal distances
        d = torch.abs(self.p.unsqueeze(1) - self.p.unsqueeze(0))  # [n, n]

        # mask out distances between samples of different classes by setting
        # them to a large value. If `min_num_of_samples_per_class >= k`, this
        # ensures that we will always find k neighbors within the same class.
        d = d + (~y_match) * 1e6

        # note: we do not mask out (i, i) pairs because k is the number of views
        # and includes the anchor itself

        # get indices of the k smallest elements for each item
        _, indices = torch.topk(d, self.k, largest=False)  # [n, k]

        if self.shuffle:
            # shuffling the order of the bags is very important for the
            # contrastive training because if we put in the same batch two bags
            # with fragments very close to each other, those will be treated as
            # negatives and this will harm the training.
            indices = indices[torch.randperm(len(indices), generator=self._g)]  # [n, k]

        # flatten to get a list of samples where each k consecutive samples
        # belong to the same bag
        indices = indices.view(-1)  # [n * k]

        # split into batches
        batches = list((indices.split(self.batch_size)))

        # drop last incomplete batch
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        # convert each batch from tensor to list, required by sampler
        batches = [b.tolist() for b in batches]

        return batches


class TemporalTemperatureBagBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Construct batches as a concatenation of bags. Each bag contains an anchor
    and `k-1` additional samples selected from the same class (y), ordered by
    proximity (p) to the anchor.

    Difficulty is controlled by a complexity parameter `c ∈ [0, 1]`:
      - c=0: easiest regime, behaves close to top-k (very close ranks dominate)
      - c=1: hardest regime, samples spread to more distant ranks

    For each anchor i, let `neighbors[i]` be the indices of same-class samples
    sorted by |p_j - p_i| ascending (rank 0 is the closest, typically i itself).
    We always include rank 0 and sample the remaining k-1 ranks without
    replacement using a temperature-controlled distribution:
        P(rank=r) ∝ exp(-r / T(c)),   for r >= 1
    where T(c) is a cosine schedule between t_min and t_max.
    """

    def __init__(
        self,
        y,
        p,
        k: int = 5,
        batch_size: int = 60,
        drop_last: bool = True,
        shuffle: bool = False,
        seed: int = 42,
        c: float = 0.0,
        t_min: float = 0.3,
        t_max: float = 12.0,
    ):
        """
        Args:
            y: Tensor of shape [n]. Bags are built within each label.
            p: Tensor of shape [n]. Temporal position of each sample within its label.
            k: Number of samples per bag (including the anchor).
            batch_size: Number of samples per batch. Must be divisible by k.
            drop_last: Whether to drop the last incomplete batch.
            shuffle: Whether to shuffle bags order (recommended).
            seed: Random seed for shuffling and sampling.
            c: Complexity in [0, 1]. You can pass c=curr_epoch/max_epoch.
            t_min: Minimum temperature (at c=0).
            t_max: Maximum temperature (at c=1).
        """
        if batch_size % k != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by k ({k})")
        if k <= 0:
            raise ValueError("k must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= float(c) <= 1.0):
            raise ValueError(f"c must be in [0, 1], got c={c}")
        if t_min <= 0 or t_max <= 0:
            raise ValueError(
                f"t_min and t_max must be > 0, got t_min={t_min}, t_max={t_max}"
            )
        if t_max < t_min:
            raise ValueError(
                f"t_max must be >= t_min, got t_min={t_min}, t_max={t_max}"
            )

        self.y = torch.as_tensor(y)
        self.p = torch.as_tensor(p, dtype=torch.float32)

        if self.y.ndim != 1 or self.p.ndim != 1:
            raise ValueError("y and p must be 1D tensors (shape [n])")
        if self.y.numel() != self.p.numel():
            raise ValueError("y and p must have the same length")

        m = batch_size // k  # number of bags per batch
        if m <= 1:
            raise ValueError(
                f"Invalid batch configuration: batch_size / k must be > 1 to form at least "
                f"two bags per batch. Got batch_size={batch_size}, k={k} (batch_size / k = {m})."
            )

        num_classes = self.y.unique().numel()
        if m > num_classes:
            warnings.warn(
                f"Requested {m} bags per batch, but the dataset contains only "
                f"{num_classes} unique classes. Some batches may contain multiple "
                f"bags from the same class.",
                category=UserWarning,
            )

        self.n = int(self.y.numel())
        self.k = int(k)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)

        self.c = float(c)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        self._g = torch.Generator(device="cpu")
        self._g.manual_seed(int(seed))

        self._indices = self._precompute_indices()  # batches of indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

    @property
    def temperature(self) -> float:
        # cosine schedule on c in [0,1]
        return self.t_min + 0.5 * (1.0 - math.cos(math.pi * self.c)) * (
            self.t_max - self.t_min
        )

    def _sample_ranks_without_replacement(
        self, M: int, k_minus_1: int, T: float
    ) -> torch.Tensor:
        """
        Sample k_minus_1 distinct ranks from {1, ..., M-1} with P(r) ∝ exp(-r/T).
        Returns a LongTensor of shape [k_minus_1] containing sampled ranks.
        """
        if k_minus_1 <= 0:
            return torch.empty((0,), dtype=torch.long)

        if M <= 1:
            raise ValueError("M must be > 1 to sample ranks from {1..M-1}")

        # ranks 1..M-1
        r = torch.arange(1, M, dtype=torch.float32)  # [M-1]
        logits = -r / float(T)  # [M-1]

        # Gumbel-topk for sampling without replacement
        u = torch.rand(M - 1, generator=self._g).clamp_(1e-12, 1.0 - 1e-12)
        g = -torch.log(-torch.log(u))
        _, topk = torch.topk(logits + g, k=k_minus_1, largest=True)

        return topk.to(torch.long) + 1  # shift back to ranks in [1..M-1]

    def _precompute_indices(self):
        # boolean matrix indicating whether fragments belong to the same class or not
        y_match = torch.eq(self.y.view(-1, 1), self.y.view(1, -1))  # [n, n]
        num_in_class = y_match.sum(dim=-1)  # [n]
        min_num_of_samples_per_class = int(num_in_class.min().item())

        if min_num_of_samples_per_class < self.k:
            raise AssertionError(
                f"Each class must contain at least k={self.k} samples; "
                f"found a class with only {min_num_of_samples_per_class} sample(s)."
            )

        # pairwise distances (smaller = closer)
        d = torch.abs(self.p.unsqueeze(1) - self.p.unsqueeze(0))  # [n, n]
        # mask out different-class distances
        d = d + (~y_match) * 1e6

        # full sorted neighbor lists by distance (within class ranks come first)
        sorted_idx = torch.argsort(d, dim=1)  # [n, n]

        # build bags using temperature sampling over ranks
        T = self.temperature
        bags = torch.empty((self.n, self.k), dtype=torch.long)

        for i in range(self.n):
            M = int(
                num_in_class[i].item()
            )  # number of valid same-class items for anchor i
            # valid neighbors are the first M entries
            nbrs = sorted_idx[i, :M]

            # always include rank 0 (closest, typically self)
            bag = [nbrs[0]]

            # sample remaining k-1 ranks from 1..M-1 with temperature bias
            ranks = self._sample_ranks_without_replacement(
                M=M, k_minus_1=self.k - 1, T=T
            )
            bag.extend(nbrs[ranks].tolist())

            bags[i] = torch.tensor(bag, dtype=torch.long)

        if self.shuffle:
            # shuffle bag order to avoid putting temporally-near bags together systematically
            bags = bags[torch.randperm(len(bags), generator=self._g)]  # [n, k]

        # flatten so each k consecutive samples belong to the same bag
        indices = bags.reshape(-1)  # [n*k]

        # split into batches
        batches = list(indices.split(self.batch_size))

        # drop last incomplete batch
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        # convert each batch from tensor to list, required by sampler
        batches = [b.tolist() for b in batches]

        return batches


class DistributedTemporalTemperatureBagBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, *args, rank, world_size, **kwargs):
        sampler = TemporalTemperatureBagBatchSampler(*args, **kwargs)
        batches = list(sampler)

        # trim to a multiple of world_size for even distribution
        n = len(batches) - (len(batches) % world_size)
        self.batches = batches[:n][rank::world_size]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class DistributedMultiPosConBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, *args, rank, world_size, **kwargs):
        sampler = MultiPosConBatchSampler(*args, **kwargs)
        batches = list(sampler)

        # trim to a multiple of world_size for even distribution
        n = len(batches) - (len(batches) % world_size)
        self.batches = batches[:n][rank::world_size]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class PartitionedTemporalTemperatureBagBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Constructs batches of 'bags' using temperature-scaled exponential sampling
    over sorted neighbors.

    For each anchor, it constructs a bag of size $k$ by selecting the anchor and
    sampling $k-1$ neighbors from the same class ($y$) and same partition ($q$)
    based on their temporal distance ($p$) from which we compute ranks.

    This sampler partitions the batches by $q$, i.e., for each batch, all bags
    must belong to the same partition $q_i$.

    ### Batch Structure

    - Total batch size $b$ consists of $m$ bags of size $k$ ($b = m * k$).
    - Each bag contains one anchor and $k-1$ neighbors.
    - Neighbors must share the same class $y$ and partition $q$ as the anchor.

    ### Sampling Mechanism

    For an anchor $i$, let $R$ be the set of same-class, same-partition candidates sorted by
    absolute proximity $|p_j - p_i|$ in ascending order. Rank $r=0$ is the anchor itself.
    The remaining $k-1$ samples are drawn without replacement from ranks $r >= 1$
    following the probability distribution:

    $$P(rank=r) -> exp (- r / T(c))$$

    ### Temperature Scheduling

    The temperature $T(c)$ is controlled by a parameter $c in [0, 1]$,
    which cosine-schedule temperature between $T_{min}$ and $T_{max}$:

      - when $c ~ 0$: $T(c)$ is small. The distribution decays sharply,
        heavily favoring low-rank (closest) neighbors. Behaves similarly to Top-k
        sampling.

      - when $c -> 1$: $T(c)$ is large. The distribution flattens,
        increasing the probability of sampling distant neighbors.
    """

    def __init__(
        self,
        y,
        p,
        q,
        k: int = 5,
        batch_size: int = 60,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        c: float = 0.0,
        t_min: float = 0.3,
        t_max: float = 12.0,
    ):
        """
        Args:
            y:
                Tensor of shape [n] with bag labels.

            p:
                Tensor of shape [n] with bag temporal positions relative to
                video.

            q:
                Tensor of shape [n] with bag partitioning labels.

            k:
                Number of samples per bag (including the anchor).

            batch_size:
                Number of samples per batch. Must be divisible by k.

            drop_last:
                Whether to drop the last incomplete batch per partition.

            shuffle:
                Whether to shuffle bags order.

            seed:
                Random seed for shuffling and sampling.

            c:
                Scheduling parameter in [0,1] for the temeprature. Determines
                the temperature with which sampling is performed (c ≈ 0 ->
                t_min, c ≈ 1 -> t_max).
                Generally used with curriculum learning, e.g. `c=curr_epoch/max_epoch`.

            t_min:
                Minimum sampling temperature (at c=0).

            t_max:
                Maximum sampling temperature (at c=1).
        """
        if batch_size % k != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by k ({k})")
        if k <= 0:
            raise ValueError("k must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= float(c) <= 1.0):
            raise ValueError(f"c must be in [0, 1], got c={c}")
        if t_min <= 0 or t_max <= 0:
            raise ValueError(
                f"t_min and t_max must be > 0, got t_min={t_min}, t_max={t_max}"
            )
        if t_max < t_min:
            raise ValueError(
                f"t_max must be >= t_min, got t_min={t_min}, t_max={t_max}"
            )

        self.y = torch.as_tensor(y)
        self.p = torch.as_tensor(p, dtype=torch.float32)
        self.q = torch.as_tensor(q)

        if self.y.ndim != 1 or self.p.ndim != 1 or self.q.ndim != 1:
            raise ValueError("y, p and q must be 1D tensors (shape [n])")
        if self.y.numel() != self.p.numel() != self.q.numel():
            raise ValueError("y, p and q must have the same length")

        m = batch_size // k  # number of bags per batch
        if m <= 1:
            raise ValueError(
                f"Invalid batch configuration: batch_size / k must be > 1 to include at least "
                f"two bags per batch. Batches with one bag are meaningless from a contrastive "
                f"learning perspective. Got batch_size={batch_size}, k={k} (batch_size / k = {m})."
            )

        for qi in q.unique():  # check num classes by partition
            num_classes = self.y[qi == q].unique().numel()
            if m > num_classes:
                warnings.warn(
                    f"Requested {m} bags per batch, but the dataset contains only "
                    f"{num_classes} unique classes for the partition {qi}. "
                    f"Some batches may contain multiple bags from the same class.",
                    category=UserWarning,
                )

        self.k = int(k)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)

        self.c = float(c)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        self._g = torch.Generator(device="cpu")
        self._g.manual_seed(int(seed))

        self._indices = self._precompute_indices()  # batches of indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

    def _precompute_indices(self):
        bags = self._construct_bags()
        batches = self._construct_batches(bags)
        return batches

    def _construct_bags(self):
        # boolean matrix indicating whether fragments belong to the same class,
        # same partition
        y_match = torch.eq(self.y.view(-1, 1), self.y.view(1, -1))  # [n, n]
        q_match = torch.eq(self.q.view(-1, 1), self.q.view(1, -1))  # [n, n]
        match = y_match & q_match

        # number of fragments
        n = self.y.numel()

        # check available classes per anchor
        num_in_class = match.sum(dim=-1)  # [n]
        min_num_of_samples_per_class = int(num_in_class.min().item())

        if min_num_of_samples_per_class < self.k:
            raise AssertionError(
                f"Each class must contain at least k={self.k} samples; "
                f"found a class with only {min_num_of_samples_per_class} sample(s)."
            )

        # pairwise distances (smaller = closer)
        d = torch.abs(self.p.unsqueeze(1) - self.p.unsqueeze(0))  # [n, n]

        # mask out different-class distances: argsort will put first the
        # farthest same-class fragments
        d = d + (~match) * 1e8

        # full sorted neighbor lists by distance (within class ranks come first)
        sorted_idx = torch.argsort(d, dim=1)  # [n, n]

        # build bags using temperature sampling over ranks
        T = self._temperature()
        bags = torch.empty((n, self.k), dtype=torch.long)

        for i in range(n):
            # number of valid same-class items for anchor i
            M = int(num_in_class[i].item())

            # prevent stocastic sampling of different-class fragments
            nbrs = sorted_idx[i, :M]  # valid neighbors are the first M entries

            # always include rank 0 (self), closes in the neighbors
            bag = [nbrs[0]]

            # sample remaining k-1 ranks from 1..M-1 with temperature-scaled
            # exponential distribution
            ranks = self._sample_ranks_without_replacement(
                M=M, k_minus_1=self.k - 1, T=T
            )
            bag.extend(nbrs[ranks].tolist())

            bags[i] = torch.tensor(bag, dtype=torch.long)  # [k]

        return bags  # [n, k]

    def _construct_batches(self, all_bags):
        """
        Construct batches from a list of bags. Batches must be partitioned by q,
        i.e. all bags in a batch must belong to the same partition.
        """
        all_batches = []

        for q in self.q.unique():
            bags = all_bags[q == self.q]

            # shuffling bags prevents the creation of batches with fragments
            # coming all from the same neighborhood even though they are from
            # different bags
            if self.shuffle:
                bags = bags[torch.randperm(len(bags), generator=self._g)]

            # flatten so each k consecutive samples belong to the same bag
            indices = bags.reshape(-1)  # [n*k]

            # split into batches
            batches = list(indices.split(self.batch_size))

            # drop last incomplete batch
            if self.drop_last and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]

            # convert each batch from tensor to list, required by sampler
            batches = [b.tolist() for b in batches]

            all_batches.extend(batches)

        # shuffling batches prevents to have a large consecutive number of
        # batches from the same partition
        if self.shuffle:
            # because drop_last can be false, we cannot convert `all_batches` to
            # torch.tensor
            all_batches = [
                all_batches[i]
                for i in torch.randperm(len(all_batches), generator=self._g)
            ]

        return all_batches

    def _sample_ranks_without_replacement(
        self, M: int, k_minus_1: int, T: float
    ) -> torch.Tensor:
        """
        Sample k_minus_1 distinct ranks from {1, ..., M-1} with P(r) ∝ exp(-r/T).
        Returns a LongTensor of shape [k_minus_1] containing sampled ranks.
        """
        if k_minus_1 <= 0:
            return torch.empty((0,), dtype=torch.long)

        if M <= 1:
            raise ValueError("M must be > 1 to sample ranks from {1..M-1}")

        # ranks 1..M-1
        r = torch.arange(1, M, dtype=torch.float32)  # [M-1]
        logits = -r / float(T)  # [M-1]

        # Gumbel-topk for sampling without replacement
        u = torch.rand(M - 1, generator=self._g).clamp_(1e-12, 1.0 - 1e-12)
        g = -torch.log(-torch.log(u))
        _, topk = torch.topk(logits + g, k=k_minus_1, largest=True)

        return topk.to(torch.long) + 1  # shift back to ranks in [1..M-1]

    def _temperature(self) -> float:
        # cosine schedule on c in [0,1]
        return self.t_min + 0.5 * (1.0 - math.cos(math.pi * self.c)) * (
            self.t_max - self.t_min
        )
