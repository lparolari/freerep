import torch


class MappingDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that maps target annotation values according to a provided
    mapping.
    """

    def __init__(self, ds, *, mapping):
        """
        Args:
            ds: The underlying dataset to wrap.
            mapping: A dict where keys are target annotation keys to map, and values
                are either "auto" (to automatically build a mapping) or a dict
                specifying the explicit mapping from original values to new values.
        """
        super().__init__()

        self.ds = ds

        self.direct_mapping = {}
        self.inverse_mapping = {}

        if not isinstance(mapping, dict):
            raise ValueError(f"Mapping must be a dict, got {mapping}.")

        self.mapping_keys = list(mapping.keys())

        for key, map_val in mapping.items():
            if isinstance(map_val, str) and map_val == "auto":
                self._build_auto_mapping(key)
            elif isinstance(map_val, dict):
                self._build_dict_mapping(key, map_val)
            elif callable(map_val):
                self._build_callable_mapping(key, map_val)
            else:
                raise ValueError(
                    f"Mapping for key '{key}' must be 'auto', dict or callable, got {map_val}."
                )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        id = self.id(index)

        img = self.ds.img(id)
        tgt_ann = self.tgt_ann(id)

        return img, tgt_ann

    def tgt_ann(self, id):
        tgt_ann = self.ds.tgt_ann(id)

        mapped = {
            k: self.direct_mapping[k](v)
            for k, v in tgt_ann.items()
            if k in self.mapping_keys
        }

        return {**tgt_ann, **mapped}

    def __getattr__(self, name):
        return getattr(self.ds, name)

    def _build_auto_mapping(self, key):
        unique_values = self._get_unique_values(key)
        sorted_values = sorted(unique_values)
        d = {v: i for i, v in enumerate(sorted_values)}
        d_inv = {i: v for v, i in d.items()}
        self.direct_mapping[key] = lambda v: d[v]
        self.inverse_mapping[key] = lambda v: d_inv[v]

    def _build_dict_mapping(self, key, d):
        unique_values = set(
            self.ds.tgt_ann(self.ds.id(i))[key] for i in range(len(self.ds))
        )
        for v in unique_values:
            if v not in d:
                raise AssertionError(
                    f"Value '{v}' for key '{key}' not found in provided mapping ({d})."
                )
        d_inv = {v: k for k, v in d.items()}
        self.direct_mapping[key] = lambda v: d[v]
        self.inverse_mapping[key] = lambda v: d_inv[v]

    def _build_callable_mapping(self, key, func):
        self.direct_mapping[key] = func
        self.inverse_mapping[key] = None  # Inverse not supported for lambda

    def _get_unique_values(self, key):
        return set(self.ds.tgt_ann(self.ds.id(i))[key] for i in range(len(self.ds)))
