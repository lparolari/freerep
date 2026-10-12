class FragmentDataset:
    """
    Aggregate instances frames and annotations into fragments, and expose a
    coco-like dataset interface with the support to fragment-level data
    handling.

    Example `img_ann`:

    ```
    {
        "id": [1, 2, 3],
        "file_name": ["img1.jpg", "img2.jpg", "img3.jpg"],
        "width": 1920,
        "height": 1080,
        "frame_id": [1, 2, 3],
        "sequence_id": 1
    }
    ```

    Example `tgt_ann`:

    ```
    {
        "id": [101, 102, 103],
        "image_id": [1, 2, 3],
        "category_id": 1,
        "bbox": [[x1, y1, w1, h1], [x2, y2, w2, h2], [x3, y3, w3, h3]],
        "area": [a1, a2, a3],
        "iscrowd": [0, 0, 0],
        "identity_id": "case56",
        "histology": "Traditional serrated adenoma",
        "morphology": "Ip",
        "size": 15,
        "location": "A"
    }
    ```

    """

    def __init__(
        self,
        ds,
        fragments,
        reduce_img_ann_keys,
        reduce_tgt_ann_keys,
    ):
        """
        Args:
            ds: An instance dataset exposing id(idx), img(id), img_ann(id), tgt_ann(id).
            fragments: A list of fragments, where each fragment is a list of instance ids.
            reduce_img_ann_keys: A list of tuples (key, method) specifying how to
                aggregate image annotations for each fragment.
            reduce_tgt_ann_keys: A list of tuples (key, method) specifying how to
                aggregate target annotations for each fragment.
        """
        self.ds = ds
        self._reduce_img_ann_keys = reduce_img_ann_keys
        self._reduce_tgt_ann_keys = reduce_tgt_ann_keys
        self._fragments = fragments
        self.ids = list(range(len(self._fragments)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.id(index)
        return self.img(id), self.tgt_ann(id)

    def id(self, index):
        return self.ids[index]

    def img(self, id):
        fragment = self._fragments[id]
        imgs = [self.ds.img(id) for id in fragment]
        return imgs

    def img_ann(self, id):
        fragment = self._fragments[id]
        anns = [self.ds.img_ann(id) for id in fragment]
        return self._reduce(anns, self._reduce_img_ann_keys)

    def tgt_ann(self, id):
        fragment = self._fragments[id]
        anns = [self.ds.tgt_ann(id) for id in fragment]
        return self._reduce(anns, self._reduce_tgt_ann_keys)

    def _reduce(self, anns, keys):
        """
        Aggregate a list of annotations into a single fragment annotation.

        Raises:
            AssertionError: If multiple values are found for a key that is
                expected to have a single value.

        Returns:
            A dict representing the aggregated fragment's annotation.

        """
        out = {}
        for key, method in keys:
            reduced = reduce(key, anns, method)
            if reduced is None:
                continue
            out[key] = reduced
        return out


def reduce(key, anns, method: str):
    if method == "all":
        return reduce_all(key, anns)
    elif method == "one":
        return reduce_one(key, anns)
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def reduce_all(key, anns):
    if key not in anns[0]:
        return None

    return [ann[key] for ann in anns]


def reduce_one(key: str, anns: list[dict]):
    if key not in anns[0]:
        return None

    assert (
        len(set([ann[key] for ann in anns])) == 1
    ), f"Cannot reduce to one {key}. Multiple values found: {[ann[key] for ann in anns]}."

    return anns[0][key]
