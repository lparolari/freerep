import logging
import random

import torch
from torchvision.transforms.v2.functional import to_image
from torchvision.tv_tensors import BoundingBoxes, Video


class TPCDataset:
    def __init__(
        self,
        ds,
        *,
        trackleter,
        fragment_length,
        stride,
        sampling_strategy,
        reduce_img_ann_keys,
        reduce_tgt_ann_keys,
        fragment_transforms=None,
    ):
        self.ds = ds
        self.trackleter = trackleter
        self.fragment_length = fragment_length
        self.stride = stride
        self.sampling_strategy = sampling_strategy
        self._reduce_img_ann_keys = reduce_img_ann_keys
        self._reduce_tgt_ann_keys = reduce_tgt_ann_keys
        self.fragment_transforms = fragment_transforms

        self._fragments = []
        self._tracklet_ids = []
        self.ids = []

        self._build_views()
        self._build_tensors()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.id(index)
        x, y = self.img(id), self.tgt_ann(id)

        # we could not map this directly in tgt_ann because tgt_ann is actually
        # used to build the mappings
        y["identity_id"] = self.y2id[y["identity_id"]]  # mapping

        bbox = y["bbox"]

        if self.fragment_transforms:
            x, bbox = self.fragment_transforms(x, bbox)

        target = {**y, "bbox": bbox}

        return x, target

    def id(self, index):
        return self.ids[index]

    def img(self, id):
        fragment = self._fragments[id]
        imgs = [self.ds.img(id) for id in fragment]
        return Video(torch.stack([to_image(img) for img in imgs]))

    def img_ann(self, id):
        fragment = self._fragments[id]
        anns = [self.ds.img_ann(id) for id in fragment]
        ann = self._reduce(anns, self._reduce_img_ann_keys)
        return ann

    def tgt_ann(self, id):
        fragment = self._fragments[id]
        tracklet_id = self._tracklet_ids[id]
        anns = [self.ds.tgt_ann(id) | {"tracklet_id": tracklet_id} for id in fragment]
        tgt_ann = self._reduce(anns, self._reduce_tgt_ann_keys)

        img_ann = self.img_ann(id)

        bbox = BoundingBoxes(
            tgt_ann["bbox"],
            format="xywh",
            canvas_size=(img_ann["height"], img_ann["width"]),
        )

        return {
            **tgt_ann,
            "bbox": bbox,
        }

    def _build_tensors(self):
        v = [self.img_ann(self.id(i))["sequence_id"] for i in range(len(self))]
        y = [self.tgt_ann(self.id(i))["identity_id"] for i in range(len(self))]

        self.v2id = {v: k for k, v in enumerate(sorted(list(set(v))))}
        self.y2id = {v: k for k, v in enumerate(sorted(list(set(y))))}

        self.v = torch.tensor([self.v2id[x] for x in v])
        self.y = torch.tensor([self.y2id[x] for x in y])
        self.t = torch.tensor(
            [self.tgt_ann(self.id(i))["tracklet_id"] for i in range(len(self))]
        )

    def _build_views(self):
        tracklet_data = [t for t in self.trackleter]
        tracklet_label = list(range(len(self.trackleter)))

        fragments = []
        tracklet_ids = []

        skipped = 0

        for i, (tracklet, label) in enumerate(zip(tracklet_data, tracklet_label)):
            if len(tracklet) < 2 * self.fragment_length * self.stride:
                logging.debug(
                    f"Skipped tracklet {i} because it is too short ({len(tracklet)})"
                )
                skipped += 1
                continue
            views = self._sample_views(tracklet)
            fragments += views
            tracklet_ids += [label] * len(views)

        logging.warn(
            f"Skipped {skipped} tracklets out of {len(self.trackleter)} because they were too short"
        )

        self._fragments = fragments
        self._tracklet_ids = tracklet_ids

        self.ids = list(range(len(self._fragments)))

    def _sample_views(self, tracklet):
        if self.sampling_strategy == "extremes":
            return self._sample_views_from_extremes_tracklet(tracklet)
        if self.sampling_strategy == "half":
            return self._sample_views_from_half_tracklet(tracklet)
        raise ValueError(f"Sample strategy '{self.sampling_strategy}' not supported")

    def _sample_views_from_extremes_tracklet(self, tracklet):
        return (
            tracklet[: self.fragment_length * self.stride : self.stride],  # first
            tracklet[-self.fragment_length * self.stride :: self.stride],  # last
        )

    def _sample_views_from_half_tracklet(self, tracklet):
        l = self.fragment_length
        s = self.stride
        n = len(tracklet)
        m = int(n / 2)  # available frames for each fragment

        dilations = [1, 2, 3, 4]

        def select_stride():
            # stride must be selected such that we have enough frames for that
            # stride, i.e. self.s * d < m. give more weight to larger strides
            weights = [int(l * d < m) * d for d in dilations]
            stride = random.choices(dilations, weights=weights)[0]
            return stride

        def select_first_frame(i, stride):
            # first frame must be selected such that we manage to stay within m frames with given stride
            first_frame_range = list(range(i, i + m - l * s)) or [0]
            return random.choice(first_frame_range)

        stride = select_stride()

        # print("stride", stride)

        view_a_i = select_first_frame(0, stride)
        view_b_i = select_first_frame(m, stride)  # or: view_a_i + l * s

        # print("view_a_i", view_a_i)
        # print("view_b_i", view_b_i)

        view_a = tracklet[view_a_i : view_a_i + l * s : s]
        view_b = tracklet[view_b_i : view_b_i + l * s : s]

        return view_a, view_b

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
