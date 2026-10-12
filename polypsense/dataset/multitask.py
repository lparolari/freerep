import torch
from torchvision.transforms.v2.functional import to_image
from torchvision.tv_tensors import BoundingBoxes, Video


class MultiTaskDataset:
    def __init__(self, ds, fragment_transforms=None):
        self.ds = ds
        self.frame_transform = None
        self.fragment_transforms = fragment_transforms

    def id(self, index):
        return self.ds.id(index)

    def img(self, id):
        return Video(
            torch.stack([to_image(frame) for frame in self.ds.img(id)])
        )

    def img_ann(self, id):
        return self.ds.img_ann(id)

    def tgt_ann(self, id):
        img_ann = self.img_ann(id)  # needed to set canvas_size for bounding boxes
        tgt_ann = self.ds.tgt_ann(id)

        bboxes = BoundingBoxes(
            tgt_ann["bbox"],
            format="xywh",
            canvas_size=(img_ann["height"], img_ann["width"]),
        )

        return {
            **tgt_ann,
            "bboxes": bboxes,
            "identity": tgt_ann["identity_id"],  # renaming for convenience
        }

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        id = self.id(index)
        x, y = self.img(id), self.tgt_ann(id)

        bboxes = y["bboxes"]

        if self.fragment_transforms:
            x, bboxes = self.fragment_transforms(x, bboxes)

        target = {**y, "bboxes": bboxes}

        return x, target
