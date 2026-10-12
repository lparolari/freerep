from torchvision.tv_tensors import BoundingBoxes
from polypsense.e2e.transforms import bboxes_diagonal_ratio
from polypsense.dataset.instance import InstanceDataset
from copy import deepcopy


def select_instances_by_bboxes_diagonal_ratio(
    instance_ds: InstanceDataset, d1: float, d2: float
):
    """
    Return a new instance dataset with instances that have a bounding box diagonal
    ratio in the range [d1, d2).
    """
    ds = deepcopy(instance_ds)

    get_canvas_size = lambda img_ann: (img_ann["height"], img_ann["width"])

    ds.ids = [
        id
        for id in ds.ids
        if d1
        <= bboxes_diagonal_ratio(
            BoundingBoxes(
                ds.tgt_ann(id)["bbox"],
                format="xywh",
                canvas_size=get_canvas_size(ds.img_ann(id)),
            )
        )
        < d2
    ]

    return ds
