import torch
from torchvision.tv_tensors import wrap
from tqdm import tqdm


def extract_features(model, dataset, *, encoder_features_source=None):
    """
    Extract features from the model using the provided dataset.
    CUDA device must be available, output will be returned on CPU.

    The dataset returns items with the following keys:

    - clip: the input data [s, c, h, w]
    - label: the corresponding label

    Args:
        model (torch.nn.Module): The model to extract features from.
        dataset (torch.utils.data.Dataset): The dataset providing
            the input data.

    Returns:
        features (torch.Tensor): The extracted features.
        labels (torch.Tensor): The labels corresponding to the extracted features.
    """
    model = model.eval().cuda()

    data = []

    for i, batch in tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc="Extracting features",
    ):
        x = batch["clip"].cuda()  # [s, c, h, w]

        with torch.no_grad():
            out = model(x.unsqueeze(0))  # [1, d]
            out = out.squeeze(0)  # [d]
            out = out.cpu()

        bbox = wrap(batch["bboxes"].float().mean(dim=0), like=batch["bboxes"])

        data.append(
            {
                "features": out,  # [d]
                "bbox": bbox,  # [4]
                "identity": batch["label"],  # [1]
                "video": batch["video"],  # [1]
                "position_video": batch["position_video"],  # [1]
                "position_identity": batch["position_identity"],  # [1]
                "category_id": batch["category_id"],  # [1]
            }
        )  # [n]

    return data


class ReidDataset:
    def __init__(self, data):
        self.data = (
            data  # [n], {features, identity, video, position_video, position_identity}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return torch.utils.data.dataloader.default_collate(batch)
