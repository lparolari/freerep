import torch

from polypsense.e2e.modeling.surgenet.metaformer import caformer_s18


class SurgenetBackbone(torch.nn.Module):
    def __init__(self, *, backbone_arch, pretrained_weights):
        super().__init__()

        if backbone_arch not in ["caformer_s18"]:
            raise ValueError(f"Unsupported architecture: {backbone_arch}")

        builder_fn = {"caformer_s18": caformer_s18}[backbone_arch]

        self.model = builder_fn(
            num_classes=1,
            # this value is needed but I think used only for classification or
            # segmentation, not used in feature extraction
            pretrained="SurgeNet",
            # sets the configuration of the model architecture compatible with
            # pretrained weights
            pretrained_weights=pretrained_weights,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: [b, s, c, h, w]

        Returns:
            x: [b, 512]
        """
        # features, features_list = self.model.forward_features(x)

        # features_list, a num_stage list of features with different resolutions
        # - torch.Size([s, 64, 64, 64])
        # - torch.Size([s, 128, 32, 32])
        # - torch.Size([s, 320, 16, 16])
        # - torch.Size([s, 512, 8, 8])

        # features is a tensor of size [s, 512], obtained as the average over
        # the spatial dimensions (h, w) of the last stage features (i.e.
        # features with size [s, 512, 8, 8])

        return torch.stack(
            [self.model.forward_features(fragment)[0] for fragment in x]
        )  # [b, p, d]
