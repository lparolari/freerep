
import torch
import torch.nn as nn
import torch.nn.functional as F


class YAwareLoss(nn.Module):
    """
    Implements the Y-Aware Loss presented in slide 41 at
    https://partage.imt.fr/index.php/s/GcwYDdgFXoi8Sci which is very close to
    the loss in paper "Weakly-supervised positional contrastive learning:
    Application to cirrhosis classification"

                  w_k             exp(s_k)
    L_i = - Σ_k ------- log( ---------------- )
                Σ_j w_j      Σ_t=1^N exp(s_t)
    """

    def __init__(self, temperature=0.07, reduction="mean", return_logits=False):
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Unknown reduction: {reduction}")

        self.temperature = temperature
        self.reduction = reduction
        self.return_logits = return_logits
        super().__init__()

    def forward(self, features, weights):
        """
        Args:
            features: A tensor of shape [b, d], the feature representations.
            weights: A tensor of shape [b, b] with positive weights. Optionally, it
                     can be a tensor of shape [b] with integer labels, in which case
                     it will automatically be converted to a binary weights matrix where
                     weights[i, j] = 1 if labels[i] == labels[j], 0 otherwise.
        """
        if weights.dim() == 1:
            weights = torch.eq(weights.view(-1, 1), weights.view(1, -1))  # [b, b]

        if not torch.is_floating_point(weights):
            weights = weights.float()  # might be boolean

        # make sure features are normalized
        features = F.normalize(features, dim=-1, p=2)  # [b, d]

        # compute temperature-scaled similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )  # [b, b]

        # stabilize logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=-1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # [b, b]

        # mask out self-similarities
        diagonal = torch.eye(logits.size(0)).bool().to(logits.device)  # [b, b]
        logits = logits + (diagonal * -1e9)

        # compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [b, b]

        # compute weighted probabilities
        weights = weights * ~diagonal  # [b, b]
        weights = weights / weights.sum(dim=1, keepdim=True)  # [b, b]

        # compute loss per sample
        loss_i = -(weights * log_probs).sum(dim=-1)  # [b]

        if self.reduction == "mean":
            loss = loss_i.mean()
        elif self.reduction == "sum":
            loss = loss_i.sum()
        else:
            loss = loss_i

        if self.return_logits:
            return loss, logits

        return loss


def mil_nce_loss(z, y, *, ignore=None, temperature=0.07, return_logits=False):
    """
    From the paper (Miech et al., 2020). "End-to-End Learning of Visual
    Representations from Uncurated Instructional Videos." CVPR 2020.
    https://doi.org/10.1109/CVPR42600.2020.00990.

    Args:
        z: A tensor of shape [b, d] with features
        y: A tensor of shape [b] with labels
        ignore: A tensor of shape [b, b] with optional mask (0 to keep, 1 to
              ignore). Diagonal masking is applied indipendently.
        temperature: A float for temperature scaling
        return_logits: Whether to return logits along with the loss
    """

    if ignore is None:
        b = y.size(0)
        ignore = torch.zeros((b, b)).bool().to(y.device)

    # make sure features are normalized
    z = F.normalize(z, dim=-1, p=2)  # [b, d]

    # compute temperature-scaled similarity
    anchor_dot_contrast = torch.div(torch.matmul(z, z.T), temperature)  # [b, b]

    # stabilize logits for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=-1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()  # [b, b]

    # mask out self-similarities
    diagonal = torch.eye(logits.size(0)).bool().to(logits.device)  # [b, b]

    # mask out ignored samples
    logits = logits.masked_fill(ignore, float("-inf"))  # [b, b]
    logits = logits.masked_fill(diagonal, float("-inf"))  # [b, b]

    # get positive mask
    pos_mask = (y[:, None] == y[None, :]) & (~ignore) & (~diagonal)  # [b, b]

    assert pos_mask.any(dim=1).all(), "No positive pairs found in at least one batch."

    # numerator: logsumexp over positives only
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))  # [b, b]
    numerator = torch.logsumexp(pos_logits, dim=1)  # [b]

    # denominator: logsumexp over all non-self samples (i.e. both positives
    # and negatives)
    denominator = torch.logsumexp(logits, dim=1)  # [b]

    loss = (denominator - numerator).mean()  # [1]

    if return_logits:
        return loss, logits

    return loss


class MILNCELoss(torch.nn.Module):
    """
    MIL-NCE loss from the paper (Miech et al., 2020).

    See `mil_nce_loss` function for details.
    """

    def __init__(self, temperature=0.07, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.return_logits = return_logits

    def forward(self, z, y, ignore=None):
        """
        z: A tensor of shape [b, d] with features
        y: A tensor of shape [b] with labels
        ignore: A tensor of shape [b, b] with optional mask (0 to keep, 1 to
              ignore). Diagonal masking is applied indipendently.
        """
        return mil_nce_loss(
            z,
            y,
            ignore=ignore,
            temperature=self.temperature,
            return_logits=self.return_logits,
        )


