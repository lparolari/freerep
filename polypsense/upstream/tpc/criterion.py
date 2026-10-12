import torch
import torch.nn as nn
import torch.nn.functional as F

class TPCLoss(nn.Module):
    def __init__(self, temperature=0.07, return_logits=False):
        self.temperature = temperature
        self.return_logits = return_logits
        super().__init__()

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]

        # 1. L2-normalize so cosine sim = dot product
        features = F.normalize(features, dim=1)  # [B, D]

        # 2. Cosine similarity matrix, scaled by temperature
        sim = features @ features.T / self.temperature  # [B, B]

        # 3. Build masks
        labels = labels.view(-1, 1)  # [B, 1]
        pos_mask = labels == labels.T  # [B, B]  True where same label
        # Remove self-pairs from positives
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~self_mask  # [B, B]

        # 4. For numerical stability: subtract row max before exp (log-sum-exp trick)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # 5. Denominator: sum over ALL pairs except self
        exp_sim = torch.exp(sim)  # [B, B]
        denom = (exp_sim * ~self_mask).sum(dim=1)  # [B]

        # 6. For each anchor, average log-prob over its positives
        # log P(positive | anchor) = sim_pos - log(denom)
        log_prob = sim - torch.log(denom.unsqueeze(1))  # [B, B]

        # Mean over positive pairs only
        num_positives = pos_mask.sum(dim=1).float()  # [B]
        loss = -(log_prob * pos_mask).sum(dim=1) / num_positives  # [B]

        # Reduce
        loss = loss.mean()

        if self.return_logits:
            return loss, sim

        return loss
