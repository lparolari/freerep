import pathlib

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecisionRecallCurve,
    BinaryROC,
)


class ReidEvaluator(L.LightningModule):
    """
    Evaluate the performance of a model on a re-identification task. We evaluate
    AUC of the ROC and PR curves obtained by binarizing a similarity matrix
    against a set of thresholds.
    """

    def __init__(self, exp_name: str):
        super().__init__()
        self.test_roc_curve = BinaryROC()
        self.test_pr_curve = BinaryPrecisionRecallCurve()
        self.test_auroc = BinaryAUROC()
        self.test_aupr = BinaryAveragePrecision()
        self.output_dir = pathlib.Path("output/reid")
        self.exp_name = exp_name

        self.test_step_outputs = []

        self.save_hyperparameters()

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(batch)

    def on_test_epoch_end(self):
        embs = torch.cat([o["features"] for o in self.test_step_outputs])  # [n, d]
        labels = torch.cat([o["identity"] for o in self.test_step_outputs])  # [n]

        embs_norm = F.normalize(embs, p=2, dim=-1)

        scores = embs_norm @ embs_norm.t()  # [n, n]
        preds = torch.sigmoid(scores)  # [n, n]

        target = labels.unsqueeze(1) == labels.unsqueeze(0)  # [n, n]
        target = target.long()

        mask_symmetric = torch.tril(torch.ones_like(target), diagonal=-1).bool()  # [n, n]  # fmt: skip

        target = target[mask_symmetric]  # [m]
        preds = preds[mask_symmetric]  # [m]

        self.test_roc_curve(preds, target)
        self.test_pr_curve(preds, target)
        self.test_auroc(preds, target)
        self.test_aupr(preds, target)

        self.log("test/auroc", self.test_auroc, prog_bar=True)
        self.log("test/aupr", self.test_aupr, prog_bar=True)

        self.logger.log_image(
            "test/roc_curve", [wandb.Image(self.test_roc_curve.plot()[0])]
        )
        self.logger.log_image(
            "test/pr_curve", [wandb.Image(self.test_pr_curve.plot()[0])]
        )

        out_file = self.output_dir / self.exp_name / "outputs.pth"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save((scores, labels, embs), out_file)
