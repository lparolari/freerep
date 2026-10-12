import pathlib

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP


class RetrievalEvaluator(L.LightningModule):
    """
    Evaluate the performance of a model on a retrieval task.

    - Obtain embeddings and labels for each sample in the dataset
    - For each sample, do retrieval and evaluate mAP and Hit Rate at k=1 and k=5
    - Average the metrics across all samples
    """

    def __init__(self, exp_name: str):
        super().__init__()
        retrieval_cfg = {"empty_target_action": "error", "ignore_index": -1}
        self.map_metric = RetrievalMAP(**retrieval_cfg)
        self.hr1_metric = RetrievalHitRate(**retrieval_cfg, top_k=1)
        self.hr5_metric = RetrievalHitRate(**retrieval_cfg, top_k=5)
        self.test_step_outputs = []
        self.output_dir = pathlib.Path("output/retrieval")
        self.exp_name = exp_name
        self.save_hyperparameters()

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(batch)

    def on_test_epoch_end(self):
        embs = torch.cat([o["features"] for o in self.test_step_outputs])  # [n, d]
        labels = torch.cat([o["identity"] for o in self.test_step_outputs])  # [n]

        n = embs.size(0)

        embs_norm = F.normalize(embs, p=2, dim=-1)

        preds = embs_norm @ embs_norm.t()  # [n, n]

        indexes = torch.arange(n, device=embs.device)
        indexes = indexes.unsqueeze(1).expand(-1, n)  # [n, n]
        # this is a matrix like
        # [[0,0,0,0 ...]
        #  [1,1,1,1 ...]
        #  ...
        #  [n,n,n,n ...]]
        # this means that all the predictions P[i,:] are associated 
        # with the query i

        target = labels.unsqueeze(1) == labels.unsqueeze(0)  # [n, n]
        target = target.long()
        target.fill_diagonal_(-1)  # mask self-similarity

        self.map_metric(preds, target, indexes)
        self.hr1_metric(preds, target, indexes)
        self.hr5_metric(preds, target, indexes)

        self.log("test_map", self.map_metric, prog_bar=True)
        self.log("test_hr1", self.hr1_metric)
        self.log("test_hr5", self.hr5_metric)

        out_file = self.output_dir / self.exp_name / "outputs.pth"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save((preds, labels, embs), out_file)
