import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP


# -------------------------------------------------------------------------
# Few-shot config
# -------------------------------------------------------------------------


@dataclass
class FewShotConfig:
    enabled: bool = False

    # Number of support examples.
    k_shots: Tuple[int, ...] = (1, 2, 5)

    # Number of random support sets sampled per identity and k.
    n_repeats: int = 50
    seed: int = 42

    # Gallery construction.
    # - "same_video": retrieve only inside the same colonoscopy/video
    # - "global": retrieve over the whole dataset
    # - "future": retrieve only future items in the same video
    gallery_mode: str = "same_video"

    # If True and video_id exists, identities are interpreted as video-local.
    group_by_video: bool = True

    # Skip episodes where no positive remains after removing the support.
    min_gallery_positives: int = 1


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def _to_python_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if torch.is_tensor(x):
        return x.detach().cpu().flatten().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _collect_field(outputs: List[Dict[str, Any]], key: str, required: bool = False):
    if not all(key in o for o in outputs):
        if required:
            raise KeyError(f"Missing required key '{key}' in test_step outputs.")
        return None

    values = [o[key] for o in outputs]

    if all(torch.is_tensor(v) for v in values):
        return torch.cat([v.detach() for v in values], dim=0)

    flat = []
    for v in values:
        flat.extend(_to_python_list(v))
    return flat


def _gather_meta_tensor(
    meta: Dict[str, Any],
    key: str,
    idx: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    value = meta.get(key, None)
    if value is None:
        return None

    if torch.is_tensor(value):
        return value.to(device)[idx]

    try:
        vals = [value[int(i)] for i in idx.detach().cpu().tolist()]
        return torch.tensor(vals, device=device)
    except Exception:
        return None


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _episode_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    top_ks: Tuple[int, ...] = (1, 5),
) -> Optional[Dict[str, float]]:
    """
    Compute retrieval metrics for a single support-set episode.

    scores: [num_gallery]
    target: [num_gallery] bool, True for same-polyp items
    """
    target = target.bool()
    n_pos = int(target.sum().item())

    if n_pos == 0:
        return None

    order = torch.argsort(scores, descending=True)
    y = target[order].float()

    ranks = torch.arange(1, y.numel() + 1, device=y.device).float()
    cum_pos = torch.cumsum(y, dim=0)

    precision_at_i = cum_pos / ranks
    ap = (precision_at_i * y).sum() / n_pos

    out = {
        "map": float(ap.item()),
    }

    for k in top_ks:
        kk = min(k, y.numel())
        out[f"hr{k}"] = float(y[:kk].any().float().item())
        out[f"recall{k}"] = float((y[:kk].sum() / n_pos).item())

    r = min(n_pos, y.numel())
    out["r_precision"] = float((y[:r].sum() / n_pos).item())

    pos_ranks = torch.nonzero(y > 0, as_tuple=False).flatten()
    last_pos_rank = int(pos_ranks[-1].item() + 1)
    out["last_pos_rank"] = float(last_pos_rank)

    # Number of negatives seen before retrieving all positives.
    out["fp_before_full_recall"] = float(last_pos_rank - n_pos)

    return out


class MetricAccumulator:
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.count = 0

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.count += 1

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        out = {k: v / self.count for k, v in self.sums.items()}
        out["num_episodes"] = float(self.count)
        return out


# -------------------------------------------------------------------------
# Support selection and scoring
# -------------------------------------------------------------------------

# The protocol reported in the paper: support tracklets are sampled uniformly
# at random, and gallery tracklets are scored by their maximum cosine
# similarity to any support tracklet.
SELECTOR_NAME = "random"
SCORER_NAME = "max_sim"


def select_support(
    candidate_idx: torch.Tensor,
    k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample k support indices uniformly among the candidates."""
    perm = torch.randperm(
        candidate_idx.numel(),
        generator=generator,
        device=candidate_idx.device,
    )
    return candidate_idx[perm[:k]]


def score_gallery(
    support_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
) -> torch.Tensor:
    """Maximum cosine similarity to any support item (embeddings are L2-normalized)."""
    sims = gallery_embs @ support_embs.t()
    return sims.max(dim=1).values


# -------------------------------------------------------------------------
# Evaluator
# -------------------------------------------------------------------------


class FewShotRetrievalEvaluator(L.LightningModule):
    """
    Evaluates both:

    1. Standard single-query retrieval:
       each tracklet/clip is used as query.

    2. Few-shot support-set retrieval:
       given K examples of a polyp, retrieve the remaining occurrences.

    Expected batch keys:
        - "features": Tensor [B, D]
        - "identity": Tensor/List [B]

    Optional batch keys:
        - "video_id": Tensor/List [B]
        - "time": Tensor [B], e.g. start frame, center frame, timestamp
        - "quality": Tensor [B], e.g. detector confidence, tracklet length,
          visibility score, bbox area, etc.
        - "tracklet_id": Tensor/List [B]
    """

    def __init__(
        self,
        exp_name: str,
        mode: str = "standard",  # "standard", "fewshot", or "both"
        fewshot_cfg: Optional[FewShotConfig] = None,
    ):
        super().__init__()

        assert mode in {"standard", "fewshot", "both"}

        retrieval_cfg = {"empty_target_action": "error", "ignore_index": -1}
        self.map_metric = RetrievalMAP(**retrieval_cfg)
        self.hr1_metric = RetrievalHitRate(**retrieval_cfg, top_k=1)
        self.hr5_metric = RetrievalHitRate(**retrieval_cfg, top_k=5)

        self.test_step_outputs = []
        self.output_dir = pathlib.Path("output/fewshotretrieval")

        self.exp_name = exp_name
        self.mode = mode
        self.fewshot_cfg = fewshot_cfg or FewShotConfig(enabled=False)

        self.save_hyperparameters(ignore=["fewshot_cfg"])

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(batch)

    def on_test_epoch_end(self):
        embs = _collect_field(self.test_step_outputs, "features", required=True)
        labels = _collect_field(self.test_step_outputs, "identity", required=True)

        video_id = _collect_field(self.test_step_outputs, "video", required=False)
        time = _collect_field(self.test_step_outputs, "position_video", required=False)
        quality = _collect_field(self.test_step_outputs, "quality", required=False)

        embs = embs.to(self.device)
        embs_norm = F.normalize(embs, p=2, dim=-1)

        meta = {
            "video_id": video_id,
            "time": time,
            "quality": quality,
        }

        results = {}

        if self.mode in {"standard", "both"}:
            results["standard"] = self._evaluate_standard(embs_norm, labels)

        if self.mode in {"fewshot", "both"} and self.fewshot_cfg.enabled:
            results["fewshot"] = self._evaluate_fewshot(embs_norm, labels, meta)

        out_file = self.output_dir / self.exp_name / "outputs.pth"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        save_obj = {
            "embs": embs.detach().cpu(),
            "labels": labels.detach().cpu() if torch.is_tensor(labels) else labels,
            "meta": {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in meta.items()
            },
            "results": results,
        }

        torch.save(save_obj, out_file)

        self.test_step_outputs.clear()

    # ------------------------------------------------------------------
    # Standard retrieval
    # ------------------------------------------------------------------

    def _evaluate_standard(self, embs_norm: torch.Tensor, labels: Any):
        if not torch.is_tensor(labels):
            raise TypeError(
                "Standard matrix retrieval currently expects tensor labels. "
                "Use tensor identities or adapt this branch to list labels."
            )

        labels = labels.to(embs_norm.device)
        n = embs_norm.size(0)

        preds = embs_norm @ embs_norm.t()  # [n, n]

        indexes = torch.arange(n, device=embs_norm.device)
        indexes = indexes.unsqueeze(1).expand(-1, n)

        target = labels.unsqueeze(1) == labels.unsqueeze(0)
        target = target.long()
        target.fill_diagonal_(-1)

        self.map_metric(preds, target, indexes)
        self.hr1_metric(preds, target, indexes)
        self.hr5_metric(preds, target, indexes)

        self.log("test_map", self.map_metric, prog_bar=True)
        self.log("test_hr1", self.hr1_metric)
        self.log("test_hr5", self.hr5_metric)

        return {
            "preds": preds.detach().cpu(),
            "labels": labels.detach().cpu(),
        }

    # ------------------------------------------------------------------
    # Few-shot retrieval
    # ------------------------------------------------------------------

    def _evaluate_fewshot(
        self,
        embs_norm: torch.Tensor,
        labels: Any,
        meta: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        cfg = self.fewshot_cfg
        device = embs_norm.device

        labels_py = _to_python_list(labels)
        video_py = _to_python_list(meta.get("video_id", None))

        n = embs_norm.size(0)

        if len(labels_py) != n:
            raise ValueError("labels must contain one identity per embedding.")

        has_video = len(video_py) == n

        if cfg.gallery_mode == "future" and meta.get("time", None) is None:
            raise ValueError("gallery_mode='future' requires meta['time'].")

        groups = self._build_identity_groups(
            labels_py=labels_py,
            video_py=video_py if has_video else None,
            group_by_video=cfg.group_by_video,
        )

        video_to_indices = self._build_video_index(
            n=n,
            video_py=video_py if has_video else None,
        )

        accumulators: Dict[str, MetricAccumulator] = {}

        for k in cfg.k_shots:
            key = f"k{k}/{SELECTOR_NAME}/{SCORER_NAME}"
            accumulators[key] = MetricAccumulator()

            for repeat in range(cfg.n_repeats):
                seed = cfg.seed + 10_000 * repeat + 1_000 * k
                generator = _make_generator(seed, device=device)

                for group_key, candidate_indices in groups.items():
                    video_key, identity = group_key

                    if len(candidate_indices) <= k:
                        continue

                    candidate_idx = torch.tensor(
                        candidate_indices,
                        device=device,
                        dtype=torch.long,
                    )

                    support_idx = select_support(candidate_idx, k, generator)

                    gallery_idx = self._build_gallery_indices(
                        n=n,
                        video_key=video_key,
                        support_idx=support_idx,
                        video_to_indices=video_to_indices,
                        meta=meta,
                        cfg=cfg,
                    )

                    if gallery_idx.numel() == 0:
                        continue

                    target = self._make_gallery_target(
                        gallery_idx=gallery_idx,
                        labels_py=labels_py,
                        identity=identity,
                        device=device,
                    )

                    if int(target.sum().item()) < cfg.min_gallery_positives:
                        continue

                    scores = score_gallery(
                        support_embs=embs_norm[support_idx],
                        gallery_embs=embs_norm[gallery_idx],
                    )

                    metrics = _episode_metrics(scores, target)

                    if metrics is None:
                        continue

                    accumulators[key].update(metrics)

        results = {
            key: acc.compute()
            for key, acc in accumulators.items()
            if acc.count > 0
        }

        for key, metrics in results.items():
            safe_key = key.replace("/", "_")

            for metric_name, value in metrics.items():
                log_name = f"test_fewshot_{safe_key}_{metric_name}"
                self.log(
                    log_name,
                    torch.tensor(value, device=device),
                    prog_bar=("map" in metric_name and "num" not in metric_name),
                )

        return results

    def _build_identity_groups(
        self,
        labels_py: List[Any],
        video_py: Optional[List[Any]],
        group_by_video: bool,
    ) -> Dict[Tuple[Any, Any], List[int]]:
        groups: Dict[Tuple[Any, Any], List[int]] = {}

        for i, identity in enumerate(labels_py):
            if group_by_video and video_py is not None:
                video_key = video_py[i]
            else:
                video_key = "__global__"

            key = (video_key, identity)
            groups.setdefault(key, []).append(i)

        return groups

    def _build_video_index(
        self,
        n: int,
        video_py: Optional[List[Any]],
    ) -> Dict[Any, List[int]]:
        if video_py is None:
            return {"__global__": list(range(n))}

        video_to_indices: Dict[Any, List[int]] = {}

        for i, video in enumerate(video_py):
            video_to_indices.setdefault(video, []).append(i)

        return video_to_indices

    def _build_gallery_indices(
        self,
        n: int,
        video_key: Any,
        support_idx: torch.Tensor,
        video_to_indices: Dict[Any, List[int]],
        meta: Dict[str, Any],
        cfg: FewShotConfig,
    ) -> torch.Tensor:
        device = support_idx.device
        support_set = set(support_idx.detach().cpu().tolist())

        if cfg.gallery_mode == "global":
            gallery = [i for i in range(n) if i not in support_set]

        elif cfg.gallery_mode == "same_video":
            base = video_to_indices.get(video_key, list(range(n)))
            gallery = [i for i in base if i not in support_set]

        elif cfg.gallery_mode == "future":
            base = video_to_indices.get(video_key, list(range(n)))

            support_time = _gather_meta_tensor(
                meta,
                "time",
                support_idx,
                device=device,
            )
            max_support_time = float(support_time.max().item())

            all_idx = torch.tensor(base, device=device, dtype=torch.long)
            all_time = _gather_meta_tensor(meta, "time", all_idx, device=device)

            gallery = []
            for idx, t in zip(base, all_time.detach().cpu().tolist()):
                if idx not in support_set and float(t) > max_support_time:
                    gallery.append(idx)

        else:
            raise ValueError(f"Unknown gallery_mode: {cfg.gallery_mode}")

        return torch.tensor(gallery, device=device, dtype=torch.long)

    def _make_gallery_target(
        self,
        gallery_idx: torch.Tensor,
        labels_py: List[Any],
        identity: Any,
        device: torch.device,
    ) -> torch.Tensor:
        target = [
            labels_py[int(i)] == identity
            for i in gallery_idx.detach().cpu().tolist()
        ]
        return torch.tensor(target, device=device, dtype=torch.bool)
