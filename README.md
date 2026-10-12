# Annotation-Free Representation Learning for Polyp Retrieval in Colonoscopy Videos

Luca Parolari¹, Lamberto Ballan¹, Carlo Biffi²*, Loic Le Folgoc³*

¹ Department of Mathematics, University of Padova, Padova, Italy \
² Cosmo Intelligent Medical Devices, Dublin, Ireland \
³ LTCI, Telecom Paris, Institut Polytechnique de Paris, Palaiseau, France \
\* Shared senior authorship.

Corresponding author: Luca Parolari \<luca.parolari@studenti.unipd.it\>

<!-- TODO: badges and link to the PDF once the paper is out -->

# Abstract

> Colonoscopy is inherently comparative: reliable visual assessment of polyps often depends on complementary views acquired under different viewpoints and imaging conditions.
> Retrieving such views requires polyp tracklet representations, which recent contrastive methods learn from temporal cues but still build from manually annotated bounding boxes.
> We propose an annotation-free framework that constructs tracklet views directly from raw colonoscopy videos, enabling temporal self-supervised learning on full-procedure videos, including frames without visible polyps.
> On REAL-Colon, our method achieves state-of-the-art results on tracklet retrieval and re-identification, outperforming both self-supervised and supervised baselines.
> We then introduce few-shot polyp retrieval, a task closer to clinical use, where a few observations of a target polyp are used to retrieve its remaining tracklets: with only two examples, our method surpasses most competing methods given ten.
> Finally, we show that a modest number of unlabeled procedures suffices for competitive performance, making on-site model development more practical.
> Code is available at https://github.com/lparolari/freerep.

**Index terms**: contrastive learning, self-supervised learning,
annotation-free, retrieval, colonoscopy videos.

# Table of Contents

- [Abstract](#abstract)
- [Structure](#structure)
- [Install](#install)
- [Training](#training)
- [Evaluation Tasks](#evaluation-tasks)
  - [Retrieval](#retrieval)
  - [Few-Shot Retrieval](#few-shot-retrieval)
  - [ReID](#reid)
- [Data](#data)
  - [Collections](#collections)
  - [Datasets](#datasets)
  - [Setup](#setup)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [Baseline Weights](#baseline-weights)
- [License](#license)
- [Cite](#cite)

# Structure

This repository provides the PyTorch reference implementation for the paper
"Annotation-Free Representation Learning for Colonoscopy Video Retrieval".

The repository is structured as follows:

```bash
.
├── data            # <-- read the Data section below to setup
├── polypsense
│   ├── dataset     # <-- tools to load datasets appropriately
│   ├── downstream  # <-- shared data/pooling helpers
│   ├── e2e         # <-- pre-training code with models implementation
│   ├── reid        # <-- retrieval, few-shot retrieval and reidentification tasks
│   └── upstream    # <-- pre-training of the baselines
├── notebooks
│   └── evaluate_fewshot.ipynb  # <-- aggregates few-shot results into tables and plots
├── tools
│   ├── make_rc_loco.ipynb      # <-- builds the leave-one-cohort-out splits
│   └── make_rc_scaling+5fold.ipynb # <-- builds the scaling-study splits
├── requirements.txt
└── README.md       # <-- this file
```

# Install

The code requires Python 3.10+ and a CUDA-capable GPU. All requirements are
listed in [requirements.txt](requirements.txt):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training and evaluation log to [Weights & Biases](https://wandb.ai). Run
`wandb login` once, then pass `--wandb_entity` (your user or team) and
`--wandb_project` to the commands below; omitting `--wandb_entity` uses the
default entity of the logged-in account.

# Training

Use the following command to pre-train the model:

```bash
python -m polypsense.e2e.cli_v3 \
  --mode=train_v2 \
  --trainer=temporalselfsupervised \
  --dataset_root=data/rc41 \
  --num_workers=8 \
  --max_epochs=25 \
  --batch_size=60 \
  --im_size=232 \
  --fragment_length=8 \
  --fragment_stride=4 \
  --fragment_drop_last \
  --min_tracklet_length=30 \
  --bbox_scale_factor=5 \
  --sampler=temporaltemperaturebag \
  --sampler_ttb_tmin=0.3 \
  --sampler_ttb_tmax=12.0 \
  --aug_randomresizedcrop \
  --aug_normalize \
  --n_views=5 \
  --eval_n_views=2 \
  --eval_batch_size=16 \
  --eval_fragment_length=8 \
  --eval_aug_anchorcrop \
  --encoder_type=mve \
  --sfe_backbone_arch=resnet50 \
  --sfe_backbone_weights=IMAGENET1K_V2 \
  --sfe_d_model=2048 \
  --mve_d_model=256 \
  --mve_d_feedforward=1024 \
  --mve_n_heads=8 \
  --mve_n_layers=3 \
  --mve_dropout=0.1 \
  --trainer_d_model=256 \
  --trainer_d_proj=128 \
  --trainer_fragment_pooling=cls \
  --opt_lr=0.00001 \
  --opt_weight_decay=0.0001 \
  --loss_temperature=0.07 \
  --wandb_project=polypsense-e2e \
  --exp_name=my_experiment
```

In the paper *K* is the bag size.
To simplify the implementation, here the bag size is expressed by the `--n_views` and its value must be set to *K+1* to account for the anchor tracklet.

The command above trains on a single GPU, as reported in the paper.

Training can be distributed over several GPUs to make it faster, by switching
the mode, the trainer and the sampler to their distributed counterparts:

```bash
python -m polypsense.e2e.cli_v3 \
  --mode=train_ddp \
  --trainer=distributedtemporalselfsupervised \
  --sampler=distributedtemporaltemperaturebag \
  --engine_num_nodes=1 \
  --engine_devices=4 \
  --engine_strategy=ddp \
  --engine_precision=16-mixed \
  ... # same arguments as above
```

Note that `--batch_size` is per device: the batches built by the sampler are
distributed across ranks, so with *N* devices each optimization step sees *N*
times as many bags as in the single-GPU run.

Checkpoints are written to
`wandb_logs/<wandb_project>/<exp_name>/checkpoints/best.ckpt`, which is what the
evaluation tasks below take as `--encoder_ckpt`.

# Evaluation Tasks

## Retrieval

This task measures the ability to retrieve same-polyp tracklets.
Given a query tracklet, we rank all other tracklets by cosine similarity in the embedding space.
Performance is measured with mean Average Precision (mAP) and Hit Rate@K (HR@K).
Use the following command to execute retrieval task:

```bash
python -m polypsense.reid.cli \
  --task=retrieval \
  --dataset_root=data/rc19 \
  --encoder_type=mve \
  --encoder_ckpt=path/to/checkpoint.ckpt \
  --encoder_pooling=cls \
  --encoder_out_dim=256 \
  --data_fragment_length=-1 \
  --data_im_size=232 \
  --data_aug_anchorcrop \
  --data_aug_normalize \
  --data_bbox_scale_factor=5 \
  --exp_name my_retrieval_exp
```

## Few-Shot Retrieval

Few-shot retrieval measures how well a polyp identity can be retrieved from a
small *support set* of *k* tracklets of that polyp, rather than from a single
query.
For every identity and every *k*, the evaluator samples *k* support tracklets
uniformly at random among the tracklets of the target polyp, scores each
remaining gallery tracklet by its maximum cosine similarity to any support
tracklet, and averages retrieval metrics over many such episodes.
Performance is measured with mAP, HR@{1,5}, Recall@{1,5}, R-Precision, the rank
of the last positive, and the number of false positives before full recall.

```bash
python -m polypsense.reid.cli \
  --task=fewshotretrieval \
  --dataset_root=data/rc19 \
  --encoder_type=mve \
  --encoder_ckpt=path/to/checkpoint.ckpt \
  --encoder_features_source=encoder \
  --encoder_pooling=cls \
  --encoder_out_dim=256 \
  --fsret_gallery_mode=same_video \
  --data_fragment_length=-1 \
  --data_im_size=232 \
  --data_aug_anchorcrop \
  --data_aug_normalize \
  --data_bbox_scale_factor=5 \
  --wandb_project=polypsense-fewshotretrieval \
  --exp_name my_fewshot_exp
```

Each run evaluates one gallery mode, so the two protocols reported in the paper
are two separate runs of the command above.

`--fsret_gallery_mode` controls how the gallery is built:

| Mode | Meaning |
| --- | --- |
| `same_video` | rank only tracklets from the same colonoscopy video as the support set |
| `global` | rank all tracklets in the dataset |

The rest of the protocol is defined by `FewShotConfig`
([`polypsense/reid/eval/fewshotretrieval.py`](polypsense/reid/eval/fewshotretrieval.py))
and is set in `get_evaluator()`
([`polypsense/reid/engine.py`](polypsense/reid/engine.py)):

| Field | Value used | Meaning |
| --- | --- | --- |
| `k_shots` | `(1, 2, 3, 5, 10)` | support-set sizes to evaluate |
| `n_repeats` | `50` | random support sets sampled per identity and *k* |
| `seed` | `42` | episode sampling seed |

The run writes the results for every *k* to
`output/fewshotretrieval/<exp_name>/outputs.pth`, under keys of the form
`k<K>/random/max_sim`.
Use [`notebooks/evaluate_fewshot.ipynb`](notebooks/evaluate_fewshot.ipynb) to
aggregate those files across models into the tables and plots: fill in the
`EXP_NAMES` dictionary in the second cell with your run names, then run the
notebook top to bottom.

The support set is sampled uniformly among the tracklets of the target polyp,
modelling an average review scenario rather than a curated selection.
Results are averaged over *R* = 50 repetitions for *K* ∈ {1, 2, 3, 5, 10}.

## ReID

Re-identification measures the ability to decide whether two tracklets depict the same polyp entity or not.
Using frozen embeddings, we compute the similarity between each embedding pair and threshold the scores to classify same-polyp vs different-polyp pairs.
Performance is measured with AUROC and AUPR.
Use the following command to execute re-identification task:

```bash
python -m polypsense.reid.cli \
  --task=reid \
  --dataset_root=data/rc19 \
  --encoder_type=mve \
  --encoder_ckpt=path/to/checkpoint.ckpt \
  --encoder_pooling=cls \
  --encoder_out_dim=256 \
  --data_fragment_length=-1 \
  --data_im_size=232 \
  --data_aug_anchorcrop \
  --data_aug_normalize \
  --data_bbox_scale_factor=5 \
  --exp_name my_reid_exp
```

Baselines are evaluated with the same commands by swapping `--encoder_type` and
the weights: `endofm`, `endofmlv`, `endovit`, `surgenet`, `dinov2`, `dinov3` and
`vjepa2` take pre-trained weights through `--encoder_pretrained` instead of
`--encoder_ckpt`, and may need a different `--encoder_pooling`
(e.g. `mean-mean` for DINOv2, `cls-mean` for DINOv3).
See [Baseline Weights](#baseline-weights) for where those weights come from.

# Data

## Collections

All our experiments use REAL-Colon, which contains 60 full-procedure
colonoscopy videos recorded at six clinical centers on three continents, with
equipment from two manufacturers.
The following table describes the dataset's main properties.
We report the link to download the raw dataset, including original videos or
frames.

| Dataset | Frames | Videos | Polyps | Link |
| --- | --- | --- | --- | --- |
| REAL-Colon | 2757723 | 60 | 132 | [[download]](https://doi.org/10.25452/figshare.plus.22202866) |

## Datasets

From this collection of datasets we derive the splits used for training the model or evaluating it on downstream tasks.
You can download the annotations and follow the instructions in [Setup](#setup) to prepare the data.

| Task | Dataset | Download |
| --- | --- | --- |
| Training | REAL-Colon (training set), aka `rc41` | [[annotations]](https://drive.google.com/drive/folders/1RlyrA3h9UM6E_14ZTolc81y3_yERTTAj?usp=drive_link) |
| Retrieval, Few-shot Retrieval & Re-identification | REAL-Colon (val+test set), aka `rc19` | [[annotations]](https://drive.google.com/drive/folders/1ig1b_6dpzzTkvf_wkB0Olox80bVCZjaZ?usp=sharing) |

## Setup

All annotation files are structured in [COCO format](https://cocodataset.org/#format-data) with the following modifications:

```python
{
  "images": [
    {
      "id": int,
      "file_name": str,
      "width": int,
      "height": int,
      "frame_id": int,    # <-- position of a frame in the video
      "sequence_id": str, # <-- video label (e.g. 001-001)
      "dataset_id": str,  # <-- dataset label (e.g. dataset0)
    },
  ],
  "annotations": [
    {
      "id", int,
      "image_id", int,
      "bbox": list[int],
      "identity_id": str,  # <-- label of the polyp entity to which this annotation belongs to
      # optional extra metadata (histology, size)
    }
  ]
}
```

**Training**. You must download the `rc41` training annotations.
These cover the 41 REAL-Colon videos used without any annotation: the 27 videos
with polyps and the 14 videos without, for 1,725,537 frames in total, of which
only 171,957 (about 10%) show at least one polyp.
Our dataloader requires the following folder structure:
```
rc41
|-- images  # <-- folder with all jpg files from REAL-Colon datasets
|   |-- 001-001_1.jpg
|   |-- 001_001_2.jpg
|       ...
|   `-- 004-015_54673.jpg
`-- annotations
    |-- instances_train.json  # <-- train annotations
    |-- instances_val.json    # <-- val annotations
    `-- instances_test.josn   # <-- test annotations
```

**Retrieval/Few-shot Retrieval/Re-identification**. You must download
retrieval/re-identification annotations.
Those include the 19-video test set (47 polyps) from REAL-Colon, which contains
only tracklets with visible polyps.
Our dataloader requires the following folder structure:
```
rc19
|-- images  # <-- same as before
|   |-- 001-001_1.jpg
|   |-- 001_001_2.jpg
|   |   ...
|   `-- 004-015_54673.jpg
`-- annotations
    `-- instances_test.josn  # <-- different from previous one, this includes 19 unseen videos from REAL-Colon
```

**Scaling-study splits**.
[`tools/make_rc_scaling+5fold.ipynb`](tools/make_rc_scaling+5fold.ipynb) builds
the nested video subsets used to measure how performance scales with the number
of training videos, with five independent folds per size, all listed in the
notebook.
Our method never uses polyp boxes, but the dataloader still reads annotation
files to build tracklets, so each fragment receives a bounding-box trajectory
sampled from a pool of real annotations and rescaled to the frame size, with the
identity set per video: the boxes carry no polyp identity information.

**Leave-one-cohort-out (LOCO) splits**. REAL-Colon videos come from four
acquisition cohorts (`001-` to `004-`).
[`tools/make_rc_loco.ipynb`](tools/make_rc_loco.ipynb) derives the four LOCO
training sets from the `rc41` training annotations by removing one cohort at a
time, writing each to `data/rc41_loco_<cohort>/annotations/instances_train.json`.
Each of them is used as `--dataset_root` for a separate training run, evaluated
on the unchanged `rc19` test set.

Both notebooks rewrite only `instances_train.json`. Since the training
dataloader also reads `instances_val.json` and `instances_test.json`, copy those
two files from `data/rc41/annotations` into each generated dataset folder before
training on it.

# Pre-trained Checkpoints

You can download and use our pre-trained model at this
[link](https://drive.google.com/file/d/1B5t5uVCv_5WVMdBbhAqs3is-PUKjq_lo/view?usp=sharing)
(`ee1175.ckpt`).

# Baseline Weights

We did not retrain the baselines: each one is evaluated with the weights
released by its authors, passed to `--encoder_pretrained`.

| `--encoder_type` | Weights |
| --- | --- |
| `endofm` | [med-air/Endo-FM](https://github.com/med-air/Endo-FM) |
| `endofmlv` | [med-air/EndoFM-LV](https://github.com/med-air/EndoFM-LV) |
| `surgenet` | [timjaspers0801/surgenet](https://github.com/timjaspers0801/surgenet) |
| `endovit` | [DominikBatic/EndoViT](https://github.com/DominikBatic/EndoViT) |
| `dinov2`, `dinov3`, `vjepa2` | Hugging Face Hub |

The first four take the path of a local checkpoint file, e.g.
`--encoder_pretrained=cache/EndoFM/endo_fm.pth`.
The Hugging Face ones accept either a model id
(e.g. `--encoder_pretrained=facebook/dinov2-base`) or the path of a locally
downloaded snapshot, which is what we use so that the runs need no network
access.

# License

This code accompanies a paper that is currently under review and is provided
for review purposes only. All rights reserved; see [LICENSE](LICENSE).
The model implementations under `polypsense/e2e/modeling` are adapted from the
repositories listed in [Baseline Weights](#baseline-weights) and remain subject
to their original licenses.

# Cite

If you find this work useful, please cite our paper:

```bibtex
TODO: bibtex
```
