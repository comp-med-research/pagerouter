# pagerouter

**Page-level analysis and adaptive routing across document parsing models.**

Post-hoc routing from prediction matrices and optional visual/layout encoders. Large artifacts stay local (`data/`, `results/`, `figures/`).

---

## Datasets

| Dataset | Description |
|---|---|
| OmniDocBench v1.5 | Training / in-domain matrix (`dataset == "omni"`) |
| Real5-OmniDocBench Scanning | Cross-domain test (`dataset == "real5"`) |
| hard296 | Held-out Omni slice (see `data/hard296/`) |

---

## Data (gitignored)

- `data/omni_predictions.csv`, `data/real5_predictions.csv`
- `data/page_images/` → symlink or copy from rankshift OmniDocBench images
- `data/embeddings/`, `data/hard296/`

Schema: see `pagerouter/load.py`.

---

## Project layout

```
pagerouter/
├── data/                   # gitignored
├── pagerouter/             # library
│   ├── load.py
│   ├── evaluate.py         # NED / routing metrics
│   ├── mlp_labels.py       # oracle & stratum labels
│   ├── mlp_features.py     # multimodal feature blocks
│   ├── multimodal_fusion.py
│   ├── stratum.py          # stratum mean-champion baseline
│   ├── docling_layout.py   # Heron layout detections
│   ├── agents.py           # VLM agent providers (optional)
│   └── …                   # profiles, clustering, oracle (library)
├── scripts/
│   ├── prepare_hard296_pagerouter.py
│   ├── vlm_finetune/       # VLM routing fine-tune (dataset, LoRA, registry)
│   ├── train_vlm_router.py
│   ├── aggregate_vlm_finetuned.py
│   ├── aggregate_validation_shortlist.py
│   └── run_vlm_router_eval.py
├── results/                # gitignored
├── figures/                # gitignored
└── tests/
```

Legacy experiment drivers (`run_routing`, `train_mlp`, `viz`, DocLayout-YOLO extractors, etc.) were removed from the repo; metrics and validation scripts above remain.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare hard296 slice (needs rankshift `scores.csv` path on your machine):

```bash
python scripts/prepare_hard296_pagerouter.py --help
```

VLM fine-tuning: `python scripts/train_vlm_router.py --help` (writes under `results/vlm_finetuned/`).

---

## Tests

```bash
pytest tests/ -q
```