# pagerouter

**Page-level analysis and adaptive routing across 14 document parsing models.**

EMNLP 2026 submission.

---

## Central claim

> Given only a page image — no ground truth, no metadata, no labeled data —
> a VLM routing agent selects the best available document parsing model per page,
> recovering X% of the oracle performance gap at the cost of a single lightweight inference call.

---

## Overview

All analysis is post-hoc. No new document parsing inference is run. The core data object is a prediction matrix **P[page, model]** of NED scores: 1,355 pages × 14 models × 2 datasets.

---

## Datasets

| Dataset | Pages | Description |
|---|---|---|
| OmniDocBench v1.5 | 1,355 | 9 doc types, 4 layout types, digital PDFs. E2E quick_match NED. |
| Real5-OmniDocBench Scanning | 1,355 | One-to-one physical scan reconstruction. Same GT as OmniDocBench. |

---

## Models (14)

| Name | Type |
|---|---|
| `chandra2` | Specialist VLM |
| `chatgpt_api` | Frontier VLM (GPT) |
| `deepseek_ocr_2` | Specialist VLM |
| `docling_ocr` | Pipeline (Docling) |
| `dolphin_1_5` | Specialist VLM |
| `dotsocr` | Specialist VLM |
| `glmocr` | Specialist VLM |
| `got_ocr2` | Specialist VLM (GOT-OCR2.0) |
| `hunyuanocr` | Specialist VLM (Hunyuan-OCR) |
| `mineru_1_2b` | Pipeline (MinerU 2.5) |
| `monkeyocr_pro_3b` | Specialist VLM (MonkeyOCR Pro) |
| `paddleocrVL_1_5` | Specialist VLM (PaddleOCR-VL) |
| `rolmocr` | Specialist VLM (RolmOCR) |
| `youtu` | Specialist VLM (Youtu-Parsing) |

---

## Data schema

**TBD — confirm before implementing `pagerouter/load.py`.**

Expected columns after loading:

| Column | Type | Description |
|---|---|---|
| `page_id` | str | TBD — filename? integer index? composite key? |
| `model` | str | Model name (one of 14 above) |
| `ned_score` | float | NED score in [0, 1] (higher = better) |
| `doc_type` | str | TBD — raw string from OmniDocBench? |
| `layout_type` | str | TBD — raw string from OmniDocBench? |
| `dataset` | str | `"omni"` or `"real5"` |

Source CSV files (gitignored):
- `data/omni_predictions.csv`
- `data/real5_predictions.csv`

---

## Project structure

```
pagerouter/
├── data/                   # gitignored
│   ├── omni_predictions.csv
│   ├── real5_predictions.csv
│   └── page_images/        # page images for Experiment 5
├── pagerouter/             # library
│   ├── load.py             # data loading and validation
│   ├── profiles.py         # Experiment 1: capability profiles
│   ├── clustering.py       # Experiment 2: behavioral clustering
│   ├── oracle.py           # Experiment 3: oracle & complementarity
│   ├── routing.py          # Experiments 4 & 5: router classes
│   ├── evaluate.py         # shared evaluation utilities
│   └── viz.py              # figure generation
├── prompts/
│   └── routing_prompt.txt  # agentic router prompt template
├── scripts/
│   ├── run_profiles.py     # Experiment 1
│   ├── run_clustering.py   # Experiment 2
│   ├── run_oracle.py       # Experiment 3
│   ├── run_routing.py      # Experiment 4
│   └── run_agentic_router.py  # Experiment 5
├── results/                # gitignored (output CSVs)
├── figures/                # gitignored (output figures)
├── tests/
│   └── test_load.py
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the experiments

**Experiment 1 — Capability profiles**
```bash
python scripts/run_profiles.py
```
Outputs: `figures/capability_heatmap_doctype.pdf`, `figures/capability_heatmap_layout.pdf`

**Experiment 2 — Behavioral clustering**
```bash
python scripts/run_clustering.py
```
Outputs: `figures/clustering_dendrogram_omni.pdf`, `results/pairwise_cosine_omni.csv`

**Experiment 3 — Oracle & complementarity**
```bash
python scripts/run_oracle.py --threshold 0.8
```
Outputs: `figures/oracle_barchart_omni.pdf`, `figures/complementarity_omni.pdf`, `figures/coverage_curves_omni.pdf`

**Experiment 4 — Lightweight routing baselines**
```bash
python scripts/run_routing.py
# Run specific routers only:
python scripts/run_routing.py --routers best metadata logistic
```
Trains on OmniDocBench, tests on Real5.
Outputs: `figures/routing_results.pdf`

**Experiment 5 — Agentic router**
```bash
export ANTHROPIC_API_KEY=sk-...
# Pilot run (50 pages):
python scripts/run_agentic_router.py --sample-n 50
# Full evaluation:
python scripts/run_agentic_router.py
```
Outputs: `figures/agentic_confusion_real5.pdf`, `results/agentic_router_responses.jsonl`

---

## Environment variables

| Variable | Required by | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Experiment 5 | Anthropic API key for VLM routing calls |

---

## Notes

- Experiment 5 requires page images in `data/page_images/{page_id}.jpg` (or `.png`). The mapping from `page_id` to image filename is TBD — confirm with data schema.
- All routers train on OmniDocBench and test on Real5 (cross-domain evaluation).
- The routing prompt in `prompts/routing_prompt.txt` contains `[TBD]` placeholders for model capability descriptions — fill these in after completing Experiment 1.
