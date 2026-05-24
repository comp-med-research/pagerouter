"""Train / eval page splits for MLP router experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def omni_holdout_page_ids(
    omni_mat: pd.DataFrame,
    omni_long: pd.DataFrame,
    *,
    holdout_frac: float,
    split_seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified train / held-out omni ``page_id`` lists by ``doc_type`` × ``layout_type``."""
    pages = sorted(omni_mat.index.astype(str).unique())
    uniq = (
        omni_long[omni_long["dataset"] == "omni"]
        .drop_duplicates(subset=["page_id"])[["page_id", "doc_type", "layout_type"]]
        .copy()
    )
    uniq["page_id"] = uniq["page_id"].astype(str)
    uniq = uniq.set_index("page_id")

    strata: list[str] = []
    for pid in pages:
        if pid in uniq.index:
            dt = str(uniq.loc[pid, "doc_type"])
            lt = str(uniq.loc[pid, "layout_type"])
        else:
            dt, lt = "unknown", "unknown"
        strata.append(f"{dt}||{lt}")

    idx = np.arange(len(pages), dtype=np.int64)
    y = np.asarray(strata, dtype=object)
    try:
        splitter: ShuffleSplit | StratifiedShuffleSplit = StratifiedShuffleSplit(
            n_splits=1, test_size=holdout_frac, random_state=split_seed
        )
        train_rel, eval_rel = next(splitter.split(idx, y))
    except ValueError:
        splitter = ShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=split_seed)
        train_rel, eval_rel = next(splitter.split(idx))
    arr = np.array(pages)
    train_ids = sorted(arr[train_rel].tolist())
    eval_ids = sorted(arr[eval_rel].tolist())
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise RuntimeError(f"OmniDoc holdout leaked pages: {list(overlap)[:5]}")
    return train_ids, eval_ids
