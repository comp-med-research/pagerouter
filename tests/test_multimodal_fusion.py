"""Tests for multimodal fusion primitives."""

from __future__ import annotations

import pandas as pd
import torch

from pagerouter.mlp_features import build_feature_matrix, fit_metadata_columns
from pagerouter.multimodal_fusion import BilinearFusion, GMUFusion


def test_norm_concat_shapes():
    n, d = 4, 8
    img = torch.randn(n, d)
    lay = torch.randn(n, d)
    dummy = pd.DataFrame(
        {
            "page_id": ["a", "b", "c", "d"],
            "doc_type": ["x"] * 4,
            "layout_type": ["y"] * 4,
            "dataset": ["omni"] * 4,
        }
    )
    cols = fit_metadata_columns(dummy)
    x = build_feature_matrix(
        "image_layout",
        fusion="norm_concat",
        page_ids=list("abcd"),
        image=img,
        layout=lay,
        df=dummy,
        metadata_columns=cols,
    )
    assert x.shape == (n, 2 * d + len(cols))


def test_weighted_avg_reduction():
    n, d = 3, 5
    img = torch.randn(n, d)
    lay = torch.randn(n, d)
    x = build_feature_matrix(
        "image_layout",
        fusion="weighted_avg",
        page_ids=["p1", "p2", "p3"],
        image=img,
        layout=lay,
    )
    assert x.shape == (n, d)


def test_gmu_bilinear_forward():
    n, d = 2, 6
    g = GMUFusion(d)
    y = g(torch.randn(n, d), torch.randn(n, d))
    assert y.shape == (n, d)

    b = BilinearFusion(d, out_dim=4)
    y2 = b(torch.randn(n, d), torch.randn(n, d))
    assert y2.shape == (n, 4)
