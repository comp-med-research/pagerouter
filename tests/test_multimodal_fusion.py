"""Tests for multimodal fusion primitives."""

from __future__ import annotations

import pandas as pd
import torch

from pagerouter.mlp_features import build_feature_matrix, fit_metadata_columns
from pagerouter.multimodal_fusion import BilinearFusion, GMUFusion, WeightedAvgFusionRouter


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


def test_weighted_avg_mismatched_dims_deferred_to_router():
    n = 4
    img = torch.randn(n, 10)
    lay = torch.randn(n, 3)
    x = build_feature_matrix(
        "image_layout",
        fusion="weighted_avg",
        page_ids=[f"p{i}" for i in range(n)],
        image=img,
        layout=lay,
    )
    assert x.shape == (n, 13)


def test_weighted_avg_fusion_router_forward():
    n = 5
    d_img, d_layout, d_meta = 1024, 16, 7
    model = WeightedAvgFusionRouter(
        d_img=d_img,
        d_layout=d_layout,
        d_meta=d_meta,
        num_classes=4,
        proj_dim=768,
    )
    x = torch.randn(n, d_img + d_layout + d_meta)
    y = model(x)
    assert y.shape == (n, 4)
    assert model.proj_dim == 768


def test_gmu_bilinear_forward():
    n, d = 2, 6
    g = GMUFusion(d)
    y = g(torch.randn(n, d), torch.randn(n, d))
    assert y.shape == (n, d)

    b = BilinearFusion(d, out_dim=4)
    y2 = b(torch.randn(n, d), torch.randn(n, d))
    assert y2.shape == (n, 4)
