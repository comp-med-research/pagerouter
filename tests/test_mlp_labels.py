import pandas as pd

from pagerouter.mlp_labels import per_doc_type_labels, per_page_oracle_labels, per_stratum_labels
from pagerouter.routing import MODELS


def test_per_page_oracle():
    matrix = pd.DataFrame(
        {
            "a": [0.1, 0.9],
            "b": [0.8, 0.2],
        },
        index=["p1", "p2"],
    )
    lab = per_page_oracle_labels(matrix)
    assert lab.loc["p1"] == "b"
    assert lab.loc["p2"] == "a"


def test_per_stratum_labels():
    matrix = pd.DataFrame(
        {
            "a": [0.4, 0.5, 0.2],
            "b": [0.6, 0.4, 0.8],
        },
        index=["p1", "p2", "p3"],
    )
    df = pd.DataFrame(
        {
            "page_id": ["p1", "p2", "p3"],
            "doc_type": ["book", "book", "note"],
            "layout_type": ["single_column", "single_column", "single_column"],
            "dataset": ["omni", "omni", "omni"],
            "model": [MODELS[0]] * 3,
            "ned_score": [0.0, 0.0, 0.0],
        }
    )
    lab = per_stratum_labels(matrix, df)
    # book: mean a=0.45, b=0.5 → b
    assert lab.loc["p1"] == "b" and lab.loc["p2"] == "b"
    # note: single page p3 → b
    assert lab.loc["p3"] == "b"


def test_per_doc_type_labels():
    matrix = pd.DataFrame(
        {"a": [0.4, 0.5, 0.2], "b": [0.6, 0.4, 0.8]},
        index=["p1", "p2", "p3"],
    )
    df = pd.DataFrame(
        {
            "page_id": ["p1", "p2", "p3"],
            "doc_type": ["book", "book", "note"],
            "layout_type": ["single_column", "double_column", "single_column"],
            "dataset": ["omni", "omni", "omni"],
            "model": [MODELS[0]] * 3,
            "ned_score": [0.0, 0.0, 0.0],
        }
    )
    lab = per_doc_type_labels(matrix, df)
    assert lab.loc["p1"] == "b" and lab.loc["p2"] == "b"
    assert lab.loc["p3"] == "b"


def test_best_single_realized():
    from pagerouter.evaluate import best_single_realized_ned

    train = pd.DataFrame({"a": [0.4, 0.35], "b": [0.5, 0.55]}, index=["p1", "p2"])
    test = pd.DataFrame({"a": [0.2, 0.2], "b": [0.9, 0.1]}, index=["p1", "p2"])
    ned, champ = best_single_realized_ned(train, test)
    assert champ == "b"
    assert abs(ned - 0.5) < 1e-6
