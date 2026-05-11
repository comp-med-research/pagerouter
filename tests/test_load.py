"""
Tests for pagerouter.load — schema validation and basic loading contract.
"""

import pytest
import pandas as pd

from pagerouter.load import validate_schema, EXPECTED_COLUMNS


def _make_valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "page_id":     ["page_001", "page_001", "page_002"],
        "model":       ["chandra2", "docling_ocr", "chandra2"],
        "ned_score":   [0.9, 0.7, 0.85],
        "doc_type":    ["academic", "academic", "book"],
        "layout_type": ["single_column", "single_column", "double_column"],
        "dataset":     ["omni", "omni", "omni"],
    })


def test_validate_schema_passes_on_valid_df():
    df = _make_valid_df()
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_column():
    df = _make_valid_df().drop(columns=["ned_score"])
    with pytest.raises(ValueError, match="ned_score"):
        validate_schema(df)


def test_validate_schema_raises_on_out_of_range_ned():
    df = _make_valid_df()
    df.loc[0, "ned_score"] = 1.5
    with pytest.raises(ValueError):
        validate_schema(df)


def test_validate_schema_raises_on_invalid_dataset():
    df = _make_valid_df()
    df.loc[0, "dataset"] = "unknown_dataset"
    with pytest.raises(ValueError, match="dataset"):
        validate_schema(df)


def test_expected_columns_unchanged():
    assert EXPECTED_COLUMNS == {"page_id", "model", "ned_score", "doc_type", "layout_type", "dataset"}
