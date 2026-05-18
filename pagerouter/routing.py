"""
Experiments 4 & 5 — Routing baselines and agentic router.

All routers implement the BaseRouter interface: fit() on training data, predict() on
page-level features or images.

Train set: OmniDocBench (omni).
Test set:  Real5-OmniDocBench Scanning (real5). Cross-domain evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MODELS = [
    "chandra2", "chatgpt_api", "deepseek_ocr_2", "docling_ocr", "dolphin_1_5",
    "dotsocr", "glmocr", "got_ocr2", "hunyuanocr", "mineru_1_2b",
    "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr", "youtu",
]


def _page_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per page_id with doc_type and layout_type."""
    return df[["page_id", "doc_type", "layout_type"]].drop_duplicates("page_id").set_index("page_id")


def _build_features(pages: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode doc_type and layout_type for ML routers."""
    return pd.get_dummies(pages[["doc_type", "layout_type"]], dtype=float)


class BaseRouter(ABC):
    @abstractmethod
    def fit(self, matrix: pd.DataFrame, df: pd.DataFrame) -> "BaseRouter":
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class BestSingleRouter(BaseRouter):
    """Always selects the model with the highest mean NED on the training set."""

    def fit(self, matrix: pd.DataFrame, df: pd.DataFrame) -> "BestSingleRouter":
        self.best_model: str = matrix.mean(axis=0).idxmax()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pages = _page_attrs(df).index
        return pd.Series(self.best_model, index=pages, name="model")


class MetadataRouter(BaseRouter):
    """Maps doc_type → best model from training profiles. Falls back to best single."""

    def fit(self, matrix: pd.DataFrame, df: pd.DataFrame) -> "MetadataRouter":
        self.best_model: str = matrix.mean(axis=0).idxmax()
        attrs = _page_attrs(df)
        page_best = matrix.idxmax(axis=1)  # best model per page
        merged = attrs.join(page_best.rename("best_model"))
        self.doc_type_to_model: dict[str, str] = (
            merged.groupby("doc_type")["best_model"]
            .agg(lambda x: x.value_counts().idxmax())
            .to_dict()
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        attrs = _page_attrs(df)
        selections = attrs["doc_type"].map(self.doc_type_to_model).fillna(self.best_model)
        return selections.rename("model")


class LogisticRouter(BaseRouter):
    """Logistic regression over doc_type + layout_type one-hot features."""

    def fit(self, matrix: pd.DataFrame, df: pd.DataFrame) -> "LogisticRouter":
        attrs = _page_attrs(df)
        target = matrix.idxmax(axis=1).rename("best_model")
        merged = attrs.join(target)

        X = _build_features(merged)
        self._feature_cols = X.columns.tolist()
        self._le = LabelEncoder().fit(merged["best_model"])
        y = self._le.transform(merged["best_model"])

        self.clf = LogisticRegression(max_iter=500, random_state=42)
        self.clf.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        attrs = _page_attrs(df)
        X = _build_features(attrs).reindex(columns=self._feature_cols, fill_value=0.0)
        preds = self._le.inverse_transform(self.clf.predict(X))
        return pd.Series(preds, index=attrs.index, name="model")


class XGBoostRouter(BaseRouter):
    """XGBoost over doc_type + layout_type one-hot features."""

    def fit(self, matrix: pd.DataFrame, df: pd.DataFrame) -> "XGBoostRouter":
        from xgboost import XGBClassifier

        attrs = _page_attrs(df)
        target = matrix.idxmax(axis=1).rename("best_model")
        merged = attrs.join(target)

        X = _build_features(merged)
        self._feature_cols = X.columns.tolist()
        self._le = LabelEncoder().fit(merged["best_model"])
        y = self._le.transform(merged["best_model"])

        self.clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        )
        self.clf.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        attrs = _page_attrs(df)
        X = _build_features(attrs).reindex(columns=self._feature_cols, fill_value=0.0)
        preds = self._le.inverse_transform(self.clf.predict(X))
        return pd.Series(preds, index=attrs.index, name="model")


class AgenticRouter(BaseRouter):
    """VLM routing agent that selects a model given only a page image.

    Zero-shot — fit() is a no-op. Calls the Anthropic Messages API; ``model_id``
    must be a vision-capable Claude model (default ``claude-sonnet-4-6``).
    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(
        self,
        prompt_path: str | Path,
        log_path: str | Path | None = None,
        model_id: str = "claude-sonnet-4-6",
        sample_n: int | None = None,
    ) -> None:
        import anthropic

        prompt_path = Path(prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Routing prompt not found: {prompt_path}")
        self.prompt = prompt_path.read_text().strip()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id
        self.sample_n = sample_n
        self.log_path = Path(log_path) if log_path else Path("results/agentic_router_responses.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def fit(self, matrix: pd.DataFrame, df: pd.DataFrame) -> "AgenticRouter":
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pages = _page_attrs(df)
        if self.sample_n is not None:
            pages = pages.iloc[: self.sample_n]

        selections: dict[str, str] = {}
        for page_id in pages.index:
            image_path = Path("data/page_images") / page_id
            for attempt in range(4):
                try:
                    selections[page_id] = self._route_page(page_id, image_path)
                    break
                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning("page %s attempt %d failed: %s — retrying in %ds",
                                   page_id, attempt + 1, exc, wait)
                    time.sleep(wait)
            else:
                logger.error("page %s: all retries failed, falling back to dotsocr", page_id)
                selections[page_id] = "dotsocr"

        return pd.Series(selections, name="model")

    def _route_page(self, page_id: str, image_path: Path) -> str:
        import base64

        suffix = image_path.suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": media_type,
                                                  "data": image_data}},
                    {"type": "text", "text": self.prompt},
                ],
            }],
        )
        response_text = response.content[0].text.strip()

        # Match response to a known model name (case-insensitive, strip punctuation)
        normalised = response_text.lower().strip(" .,\n")
        matched = next((m for m in MODELS if m == normalised), None)
        if matched is None:
            # Partial match fallback
            matched = next((m for m in MODELS if m in normalised), None)
        if matched is None:
            raise ValueError(f"Could not parse model from response: {response_text!r}")

        with self.log_path.open("a") as fh:
            fh.write(json.dumps({
                "page_id": page_id,
                "response_text": response_text,
                "parsed_model": matched,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n")

        return matched
