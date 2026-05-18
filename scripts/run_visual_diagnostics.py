"""
Visual diagnostics — tests whether visual similarity predicts model fit
before building the KNN router.

Two diagnostics on OmniDocBench only:
  1. Cluster purity:     k-means clusters, check if best_model is consistent
  2. Neighbor entropy:   for each page, entropy of k nearest neighbors' best_model

Optional ``--umap``: 2-D plots colored by doc_type vs argmax best model. With neighbor entropy
in the ~30–60% band, best-model clouds overlap by construction; compare against doc_type geometry.

Image embeddings (``--backbone``):
  - ``clip``   — OpenCLIP ViT-B/32 (default)
  - ``dinov2`` — DINOv2 ViT-B/14 via PyTorch Hub (``facebookresearch/dinov2``)
  - ``dit``    — Microsoft DiT-base (document image Transformer), sequence mean-pooled

Usage:
  python scripts/run_visual_diagnostics.py
  python scripts/run_visual_diagnostics.py --backbone dinov2 --rebuild
  python scripts/run_visual_diagnostics.py --backbone dit --rebuild
  python scripts/run_visual_diagnostics.py --k-neighbors 10 --k-clusters 5,10,20,50
  python scripts/run_visual_diagnostics.py --umap
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from pagerouter import load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
EMBEDDINGS_DIR = DATA / "embeddings"

N_MODELS = 14  # fixed for max-entropy baseline

BACKBONES = ("clip", "dinov2", "dit")


def _output_suffix(backbone: str) -> str:
    """Keep legacy filenames for the default CLIP run."""
    return "" if backbone == "clip" else f"_{backbone}"


# ─── Embedding ───────────────────────────────────────────────────────────────


class ClipEmbedder:
    """CLIP ViT-B/32 image embedder using open-clip-torch."""

    backbone_key = "clip"
    display_name = "CLIP ViT-B/32"

    def __init__(self) -> None:
        try:
            import open_clip
            import torch
        except ImportError as e:
            sys.exit(
                f"Missing dependency: {e}\n"
                "Install with: pip install open-clip-torch torch"
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
        self._torch = torch
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        print(f"[embed] {self.display_name} loaded on {self.device}")

    def embed_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
    ) -> tuple[np.ndarray, list[int]]:
        """Embed images; returns (embeddings, valid_indices) for paths that loaded."""
        from PIL import Image

        torch = self._torch
        all_embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []

        for batch_start in range(0, len(image_paths), batch_size):
            batch = image_paths[batch_start : batch_start + batch_size]
            tensors = []
            local_valid: list[int] = []
            for local_i, p in enumerate(batch):
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.preprocess(img))
                    local_valid.append(batch_start + local_i)
                except Exception:
                    pass

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy().astype(np.float32))
            valid_indices.extend(local_valid)

            n_done = min(batch_start + batch_size, len(image_paths))
            print(f"\r[embed] {n_done}/{len(image_paths)} images", end="", flush=True)

        print()
        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32), []
        return np.vstack(all_embeddings), valid_indices


class Dinov2Embedder:
    """DINOv2 ViT-B/14 CLS embeddings via ``torch.hub`` (Meta)."""

    backbone_key = "dinov2"
    display_name = "DINOv2 ViT-B/14"

    def __init__(self, hub_name: str = "dinov2_vitb14") -> None:
        try:
            import torch
            from torchvision import transforms
        except ImportError as e:
            sys.exit(
                f"Missing dependency: {e}\n"
                "Install with: pip install torch torchvision"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model = torch.hub.load(
                    "facebookresearch/dinov2",
                    hub_name,
                    trust_repo=True,
                )
            except TypeError:
                self.model = torch.hub.load("facebookresearch/dinov2", hub_name)
        self.model = self.model.to(self.device).eval()
        self.embed_dim = int(getattr(self.model, "embed_dim", 768))
        # Standard ImageNet preprocessing (DINOv2 linear-eval protocol; 224×224).
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        print(
            f"[embed] {self.display_name} ({hub_name}, torch.hub) loaded on {self.device}"
        )

    def _forward_cls(self, batch_tensor):
        """CLS vector, L2-normalized (handles hub variants)."""
        torch = self._torch
        out = self.model.forward_features(batch_tensor)
        if isinstance(out, dict):
            feat = out["x_norm_clstoken"]
        else:
            feat = out[:, 0] if out.dim() == 3 else out
        return feat / feat.norm(dim=-1, keepdim=True)

    def embed_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
    ) -> tuple[np.ndarray, list[int]]:
        from PIL import Image

        torch = self._torch
        all_embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []

        for batch_start in range(0, len(image_paths), batch_size):
            batch = image_paths[batch_start : batch_start + batch_size]
            tensors = []
            local_valid: list[int] = []
            for local_i, p in enumerate(batch):
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.preprocess(img))
                    local_valid.append(batch_start + local_i)
                except Exception:
                    pass

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self._forward_cls(batch_tensor)

            all_embeddings.append(features.cpu().numpy().astype(np.float32))
            valid_indices.extend(local_valid)

            n_done = min(batch_start + batch_size, len(image_paths))
            print(f"\r[embed] {n_done}/{len(image_paths)} images", end="", flush=True)

        print()
        if not all_embeddings:
            return np.empty((0, self.embed_dim), dtype=np.float32), []
        return np.vstack(all_embeddings), valid_indices


class DitEmbedder:
    """Microsoft DiT-base: sequence mean-pool then L2-normalise."""

    backbone_key = "dit"
    display_name = "DiT-base (microsoft/dit-base)"

    def __init__(self, hf_model: str = "microsoft/dit-base") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            sys.exit(
                f"Missing dependency: {e}\nInstall with: pip install transformers torch"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.processor = AutoImageProcessor.from_pretrained(hf_model)
            self.model = AutoModel.from_pretrained(hf_model).to(self.device).eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 768))
        print(f"[embed] {self.display_name} loaded on {self.device}")

    def embed_batch(
        self,
        image_paths: list[Path],
        batch_size: int = 8,
    ) -> tuple[np.ndarray, list[int]]:
        from PIL import Image

        torch = self._torch
        all_embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []

        for batch_start in range(0, len(image_paths), batch_size):
            batch = image_paths[batch_start : batch_start + batch_size]
            images = []
            local_valid: list[int] = []
            for local_i, p in enumerate(batch):
                try:
                    images.append(Image.open(p).convert("RGB"))
                    local_valid.append(batch_start + local_i)
                except Exception:
                    pass

            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
                hidden = out.last_hidden_state
                features = hidden.mean(dim=1)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy().astype(np.float32))
            valid_indices.extend(local_valid)

            n_done = min(batch_start + batch_size, len(image_paths))
            print(f"\r[embed] {n_done}/{len(image_paths)} images", end="", flush=True)

        print()
        if not all_embeddings:
            return np.empty((0, self.hidden_size), dtype=np.float32), []
        return np.vstack(all_embeddings), valid_indices


def make_embedder(backbone: str) -> ClipEmbedder | Dinov2Embedder | DitEmbedder:
    b = backbone.lower().strip()
    if b == "clip":
        return ClipEmbedder()
    if b == "dinov2":
        return Dinov2Embedder()
    if b == "dit":
        return DitEmbedder()
    sys.exit(f"--backbone must be one of {BACKBONES}, got {backbone!r}")


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_paths(backbone: str) -> tuple[Path, Path]:
    b = backbone.lower().strip()
    return (
        EMBEDDINGS_DIR / f"{b}_omni.npy",
        EMBEDDINGS_DIR / f"{b}_omni_ids.npy",
    )


def load_cached_embeddings(backbone: str) -> tuple[np.ndarray, np.ndarray] | None:
    emb_path, ids_path = _cache_paths(backbone)
    if emb_path.exists() and ids_path.exists():
        return np.load(emb_path), np.load(ids_path, allow_pickle=True)
    return None


def save_cached_embeddings(
    backbone: str, embeddings: np.ndarray, page_ids: np.ndarray
) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    emb_path, ids_path = _cache_paths(backbone)
    np.save(emb_path, embeddings)
    np.save(ids_path, page_ids)
    print(f"[embed] Saved {len(embeddings)} embeddings to {emb_path}")


# ─── Data preparation ────────────────────────────────────────────────────────

def prepare_omni(
    omni_path: Path,
    real5_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (matrix, best_model series, meta dataframe) for OmniDocBench."""
    df = load.load_predictions(omni_path, real5_path)
    omni_df = df[df["dataset"] == "omni"].copy()
    matrix = load.get_matrix(omni_df, "omni")
    best_model = matrix.idxmax(axis=1)  # page_id → model name
    meta = (
        omni_df[["page_id", "doc_type", "layout_type"]]
        .drop_duplicates("page_id")
        .set_index("page_id")
    )
    return matrix, best_model, meta


def resolve_image_paths(
    page_ids: np.ndarray, image_dir: Path
) -> tuple[list[Path], list[str]]:
    """Return (paths_that_exist, matching_page_ids)."""
    paths, ids = [], []
    for pid in page_ids:
        p = image_dir / str(pid)
        if p.exists():
            paths.append(p)
            ids.append(str(pid))
    return paths, ids


# ─── Diagnostic 1 — Cluster purity ───────────────────────────────────────────

def run_cluster_purity(
    embeddings: np.ndarray,
    page_ids: np.ndarray,
    best_model: pd.Series,
    k_values: list[int],
) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    rows = []
    for k in k_values:
        print(f"[diag1] k-means k={k} ...", end=" ", flush=True)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        print("done")

        purities, entropies = [], []
        cluster_rows = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.sum() == 0:
                continue
            cluster_page_ids = page_ids[mask]
            cluster_models = best_model.reindex(cluster_page_ids).dropna()
            if len(cluster_models) == 0:
                continue

            counts = cluster_models.value_counts()
            purity = counts.iloc[0] / len(cluster_models)
            probs = counts.values / counts.values.sum()
            ent = scipy_entropy(probs, base=2)

            purities.append(purity)
            entropies.append(ent)
            cluster_rows.append(
                {
                    "k": k,
                    "cluster_id": cluster_id,
                    "n_pages": int(mask.sum()),
                    "purity": purity,
                    "entropy": ent,
                    "majority_model": counts.index[0],
                }
            )
        rows.extend(cluster_rows)

        print(
            f"[diag1] k={k:>2}  mean_purity={np.mean(purities):.3f}  "
            f"mean_entropy={np.mean(entropies):.3f}  "
            f"purity_dist=[{np.min(purities):.2f}, {np.percentile(purities,25):.2f}, "
            f"{np.median(purities):.2f}, {np.percentile(purities,75):.2f}, "
            f"{np.max(purities):.2f}]"
        )

    return pd.DataFrame(rows)


def plot_cluster_purity(
    purity_df: pd.DataFrame, out_path: Path, *, embedder_label: str
) -> None:
    import matplotlib.pyplot as plt

    k_values = sorted(purity_df["k"].unique())
    n = len(k_values)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        sub = purity_df[purity_df["k"] == k]["purity"]
        ax.hist(sub, bins=15, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(sub.mean(), color="crimson", linestyle="--", linewidth=1.5,
                   label=f"mean={sub.mean():.2f}")
        ax.set_title(f"k={k}")
        ax.set_xlabel("Cluster purity")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(f"Cluster purity distribution ({embedder_label}, OmniDocBench)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag1] Saved {out_path}")


# ─── Diagnostic 2 — Neighbor label entropy ───────────────────────────────────

def run_neighbor_entropy(
    embeddings: np.ndarray,
    page_ids: np.ndarray,
    best_model: pd.Series,
    meta: pd.DataFrame,
    k_neighbors: int,
) -> pd.DataFrame:
    try:
        import faiss
    except ImportError:
        sys.exit("Missing dependency: faiss\nInstall with: pip install faiss-cpu")

    print(f"[diag2] Building FAISS index ({len(embeddings)} pages) ...", end=" ", flush=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine on L2-normalised vectors
    index.add(embeddings)
    print("done")

    # +1 because query itself is always the top-1 hit
    _, indices = index.search(embeddings, k_neighbors + 1)

    rows = []
    for i, pid in enumerate(page_ids):
        neighbor_idxs = [j for j in indices[i] if j != i][:k_neighbors]
        neighbor_pids = page_ids[neighbor_idxs]
        neighbor_models = best_model.reindex(neighbor_pids).dropna().values

        if len(neighbor_models) == 0:
            continue

        counts = pd.Series(neighbor_models).value_counts()
        probs = counts.values / counts.values.sum()
        ent = scipy_entropy(probs, base=2)

        own_model = best_model.get(str(pid), np.nan)
        doc_type = meta["doc_type"].get(str(pid), "unknown")
        layout_type = meta["layout_type"].get(str(pid), "unknown")

        rows.append(
            {
                "page_id": str(pid),
                "entropy": ent,
                "own_best_model": own_model,
                "neighbor_best_models": "|".join(neighbor_models.tolist()),
                "doc_type": doc_type,
                "layout_type": layout_type,
            }
        )

    df = pd.DataFrame(rows)

    print(f"[diag2] mean_entropy={df['entropy'].mean():.3f}")
    print("[diag2] Mean entropy by doc_type:")
    by_doc = df.groupby("doc_type")["entropy"].mean().sort_values()
    for dt, ent in by_doc.items():
        print(f"         {dt:<25} {ent:.3f}")
    print("[diag2] Mean entropy by layout_type:")
    by_layout = df.groupby("layout_type")["entropy"].mean().sort_values()
    for lt, ent in by_layout.items():
        print(f"         {lt:<25} {ent:.3f}")

    return df


def plot_neighbor_entropy(
    entropy_df: pd.DataFrame,
    out_dir: Path,
    *,
    embedder_label: str,
    file_suffix: str,
) -> None:
    import matplotlib.pyplot as plt

    # Plot 1 — histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(entropy_df["entropy"], bins=30, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.axvline(entropy_df["entropy"].mean(), color="crimson", linestyle="--",
               linewidth=1.5, label=f"mean={entropy_df['entropy'].mean():.3f}")
    ax.axvline(math.log2(N_MODELS), color="grey", linestyle=":", linewidth=1.2,
               label=f"max={math.log2(N_MODELS):.3f}")
    ax.set_xlabel("Neighbor label entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-page neighbor entropy ({embedder_label}, OmniDocBench)")
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / f"diagnostic_neighbor_entropy_hist{file_suffix}.pdf"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag2] Saved {p1}")

    # Plot 2 — mean entropy by doc_type
    by_doc = entropy_df.groupby("doc_type")["entropy"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(by_doc.index, by_doc.values, color="steelblue", edgecolor="white")
    ax.axvline(entropy_df["entropy"].mean(), color="crimson", linestyle="--",
               linewidth=1.2, label="overall mean")
    ax.axvline(math.log2(N_MODELS), color="grey", linestyle=":", linewidth=1.2,
               label=f"max={math.log2(N_MODELS):.2f}")
    for bar, val in zip(bars, by_doc.values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Mean neighbor entropy (bits)")
    ax.set_title(f"Mean neighbor entropy by doc_type ({embedder_label})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = out_dir / f"diagnostic_neighbor_entropy_by_doctype{file_suffix}.pdf"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag2] Saved {p2}")


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(
    purity_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    *,
    backbone: str,
    embedder_label: str,
) -> None:
    k10_purity = purity_df[purity_df["k"] == 10]["purity"]
    mean_purity = k10_purity.mean() if len(k10_purity) else float("nan")
    median_purity = k10_purity.median() if len(k10_purity) else float("nan")

    mean_entropy = entropy_df["entropy"].mean()
    max_entropy = math.log2(N_MODELS)
    entropy_pct = 100 * mean_entropy / max_entropy

    if entropy_pct < 30:
        verdict = "KNN routing likely viable"
    elif entropy_pct < 60:
        verdict = "partial signal, consider family-level routing"
    else:
        verdict = (
            "visual clusters do not predict model fit,\n"
            "  recommend reframing (see options in pagerouter/DIAGNOSTICS.md)"
        )

    print()
    print("=" * 52)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 52)
    print(f"  Backbone: {backbone} ({embedder_label})")
    print(f"  Cluster purity (k=10):           mean={mean_purity:.3f}, median={median_purity:.3f}")
    print(f"  Neighbor entropy (k=10 neighbors): mean={mean_entropy:.3f}")
    print(f"  Max possible entropy (14 models):  {max_entropy:.3f}")
    print()
    print(f"  Entropy as % of max: {entropy_pct:.1f}%")
    print()
    print("  Verdict:")
    print(f"  - If mean entropy < 30% of max → KNN routing likely viable")
    print(f"  - If mean entropy 30-60% of max → partial signal, consider family-level routing")
    print(f"  - If mean entropy > 60% of max → visual clusters do not predict model fit,")
    print(f"    recommend reframing (see options in pagerouter/DIAGNOSTICS.md)")
    print()
    print(f"  >>> Your result: {entropy_pct:.1f}% — {verdict}")
    print("=" * 52)


def run_umap_plots(
    embeddings: np.ndarray,
    page_ids: np.ndarray,
    best_model: pd.Series,
    meta: pd.DataFrame,
    out_dir: Path,
    *,
    file_suffix: str,
    embedder_label: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> None:
    """2-D UMAP for qualitative geometry; compare doc_type vs argmax-best-model coloring."""
    try:
        import umap
    except ImportError:
        sys.exit(
            "Missing dependency: umap-learn\n"
            "Install with: pip install umap-learn"
        )
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    pids = np.array([str(p) for p in page_ids])
    bm = best_model.reindex(pids)
    doc = meta["doc_type"].reindex(pids)
    mask = bm.notna()
    X = embeddings[mask.to_numpy()]
    bm_ok = bm[mask].astype(str).reset_index(drop=True)
    doc_ok = doc[mask].fillna("unknown").astype(str).reset_index(drop=True)
    if len(X) < 30:
        print("[umap] Skipping UMAP: fewer than 30 embedded pages with labels.")
        return

    print(
        f"[umap] Fitting UMAP (n={len(X)}, n_neighbors={n_neighbors}, "
        f"min_dist={min_dist}) ...",
        flush=True,
    )
    Xs = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        verbose=False,
    )
    z = reducer.fit_transform(Xs)
    print("[umap] Done.")

    def scatter_categories(z_xy: np.ndarray, cats: pd.Series, title: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(11, 8))
        uniques = sorted(cats.unique())
        for c in uniques:
            m = cats.values == c
            ax.scatter(
                z_xy[m, 0],
                z_xy[m, 1],
                s=12,
                alpha=0.72,
                label=c,
                rasterized=True,
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            ncol=2,
            framealpha=0.9,
        )
        fig.tight_layout()
        outp = out_dir / fname
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"[umap] Saved {outp}")

    scatter_categories(
        z,
        bm_ok,
        f"UMAP ({embedder_label}) colored by argmax best model",
        f"diagnostic_umap_best_model{file_suffix}.pdf",
    )
    scatter_categories(
        z,
        doc_ok,
        f"UMAP ({embedder_label}) colored by doc_type",
        f"diagnostic_umap_doc_type{file_suffix}.pdf",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--omni", type=Path, default=DATA / "omni_predictions.csv")
    ap.add_argument("--real5", type=Path, default=DATA / "real5_predictions.csv")
    ap.add_argument(
        "--image-dir", type=Path, default=DATA / "page_images",
        help="Directory containing page image files (default: data/page_images/)",
    )
    ap.add_argument(
        "--backbone",
        choices=BACKBONES,
        default="clip",
        help="Visual embedding model (default: clip)",
    )
    ap.add_argument(
        "--rebuild", action="store_true",
        help="Force recompute embeddings even if cache exists",
    )
    ap.add_argument(
        "--k-neighbors", type=int, default=10,
        help="Number of neighbors for Diagnostic 2 (default: 10)",
    )
    ap.add_argument(
        "--k-clusters", type=str, default="5,10,20,50",
        help="Comma-separated k values for Diagnostic 1 (default: 5,10,20,50)",
    )
    ap.add_argument(
        "--umap",
        action="store_true",
        help="Write 2-D UMAP scatter PDFs (requires umap-learn)",
    )
    ap.add_argument(
        "--umap-neighbors",
        type=int,
        default=25,
        help="UMAP n_neighbors (default: 25)",
    )
    ap.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.08,
        help="UMAP min_dist (default: 0.08)",
    )
    ap.add_argument(
        "--umap-seed",
        type=int,
        default=42,
        help="UMAP random_state (default: 42)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    k_values = [int(k.strip()) for k in args.k_clusters.split(",")]
    backbone = args.backbone.lower().strip()
    suf = _output_suffix(backbone)
    embedder = make_embedder(backbone)

    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── Load labels ──────────────────────────────────────────────────────────
    print("[data] Loading OmniDocBench predictions ...")
    _, best_model, meta = prepare_omni(args.omni, args.real5)
    all_page_ids = best_model.index.to_numpy()
    print(f"[data] {len(all_page_ids)} pages, {len(best_model.unique())} models")

    # ── Embeddings ───────────────────────────────────────────────────────────
    cached = None if args.rebuild else load_cached_embeddings(backbone)
    if cached is not None:
        embeddings, emb_page_ids = cached
        print(
            f"[embed] Loaded {len(embeddings)} cached embeddings from "
            f"{_cache_paths(backbone)[0]}"
        )
    else:
        image_paths, found_ids = resolve_image_paths(all_page_ids, args.image_dir)
        if not image_paths:
            sys.exit(
                f"No images found in {args.image_dir}.\n"
                "Populate data/page_images/ with the OmniDocBench page images\n"
                "and re-run, or point --image-dir to the correct directory."
            )
        print(f"[embed] Found {len(image_paths)}/{len(all_page_ids)} images in {args.image_dir}")
        bs = 8 if backbone == "dit" else 32
        raw_embeddings, valid_indices = embedder.embed_batch(image_paths, batch_size=bs)
        emb_page_ids = np.array([found_ids[i] for i in valid_indices])
        embeddings = raw_embeddings
        save_cached_embeddings(backbone, embeddings, emb_page_ids)

    if len(embeddings) == 0:
        sys.exit("No embeddings produced. Check that images are valid.")

    print(f"[embed] Backbone={backbone} ({embedder.display_name}); {len(embeddings)} embedded pages")

    # ── Diagnostic 1 — Cluster purity ────────────────────────────────────────
    print("\n── Diagnostic 1: Cluster purity ──────────────────────────────────")
    purity_df = run_cluster_purity(embeddings, emb_page_ids, best_model, k_values)
    purity_csv = RESULTS / f"diagnostic_cluster_purity{suf}.csv"
    purity_df.to_csv(purity_csv, index=False)
    print(f"[diag1] Saved {purity_csv}")
    plot_cluster_purity(
        purity_df,
        FIGURES / f"diagnostic_cluster_purity{suf}.pdf",
        embedder_label=embedder.display_name,
    )

    # ── Diagnostic 2 — Neighbor entropy ──────────────────────────────────────
    print(f"\n── Diagnostic 2: Neighbor entropy (k={args.k_neighbors}) ─────────────────")
    entropy_df = run_neighbor_entropy(
        embeddings, emb_page_ids, best_model, meta, args.k_neighbors
    )
    entropy_csv = RESULTS / f"diagnostic_neighbor_entropy{suf}.csv"
    entropy_df.to_csv(entropy_csv, index=False)
    print(f"[diag2] Saved {entropy_csv}")
    plot_neighbor_entropy(
        entropy_df,
        FIGURES,
        embedder_label=embedder.display_name,
        file_suffix=suf,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(purity_df, entropy_df, backbone=backbone, embedder_label=embedder.display_name)

    if args.umap:
        print("\n── Optional: UMAP embedding plots ──────────────────────────────")
        print(
            "[umap] Best-model coloring often looks mixed when neighbor entropy is ~40–60% "
            "of max; compare with doc_type coloring."
        )
        run_umap_plots(
            embeddings,
            emb_page_ids,
            best_model,
            meta,
            FIGURES,
            file_suffix=suf,
            embedder_label=embedder.display_name,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.umap_seed,
        )


if __name__ == "__main__":
    main()
