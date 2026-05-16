"""
Figure generation for all experiments.

All plot functions save to figures/ and return the matplotlib Figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_capability_heatmap(
    matrix: pd.DataFrame,
    stratum_col: str,
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Heatmap of mean NED per model per stratum (Experiment 1).

    Parameters
    ----------
    matrix:
        Model × stratum score matrix as returned by profiles.compute_score_matrix().
    stratum_col:
        "doc_type" or "layout_type" — used for axis label.
    out_path:
        Output file path (e.g. figures/capability_heatmap_doctype.pdf).
    title:
        Optional figure title.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(6, len(matrix.columns) * 1.2), max(4, len(matrix) * 0.55)))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        cbar_kws={"label": "Mean NED"},
    )
    ax.set_xlabel(stratum_col.replace("_", " ").title(), labelpad=8)
    ax.set_ylabel("Model", labelpad=8)
    ax.set_title(title or f"Mean NED by {stratum_col.replace('_', ' ').title()}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_clustering_dendrogram(
    linkage_matrix,
    model_labels: list[str],
    out_path: str | Path,
) -> None:
    """Dendrogram of model behavioral clustering (Experiment 2).

    Parameters
    ----------
    linkage_matrix:
        Scipy linkage array as returned by clustering.hierarchical_cluster().
    model_labels:
        Ordered list of model names for leaf labels.
    out_path:
        Output file path.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # Model family groupings for leaf colouring
    FRONTIER   = {"chatgpt_api", "glmocr"}
    SPECIALIST = {"got_ocr2", "hunyuanocr", "youtu", "chandra2", "dotsocr",
                  "dolphin_1_5", "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr"}
    PIPELINE   = {"docling_ocr", "mineru_1_2b", "deepseek_ocr_2"}

    def _colour(name: str) -> str:
        if name in FRONTIER:
            return "#e06c00"
        if name in PIPELINE:
            return "#0068c9"
        return "#2ca02c"

    fig, ax = plt.subplots(figsize=(10, 5))
    ddata = dendrogram(linkage_matrix, labels=model_labels, ax=ax,
                       orientation="top", leaf_rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_color(_colour(lbl.get_text()))

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#e06c00", label="Frontier"),
        Patch(color="#2ca02c", label="Specialist"),
        Patch(color="#0068c9", label="Pipeline"),
    ]
    ax.legend(handles=legend, loc="upper right", framealpha=0.8)
    ax.set_ylabel("Ward Distance")
    ax.set_title("Model Behavioral Clustering (Cosine Distance)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_vs_models(
    matrix: pd.DataFrame,
    out_path: str | Path,
    title: str | None = None,
    *,
    best_model_legend: str | None = None,
    legend_loc: str = "upper right",
    dataset_scope: str | None = None,
    stratum_context: bool = False,
    metric_display_name: str | None = None,
    x_axis_label: str | None = None,
) -> None:
    """Horizontal bars: each model's mean NED vs oracle-1 (Experiment 3).

    Oracle-1 = mean over pages of max_m NED(page, m) — per-page winner, not one model.
    Model bars = column mean (one fixed model on all pages).

    dataset_scope:
        If ``best_model_legend`` is None, choose wording: ``\"omni\"``, ``\"real5\"``, or ``None`` (generic).
    stratum_context:
        If True (stratum slice plots), label the reference line as best model **within that slice**.
    metric_display_name:
        Short name for non-NED scores (e.g. ``\"CDM\"``, ``\"TEDS\"``) — used in the oracle legend
        and default x-axis label unless ``x_axis_label`` is set.
    x_axis_label:
        Overrides the x-axis label (default: ``\"Mean NED\"`` or ``f\"Mean {metric_display_name}\"``).
    """
    import matplotlib.pyplot as plt

    oracle_one = float(matrix.max(axis=1).mean())
    model_means = matrix.mean(axis=0).sort_values(ascending=True)
    best_name = model_means.idxmax()
    best_mean = float(model_means.max())

    fig_h = max(5.0, len(model_means) * 0.38 + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    colors = ["#c44e52" if m == best_name else "#4c72b0" for m in model_means.index]
    ax.barh(model_means.index.astype(str), model_means.values, color=colors, alpha=0.88)

    metric_part = f" ({metric_display_name})" if metric_display_name else ""
    ax.axvline(
        oracle_one,
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        label=f"Oracle-1{metric_part} ({oracle_one:.4f}) — per-page argmax, mean",
    )
    if best_model_legend is not None:
        blegend = best_model_legend
    elif stratum_context:
        blegend = f"Best single model (this stratum): {best_name} ({best_mean:.4f})"
    elif dataset_scope == "real5":
        blegend = f"Best Model on Real5: {best_name} ({best_mean:.4f})"
    elif dataset_scope == "omni":
        blegend = f"Best model (OmniDocBench digital): {best_name} ({best_mean:.4f})"
    else:
        blegend = f"Best fixed model: {best_name} ({best_mean:.4f})"
    ax.axvline(
        best_mean,
        color="#c44e52",
        linestyle=":",
        linewidth=1.4,
        alpha=0.9,
        label=blegend,
    )

    lo = min(float(model_means.min()), best_mean, oracle_one)
    hi = max(float(model_means.max()), best_mean, oracle_one)
    pad = max(0.015, (hi - lo) * 0.12)
    ax.set_xlim(max(0.0, lo - pad), min(1.0, hi + pad))
    if x_axis_label is not None:
        ax.set_xlabel(x_axis_label)
    elif metric_display_name:
        ax.set_xlabel(f"Mean {metric_display_name}")
    else:
        ax.set_xlabel("Mean NED")
    ax.set_ylabel("Model")
    ax.set_title(title or "Oracle-1 vs mean NED of each fixed model")
    ax.legend(loc=legend_loc, fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_vs_models_per_stratum(
    df: pd.DataFrame,
    matrix: pd.DataFrame,
    stratum_col: str,
    out_dir: str | Path,
    *,
    dataset: str,
    min_pages: int = 5,
    legend_loc: str = "upper right",
    metric_display_name: str | None = None,
    x_axis_label: str | None = None,
) -> list[Path]:
    """One oracle-vs-models PDF per distinct stratum value (doc_type or layout_type).

    Restricts to rows where ``df[\"dataset\"] == dataset`` and ``matrix`` rows match.

    Returns list of written paths (skipped strata omitted).
    """
    import re

    out_dir = Path(out_dir)
    sub_df = df[df["dataset"] == dataset]
    attrs = sub_df[["page_id", stratum_col]].drop_duplicates("page_id")

    written: list[Path] = []
    base = out_dir / f"oracle_vs_models_by_{stratum_col}_{dataset}"
    base.mkdir(parents=True, exist_ok=True)

    for raw_val in sorted(attrs[stratum_col].astype(str).unique()):
        page_ids = attrs.loc[attrs[stratum_col].astype(str) == raw_val, "page_id"]
        idx = [p for p in page_ids if p in matrix.index]
        if len(idx) < min_pages:
            continue
        sub_m = matrix.loc[idx]
        if sub_m.shape[0] < min_pages:
            continue

        slug = re.sub(r"[^\w\-+.]+", "_", str(raw_val), flags=re.UNICODE).strip("_")
        slug = slug[:80] if slug else "unknown"
        out_path = base / f"{slug}.pdf"

        title = (
            f"{stratum_col.replace('_', ' ').title()}: {raw_val}\n"
            f"{dataset.upper()} — n_pages={len(idx)}"
        )
        plot_oracle_vs_models(
            sub_m,
            out_path,
            title=title,
            legend_loc=legend_loc,
            dataset_scope=dataset,
            stratum_context=True,
            metric_display_name=metric_display_name,
            x_axis_label=x_axis_label,
        )
        written.append(out_path)

    return written


def plot_oracle_barchart(
    oracle_curve: pd.Series,
    best_single_ned: float,
    out_path: str | Path,
) -> None:
    """Bar chart of oracle NED for k = 1, 2, 3, 5, all (Experiment 3, headline figure).

    Parameters
    ----------
    oracle_curve:
        Series indexed by k, values are oracle mean NED.
    best_single_ned:
        Mean NED of the best **fixed** model (max over models of column mean NED).
        Horizontal reference line — not oracle-1 (that is the k=1 bar).
    out_path:
        Output file path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ks = [str(k) for k in oracle_curve.index]
    ax.bar(ks, oracle_curve.values, color="#4c72b0", alpha=0.85, label="Oracle-k")
    ax.axhline(best_single_ned, color="#c44e52", linestyle="--", linewidth=1.4,
               label=f"Best single ({best_single_ned:.3f})")
    ax.set_xlabel("k (models per page)")
    ax.set_ylabel("Mean NED")
    ax.set_ylim(max(0, best_single_ned - 0.02), min(1.0, oracle_curve.max() + 0.01))
    ax.set_title("Oracle NED vs. Number of Models")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_complementarity_heatmap(
    comp_matrix: pd.DataFrame,
    out_path: str | Path,
) -> None:
    """Heatmap of pairwise model complementarity Φ(i, j) (Experiment 3).

    Parameters
    ----------
    comp_matrix:
        Asymmetric (n_models, n_models) complementarity matrix.
    out_path:
        Output file path.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        comp_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.3,
        cbar_kws={"label": "Φ(i,j): P(i fails, j succeeds)"},
    )
    ax.set_xlabel("j succeeds")
    ax.set_ylabel("i fails")
    ax.set_title("Pairwise Complementarity Φ(i, j)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_curves(
    oracle_curve: pd.Series,
    out_path: str | Path,
) -> None:
    """Line plot of cumulative oracle NED vs. number of models in the pool (Experiment 3).

    Parameters
    ----------
    oracle_curve:
        Series indexed by k, values are oracle mean NED.
    out_path:
        Output file path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    ks = list(oracle_curve.index)
    vals = oracle_curve.values
    ax.plot(ks, vals, marker="o", color="#4c72b0", linewidth=2)

    # Mark elbow: largest drop in marginal gain
    gains = np.diff(vals)
    if len(gains) > 1:
        elbow_idx = int(np.argmax(np.diff(gains) < 0)) + 1
        ax.axvline(ks[elbow_idx], color="#c44e52", linestyle=":", linewidth=1.2,
                   label=f"Elbow k={ks[elbow_idx]}")
        ax.legend()

    ax.set_xlabel("k (models in pool)")
    ax.set_ylabel("Oracle Mean NED")
    ax.set_title("Cumulative Oracle NED vs. Pool Size")
    ax.set_xticks(ks)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_routing_results(
    summaries: list[dict],
    oracle_ned: float,
    out_path: str | Path,
    *,
    best_fixed_ned: float | None = None,
) -> None:
    """Grouped bar or dot chart comparing all router mean NEDs (Experiments 4 & 5).

    Parameters
    ----------
    summaries:
        List of dicts from evaluate.routing_summary(); each has keys
        {label, mean_ned, oracle_gap_pct}.
    oracle_ned:
        Oracle-1 upper bound on the **test** matrix (mean row-wise max).
    best_fixed_ned:
        Mean NED on the **test** set using the single best fixed parser (max over models
        of mean page NED). Shown as vertical reference line (“Best Model on Real5”).
    out_path:
        Output file path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    summaries_sorted = sorted(summaries, key=lambda s: s["mean_ned"])
    labels   = [s["label"] for s in summaries_sorted]
    neds     = [s["mean_ned"] for s in summaries_sorted]
    gap_pcts = [s.get("oracle_gap_pct", 0.0) for s in summaries_sorted]

    if best_fixed_ned is None:
        best_fixed_ned = float(np.max(neds))

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.55 + 1)))
    bars = ax.barh(labels, neds, color="#4c72b0", alpha=0.85)

    for bar, gp in zip(bars, gap_pcts):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{gp:.1%}", va="center", fontsize=8)

    ax.axvline(oracle_ned, color="#2ca02c", linestyle="--", linewidth=1.3,
               label=f"Oracle-1 ({oracle_ned:.3f})")
    ax.axvline(best_fixed_ned, color="#c44e52", linestyle=":", linewidth=1.3,
               label=f"Best Model on Real5 ({best_fixed_ned:.3f})")

    lo = min(min(neds), best_fixed_ned)
    hi = max(max(neds), oracle_ned, best_fixed_ned)
    margin = max(0.01, (hi - lo) * 0.35)
    ax.set_xlim(max(0.0, lo - margin), min(1.0, hi + margin))
    ax.set_xlabel("Mean NED (Real5 test set)")
    ax.set_title("Routing Baseline Comparison")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_agentic_confusion_matrix(
    selections: pd.Series,
    oracle_selections: pd.Series,
    out_path: str | Path,
) -> None:
    """Confusion matrix of agentic router selections vs. oracle selections (Experiment 5).

    Parameters
    ----------
    selections:
        Series of agent-selected model names (index = page_id).
    oracle_selections:
        Series of oracle-selected model names (index = page_id).
    out_path:
        Output file path.
    """
    # TODO: sklearn confusion_matrix, rows = oracle, cols = agent
    # TODO: normalise by row so diagonal shows per-model accuracy
    raise NotImplementedError


def plot_agent_agreement_heatmap(
    ablation_df: "pd.DataFrame",
    out_path: str | Path,
) -> None:
    """Pairwise agent agreement heatmap for the multi-agent ablation.

    Cell [i, j] = fraction of pages where agent i and agent j selected the
    same model (0 = never agree, 1 = always agree; diagonal = 1 by definition).

    Parameters
    ----------
    ablation_df:
        DataFrame loaded from ablation_results.jsonl with columns
        [page_id, agent_name, parsed_model].
    out_path:
        Output file path (e.g. figures/agent_agreement_heatmap.pdf).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    pivot = ablation_df.pivot_table(
        index="page_id", columns="agent_name", values="parsed_model", aggfunc="first"
    )
    agents = sorted(pivot.columns.tolist())
    pivot = pivot[agents]

    n = len(agents)
    agreement = np.zeros((n, n))
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            shared = pivot[[a, b]].dropna()
            if len(shared) == 0:
                agreement[i, j] = float("nan")
            else:
                agreement[i, j] = (shared[a] == shared[b]).mean()

    agreement_df = pd.DataFrame(agreement, index=agents, columns=agents)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4, n * 0.9)))
    sns.heatmap(
        agreement_df,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Fraction of pages in agreement"},
    )
    ax.set_title("Agent pairwise agreement rate")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
