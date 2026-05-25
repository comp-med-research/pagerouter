"""
Figure generation for all experiments.

All plot functions save to figures/ and return the matplotlib Figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MLP_LABEL_ORDER = ["per_page", "per_doc_type", "per_layout", "per_stratum"]
MLP_LABEL_COLORS = {
    "per_page": "#4c72b0",
    "per_doc_type": "#55a868",
    "per_layout": "#8172b3",
    "per_stratum": "#c44e52",
}
MLP_LABEL_PRETTY = {
    "per_page": "per-page label",
    "per_doc_type": "per-page-type label",
    "per_layout": "per-layout label",
    "per_stratum": "per-stratum label",
}


def _baseline_ned_from_df(baselines: pd.DataFrame, method: str, ned_col: str) -> float | None:
    sub = baselines[baselines["method"] == method]
    if sub.empty:
        return None
    return float(sub.iloc[0][ned_col])


def _active_mlp_label_order(mlp: pd.DataFrame) -> list[str]:
    present = set(mlp["label_type"].dropna().astype(str))
    return [lt for lt in MLP_LABEL_ORDER if lt in present]


def _draw_mlp_ablation_baselines(
    ax,
    baselines: pd.DataFrame,
    *,
    ned_col: str,
    test_set_label: str,
) -> list[float]:
    """Reference lines for oracle and train/test lookup baselines."""
    refs: list[float] = []

    def _ned(method: str) -> float | None:
        return _baseline_ned_from_df(baselines, method, ned_col)

    oracle_ned = _ned("oracle_upper_bound")
    best_omni_single = _ned("best_single_train_champion")
    best_fixed_test = _ned("best_fixed_on_test")
    doc_type_ned = _ned("best_per_doc_type_table")
    layout_ned = _ned("best_per_layout_table")
    stratum_ned = _ned("best_per_stratum_table")

    if oracle_ned is not None:
        ax.axvline(
            oracle_ned,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.3,
            label=f"Oracle-1 ({oracle_ned:.3f})",
        )
        refs.append(oracle_ned)
    if best_fixed_test is not None:
        ax.axvline(
            best_fixed_test,
            color="#9467bd",
            linestyle="-.",
            linewidth=1.4,
            label=f"Best fixed model ({test_set_label}) ({best_fixed_test:.3f})",
        )
        refs.append(best_fixed_test)
    if best_omni_single is not None:
        ax.axvline(
            best_omni_single,
            color="#8c8c8c",
            linestyle=":",
            linewidth=1.2,
            label=f"Best Omni single (train set) ({best_omni_single:.3f})",
        )
        refs.append(best_omni_single)
    if doc_type_ned is not None:
        ax.axvline(
            doc_type_ned,
            color="#55a868",
            linestyle=(0, (5, 2)),
            linewidth=1.2,
            alpha=0.85,
            label=f"Best Omni mean page type (train set) ({doc_type_ned:.3f})",
        )
        refs.append(doc_type_ned)
    if layout_ned is not None:
        ax.axvline(
            layout_ned,
            color="#8172b3",
            linestyle=(0, (3, 2, 1, 2)),
            linewidth=1.2,
            alpha=0.85,
            label=f"Best Omni mean layout (train set) ({layout_ned:.3f})",
        )
        refs.append(layout_ned)
    if stratum_ned is not None:
        ax.axvline(
            stratum_ned,
            color="#e6a817",
            linestyle=(0, (3, 1, 1, 1)),
            linewidth=1.3,
            label=f"Best Omni mean doc×layout (train set) ({stratum_ned:.3f})",
        )
        refs.append(stratum_ned)
    return refs


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


def plot_mlp_ablation(
    ablation_df: "pd.DataFrame",
    out_path: str | Path,
    *,
    test_set_label: str = "Real5",
) -> None:
    """Grouped horizontal bars: per encoder, per-page vs per-stratum MLP vs reference lines."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = ablation_df.copy()
    ned_col = "mean_ned_real5"
    std_col = "std_ned_real5"
    gap_col = "oracle_gap_recovered"

    baselines = df[df["kind"] == "baseline"]
    mlp = df[df["kind"] == "mlp"].copy()
    if mlp.empty:
        raise ValueError("No MLP rows in ablation table.")

    enc_order = sorted(mlp["encoder"].unique(), key=lambda x: str(x))
    enc_display = [str(e).replace("_omni", "") for e in enc_order]
    y_label = "Layout encoder"
    if "feature_mode" in mlp.columns:
        modes = set(mlp["feature_mode"].dropna().astype(str).str.strip()) - {"", "nan"}
        if modes == {"image"} or (len(modes) == 1 and "image" in modes):
            y_label = "Visual encoder"
        elif modes != {"layout"}:
            y_label = "Encoder / feature mode"
    label_order = _active_mlp_label_order(mlp)
    if not label_order:
        label_order = ["per_page", "per_stratum"]

    n_enc = len(enc_order)
    n_labels = len(label_order)
    group_gap = 1.0 + 0.12 * max(0, n_labels - 2)
    bar_h = min(0.32, 0.78 / max(n_labels, 1))
    y_centers = np.arange(n_enc) * group_gap

    fig_h = max(4.0, n_enc * (0.28 * n_labels + 0.55) + 2.2)
    fig, ax = plt.subplots(figsize=(9.0 if n_labels > 2 else 8.5, fig_h))

    all_neds: list[float] = []
    for li, lt in enumerate(label_order):
        offset = (li - (n_labels - 1) / 2.0) * bar_h
        neds, stds, gaps = [], [], []
        for enc in enc_order:
            row = mlp[(mlp["encoder"] == enc) & (mlp["label_type"] == lt)]
            if row.empty:
                neds.append(np.nan)
                stds.append(0.0)
                gaps.append(0.0)
            else:
                r = row.iloc[0]
                neds.append(float(r[ned_col]))
                stds.append(float(r[std_col]) if pd.notna(r[std_col]) else 0.0)
                gaps.append(float(r[gap_col]) if pd.notna(r[gap_col]) else 0.0)
        y_pos = y_centers + offset
        bars = ax.barh(
            y_pos,
            neds,
            height=bar_h * 0.92,
            xerr=stds,
            color=MLP_LABEL_COLORS.get(lt, "#999999"),
            alpha=0.9,
            capsize=3,
            label=MLP_LABEL_PRETTY.get(lt, lt),
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        for bar, gp, xerr, ned in zip(bars, gaps, stds, neds):
            if np.isnan(ned):
                continue
            all_neds.append(ned)
            ax.text(
                bar.get_width() + max(xerr, 0.0005) + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{gp:.1%}",
                va="center",
                fontsize=7,
            )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(enc_display)
    ax.set_ylabel(y_label)

    ref_vals = _draw_mlp_ablation_baselines(
        ax, baselines, ned_col=ned_col, test_set_label=test_set_label
    )

    lo = min(all_neds + ref_vals) if (all_neds or ref_vals) else 0.0
    hi = max(all_neds + ref_vals) if (all_neds or ref_vals) else 1.0
    margin = max(0.012, (hi - lo) * 0.28)
    ax.set_xlim(max(0.0, lo - margin), min(1.0, hi + margin))
    ax.set_xlabel(f"Mean NED ({test_set_label})")
    title_kind = "Layout" if y_label == "Layout encoder" else "MLP"
    ax.set_title(f"{title_kind} router ablation ({test_set_label})")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_logistic_ablation(
    ablation_df: "pd.DataFrame",
    out_path: str | Path,
    *,
    test_set_label: str = "Real5",
) -> None:
    """Grouped bars: MLP vs logistic logistic regression on the same frozen embeddings."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = ablation_df.copy()
    ned_col = "mean_ned_real5"
    std_col = "std_ned_real5"
    gap_col = "oracle_gap_recovered"

    baselines = df[df["kind"] == "baseline"]
    routers = df[df["kind"].isin(["mlp", "logistic"])].copy()
    if routers.empty:
        raise ValueError("No MLP/logistic rows in ablation table.")

    enc_order = sorted(routers["encoder"].unique(), key=lambda x: str(x))
    enc_display = [str(e).replace("_omni", "") for e in enc_order]
    series = [
        ("mlp", "per_page", "#4c72b0", "MLP per-page"),
        ("mlp", "per_stratum", "#8da0cb", "MLP per-stratum"),
        ("logistic", "per_page", "#c44e52", "Logistic per-page"),
        ("logistic", "per_stratum", "#e7969c", "Logistic per-stratum"),
    ]

    n_enc = len(enc_order)
    group_gap = 1.15
    bar_h = 0.18
    y_centers = np.arange(n_enc) * group_gap

    fig_h = max(4.5, n_enc * 1.35 + 2.0)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))

    all_neds: list[float] = []
    for si, (router_kind, lt, color, label) in enumerate(series):
        offset = (si - 1.5) * bar_h
        neds, stds, gaps = [], [], []
        for enc in enc_order:
            row = routers[
                (routers["encoder"] == enc)
                & (routers["kind"] == router_kind)
                & (routers["label_type"] == lt)
            ]
            if row.empty:
                neds.append(np.nan)
                stds.append(0.0)
                gaps.append(0.0)
            else:
                r = row.iloc[0]
                neds.append(float(r[ned_col]))
                stds.append(float(r[std_col]) if pd.notna(r[std_col]) else 0.0)
                gaps.append(float(r[gap_col]) if pd.notna(r[gap_col]) else 0.0)
        y_pos = y_centers + offset
        bars = ax.barh(
            y_pos,
            neds,
            height=bar_h * 0.92,
            xerr=stds,
            color=color,
            alpha=0.92,
            capsize=2,
            label=label,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        for bar, gp, xerr, ned in zip(bars, gaps, stds, neds):
            if np.isnan(ned):
                continue
            all_neds.append(ned)
            ax.text(
                bar.get_width() + max(xerr, 0.0005) + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{gp:.1%}",
                va="center",
                fontsize=6,
            )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(enc_display)
    ax.set_ylabel("Encoder")

    ref_vals = _draw_mlp_ablation_baselines(
        ax, baselines, ned_col=ned_col, test_set_label=test_set_label
    )

    lo = min(all_neds + ref_vals) if (all_neds or ref_vals) else 0.0
    hi = max(all_neds + ref_vals) if (all_neds or ref_vals) else 1.0
    margin = max(0.012, (hi - lo) * 0.28)
    ax.set_xlim(max(0.0, lo - margin), min(1.0, hi + margin))
    ax.set_xlabel(f"Mean NED ({test_set_label})")
    ax.set_title(f"MLP vs logistic router ({test_set_label})")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_multimodal_ablation(
    ablation_df: "pd.DataFrame",
    out_path: str | Path,
    *,
    test_set_label: str = "Real5",
) -> None:
    """Grouped bars by feature modality (image / layout / image+layout / …)."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from pagerouter.mlp_features import FEATURE_MODE_LABELS

    df = ablation_df.copy()
    ned_col = "mean_ned_real5"
    std_col = "std_ned_real5"
    gap_col = "oracle_gap_recovered"

    baselines = df[df["kind"] == "baseline"]
    mlp = df[df["kind"] == "mlp"].copy()
    if mlp.empty:
        raise ValueError("No MLP rows in ablation table.")
    if "feature_mode" not in mlp.columns:
        raise ValueError("Multimodal plot requires feature_mode column.")

    mode_order = ["image", "layout", "image_layout", "layout_metadata", "image_metadata", "all"]
    modes = [m for m in mode_order if m in set(mlp["feature_mode"].astype(str))]
    mode_labels = [FEATURE_MODE_LABELS.get(m, m) for m in modes]

    label_order = _active_mlp_label_order(mlp) or ["per_page", "per_stratum"]

    n_modes = len(modes)
    n_labels = len(label_order)
    group_gap = 1.0 + 0.12 * max(0, n_labels - 2)
    bar_h = min(0.32, 0.78 / max(n_labels, 1))
    y_centers = np.arange(n_modes) * group_gap

    fig_h = max(4.5, n_modes * (0.28 * n_labels + 0.55) + 2.0)
    fig, ax = plt.subplots(figsize=(9.0 if n_labels > 2 else 8.5, fig_h))

    all_neds: list[float] = []
    for li, lt in enumerate(label_order):
        offset = (li - (n_labels - 1) / 2.0) * bar_h
        neds, stds, gaps = [], [], []
        for mode in modes:
            row = mlp[(mlp["feature_mode"] == mode) & (mlp["label_type"] == lt)]
            if row.empty:
                neds.append(np.nan)
                stds.append(0.0)
                gaps.append(0.0)
            else:
                r = row.iloc[0]
                neds.append(float(r[ned_col]))
                stds.append(float(r[std_col]) if pd.notna(r[std_col]) else 0.0)
                gaps.append(float(r[gap_col]) if pd.notna(r[gap_col]) else 0.0)
        y_pos = y_centers + offset
        bars = ax.barh(
            y_pos,
            neds,
            height=bar_h * 0.92,
            xerr=stds,
            color=MLP_LABEL_COLORS.get(lt, "#999999"),
            alpha=0.9,
            capsize=3,
            label=MLP_LABEL_PRETTY.get(lt, lt),
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        for bar, gp, xerr, ned in zip(bars, gaps, stds, neds):
            if np.isnan(ned):
                continue
            all_neds.append(ned)
            ax.text(
                bar.get_width() + max(xerr, 0.0005) + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{gp:.1%}",
                va="center",
                fontsize=7,
            )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(mode_labels)
    ax.set_ylabel("Input features")

    ref_vals = _draw_mlp_ablation_baselines(
        ax, baselines, ned_col=ned_col, test_set_label=test_set_label
    )

    lo = min(all_neds + ref_vals) if (all_neds or ref_vals) else 0.0
    hi = max(all_neds + ref_vals) if (all_neds or ref_vals) else 1.0
    margin = max(0.012, (hi - lo) * 0.28)
    ax.set_xlim(max(0.0, lo - margin), min(1.0, hi + margin))
    ax.set_xlabel(f"Mean NED ({test_set_label})")
    ax.set_title(f"Fusion router ablation ({test_set_label})")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _short_encoder(name: str) -> str:
    s = str(name)
    for old, new in (
        ("doclayout-yolo-stats", "yolo"),
        ("docling-heron-101-stats", "heron"),
        ("layoutlmv3-detected", "layoutlmv3"),
        ("colqwen2-v1.0", "colqwen"),
        ("jina-clip-v2", "jina-clip"),
        ("dinov2-small", "dinov2-s"),
        ("clip-l-14", "clip"),
        ("dit-large", "dit-l"),
    ):
        s = s.replace(old, new)
    return s


def _validation_combo_bar_label(row: pd.Series) -> str:
    enc = str(row["encoder"])
    mode = str(row.get("feature_mode", ""))
    if "+metadata" in enc or mode == "all":
        suffix = "+meta"
    else:
        suffix = ""
    parts = enc.replace("+metadata", "").split("+")
    if len(parts) >= 2:
        return f"{_short_encoder(parts[0])}+{_short_encoder(parts[1])}{suffix}"
    return _short_encoder(enc)


def plot_mlp_validation_shortlist(
    ablation_df: "pd.DataFrame",
    out_path: str | Path,
    *,
    test_set_label: str = "Real5",
) -> None:
    """Three-panel plot for validation shortlist (visual / layout / combo grids).

    Unlike ``plot_mlp_multimodal_ablation``, which assumes one config per
    (feature_mode, label_type), this shows every shortlist configuration.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = ablation_df.copy()
    ned_col = "mean_ned_real5"
    std_col = "std_ned_real5"
    gap_col = "oracle_gap_recovered"

    baselines = df[df["kind"] == "baseline"]
    mlp = df[df["kind"] == "mlp"].copy()
    if mlp.empty:
        raise ValueError("No MLP rows in ablation table.")

    visual = mlp[mlp["feature_mode"] == "image"].copy()
    layout = mlp[mlp["feature_mode"] == "layout"].copy()
    combo = mlp[mlp["feature_mode"].isin(["image_layout", "all"])].copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, max(5.0, 0.38 * max(len(visual), len(layout), len(combo) // 2) + 3)),
        gridspec_kw={"width_ratios": [1.0, 1.0, 2.4]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    def _panel(
        ax,
        panel_df: pd.DataFrame,
        y_labels: list[str],
        *,
        title: str,
        color_by_label: bool = True,
        bar_colors: list[str] | None = None,
    ) -> list[float]:
        panel_df = panel_df.copy()
        panel_df["_y"] = pd.Categorical(
            panel_df["_plot_y"].astype(str),
            categories=y_labels,
            ordered=True,
        )
        panel_df = panel_df.sort_values("_y")
        n = len(y_labels)
        y_pos = np.arange(n)
        neds, stds, gaps, colors = [], [], [], []
        for i, yl in enumerate(y_labels):
            row = panel_df[panel_df["_plot_y"].astype(str) == yl]
            if row.empty:
                neds.append(np.nan)
                stds.append(0.0)
                gaps.append(0.0)
                colors.append("#cccccc")
            else:
                r = row.iloc[0]
                neds.append(float(r[ned_col]))
                stds.append(float(r[std_col]) if pd.notna(r[std_col]) else 0.0)
                gaps.append(float(r[gap_col]) if pd.notna(r[gap_col]) else 0.0)
                if bar_colors is not None:
                    colors.append(bar_colors[i % len(bar_colors)])
                elif color_by_label:
                    colors.append(MLP_LABEL_COLORS.get(str(r["label_type"]), "#999999"))
                else:
                    colors.append("#4c72b0")

        bars = ax.barh(
            y_pos,
            neds,
            height=0.72,
            xerr=stds,
            color=colors,
            alpha=0.92,
            capsize=2,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        all_neds: list[float] = []
        for bar, gp, xerr, ned in zip(bars, gaps, stds, neds):
            if np.isnan(ned):
                continue
            all_neds.append(ned)
            ax.text(
                bar.get_width() + max(xerr, 0.0005) + 0.0015,
                bar.get_y() + bar.get_height() / 2,
                f"{ned:.3f} ({gp:.0%})",
                va="center",
                fontsize=6,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"Mean NED ({test_set_label})", fontsize=8)
        ref_vals = _draw_mlp_ablation_baselines(
            ax, baselines, ned_col=ned_col, test_set_label=test_set_label
        )
        vals = all_neds + ref_vals
        if vals:
            lo, hi = min(vals), max(vals)
            margin = max(0.008, (hi - lo) * 0.22)
            ax.set_xlim(max(0.0, lo - margin), min(1.0, hi + margin))
        return all_neds

    # Visual-only panel
    visual = visual.sort_values(ned_col, ascending=True)
    visual["_plot_y"] = visual["encoder"].map(_short_encoder)
    visual_y = visual["_plot_y"].astype(str).tolist()
    _panel(axes[0], visual, visual_y, title="Visual-only")

    # Layout-only panel
    layout = layout.sort_values(ned_col, ascending=True)
    layout["_plot_y"] = layout.apply(
        lambda r: f"{_short_encoder(r['encoder'])} ({MLP_LABEL_PRETTY.get(str(r['label_type']), r['label_type'])})",
        axis=1,
    )
    layout_y = layout["_plot_y"].astype(str).tolist()
    _panel(axes[1], layout, layout_y, title="Layout-only")

    # Combo panel — one bar per config, sorted by NED
    combo = combo.copy()
    if "config_id" in combo.columns:
        combo["_plot_y"] = combo["config_id"].astype(str).str.replace("combo|", "", regex=False)
    else:
        combo["_plot_y"] = combo.apply(_validation_combo_bar_label, axis=1)
    combo["_fusion_kind"] = combo["feature_fusion"].astype(str)
    combo = combo.sort_values(["_fusion_kind", ned_col], ascending=[True, True])
    combo_y = combo["_plot_y"].astype(str).tolist()
    combo_colors = []
    for fusion, mode in zip(combo["feature_fusion"].astype(str), combo["feature_mode"].astype(str)):
        if fusion == "weighted_avg":
            combo_colors.append("#59a14f" if mode == "image_layout" else "#8cd17d")
        elif mode == "all":
            combo_colors.append("#dd8452")
        else:
            combo_colors.append("#4c72b0")
    _panel(
        axes[2],
        combo,
        combo_y,
        title="Combinations (blue/orange=norm_concat, green=weighted_avg proj)",
        color_by_label=False,
        bar_colors=combo_colors,
    )

    handles = [
        plt.Line2D([0], [0], color="#4c72b0", lw=6, label="norm_concat"),
        plt.Line2D([0], [0], color="#dd8452", lw=6, label="norm_concat + metadata"),
        plt.Line2D([0], [0], color="#59a14f", lw=6, label="weighted_avg (proj)"),
        plt.Line2D([0], [0], color="#8cd17d", lw=6, label="weighted_avg (proj) + metadata"),
    ]
    for lt in _active_mlp_label_order(mlp):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=MLP_LABEL_COLORS.get(lt, "#999"),
                lw=6,
                label=MLP_LABEL_PRETTY.get(lt, lt),
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle(f"Validation shortlist ({test_set_label})", fontsize=11, y=1.01)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_routing_results(
    summaries: list[dict],
    oracle_ned: float,
    out_path: str | Path,
    *,
    best_fixed_ned: float | None = None,
    test_set_label: str = "Real5",
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
        of mean page NED). Shown as vertical reference line.
    test_set_label:
        Short name for the target split (e.g. ``\"Real5\"``, ``\"hard296\"``) used in
        axis label and best-fixed-model legend.
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
               label=f"Best fixed model ({test_set_label}) ({best_fixed_ned:.3f})")

    lo = min(min(neds), best_fixed_ned)
    hi = max(max(neds), oracle_ned, best_fixed_ned)
    margin = max(0.01, (hi - lo) * 0.35)
    ax.set_xlim(max(0.0, lo - margin), min(1.0, hi + margin))
    ax.set_xlabel(f"Mean NED ({test_set_label})")
    ax.set_title("Routing Baseline Comparison")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_fusion_method_ablation(
    ablation_df: "pd.DataFrame",
    out_path: str | Path,
    *,
    test_set_label: str = "Real5",
) -> None:
    """Grouped bars by fusion rule (norm_concat, GMU, bilinear, …) on image+layout."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from pagerouter.multimodal_fusion import FUSION_LABELS

    df = ablation_df.copy()
    ned_col = "mean_ned_real5"
    std_col = "std_ned_real5"
    gap_col = "oracle_gap_recovered"

    baselines = df[df["kind"] == "baseline"]
    mlp = df[df["kind"] == "mlp"].copy()
    if mlp.empty:
        raise ValueError("No MLP rows in ablation table.")
    if "feature_fusion" not in mlp.columns:
        raise ValueError("Fusion-method plot requires feature_fusion column.")

    fusion_order = ["norm_concat", "weighted_avg", "gmu", "bilinear", "concat"]
    fusions = [f for f in fusion_order if f in set(mlp["feature_fusion"].astype(str))]
    fusion_labels = [FUSION_LABELS.get(f, f) for f in fusions]

    label_order = _active_mlp_label_order(mlp) or ["per_page", "per_stratum"]

    n_fusions = len(fusions)
    n_labels = len(label_order)
    group_gap = 1.0 + 0.12 * max(0, n_labels - 2)
    bar_h = min(0.32, 0.78 / max(n_labels, 1))
    y_centers = np.arange(n_fusions) * group_gap

    fig_h = max(4.5, n_fusions * (0.28 * n_labels + 0.55) + 2.0)
    fig, ax = plt.subplots(figsize=(9.0 if n_labels > 2 else 8.5, fig_h))

    all_neds: list[float] = []
    for li, lt in enumerate(label_order):
        offset = (li - (n_labels - 1) / 2.0) * bar_h
        neds, stds, gaps = [], [], []
        for fusion in fusions:
            row = mlp[(mlp["feature_fusion"] == fusion) & (mlp["label_type"] == lt)]
            if row.empty:
                neds.append(np.nan)
                stds.append(0.0)
                gaps.append(0.0)
            else:
                r = row.iloc[0]
                neds.append(float(r[ned_col]))
                stds.append(float(r[std_col]) if pd.notna(r[std_col]) else 0.0)
                gaps.append(float(r[gap_col]) if pd.notna(r[gap_col]) else 0.0)
        y_pos = y_centers + offset
        bars = ax.barh(
            y_pos,
            neds,
            height=bar_h * 0.92,
            xerr=stds,
            color=MLP_LABEL_COLORS.get(lt, "#999999"),
            alpha=0.9,
            capsize=3,
            label=MLP_LABEL_PRETTY.get(lt, lt),
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        for bar, gp, xerr, ned in zip(bars, gaps, stds, neds):
            if np.isnan(ned):
                continue
            all_neds.append(ned)
            ax.text(
                bar.get_width() + max(xerr, 0.0005) + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{gp:.1%}",
                va="center",
                fontsize=7,
            )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(fusion_labels)
    ax.set_ylabel("Fusion method")

    ref_vals = _draw_mlp_ablation_baselines(
        ax, baselines, ned_col=ned_col, test_set_label=test_set_label
    )

    lo = min(all_neds + ref_vals) if (all_neds or ref_vals) else 0.0
    hi = max(all_neds + ref_vals) if (all_neds or ref_vals) else 1.0
    margin = max(0.012, (hi - lo) * 0.28)
    ax.set_xlim(max(0.0, lo - margin), min(1.0, hi + margin))
    ax.set_xlabel(f"Mean NED ({test_set_label})")
    ax.set_title(f"Image+layout fusion method ablation ({test_set_label})")
    ax.legend(fontsize=7, loc="lower right")
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
