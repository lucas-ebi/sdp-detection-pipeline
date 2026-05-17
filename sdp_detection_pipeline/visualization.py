"""Plotting and visualisation helpers."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logomaker as lm
from prince import MCA
from wordcloud import WordCloud
from Bio.SeqUtils import seq3

# Default output directory – created on first save
OUTPUT_DIR = Path("./output")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Persist a figure to the default output directory."""
    _ensure_output_dir()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")


# ------------------------------------------------------------------- cleansing heatmaps
def plot_cleanse_heatmaps(
    dirty: pd.DataFrame,
    clean: pd.DataFrame,
    save: bool = False,
    show: bool = False,
) -> plt.Figure:
    """Side‑by‑side gap heatmaps before and after cleansing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(dirty.isna().astype(int), cmap="viridis", aspect="auto", extent=[0, 1, 0, 1])
    hm_after = ax2.imshow(
        clean.isna().astype(int), cmap="viridis", aspect="auto", extent=[0, 1, 0, 1]
    )
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(hm_after, cax=cax).set_label("Gaps (Indels)")
    ax1.axis("off")
    ax2.axis("off")

    if save:
        save_figure(fig, "cleanse_heatmaps.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# -------------------------------------------------------------------- Pareto chart
def plot_pareto(
    column_importances: pd.Series,
    save: bool = False,
    show: bool = False,
) -> plt.Figure:
    """Bar chart of per‑column importance with cumulative line (Pareto)."""
    sorted_cols = column_importances.index
    xvalues = range(len(sorted_cols))

    fig, ax1 = plt.subplots(figsize=(16, 4))
    ax1.bar(xvalues, column_importances, color="cyan")
    ax1.set_ylabel("Summed Importance", fontsize=16)
    ax1.tick_params(axis="y", labelsize=12)

    ax2 = ax1.twinx()
    ax2.plot(
        xvalues,
        np.cumsum(column_importances) / np.sum(column_importances),
        color="magenta",
        marker=".",
    )
    ax2.set_ylabel("Cumulative Importance", fontsize=16)
    ax2.tick_params(axis="y", labelsize=12)

    plt.xticks(xvalues, sorted_cols)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

    if save:
        save_figure(fig, "pareto_chart.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# --------------------------------------------------------- Perceptual map (MCA plot)
def plot_perceptual_map(
    mca: MCA,
    unique_sequences: pd.DataFrame,
    coordinates: np.ndarray,
    labels: np.ndarray,
    selected_columns: List[int],
    selected_feature_names: np.ndarray,
    save: bool = False,
    show: bool = False,
) -> plt.Figure:
    """Scatter plot of sequences (MCA coordinates) annotated with SDP residues."""
    # Filter to features belonging to the selected columns
    residue_features = [
        f
        for f in selected_feature_names
        if int(f.rsplit("_", 1)[0]) in selected_columns
    ]

    # Prince index entries use "__" as separator (e.g. "200__A")
    # but residue_features uses "_" (e.g. "200_A")
    coord_df = mca.column_coordinates(unique_sequences[selected_columns])
    pairs = [entry.rsplit("__", 1) for entry in coord_df.index]
    mask = [f"{c}_{a}" in residue_features for c, a in pairs]
    df_res = coord_df[mask]
    df_res.index = pd.Index([f"{seq3(a)}{c}" for c, a in [p for p, m in zip(pairs, mask) if m]])

    fig = plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    legend_handles = []

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        color = plt.cm.viridis(label / len(unique_labels))
        plt.scatter(coordinates[idx, 0], coordinates[idx, 1], color=color, alpha=0.5)
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Cluster {label}",
                markersize=10,
                markerfacecolor=color,
            )
        )

    plt.scatter(df_res[0], df_res[1], marker="*", color="black", s=50)
    for i, (x, y) in enumerate(zip(df_res[0], df_res[1])):
        plt.annotate(df_res.index[i], (x, y), textcoords="offset points", xytext=(0, 10), ha="center")

    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="k", label="Selected Residues", markersize=10)
    )
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(handles=legend_handles, title="Sequence Clusters")
    plt.grid()

    if save:
        save_figure(fig, "perceptual_map.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ------------------------------------------------------------------ Word clouds
def generate_wordclouds(
    cleaned_index: pd.Index,
    unique_sequences: pd.DataFrame,
    labels: np.ndarray,
    metadata: Optional[Union[str, Path]] = None,
    protein_column: str = "Protein names",
    save: bool = False,
    show: bool = False,
) -> Dict[int, str]:
    """
    Build (and optionally plot) word clouds of protein descriptions per cluster.

    `cleaned_index` must be the index of the non‑deduplicated, cleansed DataFrame
    (i.e. `MCAClusterFeatureSelector.input_index_`).
    """
    if metadata is None:
        return {}

    meta_df = pd.read_csv(metadata, delimiter="\t")
    wordcloud_data: Dict[int, str] = {}

    for label in np.unique(labels):
        dedup_indices = unique_sequences.iloc[np.where(labels == label)[0]].index
        headers = cleaned_index[dedup_indices]
        entry_names = [h.split("/")[0] for h in headers]

        subset = meta_df[meta_df["Entry Name"].isin(entry_names)].copy()
        if subset.empty:
            continue

        matches = subset[protein_column].str.extract(
            r"(.+?) ([\w\-,]+ase)", flags=re.IGNORECASE
        )
        substrate = (
            matches[0]
            .fillna("")
            .apply(lambda x: "/".join(re.findall(r"\b(\w+(?:ene|ine|ate|yl))\b", x, flags=re.IGNORECASE)))
            .str.lower()
        )
        enzyme = (
            matches[1]
            .fillna("")
            .apply(lambda x: x.split("-")[-1] if "-" in x else x)
            .str.lower()
        )
        combined = (substrate.str.cat(enzyme, sep=" ")).str.strip()
        text = " ".join(sorted({s for s in combined if s}))
        if text:
            wordcloud_data[label] = text

    if (show or save) and wordcloud_data:
        _plot_wordclouds(wordcloud_data, save=save, show=show)

    return wordcloud_data


def _plot_wordclouds(
    wordcloud_data: Dict[int, str],
    save: bool = False,
    show: bool = False,
) -> None:
    n = len(wordcloud_data)
    fig, axes = plt.subplots(n, 1, figsize=(10, 6 * n))
    if n == 1:
        axes = [axes]

    for ax, (label, text) in zip(axes, wordcloud_data.items()):
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

    if save:
        save_figure(fig, "wordcloud.png")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -------------------------------------------------------------------- Sequence logos
def generate_logos(
    unique_sequences: pd.DataFrame,
    labels: np.ndarray,
    selected_columns: List[int],
    color_scheme: str = "NajafabadiEtAl2017",
    save: bool = False,
    show: bool = False,
) -> Dict[int, pd.DataFrame]:
    """
    Generate sequence logos for each cluster over the selected SDP columns.

    Returns a dictionary mapping cluster label to a frequency DataFrame suitable for `logomaker`.
    """
    valid_schemes = sorted(
        lm.list_color_schemes()
        .loc[lm.list_color_schemes().characters == "ACDEFGHIKLMNPQRSTVWY"]
        .color_scheme.values
    )
    if color_scheme not in valid_schemes:
        raise ValueError(
            f"color_scheme must be one of {valid_schemes}, got {color_scheme!r}"
        )

    logos_data: Dict[int, pd.DataFrame] = {}
    for label in np.unique(labels):
        subset = unique_sequences[sorted(selected_columns)].iloc[
            np.where(labels == label)[0]
        ]
        freq = subset.T.apply(lambda col: col.value_counts(normalize=True), axis=1).fillna(0)
        logos_data[label] = freq

        if save or show:
            msa_columns = freq.index.tolist()
            logo = lm.Logo(
                freq.reset_index(drop=True),
                color_scheme=color_scheme,
                vpad=0.1,
                width=0.8,
            )
            logo.style_spines(visible=False)
            logo.ax.set_xticks(range(len(msa_columns)))
            logo.ax.set_xticklabels(msa_columns, fontsize=24)
            fig = logo.ax.figure
            if save:
                save_figure(fig, f"sdp_logo_{label}.png")
            if show:
                plt.show()
            else:
                plt.close(fig)

    return logos_data