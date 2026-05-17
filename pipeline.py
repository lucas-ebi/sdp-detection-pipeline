#!/usr/bin/env python
"""
SDP Detection Pipeline applies Machine Learning techniques to extract meaningful information from
Multiple Sequence Alignments (MSA) of homologous protein families.

Copyright (C) 2023, Lucas Carrijo de Oliveira (lucas@ebi.ac.uk)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import re
import sys
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker as lm
import nltk
from Bio import SeqIO
from Bio.SeqUtils import seq3
from prince import MCA
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from wordcloud import WordCloud
import fastcluster

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_msa(file_path, file_format='fasta'):
    """Parse an MSA file into a DataFrame (rows = sequences, columns = alignment positions)."""
    headers, sequences = [], []
    for record in SeqIO.parse(file_path, file_format):
        headers.append(record.id)
        sequences.append(list(str(record.seq)))
    return pd.DataFrame(sequences, index=headers)


def map_positions(msa_df):
    """
    Map alignment column indices to per-sequence residue positions.

    Only processes headers matching the 'ID/start-end' convention.
    Returns {header: {col_index: residue_position}}.
    """
    positions_map = defaultdict(dict)
    for header in msa_df.index:
        if re.search(r'.+/\d+-\d+', header):
            sequence = msa_df.loc[header]
            offset, _ = header.split('/')[1].split('-')
            position = int(offset) - 1
            for col, val in zip(sequence.index, sequence.values):
                if val != '-':
                    position += 1
                    positions_map[header][col] = position
    return positions_map


def build_profiles(msa_df, positions_map, selected_columns):
    """
    Build a human-readable profile DataFrame using 3-letter AA codes and residue positions.

    Returns a DataFrame with the same index as msa_df, one column per selected position.
    """
    rows = []
    for header, row in msa_df[sorted(selected_columns)].iterrows():
        pos_map = positions_map.get(header, {})
        rows.append({
            col: f"{seq3(aa.upper())}{pos_map.get(col, '?')}" if aa != '-' else '-'
            for col, aa in row.items()
        })
    return pd.DataFrame(rows, index=msa_df.index, columns=sorted(selected_columns))


# ─── Transformers ─────────────────────────────────────────────────────────────

class CleanseTransformer(BaseEstimator, TransformerMixin):
    """
    Remove gap-heavy columns and rows from an MSA DataFrame.

    fit() learns which columns survive the column-gap threshold on training data.
    transform() reapplies that fixed column set, then prunes rows and deduplicates.
    After transform(), `dirty_` and `clean_` hold pre/post-drop gap masks for plotting.

    Parameters
    ----------
    indel : str
        Gap character (default '-').
    remove_lowercase : bool
        Treat lowercase residues (insert states) as gaps.
    threshold : float
        Minimum fraction of non-gap values required to keep a column or row.
    """

    def __init__(self, indel='-', remove_lowercase=True, threshold=0.9):
        self.indel = indel
        self.remove_lowercase = remove_lowercase
        self.threshold = threshold

    def _mask(self, X):
        # Work on a copy to avoid side effects
        X = X.copy()
        bad = [self.indel]
        if self.remove_lowercase:
            bad += [chr(c) for c in range(ord('a'), ord('z') + 1)]
        else:
            X = X.map(str.upper)
        return X.replace(bad, np.nan)

    def fit(self, X, y=None):
        masked = self._mask(X)
        min_rows = int(self.threshold * masked.shape[0])
        self.kept_columns_ = list(masked.columns[masked.notna().sum(axis=0) >= min_rows])
        if not self.kept_columns_:
            raise ValueError("All columns filtered out; lower threshold.")
        return self

    def transform(self, X):
        masked = self._mask(X)
        self.dirty_ = masked.copy()

        masked = masked[self.kept_columns_]
        min_cols = int(self.threshold * masked.shape[1])
        masked = masked.dropna(thresh=min_cols, axis=0)
        self.clean_ = masked.copy()

        return masked.fillna(self.indel).drop_duplicates()


class MCAClusterFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Chain MCA dimensionality reduction, sequence clustering, and random-forest
    feature selection to identify Specificity-Determining Positions (SDPs).

    `transform` returns the input MSA filtered to `selected_columns_`. All
    intermediate results (MCA, cluster labels, importances) are stored as
    fitted attributes for downstream visualisation.

    Parameters
    ----------
    mca_n_components : int
        Number of MCA dimensions (default 2).
    clustering : {'single-linkage', 'k-means'}
        Clustering algorithm (default 'single-linkage').
    min_clusters, max_clusters : int
        Search range for the optimal cluster count via silhouette score.
    rf_n_estimators : int
        Trees in the random forest (default 1000).
    random_state : int or None
        Random seed applied to both clustering (k-means) and the random forest.
    feature_threshold : str or float
        SelectFromModel importance threshold (e.g. 'median', 'mean').
    importance_cutoff : float
        Cumulative importance fraction used to trim selected columns (default 0.9).
    top_n : int or None
        If set, keep only the top-N most important columns regardless of cutoff.
    """

    def __init__(self, mca_n_components=2, clustering='single-linkage',
                 min_clusters=2, max_clusters=10,
                 rf_n_estimators=1000, random_state=42,
                 feature_threshold='median',
                 importance_cutoff=0.9, top_n=None):
        self.mca_n_components = mca_n_components
        self.clustering = clustering
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.rf_n_estimators = rf_n_estimators
        self.random_state = random_state
        self.feature_threshold = feature_threshold
        self.importance_cutoff = importance_cutoff
        self.top_n = top_n

    def fit(self, X, y=None):
        if self.clustering not in ('k-means', 'single-linkage'):
            raise ValueError("clustering must be 'k-means' or 'single-linkage'")

        # FIX: save original index for mapping back to sequence headers later
        self.input_index_ = X.index

        # X is already cleansed and deduplicated by CleanseTransformer
        self.unique_sequences_ = X.reset_index(drop=True)

        # MCA
        self.mca_ = MCA(n_components=self.mca_n_components)
        self.mca_.fit(self.unique_sequences_)
        self.coordinates_ = np.array(self.mca_.transform(self.unique_sequences_))

        # Clustering
        self.labels_ = self._best_cluster_labels(self.coordinates_)

        # One-hot encode sequences; use column names as feature name prefix
        self.encoder_ = OneHotEncoder(sparse_output=True)
        X_enc = self.encoder_.fit_transform(self.unique_sequences_)
        feature_names = self.encoder_.get_feature_names_out(
            self.unique_sequences_.columns.astype(str)
        )

        # Random forest with cluster labels as target
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state,
        )
        rf.fit(X_enc, self.labels_)

        # Feature selection
        self.selector_ = SelectFromModel(rf, threshold=self.feature_threshold, prefit=True)
        selected_idx = self.selector_.get_support(indices=True)
        self.selected_feature_names_ = feature_names[selected_idx]

        # Aggregate importance per alignment column and rank
        col_importance = pd.DataFrame({
            'feature': self.selected_feature_names_,
            'importance': rf.feature_importances_[selected_idx],
        })
        # rsplit from the right to safely handle column names that contain "_"
        col_importance['column'] = col_importance['feature'].str.rsplit('_', n=1).str[0].astype(int)
        self.column_importances_ = (
            col_importance.groupby('column')['importance']
            .sum()
            .sort_values(ascending=False)
        )

        # Determine which columns to keep
        cumulative = np.cumsum(self.column_importances_) / self.column_importances_.sum()
        if self.top_n is not None:
            self.selected_columns_ = self.column_importances_.index[:self.top_n].tolist()
        else:
            cutoff = int(np.searchsorted(cumulative.values, self.importance_cutoff))
            self.selected_columns_ = self.column_importances_.index[:cutoff + 1].tolist()

        return self

    def transform(self, X):
        """Return the MSA filtered to the selected alignment columns."""
        return X[sorted(self.selected_columns_)]

    def _best_cluster_labels(self, coords):
        k_range = range(self.min_clusters, self.max_clusters + 1)
        silhouette_scores = []
        candidates = []

        if self.clustering == 'single-linkage':
            Z = fastcluster.linkage(coords, method='ward')
            for k in k_range:
                labels = fcluster(Z, k, criterion='maxclust')
                silhouette_scores.append(silhouette_score(coords, labels))
                candidates.append(labels)
            return candidates[int(np.argmax(silhouette_scores))]

        # k-means
        for k in k_range:
            model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
            labels = model.fit_predict(coords)
            silhouette_scores.append(silhouette_score(coords, labels))
            candidates.append(labels)
        return candidates[int(np.argmax(silhouette_scores))]


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_cleanse_heatmaps(dirty, clean, save=False, show=False):
    """Plot gap heatmaps before and after cleansing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    hm_before = ax1.imshow(dirty.isna().astype(int), cmap='viridis', aspect='auto', extent=[0, 1, 0, 1])
    hm_after = ax2.imshow(clean.isna().astype(int), cmap='viridis', aspect='auto', extent=[0, 1, 0, 1])

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(hm_after, cax=cax)
    cbar.set_label('Gaps (Indels)')

    ax1.axis('off')
    ax2.axis('off')

    if show:
        plt.show()
    if save:
        plt.savefig('./output/cleanse_heatmaps.png', dpi=300)


def plot_pareto(column_importances, save=False, show=False):
    """Bar + cumulative line chart of column importances (Pareto chart)."""
    sorted_cols = column_importances.index
    xvalues = range(len(sorted_cols))

    fig, ax1 = plt.subplots(figsize=(16, 4))
    ax1.bar(xvalues, column_importances, color='cyan')
    ax1.set_ylabel('Summed Importance', fontsize=16)
    ax1.tick_params(axis='y', labelsize=12)

    ax2 = ax1.twinx()
    ax2.plot(
        xvalues,
        np.cumsum(column_importances) / np.sum(column_importances),
        color='magenta', marker='.',
    )
    ax2.set_ylabel('Cumulative Importance', fontsize=16)
    ax2.tick_params(axis='y', labelsize=12)

    plt.xticks(xvalues, sorted_cols)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

    if show:
        plt.show()
    if save:
        plt.savefig('./output/pareto_chart.png')


def plot_perceptual_map(mca, unique_sequences, coordinates, labels,
                        selected_columns, selected_feature_names,
                        save=False, show=False):
    """Scatter plot of MCA sequence coordinates with annotated SDP residues."""
    residue_features = [
        f for f in selected_feature_names
        if int(f.rsplit('_', 1)[0]) in selected_columns
    ]
    df_res = mca.column_coordinates(unique_sequences[selected_columns]).loc[residue_features]
    df_res.index = pd.Index([
        seq3(feat.rsplit('_', 1)[1]) + feat.rsplit('_', 1)[0] for feat in df_res.index
    ])

    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    legend_handles = []

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        color = plt.cm.viridis(label / len(unique_labels))
        plt.scatter(coordinates[idx, 0], coordinates[idx, 1], color=color, alpha=0.5)
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
                       markersize=10, markerfacecolor=color)
        )

    plt.scatter(df_res[0], df_res[1], marker='*', color='black', s=50)
    for i, (x, y) in enumerate(zip(df_res[0], df_res[1])):
        plt.annotate(df_res.index[i], (x, y), textcoords='offset points', xytext=(0, 10), ha='center')

    legend_handles.append(
        plt.Line2D([0], [0], marker='*', color='k', label='Selected Residues', markersize=10)
    )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(handles=legend_handles, title='Sequence Clusters')
    plt.grid()

    if show:
        plt.show()
    if save:
        plt.savefig('./output/perceptual_map.png')


# FIX: renamed first parameter from `original_index` to `cleaned_index`
# and updated the logic to use the correct index (the one saved during fit).
def generate_wordclouds(cleaned_index, unique_sequences, labels, metadata=None,
                        column='Protein names', save=False, show=False):
    """
    Build and optionally plot word clouds for each cluster.

    Parameters
    ----------
    cleaned_index : pd.Index
        The index of the cleaned, non‑deduplicated sequences
        (MCAClusterFeatureSelector.input_index_), matching the rows
        of `unique_sequences` before deduplication.
    unique_sequences : pd.DataFrame
        Deduplicated sequence DataFrame (MCAClusterFeatureSelector.unique_sequences_).
    labels : array-like
        Cluster label per row of unique_sequences.
    metadata : str or None
        Path to TSV with columns 'Entry Name' and `column`.
    column : str
        Metadata column containing protein names.
    """
    if metadata is None:
        return {}

    meta_df = pd.read_csv(metadata, delimiter='\t')
    wordcloud_data = {}

    for label in set(labels):
        # Indices of the deduplicated table that belong to this cluster
        indices = unique_sequences.iloc[np.where(labels == label)[0]].index
        # Map back to the original headers using the saved index
        headers = cleaned_index[indices]
        entry_names = [h.split('/')[0] for h in headers]

        result = meta_df[meta_df['Entry Name'].isin(entry_names)].copy()
        matches = result[column].str.extract(r'(.+?) ([\w\-,]+ase)', flags=re.IGNORECASE)

        result['Substrate'] = (
            matches[0].fillna('')
            .apply(lambda x: '/'.join(re.findall(r'\b(\w+(?:ene|ine|ate|yl))\b', x, flags=re.IGNORECASE)))
            .str.lower()
        )
        result['Enzyme'] = (
            matches[1].fillna('')
            .apply(lambda x: x.split('-')[-1] if '-' in x else x)
            .str.lower()
        )
        result['Label'] = result['Substrate'].str.cat(result['Enzyme'], sep=' ').str.strip()

        wordcloud_text = ' '.join(
            sorted({s for s in result.Label.tolist() if len(s) > 0})
        )
        wordcloud_data[label] = wordcloud_text

    if show or save:
        _plot_wordclouds(wordcloud_data, save=save, show=show)

    return wordcloud_data


def _plot_wordclouds(wordcloud_data, save=False, show=False):
    num_clusters = len(wordcloud_data)
    fig, axs = plt.subplots(num_clusters, 1, figsize=(10, 6 * num_clusters))

    for i, (label, text) in enumerate(wordcloud_data.items()):
        ax = axs[i] if num_clusters > 1 else axs
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')

    if show:
        plt.show()
    if save:
        plt.savefig('./output/wordcloud.png')


def generate_logos(unique_sequences, labels, selected_columns,
                   color_scheme='NajafabadiEtAl2017', save=False, show=False):
    """
    Plot sequence logos for each cluster over the selected SDP columns.

    Parameters
    ----------
    unique_sequences : pd.DataFrame
        Deduplicated MSA (MCAClusterFeatureSelector.unique_sequences_).
    labels : array-like
        Cluster label per row of unique_sequences.
    selected_columns : list
        Alignment column indices to include (MCAClusterFeatureSelector.selected_columns_).
    color_scheme : str
        Logomaker color scheme for amino acids.
    """
    valid_schemes = sorted(
        lm.list_color_schemes()
        .loc[lm.list_color_schemes().characters == 'ACDEFGHIKLMNPQRSTVWY']
        .color_scheme.values
    )
    if color_scheme not in valid_schemes:
        raise ValueError(f"color_scheme must be one of {valid_schemes}")

    logos_data = {}
    for label in set(labels):
        sub = unique_sequences[sorted(selected_columns)].iloc[np.where(labels == label)[0]]
        freq = sub.T.apply(lambda col: col.value_counts(normalize=True), axis=1).fillna(0)
        logos_data[label] = freq

        if save or show:
            msa_columns = freq.index.tolist()
            data = freq.reset_index(drop=True)
            logo = lm.Logo(data, color_scheme=color_scheme, vpad=0.1, width=0.8)
            logo.style_spines(visible=False)
            logo.ax.set_xticks(range(len(msa_columns)))
            logo.ax.set_xticklabels(msa_columns, fontsize=24)
            if save:
                plt.savefig(f'./output/sdp_logo_{label}.png')

    if show:
        plt.show()

    return logos_data


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Process Multiple Sequence Alignment (MSA) data.',
        usage='python pipeline.py MSA_FILE [OPTIONS]',
    )
    parser.add_argument('data', type=str, help='Path to the MSA file')
    parser.add_argument('--metadata', type=str, help='Path to the metadata file in TSV format')
    parser.add_argument('--plot', action='store_true', help='Whether to plot data')
    parser.add_argument('--save', action='store_true', help='Whether to save plots')
    parser.add_argument('--show', action='store_true', help='Whether to show plots')
    args = parser.parse_args()

    raw_msa = load_msa(args.data)
    positions_map = map_positions(raw_msa)

    pipeline = Pipeline([
        ('cleanse', CleanseTransformer()),
        ('analysis', MCAClusterFeatureSelector(
            clustering='single-linkage',
            min_clusters=3,
            rf_n_estimators=1000,
            random_state=42,
            top_n=3,
        )),
    ])
    pipeline.fit(raw_msa)

    cleanse_step = pipeline.named_steps['cleanse']
    analysis_step = pipeline.named_steps['analysis']

    if args.plot:
        plot_cleanse_heatmaps(cleanse_step.dirty_, cleanse_step.clean_,
                              save=args.save, show=args.show)
        plot_pareto(analysis_step.column_importances_,
                    save=args.save, show=args.show)
        plot_perceptual_map(
            analysis_step.mca_,
            analysis_step.unique_sequences_,
            analysis_step.coordinates_,
            analysis_step.labels_,
            analysis_step.selected_columns_,
            analysis_step.selected_feature_names_,
            save=args.save, show=args.show,
        )

    if args.metadata:
        # FIX: use the saved input index, not raw_msa.index
        generate_wordclouds(
            analysis_step.input_index_,          # <-- correct headers after cleansing
            analysis_step.unique_sequences_,
            analysis_step.labels_,
            metadata=args.metadata,
            save=args.save, show=args.show,
        )

    generate_logos(
        analysis_step.unique_sequences_,
        analysis_step.labels_,
        analysis_step.selected_columns_,
        save=args.save, show=args.show,
    )

    profiles = build_profiles(raw_msa, positions_map, analysis_step.selected_columns_)
    print(profiles.to_string())

    return True


if __name__ == '__main__':
    try:
        sys.exit(0 if main() else 1)
    except Exception as e:
        print(f'An error occurred: {type(e).__name__} - {e}')
        sys.exit(2)