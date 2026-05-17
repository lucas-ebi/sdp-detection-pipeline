# SDP Detection Pipeline

**Machine‑learning driven identification of Specificity‑Determining Positions (SDPs) in protein multiple sequence alignments.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

Protein families often contain functional subgroups (e.g. different substrate specificities) that cannot be distinguished from overall sequence conservation alone. The **SDP Detection Pipeline** applies a combination of dimensionality reduction, clustering, and supervised feature selection to an input multiple sequence alignment (MSA) in order to pinpoint the columns (alignment positions) that best discriminate among the natural subgroups present in the data.

The pipeline uses:

- **Multiple Correspondence Analysis (MCA)** to project categorical MSA data into a low‑dimensional space.
- **Agglomerative (single‑linkage) or k‑means clustering** to identify inherent sequence groups, with the optimal number of clusters chosen via silhouette score.
- **Random Forest classification** using the cluster labels as target, followed by **feature importance ranking** to select the most discriminative alignment positions.
- The top positions are designated **Specificity‑Determining Positions (SDPs)** and are reported as a human‑readable profile (residue + position number). Optional visualisations – Pareto charts, perceptual maps, sequence logos per cluster, and word clouds of protein descriptions – facilitate biological interpretation.

The tool is implemented as a modular `scikit‑learn` compatible pipeline, making it easy to embed in larger workflows, serialise, and hyperparameter‑tune.

---

## Installation

### Requirements

- Python ≥ 3.10
- `pip` (or `conda`) for dependency management

### From source

```bash
git clone https://github.com/your-org/sdp-detection-pipeline.git
cd sdp-detection-pipeline
pip install .
```

The command above installs all required dependencies automatically.

If you plan to modify the code, use an editable installation:

```bash
pip install -e .
```

### NLTK data (for word clouds)

The word cloud functionality uses `nltk` tokenisers. The first time you use the `--metadata` option, you may need to download the required NLTK resources. This can be done once with:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

or simply by running the pipeline with the `--metadata` flag; the data is downloaded automatically on first use if not already present.

---

## Quick Start

The installed package provides a command‑line script `sdp-pipeline`:

```bash
sdp-pipeline msa.fasta --plot --save
```

- Processes the alignment in `msa.fasta`.
- Generates all plots and saves them to `./output/`.
- Prints a table of the top‑3 specificity‑determining positions to the console.

To include word clouds of protein descriptions, supply a metadata file in **UniProt‑style TSV format** (columns must include “Entry Name” and “Protein names”):

```bash
sdp-pipeline msa.fasta --metadata uniprot_metadata.tsv --plot --show
```

Use `--show` to interactively display plots instead of (or in addition to) saving them.

---

## Input

### MSA file

A multiple sequence alignment in **FASTA format**. Sequences must be aligned – all entries must have the same length. Sequence headers should ideally follow the convention `EntryName/start-end` (e.g. `PFK1_HUMAN/1-780`) to enable residue numbering; if not, the residue position map will use `?` for the numbers.

### Metadata file (optional)

A tab‑separated file (TSV) with at least the following columns:
- `Entry Name` – matches the first part of the FASTA header before `/`.
- `Protein names` – descriptive name used for word cloud generation.

This file can be obtained from UniProt for a given set of accessions.

---

## Pipeline Steps

1. **Loading & indexing** – The MSA is parsed into a `pandas.DataFrame` (rows = sequences, columns = alignment positions). Residue numbers are mapped for headers matching the `Entry/start-end` pattern.
2. **Cleansing** – Columns and rows with more than a specified fraction of gaps (default 90% allowed gaps) are removed. Lower‑case residues (often insertion states) can optionally be treated as gaps. Duplicate sequences are collapsed.
3. **Dimensionality reduction** – Multiple Correspondence Analysis (MCA) is performed on the categorical MSA (amino acid letters at each position). The sequences are projected into a continuous 2‑dimensional space.
4. **Clustering** – Optimal number of clusters is determined by silhouette score, testing a range (default 2–10). Two algorithms are supported: **single‑linkage** (via `fastcluster`) and **k‑means**. The cluster labels serve as the target variable for the subsequent feature selection.
5. **Random Forest classification & feature importance** – The cleansed MSA is one‑hot encoded, and a Random Forest classifier is trained to predict the cluster labels. Feature importances are aggregated per alignment column (by summing the importances of all amino‑acid one‑hot features that map to the same column). Columns are then ranked by summed importance.
6. **SDP selection** – The most important columns are selected either by taking the top *N* columns (`--top_n`) or by a cumulative importance threshold (default 0.9). These are the candidate Specificity‑Determining Positions.
7. **Output generation** – A profile table listing the amino acid and residue number at each SDP for every sequence is printed. Optionally, a series of visualisations are produced (see below).

---

## Output

### Console

A table (pandas `DataFrame`) with:
- one row per **original** sequence (before deduplication),
- one column per selected SDP,
- cells contain the **3‑letter amino acid code** followed by the **residue number** (e.g. `His214`). If the header does not contain position information, the number is replaced by `?`.

### Plot files (saved to `./output/` when `--save` is used)

| Plot | File | Description |
|------|------|-------------|
| Gap heatmaps | `cleanse_heatmaps.png` | Before/after visualisation of gap distribution |
| Pareto chart | `pareto_chart.png` | Bar chart of per‑column summed importance with cumulative curve |
| Perceptual map | `perceptual_map.png` | MCA projection of sequences coloured by cluster; SDP residues shown as stars |
| Word cloud(s) | `wordcloud.png` | Word clouds of protein descriptions per cluster (requires `--metadata`) |
| Sequence logos | `sdp_logo_<cluster>.png` | Sequence conservation logos for each cluster at the selected SDP positions |

---

## Advanced Usage

The pipeline is built from `scikit‑learn` compatible transformers. You can import it directly into your Python scripts:

```python
from sdp_detection_pipeline.io import load_msa, map_positions, build_profiles
from sdp_detection_pipeline.preprocessing import CleanseTransformer
from sdp_detection_pipeline.modeling import MCAClusterFeatureSelector
from sklearn.pipeline import Pipeline

raw = load_msa("msa.fasta")
positions = map_positions(raw)

pipe = Pipeline([
    ("clean", CleanseTransformer(threshold=0.9)),
    ("sdp", MCAClusterFeatureSelector(
        cluster_method="single-linkage",
        top_n=5,
        random_state=42
    ))
])
pipe.fit(raw)

# Access results
print(pipe.named_steps["sdp"].selected_columns_)   # list of SDP column indices
print(pipe.named_steps["sdp"].column_importances_) # ranked importances
```

The fitted pipeline can be serialised with `joblib.dump` and reused on new alignments (provided they share the same set of columns after cleansing).

Hyperparameter search is also straightforward:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "clean__threshold": [0.8, 0.9],
    "sdp__cluster_method": ["single-linkage", "k-means"],
    "sdp__top_n": [3, 5, None]
}
grid = GridSearchCV(pipe, param_grid, cv=3, scoring=my_custom_scorer)
grid.fit(raw)
```

*(A custom scorer must be defined to evaluate the unsupervised clustering – for example, silhouette score on the MCA coordinates.)*

---

## Citation

If you use this pipeline in your research, please cite the original software and the associated publication (if any). The original module was developed by **Lucas Carrijo de Oliveira** (lucas@ebi.ac.uk). For now, you may reference the repository directly:

```
Lucas C. de Oliveira. SDP Detection Pipeline. 2023–2025.
https://github.com/your-org/sdp-detection-pipeline
```

A formal publication is in preparation; check the repository for updates.

---

## License

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) for the full text.

---

## Acknowledgements

This work makes use of:

- [`scikit-learn`](https://scikit-learn.org/)
- [`prince`](https://github.com/MaxHalford/prince) for MCA
- [`fastcluster`](http://danifold.net/fastcluster.html) for efficient hierarchical clustering
- [`logomaker`](https://logomaker.readthedocs.io/)
- [`Biopython`](https://biopython.org/)
- [`wordcloud`](https://amueller.github.io/word_cloud/)

Development has been supported by the European Bioinformatics Institute (EMBL‑EBI).
