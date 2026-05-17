import numpy as np
import pandas as pd
import prince
import pytest
import matplotlib
matplotlib.use("Agg")

from sdp_detection_pipeline.visualization import plot_perceptual_map


@pytest.fixture
def vis_inputs():
    """Minimal inputs for plot_perceptual_map that exercise the residue-filtering path."""
    np.random.seed(0)
    residues = list("ACGT")
    cols = [10, 20, 30]
    data = pd.DataFrame(
        np.random.choice(residues, size=(12, 3)),
        columns=cols,
    )
    mca = prince.MCA(n_components=2, random_state=0)
    mca.fit(data)

    coords = np.random.randn(12, 2)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    # selected_feature_names uses "col_cat" format (single underscore)
    selected_feature_names = np.array([f"{col}_{res}" for col in cols for res in residues])
    selected_columns = cols

    return mca, data, coords, labels, selected_columns, selected_feature_names


def test_plot_perceptual_map_runs(vis_inputs):
    mca, data, coords, labels, selected_columns, selected_feature_names = vis_inputs
    fig = plot_perceptual_map(
        mca, data, coords, labels, selected_columns, selected_feature_names, show=False
    )
    assert fig is not None
