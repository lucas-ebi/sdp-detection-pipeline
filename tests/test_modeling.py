import pandas as pd
from sdp_detection_pipeline.modeling import MCAClusterFeatureSelector

def test_mca_cluster_selector_basic():
    data = [
        ['A','C','G'],
        ['A','C','G'],
        ['G','C','T'],
        ['G','C','T'],
    ]
    df = pd.DataFrame(data, index=[f's{i}' for i in range(4)], columns=[0,1,2])
    selector = MCAClusterFeatureSelector(
        mca_n_components=2,
        cluster_method='k-means',
        min_clusters=2, max_clusters=3,
        top_n=2, random_state=0
    )
    selector.fit(df)
    assert len(set(selector.labels_)) == 2
    assert selector.column_importances_.idxmax() == 0
    assert 0 in selector.selected_columns_

def test_selector_with_small_msa(small_msa_df):
    selector = MCAClusterFeatureSelector(
        cluster_method='single-linkage',
        min_clusters=2, max_clusters=5,
        top_n=3, random_state=42
    )
    selector.fit(small_msa_df)
    assert len(selector.labels_) == small_msa_df.shape[0]
    assert len(selector.selected_columns_) > 0
