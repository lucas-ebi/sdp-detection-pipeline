from sklearn.pipeline import Pipeline
from specifinder.preprocessing import CleanseTransformer
from specifinder.modeling import MCAClusterFeatureSelector

def test_full_pipeline_runs(small_msa_df):
    pipe = Pipeline([
        ('cleanse', CleanseTransformer(threshold=0.5)),
        ('sdp', MCAClusterFeatureSelector(
            cluster_method='k-means',
            min_clusters=2, max_clusters=4,
            top_n=2, random_state=0
        ))
    ])
    pipe.fit(small_msa_df)
    sdp_step = pipe.named_steps['sdp']
    assert len(sdp_step.selected_columns_) == 2
