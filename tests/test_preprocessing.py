import pandas as pd
from specifinder.preprocessing import CleanseTransformer

def test_cleanse_fit_transform(tiny_msa_df):
    ct = CleanseTransformer(threshold=0.5)
    ct.fit(tiny_msa_df)
    assert 2 in ct.kept_columns_
    cleaned = ct.transform(tiny_msa_df)
    assert not cleaned.isna().any().any()
    assert len(cleaned) == 4

def test_cleanse_threshold_column_removal():
    df = pd.DataFrame(
        [['A','-','C'], ['A','-','C'], ['A','-','C']],
        index=['s1','s2','s3'], columns=[0,1,2]
    )
    ct = CleanseTransformer(threshold=0.9)
    ct.fit(df)
    assert 1 not in ct.kept_columns_
    cleaned = ct.transform(df)
    assert 1 not in cleaned.columns
    assert cleaned.shape == (1,2)
