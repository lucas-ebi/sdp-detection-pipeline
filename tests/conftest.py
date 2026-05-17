import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def tiny_msa_df():
    data = [
        ['A', 'A', '-', 'A', 'G'],
        ['A', 'G', 'G', 'A', 'G'],
        ['G', 'G', '-', 'T', '-'],
        ['A', 'A', 'A', 'T', 'G'],
    ]
    headers = ['seq1/1-5', 'seq2/10-14', 'seq3/20-24', 'seq4/5-9']
    return pd.DataFrame(data, index=headers, columns=[0,1,2,3,4])

@pytest.fixture
def small_msa_df():
    np.random.seed(42)
    residues = list('ACDEFGHIKLMNPQRSTVWY')
    data = np.random.choice(residues + ['-'], size=(8, 6))
    headers = [f'seq{i}/1-6' for i in range(8)]
    return pd.DataFrame(data, index=headers, columns=range(6))
