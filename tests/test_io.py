import pytest
import pandas as pd
from pathlib import Path
import tempfile
from specifinder.io import load_msa, map_positions, build_profiles

def test_load_msa():
    fasta_content = """>seq1/1-3
AAA
>seq2/4-6
AGA
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fasta_content)
        tmpfile = Path(f.name)
    try:
        df = load_msa(tmpfile)
        assert df.shape == (2, 3)
        assert df.index.tolist() == ['seq1/1-3', 'seq2/4-6']
    finally:
        tmpfile.unlink()

def test_map_positions(tiny_msa_df):
    pos_map = map_positions(tiny_msa_df)
    assert pos_map['seq1/1-5'][0] == 1
    assert pos_map['seq1/1-5'][1] == 2
    assert pos_map['seq1/1-5'][3] == 3
    assert pos_map['seq1/1-5'][4] == 4
    assert pos_map['seq4/5-9'][2] == 7

def test_build_profiles(tiny_msa_df):
    pos_map = map_positions(tiny_msa_df)
    profiles = build_profiles(tiny_msa_df, pos_map, selected_columns=[0,2,4])
    assert profiles.loc['seq1/1-5', 0] == 'Ala1'
    assert profiles.loc['seq1/1-5', 2] == '-'
    assert profiles.loc['seq1/1-5', 4] == 'Gly4'
