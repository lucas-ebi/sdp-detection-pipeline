"""File I/O and basic MSA helpers."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import seq3


def load_msa(file_path: Union[str, Path], file_format: str = "fasta") -> pd.DataFrame:
    """
    Parse a Multiple Sequence Alignment file into a DataFrame.

    Rows correspond to sequences (indexed by record ID), columns to alignment positions.
    """
    headers: List[str] = []
    sequences: List[List[str]] = []
    for record in SeqIO.parse(str(file_path), file_format):
        headers.append(record.id)
        sequences.append(list(str(record.seq)))
    return pd.DataFrame(sequences, index=headers)


def map_positions(msa_df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    """
    Map alignment column indices to per‑sequence residue positions.

    Only headers matching the 'ID/start‑end' convention are processed.
    Returns {header: {col_index: residue_position}}.
    """
    positions_map: Dict[str, Dict[int, int]] = defaultdict(dict)
    for header in msa_df.index:
        if re.search(r".+/\d+-\d+", header):
            sequence = msa_df.loc[header]
            offset_str, _ = header.split("/")[1].split("-")
            position = int(offset_str) - 1
            for col, val in zip(sequence.index, sequence.values):
                if val != "-":
                    position += 1
                    positions_map[header][col] = position
    return positions_map


def build_profiles(
    msa_df: pd.DataFrame,
    positions_map: Dict[str, Dict[int, int]],
    selected_columns: List[int],
) -> pd.DataFrame:
    """
    Create a human‑readable profile using 3‑letter amino‑acid codes and residue numbers.

    The returned DataFrame shares the index of `msa_df` and one column per selected position.
    """
    sorted_cols = sorted(selected_columns)
    rows = []
    for header, row in msa_df[sorted_cols].iterrows():
        pos_map = positions_map.get(header, {})
        profile: Dict[int, str] = {}
        for col, aa in row.items():
            if aa == "-":
                profile[col] = "-"
            else:
                profile[col] = f"{seq3(aa.upper())}{pos_map.get(col, '?')}"
        rows.append(profile)
    return pd.DataFrame(rows, index=msa_df.index, columns=sorted_cols)