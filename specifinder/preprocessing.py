"""Cleansing transformer for removing gap-rich columns and rows."""

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.base import BaseEstimator, TransformerMixin


class CleanseTransformer(BaseEstimator, TransformerMixin):
    """
    Remove alignment columns and rows that exceed a gap fraction threshold.

    `fit` memorises which columns survive; `transform` applies those columns,
    prunes rows, and returns a deduplicated DataFrame.

    Parameters
    ----------
    indel : str, default '-'
        Character used for gaps / indels.
    remove_lowercase : bool, default True
        If True, lowercase residues (often insert states) are treated as gaps.
    threshold : float, default 0.9
        Minimum fraction of non-gap characters required to keep a column or row.
    """

    def __init__(
        self,
        indel: str = "-",
        remove_lowercase: bool = True,
        threshold: float = 0.9,
    ) -> None:
        self.indel = indel
        self.remove_lowercase = remove_lowercase
        self.threshold = threshold

    def _mask(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        bad = [self.indel]
        if self.remove_lowercase:
            bad += [chr(c) for c in range(ord("a"), ord("z") + 1)]
        else:
            X = X.map(str.upper)
        X = X.replace(bad, np.nan).infer_objects(copy=False)
        return X.infer_objects(copy=False)

    def fit(self, X: pd.DataFrame, y=None) -> "CleanseTransformer":
        masked = self._mask(X)
        min_rows = int(self.threshold * masked.shape[0])
        self.kept_columns_ = list(
            masked.columns[masked.notna().sum(axis=0) >= min_rows]
        )
        if not self.kept_columns_:
            raise ValueError("All columns were filtered out; consider lowering the threshold.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        masked = self._mask(X)
        self.dirty_ = masked.copy()

        masked = masked[self.kept_columns_]
        min_cols = int(self.threshold * masked.shape[1])
        masked = masked.dropna(thresh=min_cols, axis=0)
        self.clean_ = masked.copy()

        # Fill remaining gaps and remove duplicate sequences
        return masked.fillna(self.indel).drop_duplicates()