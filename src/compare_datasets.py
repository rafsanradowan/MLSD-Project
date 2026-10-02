"""Check that two dataset files hold the same data (used after switching to Feast).

  python src/compare_datasets.py tmp/dataset_pandas.parquet data/processed/dataset.parquet
"""
import sys

import numpy as np
import pandas as pd

from common import KEYS, read_parquet


def compare(a, b):
    problems = []
    if list(a.columns) != list(b.columns):
        problems.append(f"column lists differ:\n  A={list(a.columns)}\n  B={list(b.columns)}")
        return problems
    if a.shape != b.shape:
        problems.append(f"shapes differ: {a.shape} vs {b.shape}")
        return problems
    a = a.sort_values(KEYS).reset_index(drop=True)
    b = b.sort_values(KEYS).reset_index(drop=True)
    for col in a.columns:
        if col == "event_timestamp":
            same = (pd.to_datetime(a[col], utc=True).astype("int64")
                    == pd.to_datetime(b[col], utc=True).astype("int64")).all()
        elif pd.api.types.is_numeric_dtype(a[col]):
            same = np.allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
                               equal_nan=True, rtol=1e-9, atol=1e-9)
        else:
            same = (a[col] == b[col]).all()
        if not same:
            problems.append(f"values differ in column '{col}'")
    return problems


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python src/compare_datasets.py FILE_A FILE_B")
    problems = compare(read_parquet(sys.argv[1]), read_parquet(sys.argv[2]))
    if problems:
        print("DIFFERENT:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print("IDENTICAL: same columns, same shape, same values.")


if __name__ == "__main__":
    main()
