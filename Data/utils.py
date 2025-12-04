import pandas as pd
from typing import Any

# --- get the unique value of the column
def get_unique_values(column_name: str, data: pd.DataFrame) -> pd.Series:
    """Return unique values from a DataFrame column as a pandas Series.

    This ensures callers get a Series (not a numpy.ndarray), which is easier
    to save or manipulate downstream.
    """
    return pd.Series(data[column_name].unique())


def save_as_csv(data: Any, filename: str):
    """Save array-like / Series / DataFrame to CSV.

    Accepts pandas.Series, pandas.DataFrame, numpy.ndarray, list, tuple or set.
    Converts non-DataFrame inputs into a single-column DataFrame and writes
    to `filename` without an index.
    """
    # If it's already a DataFrame, use as-is
    if isinstance(data, pd.DataFrame):
        df = data
    # If it's a Series, convert to DataFrame (keeps column name if present)
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        # For array-like inputs, try to coerce into a DataFrame
        try:
            # Convert sets to list so ordering is deterministic-ish
            if isinstance(data, set):
                data = list(data)
            df = pd.DataFrame(data)
        except Exception:
            # Last resort: wrap single value into a one-row DataFrame
            df = pd.DataFrame([data])

    # If single-column and unnamed (column label 0), give it a sensible name
    if df.shape[1] == 1 and (df.columns.tolist() == [0] or df.columns.tolist() == [None]):
        col_name = getattr(data, "name", None) or "value"
        df.columns = [col_name]

    df.to_csv(filename, index=False)