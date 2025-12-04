#!/usr/bin/env python3
"""Clean Finan.csv columns: trim strings, coerce numeric-like columns to int/float.

Saves cleaned CSV next to input with suffix `_cleaned.csv` by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re
import json
import pandas as pd


def is_integer_series(s: pd.Series) -> bool:
    """Return True if all non-null values in s are integer-valued."""
    if s.dropna().empty:
        return False
    try:
        return all(float(x).is_integer() for x in s.dropna())
    except Exception:
        return False


def clean_column_values(s: pd.Series) -> pd.Series:
    """Normalize string values: strip whitespace and convert empty strings to NaN."""
    s = s.astype(object)
    s = s.apply(lambda x: x.strip() if isinstance(x, str) else x)
    # Treat empty strings as NaN
    s = s.replace({"": pd.NA})
    return s


def try_parse_numeric(s: pd.Series) -> pd.Series:
    """Try to parse a string Series into numeric values, handling commas and parentheses.

    Returns a numeric Series (float) with NaNs where conversion failed.
    """
    # Remove thousands separators and extraneous whitespace
    cleaned = s.astype(object).apply(lambda x: re.sub(r"[,\s]+", "", x) if isinstance(x, str) else x)
    # Remove surrounding parentheses used in some accounting formats (e.g., (123))
    cleaned = cleaned.apply(lambda x: x[1:-1] if isinstance(x, str) and x.startswith("(") and x.endswith(")") else x)
    # Convert to numeric
    return pd.to_numeric(cleaned, errors="coerce")



def load_attributes(attr_path: Path) -> dict:
    """Load Finan_attributes.json and return mapping column->value_type."""
    if not attr_path.exists():
        return {}
    with open(attr_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    mapping = {}
    # doc expected to have top-level table key, but we don't assume the table name
    for table, attrs in doc.items():
        for a in attrs:
            mapping[a["name"]] = a.get("value_type", "str")
    return mapping


def clean_csv(infile: Path, outfile: Path | None = None, attributes: dict | None = None) -> dict:
    """Clean CSV and coerce columns according to attributes mapping when provided.

    attributes: mapping column name -> value_type (e.g., 'int', 'float', 'str')
    """
    if outfile is None:
        outfile = infile.with_name(infile.stem + "_cleaned" + infile.suffix)

    df = pd.read_csv(infile, dtype=str, keep_default_na=False)

    if attributes is None:
        attributes = {}

    summary = {"converted_to_int": [], "converted_to_float": [], "left_as_object": [], "mismatch_int_to_float": []}

    for col in df.columns:
        s = df[col]
        s = clean_column_values(s)

        # Count original non-missing values (after treating empty as NA)
        orig_non_null = s.dropna().shape[0]

        if orig_non_null == 0:
            df[col] = s
            summary["left_as_object"].append(col)
            continue

        expected_type = attributes.get(col, None)

        # If attribute says int or float, force parse and coerce accordingly
        if expected_type in ("int", "float"):
            numeric = try_parse_numeric(s)
            # Convert according to expected type
            if expected_type == "int":
                # If all numeric values are integer-valued -> Int64
                nonnull = numeric.dropna()
                if nonnull.empty:
                    # nothing parsed as numeric, leave as object
                    df[col] = s
                    summary["left_as_object"].append(col)
                elif is_integer_series(nonnull):
                    df[col] = numeric.astype("Int64")
                    summary["converted_to_int"].append(col)
                else:
                    # mismatch: contains fractions -> keep as float but record
                    df[col] = numeric.astype("float64")
                    summary["converted_to_float"].append(col)
                    summary["mismatch_int_to_float"].append(col)
            else:  # expected float
                numeric = try_parse_numeric(s)
                if numeric.dropna().empty:
                    df[col] = s
                    summary["left_as_object"].append(col)
                else:
                    df[col] = numeric.astype("float64")
                    summary["converted_to_float"].append(col)
        else:
            # No expectation or non-numeric expected: leave as cleaned string
            df[col] = s
            summary["left_as_object"].append(col)

    df.to_csv(outfile, index=False)
    return {"outfile": str(outfile), "summary": summary}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean Finan.csv and coerce numeric columns")
    parser.add_argument("infile", nargs="?", default="Finan.csv", help="Input CSV file path")
    parser.add_argument("--out", "-o", dest="outfile", help="Output CSV path (optional)")
    parser.add_argument("--attributes", "-a", dest="attributes", default="Finan_attributes.json",
                        help="Path to attributes JSON that maps column names to value_type")
    args = parser.parse_args(argv)

    infile = Path(args.infile)
    if not infile.exists():
        print(f"Input file does not exist: {infile}", file=sys.stderr)
        return 2

    attr_path = Path(args.attributes)
    attributes = load_attributes(attr_path)

    result = clean_csv(infile, Path(args.out) if args.out else None, attributes=attributes)
    print("Wrote:", result["outfile"])
    print("Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {len(v)} columns")
        if v:
            print("    ", ", ".join(v))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
