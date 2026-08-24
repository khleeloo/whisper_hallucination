#!/usr/bin/env python3
"""Finalize MTurk audio URLs and add safe base64 hypothesis fields."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import pandas as pd


def b64(text: object) -> str:
    return base64.b64encode(str(text).encode("utf-8")).decode("ascii")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--base_url", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    base = args.base_url.rstrip("/") + "/"
    file_cols = sorted(c for c in df.columns if c.startswith("audio_filename_"))
    if not file_cols:
        raise ValueError("No audio_filename_XX columns found")

    for fcol in file_cols:
        suffix = fcol.rsplit("_", 1)[-1]
        ucol = f"audio_url_{suffix}"
        hcol = f"hypothesis_{suffix}"
        bcol = f"hypothesis_b64_{suffix}"
        if ucol not in df.columns:
            raise ValueError(f"Missing matching URL column {ucol}")
        if hcol not in df.columns:
            raise ValueError(f"Missing matching hypothesis column {hcol}")
        df[ucol] = df[fcol].map(lambda x: base + str(x))
        df[bcol] = df[hcol].map(b64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(f"HTTPS base: {base}")
    print(f"Updated URL/base64 columns: {len(file_cols)}")


if __name__ == "__main__":
    main()
