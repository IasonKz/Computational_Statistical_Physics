#!/usr/bin/env python3
"""
Merge per-run metadata.csv files from results/configs_ml/run_<id>/ into
one combined results/configs_ml/metadata.csv.

This avoids concurrent writes by Slurm array jobs.

Usage:
    python3 scripts/merge_ml_metadata.py --config-root results/configs_ml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-root", default="results/configs_ml")
    p.add_argument("--outfile", default=None)
    args = p.parse_args()

    root = Path(args.config_root)
    out = Path(args.outfile) if args.outfile else root / "metadata.csv"

    if not root.exists():
        raise SystemExit(f"Config root does not exist: {root}")

    metadata_files = sorted(
        m for m in root.rglob("metadata.csv")
        if m.resolve() != out.resolve()
    )

    if not metadata_files:
        raise SystemExit(f"No per-run metadata.csv files found under {root}")

    rows = []
    fieldnames = None
    path_field = None

    for md in metadata_files:
        rel_dir = md.parent.relative_to(root)

        with md.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue

            if fieldnames is None:
                fieldnames = list(reader.fieldnames)

                if "filename" in fieldnames:
                    path_field = "filename"
                elif "file" in fieldnames:
                    path_field = "file"
                elif "path" in fieldnames:
                    path_field = "path"
                else:
                    raise SystemExit(
                        "Could not find a filename/file/path column in metadata."
                    )

            for row in reader:
                # Convert per-run local filename to path relative to config root.
                val = row[path_field]
                val_path = Path(val)
                if not val_path.is_absolute():
                    row[path_field] = str(rel_dir / val_path)
                rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(metadata_files)} metadata files")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
