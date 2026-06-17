from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping


def append_csv_row(path: str | Path, row: Mapping[str, object]) -> None:
    """Append one row to a CSV file, archiving incompatible existing headers."""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, None)
        if existing_header and existing_header != fieldnames:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            archive_path = csv_path.with_name(f"{csv_path.stem}.{stamp}.archived{csv_path.suffix}")
            csv_path.rename(archive_path)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv_rows(path: str | Path, rows: list[Mapping[str, object]]) -> None:
    """Write rows to a CSV file, replacing any existing file."""

    if not rows:
        raise ValueError("At least one row is required to write a CSV file.")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if list(row.keys()) != fieldnames:
                raise ValueError("All rows must have the same field order.")
            writer.writerow(row)
