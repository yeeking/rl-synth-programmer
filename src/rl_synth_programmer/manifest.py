from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .curriculum import TargetSpec


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def target_record(target: TargetSpec) -> dict[str, Any]:
    record = asdict(target)
    record.pop("embedding", None)
    record.pop("audio", None)
    return record
