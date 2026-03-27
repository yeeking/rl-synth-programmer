from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .optional_deps import require_dependency


class NullWriter:
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        _ = (tag, scalar_value, global_step)

    def add_text(self, tag: str, text_string: str, global_step: int = 0) -> None:
        _ = (tag, text_string, global_step)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class DummyProgressBar:
    def __init__(self, total: int | None = None, desc: str = "", enabled: bool = True):
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.n = 0

    def update(self, count: int = 1) -> None:
        self.n += count

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any) -> None:
        _ = (ordered_dict, refresh, kwargs)

    def write(self, message: str) -> None:
        if self.enabled:
            print(message)

    def close(self) -> None:
        return None


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def stage_log(message: str) -> None:
    print(f"[{_timestamp()}] {message}")


def make_progress_bar(total: int | None, desc: str, enabled: bool = True):
    if not enabled:
        return DummyProgressBar(total=total, desc=desc, enabled=False)
    try:
        tqdm_auto = require_dependency("tqdm.auto", "tqdm")
        return tqdm_auto.tqdm(total=total, desc=desc, dynamic_ncols=True, leave=False)
    except RuntimeError:
        stage_log(f"{desc}: progress bar unavailable, using plain logging.")
        return DummyProgressBar(total=total, desc=desc, enabled=True)


def progress_iter(iterable: Iterable[Any], total: int | None, desc: str, enabled: bool = True):
    if not enabled:
        return iterable
    try:
        tqdm_auto = require_dependency("tqdm.auto", "tqdm")
        return tqdm_auto.tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)
    except RuntimeError:
        stage_log(f"{desc}: progress bar unavailable, using plain logging.")
        return iterable


def create_summary_writer(enabled: bool, log_dir: Path | None):
    if not enabled:
        return NullWriter()
    assert log_dir is not None, "TensorBoard log_dir must be provided when tensorboard logging is enabled."
    require_dependency("torch", "ml")
    require_dependency("tensorboard", "ml")
    summary_module = require_dependency("torch.utils.tensorboard", "ml")
    log_dir.mkdir(parents=True, exist_ok=True)
    return summary_module.SummaryWriter(log_dir=str(log_dir))


def log_run_metadata(writer: Any, metadata: dict[str, Any]) -> None:
    writer.add_text("run/config", json.dumps(metadata, indent=2, sort_keys=True), 0)
    for key, value in metadata.items():
        if isinstance(value, (int, float, str, bool)):
            writer.add_text(f"run/{key}", str(value), 0)

