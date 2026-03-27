"""Helpers for optional runtime dependencies."""

from __future__ import annotations

import importlib


def require_dependency(module_name: str, package_name: str | None = None):
    """Import a module or raise a helpful runtime error."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = package_name or module_name
        raise RuntimeError(
            f"Optional dependency '{package}' is required for this operation. "
            f"Install it with `pip install -e .[{package}]` or `pip install {package}`."
        ) from exc
