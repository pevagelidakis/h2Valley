"""Shared, dataset-independent building blocks used by every analysis module."""
from core.dataset import DatasetSpec, get_dataset, load_spec

__all__ = ["DatasetSpec", "get_dataset", "load_spec"]
