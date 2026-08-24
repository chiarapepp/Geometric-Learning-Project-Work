"""Reproducible, representative subsets for evaluation datasets.

Training can shuffle every epoch, while an evaluation that stops after the
first N batches can accidentally contain only the first class or a few long
recordings.  This module keeps the selection explicit and serialisable.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class IndexedSubsetDataset(Dataset):
    """A subset that retains the position in the original evaluation dataset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, subset_index):
        dataset_index = self.indices[subset_index]
        item = self.dataset[dataset_index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError("Evaluation datasets must return (points, target).")
        return item[0], item[1], dataset_index


def _python_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _class_label(value):
    """A stable, CSV-friendly class label which also supports string labels."""
    return str(_python_value(value))


def _base_dataset(dataset):
    return dataset.dataset if hasattr(dataset, "windows") else dataset


def _raw_target(base_dataset, source_index):
    """Read a target without applying the point-cloud transform when possible."""
    source_id = int(base_dataset.indices[source_index]) if hasattr(base_dataset, "indices") else int(source_index)
    if hasattr(base_dataset, "dataset"):
        return base_dataset.dataset[source_id][1]
    return base_dataset[source_index][1]


def _original_sample_id(base_dataset, source_index):
    if hasattr(base_dataset, "indices"):
        return int(base_dataset.indices[source_index])
    return int(source_index)


def build_dataset_manifest(dataset, dataset_name):
    """Return one metadata row per dataset item/window.

    A *source* is one original recording.  Several windowed items can share a
    source; this is exactly the correlation that ``max_windows_per_source``
    prevents from dominating the evaluation.
    """
    base_dataset = _base_dataset(dataset)
    target_cache = {}
    window_counts = defaultdict(int)
    rows = []

    for dataset_index in range(len(dataset)):
        if hasattr(dataset, "windows"):
            source_index, start, end = dataset.windows[dataset_index]
            source_index, start, end = int(source_index), int(start), int(end)
            if source_index not in target_cache:
                target_cache[source_index] = _raw_target(base_dataset, source_index)
            target = target_cache[source_index]
            window_id = window_counts[source_index]
            window_counts[source_index] += 1
        else:
            source_index = int(dataset_index)
            start = end = None
            target = _raw_target(base_dataset, source_index)
            window_id = 0

        rows.append({
            "dataset": dataset_name,
            "dataset_index": int(dataset_index),
            "class": _class_label(target),
            "source_index": source_index,
            "original_sample_id": _original_sample_id(base_dataset, source_index),
            "window_id": window_id,
            "window_start": "" if start is None else start,
            "window_end": "" if end is None else end,
        })
    return rows


def select_manifest_rows(
    manifest,
    mode,
    max_items=None,
    items_per_class=None,
    max_windows_per_source=1,
    seed=13,
):
    """Select ordered, random, or class-stratified rows from a manifest."""
    if mode not in {"ordered", "random", "stratified"}:
        raise ValueError(f"Unknown evaluation sampling mode: {mode}")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive when supplied")
    if max_windows_per_source < 1:
        raise ValueError("max_windows_per_source must be at least 1")

    rng = np.random.default_rng(seed)
    if mode == "ordered":
        return manifest[:max_items]
    if mode == "random":
        count = len(manifest) if max_items is None else min(max_items, len(manifest))
        selected = rng.choice(len(manifest), size=count, replace=False)
        return [manifest[int(index)] for index in sorted(selected)]

    if items_per_class is None or items_per_class < 1:
        raise ValueError("--eval-items-per-class is required for stratified evaluation")

    by_class_and_source = defaultdict(lambda: defaultdict(list))
    for row in manifest:
        by_class_and_source[row["class"]][row["source_index"]].append(row)

    selected = []
    for class_name in sorted(by_class_and_source):
        source_rows = list(by_class_and_source[class_name].values())
        for rows in source_rows:
            rng.shuffle(rows)
            del rows[max_windows_per_source:]
        rng.shuffle(source_rows)

        # Round-robin selection spreads a class across recordings before using
        # a second window from any recording.
        class_rows = []
        depth = 0
        while len(class_rows) < items_per_class:
            added = False
            for rows in source_rows:
                if depth < len(rows):
                    class_rows.append(rows[depth])
                    added = True
                    if len(class_rows) == items_per_class:
                        break
            if not added:
                break
            depth += 1
        selected.extend(class_rows)
    return sorted(selected, key=lambda row: row["dataset_index"])


def composition_by_class(manifest):
    counts = defaultdict(lambda: {"items": 0, "sources": set()})
    for row in manifest:
        current = counts[row["class"]]
        current["items"] += 1
        current["sources"].add(row["source_index"])
    return [
        {"class": class_name, "items": value["items"], "source_samples": len(value["sources"])}
        for class_name, value in sorted(counts.items())
    ]


def composition_by_source(manifest):
    counts = defaultdict(int)
    provenance = {}
    for row in manifest:
        key = (row["class"], row["source_index"])
        counts[key] += 1
        provenance[key] = row
    return [
        {
            "class": class_name,
            "source_index": source_index,
            "original_sample_id": provenance[(class_name, source_index)]["original_sample_id"],
            "items": count,
        }
        for (class_name, source_index), count in sorted(counts.items())
    ]


def write_csv_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_indices(path, dataset_length):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "dataset_index" not in rows[0]:
        raise ValueError(f"{path} is not an evaluation-subset manifest")
    indices = [int(row["dataset_index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{path} contains duplicate dataset indices")
    if any(index < 0 or index >= dataset_length for index in indices):
        raise ValueError(f"{path} does not match the current dataset length ({dataset_length})")
    return indices


def make_evaluation_loader(
    dataset,
    dataset_name,
    batch_size,
    num_workers,
    mode="ordered",
    max_items=None,
    items_per_class=None,
    max_windows_per_source=1,
    seed=13,
    subset_from=None,
):
    """Build a deterministic loader and return its selected-manifest rows."""
    manifest = build_dataset_manifest(dataset, dataset_name)
    if subset_from is not None:
        selected_indices = set(load_manifest_indices(subset_from, len(dataset)))
        selected = [row for row in manifest if row["dataset_index"] in selected_indices]
    else:
        selected = select_manifest_rows(
            manifest, mode=mode, max_items=max_items,
            items_per_class=items_per_class,
            max_windows_per_source=max_windows_per_source, seed=seed,
        )
    if not selected:
        raise RuntimeError("The evaluation subset is empty.")
    subset = IndexedSubsetDataset(dataset, [row["dataset_index"] for row in selected])
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, selected
