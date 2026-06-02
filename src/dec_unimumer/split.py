"""Create a fixed 95/5 MathWriting train/validation split.

The output contains non-overlapping `train_pool` and `validation` partitions.
`train_pool` additionally contains balanced S1..S5 shard assignments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def stable_hash(seed: int, sample_id: str, source_index: int) -> str:
    payload = f"{seed}:{sample_id}:{source_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(ids):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def split_assignments(
    sample_ids: list[str],
    *,
    seed: int,
    validation_ratio: float,
    num_shards: int,
) -> list[dict[str, Any]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if num_shards < 1:
        raise ValueError("num_shards must be positive.")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample ids must be unique before making the experiment split.")

    ordered = sorted(
        range(len(sample_ids)),
        key=lambda index: stable_hash(seed, sample_ids[index], index),
    )
    validation_count = round(len(ordered) * validation_ratio)
    if validation_count < 1 or validation_count >= len(ordered):
        raise ValueError("validation_ratio must leave at least one sample in each partition.")
    train_count = len(ordered) - validation_count
    train_indices = ordered[:train_count]
    validation_indices = ordered[train_count:]

    assignments: list[dict[str, Any]] = []
    for position, source_index in enumerate(train_indices):
        assignments.append(
            {
                "id": sample_ids[source_index],
                "source_index": source_index,
                "experiment_split": "train_pool",
                "train_shard": f"S{position % num_shards + 1}",
                "split_order": position,
            }
        )
    for position, source_index in enumerate(validation_indices):
        assignments.append(
            {
                "id": sample_ids[source_index],
                "source_index": source_index,
                "experiment_split": "validation",
                "train_shard": "",
                "split_order": position,
            }
        )
    return assignments


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source_dataset(args: argparse.Namespace) -> Any:
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets first: pip install datasets") from exc

    if args.dataset_path is not None:
        loaded = load_from_disk(str(args.dataset_path))
        if isinstance(loaded, DatasetDict):
            if args.source_split not in loaded:
                raise ValueError(
                    f"Split {args.source_split!r} not found. Available: {', '.join(loaded)}"
                )
            dataset = loaded[args.source_split]
        else:
            dataset = loaded
    else:
        dataset = load_dataset(
            args.dataset_id,
            name=args.config,
            split=args.source_split,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    return dataset


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import DatasetDict

    dataset = load_source_dataset(args)
    if args.id_field not in dataset.column_names:
        raise ValueError(
            f"ID field {args.id_field!r} is missing. Columns: {dataset.column_names}"
        )
    sample_ids = [str(value) for value in dataset[args.id_field]]
    assignments = split_assignments(
        sample_ids,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        num_shards=args.num_shards,
    )

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {args.output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict[str, Any]]] = {
        split_name: [row for row in assignments if row["experiment_split"] == split_name]
        for split_name in ("train_pool", "validation")
    }
    output_splits: dict[str, Any] = {}
    for split_name, rows in by_split.items():
        indices = [int(row["source_index"]) for row in rows]
        selected = dataset.select(indices)
        selected = selected.add_column(
            "source_index",
            [int(row["source_index"]) for row in rows],
        )
        selected = selected.add_column("experiment_split", [split_name] * len(rows))
        selected = selected.add_column(
            "train_shard",
            [str(row["train_shard"]) for row in rows],
        )
        output_splits[split_name] = selected

    dataset_dict = DatasetDict(output_splits)
    dataset_path = args.output_dir / "dataset"
    dataset_dict.save_to_disk(str(dataset_path))
    assignments_path = args.output_dir / "assignments.jsonl"
    write_jsonl(assignments_path, assignments)

    shard_counts = Counter(
        row["train_shard"] for row in by_split["train_pool"] if row["train_shard"]
    )
    summary = {
        "dataset_id": args.dataset_id,
        "config": args.config,
        "dataset_path": str(args.dataset_path or ""),
        "source_split": args.source_split,
        "seed": args.seed,
        "ratios": {
            "train_pool": 1.0 - args.validation_ratio,
            "validation": args.validation_ratio,
        },
        "counts": {name: len(rows) for name, rows in by_split.items()},
        "train_shards": dict(sorted(shard_counts.items())),
        "fingerprints": {
            name: fingerprint([str(row["id"]) for row in rows])
            for name, rows in by_split.items()
        },
        "dataset_output": str(dataset_path),
        "assignments": str(assignments_path),
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-id", default="phxember/Uni-MuMER-Data")
    parser.add_argument("--config", default="mathwriting_train")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mathwriting/split"))
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = prepare(parse_args())
    print("MathWriting split prepared")
    print(f"counts: {summary['counts']}")
    print(f"train shards: {summary['train_shards']}")
    print(f"dataset: {summary['dataset_output']}")
    print(f"assignments: {summary['assignments']}")


if __name__ == "__main__":
    main()
