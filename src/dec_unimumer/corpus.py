"""Build one Uni-MuMER error corpus from current model predictions.

Selection is performed from predictions over the complete fixed train pool.
All current errors are included. They are annotated by edit type and ranked by
CER, confidence, and rotating S1..S5 shard priority for analysis. Replay,
KEEP and recognition rows are sampled proportionally. Every image contributes at
most one SFT row per cycle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from dec_unimumer.data import (
    IMAGE_FIELDS,
    ID_FIELDS,
    image_to_pil,
    infer_field,
)
from dec_unimumer.error_markup import build_marked_formula, safe_filename
from dec_unimumer.prompts import (
    RECOGNITION_PROMPT,
    error_correction_prompt,
    error_detection_prompt,
)
from dec_unimumer.latex.metrics import compute_metrics
from dec_unimumer.latex.normalize import clean_model_prediction, normalize_latex
from dec_unimumer.latex.tokenizer import tokenize_latex


@dataclass(frozen=True)
class CycleComposition:
    total: int
    current_errors: int
    replay: int
    keep: int
    recognition: int


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported row format: {path.suffix}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def deterministic_key(seed: int, *parts: str) -> str:
    value = ":".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def active_shards_for_cycle(cycle: int, num_shards: int = 5) -> tuple[str, str]:
    if num_shards != 5:
        first = (2 * max(cycle - 1, 0)) % num_shards
        return f"S{first + 1}", f"S{(first + 1) % num_shards + 1}"
    schedule = (
        ("S1", "S2"),
        ("S3", "S4"),
        ("S5", "S1"),
        ("S2", "S3"),
        ("S4", "S5"),
    )
    return schedule[(max(cycle, 1) - 1) % len(schedule)]


def composition_from_errors(
    current_errors: int,
    *,
    replay: int,
    error_fraction: float,
    keep_fraction: float,
) -> CycleComposition:
    if current_errors < 1:
        raise ValueError("No current errors remain after cooldown filtering.")
    if error_fraction <= 0 or keep_fraction < 0 or error_fraction + keep_fraction > 1:
        raise ValueError("Invalid error/keep fractions.")
    error_rows = current_errors + replay
    total = max(error_rows, round(error_rows / error_fraction))
    keep = round(total * keep_fraction)
    recognition = max(0, total - error_rows - keep)
    return CycleComposition(
        total=total,
        current_errors=current_errors,
        replay=replay,
        keep=keep,
        recognition=recognition,
    )


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def edit_signature(reference: str, prediction: str) -> tuple[str, dict[str, int]]:
    reference_tokens = tokenize_latex(reference)
    prediction_tokens = tokenize_latex(prediction)
    counts: Counter[str] = Counter()
    matcher = SequenceMatcher(a=prediction_tokens, b=reference_tokens, autojunk=False)
    for tag, *_bounds in matcher.get_opcodes():
        if tag == "equal":
            continue
        counts[tag] += 1
    present = [name for name in ("replace", "delete", "insert") if counts[name]]
    if not present:
        return "none", dict(counts)
    if len(present) == 1:
        return present[0], dict(counts)
    return "mixed", dict(counts)


def load_assignment_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in read_rows(path)
        if str(row.get("id") or "").strip()
    }


def priority_score(
    *,
    cer: float,
    confidence: float | None,
    shard: str,
    active_shards: set[str],
    cer_weight: float,
    confidence_weight: float,
    active_shard_bonus: float,
    confidence_priority: str,
) -> float:
    confidence_value = 0.5 if confidence is None else min(max(confidence, 0.0), 1.0)
    confidence_component = (
        1.0 - confidence_value if confidence_priority == "uncertain" else confidence_value
    )
    return (
        cer_weight * min(max(cer, 0.0), 1.0)
        + confidence_weight * confidence_component
        + (active_shard_bonus if shard in active_shards else 0.0)
    )


def prediction_records(
    rows: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    *,
    prediction_field: str,
    id_field: str,
    ground_truth_field: str,
    status_field: str,
    confidence_field: str,
    normalize: bool,
    active_shards: set[str],
    cer_weight: float,
    confidence_weight: float,
    active_shard_bonus: float,
    confidence_priority: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get(id_field) or "").strip()
        if not sample_id or str(row.get(status_field) or "ok") != "ok":
            continue
        ground_truth = str(row.get(ground_truth_field) or "").strip()
        prediction = clean_model_prediction(str(row.get(prediction_field) or ""))
        if not ground_truth or not prediction:
            continue
        metrics = compute_metrics(ground_truth, prediction)
        correct = (
            bool(metrics.normalized_exact_match)
            if normalize
            else bool(metrics.raw_exact_match)
        )
        cer = float(metrics.normalized_cer if normalize else metrics.cer)
        reference = normalize_latex(ground_truth) if normalize else ground_truth
        candidate = normalize_latex(prediction) if normalize else prediction
        error_type, edit_counts = edit_signature(reference, candidate)
        assignment = assignments.get(sample_id, {})
        shard = str(assignment.get("train_shard") or "")
        confidence = parse_float(row.get(confidence_field))
        records.append(
            {
                "source_sample_id": sample_id,
                "ground_truth": reference,
                "candidate_prediction": candidate,
                "raw_prediction": str(row.get(prediction_field) or ""),
                "correct": correct,
                "selected_cer": cer,
                "confidence": confidence,
                "confidence_available": confidence is not None,
                "avg_logprob": parse_float(row.get("avg_logprob")),
                "error_type": error_type,
                "edit_counts": edit_counts,
                "train_shard": shard,
                "priority_score": priority_score(
                    cer=cer,
                    confidence=confidence,
                    shard=shard,
                    active_shards=active_shards,
                    cer_weight=cer_weight,
                    confidence_weight=confidence_weight,
                    active_shard_bonus=active_shard_bonus,
                    confidence_priority=confidence_priority,
                ),
            }
        )
    return records


def balanced_error_select(
    records: list[dict[str, Any]],
    limit: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("error_type") or "unknown")].append(record)
    for name, items in groups.items():
        items.sort(
            key=lambda row: (
                -float(row.get("priority_score") or 0.0),
                deterministic_key(seed, name, str(row["source_sample_id"])),
            )
        )

    selected: list[dict[str, Any]] = []
    group_names = sorted(groups)
    while len(selected) < limit:
        progress = False
        for name in group_names:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop(0))
                progress = True
        if not progress:
            break
    return selected


def deterministic_sample(
    records: list[dict[str, Any]],
    limit: int,
    *,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: deterministic_key(seed, label, str(row["source_sample_id"])),
    )
    return ordered[:limit]


def selected_ids_from_manifests(paths: list[Path]) -> set[str]:
    selected: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            sample_id = str(row.get("source_sample_id") or row.get("id") or "").strip()
            if sample_id:
                selected.add(sample_id)
    return selected


def load_replay_records(paths: list[Path]) -> list[dict[str, Any]]:
    replay: list[dict[str, Any]] = []
    for path in paths:
        for row in read_rows(path):
            if str(row.get("bucket") or "") not in {"new_error", "replay"}:
                continue
            if not row.get("candidate_prediction") or not row.get("ground_truth"):
                continue
            item = dict(row)
            item["replay_source"] = str(path)
            replay.append(item)
    return replay


def choose_cycle_examples(
    records: list[dict[str, Any]],
    *,
    replay_records: list[dict[str, Any]],
    cooldown_ids: set[str],
    error_fraction: float,
    keep_fraction: float,
    replay_fraction_of_errors: float,
    seed: int,
) -> tuple[list[dict[str, Any]], CycleComposition]:
    if not 0 <= replay_fraction_of_errors < 1:
        raise ValueError("Replay fraction must be in [0, 1).")
    available = [
        row for row in records if str(row["source_sample_id"]) not in cooldown_ids
    ]
    wrong = [row for row in available if not bool(row["correct"])]
    correct = [row for row in available if bool(row["correct"])]
    replay_pool = [
        row
        for row in replay_records
        if str(row["source_sample_id"]) not in cooldown_ids
    ]

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    new_selected = balanced_error_select(wrong, len(wrong), seed=seed)
    for row in new_selected:
        selected.append({**row, "bucket": "new_error"})
        used_ids.add(str(row["source_sample_id"]))

    desired_replay = round(
        len(new_selected)
        * replay_fraction_of_errors
        / max(1e-12, 1.0 - replay_fraction_of_errors)
    )
    replay_selected = deterministic_sample(
        [
            row
            for row in replay_pool
            if str(row["source_sample_id"]) not in used_ids
        ],
        desired_replay,
        seed=seed,
        label="replay",
    )
    for row in replay_selected:
        sample_id = str(row["source_sample_id"])
        if sample_id in used_ids:
            continue
        selected.append({**row, "bucket": "replay"})
        used_ids.add(sample_id)

    composition = composition_from_errors(
        len(new_selected),
        replay=len(replay_selected),
        error_fraction=error_fraction,
        keep_fraction=keep_fraction,
    )

    keep_selected = deterministic_sample(
        [row for row in correct if str(row["source_sample_id"]) not in used_ids],
        composition.keep,
        seed=seed,
        label="keep",
    )
    for row in keep_selected:
        selected.append({**row, "bucket": "keep"})
        used_ids.add(str(row["source_sample_id"]))

    recognition_selected = deterministic_sample(
        [row for row in available if str(row["source_sample_id"]) not in used_ids],
        composition.recognition,
        seed=seed,
        label="recognition",
    )
    for row in recognition_selected:
        selected.append({**row, "bucket": "recognition"})
        used_ids.add(str(row["source_sample_id"]))

    return selected, composition


def load_source_dataset(path: Path, split: str) -> Any:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets first: pip install datasets") from exc
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise ValueError(f"Split {split!r} not found. Available: {', '.join(loaded)}")
        return loaded[split]
    return loaded


def save_selected_images(
    selected: list[dict[str, Any]],
    *,
    dataset: Any,
    id_field: str | None,
    image_field: str | None,
    output_dir: Path,
) -> dict[str, str]:
    columns = list(dataset.column_names)
    resolved_id = id_field or infer_field(columns, ID_FIELDS, "id")
    resolved_image = image_field or infer_field(columns, IMAGE_FIELDS, "image")
    ids = [str(value) for value in dataset[resolved_id]]
    index_by_id = {sample_id: index for index, sample_id in enumerate(ids)}
    if len(index_by_id) != len(ids):
        raise ValueError("Source dataset ids are not unique.")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for record in selected:
        sample_id = str(record["source_sample_id"])
        if sample_id not in index_by_id:
            raise KeyError(f"Selected id is missing from source train_pool: {sample_id}")
        image = image_to_pil(dataset[index_by_id[sample_id]][resolved_image])
        path = image_dir / f"{safe_filename(sample_id)}.png"
        image.save(path, format="PNG")
        paths[sample_id] = str(path.relative_to(output_dir)).replace("\\", "/")
    return paths


def error_task_for_record(record: dict[str, Any], *, seed: int, mode: str) -> str:
    if mode in {"detection", "correction"}:
        return "error_find" if mode == "detection" else "error_fix"
    key = deterministic_key(seed, "error-task", str(record["source_sample_id"]))
    return "error_find" if int(key[:8], 16) % 2 == 0 else "error_fix"


def training_row(
    record: dict[str, Any],
    *,
    image_path: str,
    cycle: int,
    seed: int,
    error_task_mode: str,
) -> dict[str, Any]:
    sample_id = str(record["source_sample_id"])
    bucket = str(record["bucket"])
    ground_truth = str(record["ground_truth"])
    candidate = str(record.get("candidate_prediction") or ground_truth)
    common = {
        **record,
        "id": f"{sample_id}::cycle{cycle}::{bucket}",
        "source_sample_id": sample_id,
        "image_path": image_path,
        "dynamic_cycle": cycle,
    }
    if bucket in {"new_error", "replay"}:
        marked = build_marked_formula(ground_truth, candidate)
        if not marked.edit_count:
            raise ValueError(f"Error row has no token edits: {sample_id}")
        task = error_task_for_record(record, seed=seed, mode=error_task_mode)
        if task == "error_find":
            return {
                **common,
                "task": task,
                "prompt": error_detection_prompt(candidate),
                "label": marked.marked,
                "marked_formula": marked.marked,
                "correction_log": marked.correction_log,
            }
        answer = ground_truth
        if marked.correction_log:
            answer = f"{marked.correction_log}\n{ground_truth}"
        return {
            **common,
            "task": task,
            "prompt": error_correction_prompt(marked.marked),
            "label": answer,
            "marked_formula": marked.marked,
            "correction_log": marked.correction_log,
        }
    if bucket == "keep":
        return {
            **common,
            "task": "error_find",
            "prompt": error_detection_prompt(candidate),
            "label": candidate,
            "marked_formula": candidate,
            "correction_log": "",
        }
    return {
        **common,
        "task": "recognition",
        "prompt": RECOGNITION_PROMPT,
        "label": ground_truth,
        "marked_formula": "",
        "correction_log": "",
    }


def build_cycle(args: argparse.Namespace) -> dict[str, Any]:
    selection_cycle = args.cycle
    active_shards = (
        set(args.active_shard)
        if args.active_shard
        else set(active_shards_for_cycle(max(args.cycle, 1)))
    )
    predictions = read_rows(args.predictions)
    assignments = load_assignment_map(args.assignments)
    records = prediction_records(
        predictions,
        assignments,
        prediction_field=args.prediction_field,
        id_field=args.prediction_id_field,
        ground_truth_field=args.ground_truth_field,
        status_field=args.status_field,
        confidence_field=args.confidence_field,
        normalize=args.normalize,
        active_shards=active_shards,
        cer_weight=args.cer_weight,
        confidence_weight=args.confidence_weight,
        active_shard_bonus=args.active_shard_bonus,
        confidence_priority=args.confidence_priority,
    )
    cooldown_ids = selected_ids_from_manifests(args.cooldown_manifest)
    replay_records = load_replay_records(args.replay_manifest)
    selected, composition = choose_cycle_examples(
        records,
        replay_records=replay_records,
        cooldown_ids=cooldown_ids,
        error_fraction=args.error_fraction,
        keep_fraction=args.keep_fraction,
        replay_fraction_of_errors=args.replay_fraction_of_errors,
        seed=args.seed + selection_cycle,
    )

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {args.output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_source_dataset(args.dataset_path, args.dataset_split)
    image_paths = save_selected_images(
        selected,
        dataset=dataset,
        id_field=args.id_field,
        image_field=args.image_field,
        output_dir=args.output_dir,
    )
    train_rows = [
        training_row(
            record,
            image_path=image_paths[str(record["source_sample_id"])],
            cycle=args.cycle,
            seed=args.seed + selection_cycle,
            error_task_mode=args.error_task_mode,
        )
        for record in selected
    ]
    random.Random(args.seed + selection_cycle).shuffle(train_rows)

    selected_path = args.output_dir / "selected_examples.jsonl"
    train_path = args.output_dir / "train.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(train_path, train_rows)

    bucket_counts = Counter(str(row["bucket"]) for row in selected)
    task_counts = Counter(str(row["task"]) for row in train_rows)
    error_type_counts = Counter(
        str(row.get("error_type") or "none")
        for row in selected
        if str(row["bucket"]) in {"new_error", "replay"}
    )
    shard_counts = Counter(str(row.get("train_shard") or "") for row in selected)
    composition_shortfall = {
        "replay": max(0, composition.replay - bucket_counts["replay"]),
        "keep": max(0, composition.keep - bucket_counts["keep"]),
        "recognition": max(
            0,
            composition.recognition - bucket_counts["recognition"],
        ),
    }
    summary = {
        "cycle": args.cycle,
        "selection_cycle": selection_cycle,
        "predictions": str(args.predictions),
        "assignments": str(args.assignments),
        "dataset_path": str(args.dataset_path),
        "dataset_split": args.dataset_split,
        "active_shards": sorted(active_shards),
        "cooldown_ids": len(cooldown_ids),
        "prediction_records": len(records),
        "wrong_prediction_records": sum(1 for row in records if not row["correct"]),
        "correct_prediction_records": sum(1 for row in records if row["correct"]),
        "confidence_available": sum(1 for row in records if row["confidence_available"]),
        "derived_composition": composition.__dict__,
        "composition_shortfall": composition_shortfall,
        "training_rows": len(train_rows),
        "unique_images": len({row["source_sample_id"] for row in selected}),
        "bucket_counts": dict(bucket_counts),
        "task_counts": dict(task_counts),
        "error_type_counts": dict(error_type_counts),
        "shard_counts": dict(sorted(shard_counts.items())),
        "prompt_characters": sum(len(str(row["prompt"])) for row in train_rows),
        "label_characters": sum(len(str(row["label"])) for row in train_rows),
        "max_examples_per_image": max(
            Counter(str(row["source_sample_id"]) for row in train_rows).values(),
            default=0,
        ),
        "train_jsonl": str(train_path),
        "selected_examples": str(selected_path),
        "selection_fingerprint": hashlib.sha256(
            "\n".join(
                sorted(
                    f"{row['source_sample_id']}:{row['bucket']}:{row['task']}"
                    for row in train_rows
                )
            ).encode("utf-8")
        ).hexdigest(),
        "args": {
            key: [str(item) for item in value]
            if isinstance(value, list) and value and isinstance(value[0], Path)
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-split", default="train_pool")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--prediction-id-field", default="id")
    parser.add_argument("--ground-truth-field", default="ground_truth")
    parser.add_argument("--status-field", default="status")
    parser.add_argument("--confidence-field", default="confidence")
    parser.add_argument("--id-field")
    parser.add_argument("--image-field")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--error-fraction", type=float, default=0.50)
    parser.add_argument("--keep-fraction", type=float, default=0.35)
    parser.add_argument("--replay-fraction-of-errors", type=float, default=0.20)
    parser.add_argument(
        "--error-task-mode",
        choices=["alternate", "detection", "correction"],
        default="alternate",
    )
    parser.add_argument("--replay-manifest", type=Path, action="append", default=[])
    parser.add_argument("--cooldown-manifest", type=Path, action="append", default=[])
    parser.add_argument("--active-shard", action="append")
    parser.add_argument("--cer-weight", type=float, default=0.65)
    parser.add_argument("--confidence-weight", type=float, default=0.20)
    parser.add_argument("--active-shard-bonus", type=float, default=0.15)
    parser.add_argument(
        "--confidence-priority",
        choices=["uncertain", "confident"],
        default="uncertain",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build_cycle(parse_args())
    print("Error corpus built")
    print(f"cycle: {summary['cycle']}")
    print(f"active shards: {summary['active_shards']}")
    print(f"training rows: {summary['training_rows']}")
    print(f"unique images: {summary['unique_images']}")
    print(f"buckets: {summary['bucket_counts']}")
    print(f"tasks: {summary['task_counts']}")
    print(f"train manifest: {summary['train_jsonl']}")


if __name__ == "__main__":
    main()
