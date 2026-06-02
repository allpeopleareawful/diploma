"""Download selected configs from phxember/Uni-MuMER-Data.

The default preset downloads normalized MathWriting recognition data used by
the dynamic error-corpus experiment. Other Uni-MuMER configs remain available
through explicit presets or repeated ``--config`` arguments.

Each config is saved separately, and the selected configs can also be
concatenated into one local HF-disk dataset with a single `train` split.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from dec_unimumer.data import extract_conversation_prompt_answer
from dec_unimumer.prompts import TASK_PROMPTS


DATASET_ID = "phxember/Uni-MuMER-Data"

PRESETS = {
    "mathwriting-recognition": ("mathwriting_train",),
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def nonempty_dir(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def prepare_empty_dir(path: Path, *, overwrite: bool) -> None:
    if nonempty_dir(path):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def selected_configs(args: argparse.Namespace) -> list[str]:
    configs = list(PRESETS[args.preset])
    if args.config:
        configs = args.config
    return list(dict.fromkeys(configs))


def task_from_config(config: str) -> str:
    if config.endswith("_tree"):
        return "tree"
    if config.endswith("_can"):
        return "counting"
    if config.endswith("_error_find"):
        return "error_find"
    if config.endswith("_error_fix"):
        return "error_fix"
    return "recognition"


def dataset_summary(dataset: Any) -> dict[str, Any]:
    return {
        "num_rows": len(dataset),
        "columns": list(dataset.column_names),
        "features": {name: str(feature) for name, feature in dataset.features.items()},
    }


def first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def target_from_row(row: dict[str, Any]) -> str:
    if row.get("conversations") is not None:
        _prompt, answer = extract_conversation_prompt_answer(row.get("conversations"))
        if answer:
            return answer.strip()
    if row.get("messages") is not None:
        _prompt, answer = extract_conversation_prompt_answer(row.get("messages"))
        if answer:
            return answer.strip()
    return first_nonempty(row, ("target", "answer", "gt", "latex", "label", "text", "caption"))


def validate_configs(dataset_id: str, configs: list[str], args: argparse.Namespace) -> None:
    if args.skip_config_validation:
        return

    try:
        from datasets import get_dataset_config_names
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets first: pip install datasets") from exc

    kwargs: dict[str, Any] = {}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if args.revision:
        kwargs["revision"] = args.revision

    available = set(get_dataset_config_names(dataset_id, **kwargs))
    missing = [config for config in configs if config not in available]
    if missing:
        preview = ", ".join(sorted(available)[:50])
        raise ValueError(
            "Some configs are not available in the dataset: "
            f"{', '.join(missing)}. Available configs include: {preview}"
        )


def load_or_download_config(config: str, args: argparse.Namespace) -> Any:
    from datasets import load_dataset, load_from_disk

    config_dir = args.output_dir / config
    if nonempty_dir(config_dir) and not args.overwrite:
        print(f"[{config}] exists, loading from disk: {config_dir}")
        return load_from_disk(str(config_dir))

    if config_dir.exists() and args.overwrite:
        shutil.rmtree(config_dir)

    load_kwargs: dict[str, Any] = {}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    if args.revision:
        load_kwargs["revision"] = args.revision

    print(f"[{config}] downloading {args.dataset_id} / {config} / split={args.split}")
    dataset = load_dataset(args.dataset_id, name=config, split=args.split, **load_kwargs)
    if args.limit:
        limit = min(args.limit, len(dataset))
        dataset = dataset.select(range(limit))

    dataset = add_metadata_columns(dataset, config, dataset_id=args.dataset_id)

    config_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(config_dir))
    write_json(
        config_dir / "download_summary.json",
        {
            "dataset_id": args.dataset_id,
            "config": config,
            "split": args.split,
            "task": task_from_config(config),
            "limit": args.limit or None,
            "output_dir": config_dir,
            "summary": dataset_summary(dataset),
        },
    )
    print(f"[{config}] saved: {config_dir} ({len(dataset)} rows)")
    return dataset


def add_metadata_columns(dataset: Any, config: str, *, dataset_id: str) -> Any:
    task = task_from_config(config)
    prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["recognition"])
    target_columns = [
        column
        for column in ("conversations", "messages", "target", "answer", "gt", "latex", "label", "text", "caption")
        if column in dataset.column_names
    ]
    if not target_columns:
        raise ValueError(
            f"Cannot build targets for config {config}. Available columns: {dataset.column_names}"
        )

    def build_batch(*column_values: list[Any]) -> dict[str, list[str]]:
        batch_size = len(column_values[0])
        targets: list[str] = []
        for row_index in range(batch_size):
            row = {
                column: values[row_index]
                for column, values in zip(target_columns, column_values, strict=True)
            }
            targets.append(target_from_row(row))
        return {
            "source_config": [config] * batch_size,
            "task": [task] * batch_size,
            "source_dataset": [dataset_id] * batch_size,
            "prompt": [prompt] * batch_size,
            "target": targets,
        }

    return dataset.map(
        build_batch,
        batched=True,
        input_columns=target_columns,
        desc=f"Preparing {config} prompts and targets",
    )


def combine_configs(datasets_by_config: dict[str, Any], args: argparse.Namespace) -> Any:
    from datasets import DatasetDict, concatenate_datasets

    prepare_empty_dir(args.combined_output_dir, overwrite=args.overwrite)
    datasets = list(datasets_by_config.values())
    try:
        combined = concatenate_datasets(datasets)
    except Exception as exc:  # noqa: BLE001 - re-raise with actionable context.
        columns = {config: dataset.column_names for config, dataset in datasets_by_config.items()}
        raise RuntimeError(
            "Could not concatenate selected configs. Their schemas may differ. "
            f"Columns by config: {columns}"
        ) from exc

    dataset_dict = DatasetDict({args.combined_split_name: combined})
    dataset_dict.save_to_disk(str(args.combined_output_dir))
    write_json(
        args.combined_output_dir / "download_summary.json",
        {
            "dataset_id": args.dataset_id,
            "configs": list(datasets_by_config.keys()),
            "split": args.split,
            "combined_split_name": args.combined_split_name,
            "limit_per_config": args.limit or None,
            "output_dir": args.combined_output_dir,
            "summary": {
                args.combined_split_name: {
                    "num_rows": len(combined),
                    "columns": list(combined.column_names),
                }
            },
        },
    )
    print(f"[combined] saved: {args.combined_output_dir} ({len(combined)} rows)")
    return dataset_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="mathwriting-recognition",
        help="Shortcut for common Uni-MuMER config groups.",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Config to download. Repeat to override the MathWriting recognition preset.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/unimumer_mathwriting_configs"))
    parser.add_argument(
        "--combine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concatenate selected configs into a single HF-disk DatasetDict.",
    )
    parser.add_argument(
        "--combined-output-dir",
        type=Path,
        default=Path("data/raw/unimumer_mathwriting_hf"),
    )
    parser.add_argument("--combined-split-name", default="train")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-config-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per config; 0 means full config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = selected_configs(args)
    validate_configs(args.dataset_id, configs, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets_by_config: dict[str, Any] = {}
    config_summaries: dict[str, Any] = {}

    for config in configs:
        dataset = load_or_download_config(config, args)
        datasets_by_config[config] = dataset
        config_summaries[config] = {
            "task": task_from_config(config),
            "local_path": str(args.output_dir / config),
            "summary": dataset_summary(dataset),
        }

    combined_summary: dict[str, Any] | None = None
    if args.combine:
        combined = combine_configs(datasets_by_config, args)
        combined_split = combined[args.combined_split_name]
        combined_summary = {
            "local_path": str(args.combined_output_dir),
            "split": args.combined_split_name,
            "num_rows": len(combined_split),
            "columns": list(combined_split.column_names),
        }

    summary = {
        "dataset_id": args.dataset_id,
        "preset": args.preset,
        "configs": configs,
        "split": args.split,
        "limit_per_config": args.limit or None,
        "output_dir": str(args.output_dir),
        "combined": combined_summary,
        "configs_summary": config_summaries,
    }
    summary_path = args.output_dir / "download_summary.json"
    write_json(summary_path, summary)

    print()
    print("Done.")
    print(f"Config datasets: {args.output_dir}")
    if combined_summary:
        print(f"Combined dataset: {args.combined_output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
