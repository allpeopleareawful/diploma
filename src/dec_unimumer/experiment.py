"""Run the Uni-MuMER frozen, recognition, static, and dynamic experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from dec_unimumer.crohme import CROHME_YEARS
from dec_unimumer.paths import PROJECT_ROOT
from dec_unimumer.reporting import aggregate_reports


MODEL_ID = "phxember/Uni-MuMER-Qwen2.5-VL-3B"


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_stage(
    name: str,
    command: list[str],
    *,
    marker: Path,
    cwd: Path = PROJECT_ROOT,
) -> None:
    if marker.exists():
        print(f"[skip] {name}: {marker}", flush=True)
        return
    print(f"\n[stage] {name}", flush=True)
    print(display_command(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(name, encoding="utf-8")


def module_command(python: str, module: str, *arguments: str) -> list[str]:
    return [python, "-m", f"dec_unimumer.{module}", *arguments]


def crohme_manifests(args: argparse.Namespace) -> dict[int, Path]:
    return {year: args.crohme_root / str(year) / "test.jsonl" for year in CROHME_YEARS}


def inference_command(
    args: argparse.Namespace,
    *,
    split: str,
    output_dir: Path,
    adapter: Path | None = None,
    metadata: Path | None = None,
) -> list[str]:
    command = module_command(
        args.python,
        "inference",
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--max-tokens",
        str(args.max_new_tokens),
        "--inference-batch-size",
        str(args.inference_batch_size),
        "--flush-csv-every",
        str(args.flush_csv_every),
        "--output-dir",
        str(output_dir),
        "--resume",
    )
    if metadata is None:
        command.extend(
            [
                "--source",
                "hf_disk",
                "--dataset-path",
                str(args.split_dir / "dataset"),
                "--split",
                split,
                "--conversation-field",
                "conversations",
            ]
        )
    else:
        command.extend(
            [
                "--source",
                "jsonl",
                "--metadata",
                str(metadata),
                "--root-dir",
                str(metadata.parent),
                "--split",
                split,
                "--label-field",
                "ground_truth",
                "--prompt-field",
                "prompt",
            ]
        )
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    return command


def training_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    adapter: Path | None,
    corpus_dir: Path | None = None,
    recognition_split: str | None = None,
    epochs: float | None = None,
) -> list[str]:
    if (corpus_dir is None) == (recognition_split is None):
        raise ValueError("Specify exactly one training source.")
    if corpus_dir is not None:
        source_arguments = [
            "--source",
            "jsonl",
            "--metadata",
            str(corpus_dir / "train.jsonl"),
            "--root-dir",
            str(corpus_dir),
            "--label-field",
            "label",
            "--prompt-field",
            "prompt",
        ]
    else:
        source_arguments = [
            "--source",
            "hf_disk",
            "--dataset-path",
            str(args.split_dir / "dataset"),
            "--split",
            str(recognition_split),
            "--conversation-field",
            "conversations",
        ]
    try:
        run_name = "_".join(output_dir.relative_to(args.output_root).parts)
    except ValueError:
        run_name = output_dir.name
    command = module_command(
        args.python,
        "training",
        "--model",
        args.model,
        *source_arguments,
        "--validation-source",
        "hf_disk",
        "--validation-dataset-path",
        str(args.split_dir / "dataset"),
        "--validation-split",
        "validation",
        "--validation-conversation-field",
        "conversations",
        "--validation-baseline-predictions",
        str(args.output_root / "frozen" / "validation" / "predictions.jsonl"),
        "--num-train-epochs",
        str(args.epochs_per_cycle if epochs is None else epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-length",
        str(args.max_length),
        "--min-pixels",
        str(args.min_pixels),
        "--max-pixels",
        str(args.max_pixels),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        "0",
        "--save-steps",
        "0",
        "--dtype",
        args.dtype,
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--lora-target-modules",
        args.lora_target_modules,
        "--vision-lora",
        "--vision-lora-target-modules",
        args.vision_lora_target_modules,
        "--seed",
        str(args.seed),
        "--generation-eval-batch-size",
        str(args.generation_eval_batch_size),
        "--generation-eval-max-new-tokens",
        str(args.max_new_tokens),
        "--report-to",
        args.report_to,
        "--wandb-project",
        args.wandb_project,
        "--wandb-run-name",
        run_name,
        "--output-dir",
        str(output_dir),
    )
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    if args.device_map:
        command.extend(["--device-map", args.device_map])
    if args.load_in_4bit:
        command.append("--load-in-4bit")
    if args.cdm_evaluator is not None:
        command.extend(
            [
                "--cdm-evaluator",
                str(args.cdm_evaluator),
                "--cdm-pools",
                str(args.cdm_pools),
            ]
        )
        if args.cdm_python:
            command.extend(["--cdm-python", args.cdm_python])
        if args.cdm_docker_image:
            command.extend(["--cdm-docker-image", args.cdm_docker_image])
    return command


def corpus_command(
    args: argparse.Namespace,
    *,
    cycle: int,
    predictions: Path,
    output_dir: Path,
) -> list[str]:
    command = module_command(
        args.python,
        "corpus",
        "--cycle",
        str(cycle),
        "--predictions",
        str(predictions),
        "--assignments",
        str(args.split_dir / "assignments.jsonl"),
        "--dataset-path",
        str(args.split_dir / "dataset"),
        "--dataset-split",
        "train_pool",
        "--error-fraction",
        str(args.error_fraction),
        "--keep-fraction",
        str(args.keep_fraction),
        "--replay-fraction-of-errors",
        str(args.replay_fraction_of_errors),
        "--error-task-mode",
        args.error_task_mode,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    )
    if cycle == 0:
        for shard in ("S1", "S2", "S3", "S4", "S5"):
            command.extend(["--active-shard", shard])
    else:
        for replay_cycle in range(cycle):
            command.extend(
                [
                    "--replay-manifest",
                    str(args.corpus_root / f"D{replay_cycle}" / "selected_examples.jsonl"),
                ]
            )
        command.extend(
            [
                "--cooldown-manifest",
                str(args.corpus_root / f"D{cycle - 1}" / "selected_examples.jsonl"),
            ]
        )
    return command


def comparison_command(
    args: argparse.Namespace,
    *,
    predictions: Path,
    baseline_predictions: Path,
    output_dir: Path,
) -> list[str]:
    command = module_command(
        args.python,
        "evaluation",
        "--input",
        str(predictions),
        "--baseline-input",
        str(baseline_predictions),
        "--prediction-field",
        "prediction",
        "--baseline-input-prediction-field",
        "prediction",
        "--output-dir",
        str(output_dir),
        "--cdm-evaluator",
        str(args.cdm_evaluator),
        "--cdm-pools",
        str(args.cdm_pools),
    )
    if args.cdm_python:
        command.extend(["--cdm-python", args.cdm_python])
    if args.cdm_docker_image:
        command.extend(["--cdm-docker-image", args.cdm_docker_image])
    return command


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_fixed_split(args: argparse.Namespace) -> None:
    marker = args.split_dir / ".complete"
    summary_path = args.split_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        ratios = summary.get("ratios", {})
        counts = summary.get("counts", {})
        compatible = (
            set(counts) == {"train_pool", "validation"}
            and abs(float(ratios.get("validation", -1)) - args.validation_ratio) < 1e-12
        )
        if not compatible:
            raise RuntimeError(
                f"Existing split at {args.split_dir} does not match the requested protocol. "
                "Remove that generated directory before starting this experiment."
            )
    if (
        not marker.exists()
        and summary_path.exists()
        and (args.split_dir / "dataset").exists()
        and (args.split_dir / "assignments.jsonl").exists()
    ):
        marker.write_text("existing fixed split", encoding="utf-8")
    run_stage(
        "prepare fixed MathWriting train/validation split",
        module_command(
            args.python,
            "split",
            "--dataset-path",
            str(args.dataset_path),
            "--source-split",
            args.source_split,
            "--output-dir",
            str(args.split_dir),
            "--validation-ratio",
            str(args.validation_ratio),
            "--num-shards",
            "5",
            "--seed",
            str(args.seed),
        ),
        marker=marker,
    )


def run_frozen(args: argparse.Namespace) -> None:
    for split in ("train_pool", "validation"):
        output_dir = args.output_root / "frozen" / split
        run_stage(
            f"B0 frozen recognition: {split}",
            inference_command(args, split=split, output_dir=output_dir),
            marker=output_dir / ".complete",
        )
    for year, manifest in crohme_manifests(args).items():
        output_dir = args.output_root / "frozen" / f"crohme_{year}"
        run_stage(
            f"B0 frozen recognition: CROHME {year}",
            inference_command(
                args,
                split="test",
                output_dir=output_dir,
                metadata=manifest,
            ),
            marker=output_dir / ".complete",
        )


def run_frozen_reports(args: argparse.Namespace) -> None:
    names = ("validation", *(f"crohme_{year}" for year in CROHME_YEARS))
    for name in names:
        predictions = args.output_root / "frozen" / name / "predictions.jsonl"
        report_dir = args.report_root / "frozen" / name
        run_stage(
            f"B0 metrics: {name}",
            comparison_command(
                args,
                predictions=predictions,
                baseline_predictions=predictions,
                output_dir=report_dir,
            ),
            marker=report_dir / ".complete",
        )


def train_error_cycle(
    args: argparse.Namespace,
    *,
    variant: str,
    checkpoint_cycle: int,
    corpus_cycle: int,
    previous_adapter: Path | None,
) -> Path:
    corpus_dir = args.corpus_root / f"D{corpus_cycle}"
    checkpoint_dir = args.output_root / variant / f"C{checkpoint_cycle}"
    run_stage(
        f"train {variant} C{checkpoint_cycle} on D{corpus_cycle}",
        training_command(
            args,
            corpus_dir=corpus_dir,
            output_dir=checkpoint_dir,
            adapter=previous_adapter,
        ),
        marker=checkpoint_dir / ".complete",
    )
    return checkpoint_dir / "adapter"


def update_dynamic_state(
    args: argparse.Namespace,
    *,
    cycle: int,
    validation_summary: Path,
) -> dict[str, Any]:
    marker = args.output_root / "dynamic_error" / "state_markers" / f"C{cycle}.complete"
    if not marker.exists():
        run_stage(
            f"update dynamic early stopping after C{cycle}",
            module_command(
                args.python,
                "state",
                "--state",
                str(args.state_path),
                "--cycle",
                str(cycle),
                "--validation-summary",
                str(validation_summary),
                "--metric",
                args.early_stop_metric,
                "--patience",
                str(args.patience),
                "--min-delta",
                str(args.min_delta),
            ),
            marker=marker,
        )
    return load_json(args.state_path)


def run_dynamic_error(args: argparse.Namespace) -> dict[str, Any]:
    adapter = train_error_cycle(
        args,
        variant="dynamic_error",
        checkpoint_cycle=1,
        corpus_cycle=0,
        previous_adapter=None,
    )
    state = update_dynamic_state(
        args,
        cycle=1,
        validation_summary=args.output_root
        / "dynamic_error"
        / "C1"
        / "validation"
        / "final"
        / "summary.json",
    )
    last_cycle = 1

    for checkpoint_cycle in range(1, args.max_cycles):
        if state["should_stop"]:
            break
        mining_dir = args.output_root / "dynamic_error" / "mining" / f"C{checkpoint_cycle}"
        run_stage(
            f"recognize train pool with dynamic C{checkpoint_cycle}",
            inference_command(
                args,
                split="train_pool",
                output_dir=mining_dir,
                adapter=adapter,
            ),
            marker=mining_dir / ".complete",
        )

        corpus_cycle = checkpoint_cycle
        corpus_dir = args.corpus_root / f"D{corpus_cycle}"
        run_stage(
            f"build dynamic D{corpus_cycle}",
            corpus_command(
                args,
                cycle=corpus_cycle,
                predictions=mining_dir / "predictions.jsonl",
                output_dir=corpus_dir,
            ),
            marker=corpus_dir / ".complete",
        )

        next_cycle = checkpoint_cycle + 1
        adapter = train_error_cycle(
            args,
            variant="dynamic_error",
            checkpoint_cycle=next_cycle,
            corpus_cycle=corpus_cycle,
            previous_adapter=adapter,
        )
        state = update_dynamic_state(
            args,
            cycle=next_cycle,
            validation_summary=args.output_root
            / "dynamic_error"
            / f"C{next_cycle}"
            / "validation"
            / "final"
            / "summary.json",
        )
        last_cycle = next_cycle

    best_cycle = int(state["best_cycle"])
    return {
        "adapter": args.output_root / "dynamic_error" / f"C{best_cycle}" / "adapter",
        "best_cycle": best_cycle,
        "last_cycle": last_cycle,
        "state": state,
    }


def run_static_error(args: argparse.Namespace, *, cycles: int) -> Path:
    adapter: Path | None = None
    for cycle in range(1, cycles + 1):
        adapter = train_error_cycle(
            args,
            variant="static_error",
            checkpoint_cycle=cycle,
            corpus_cycle=0,
            previous_adapter=adapter,
        )
    if adapter is None:
        raise ValueError("Static training requires at least one cycle.")
    return adapter


def run_recognition_lora(args: argparse.Namespace) -> Path:
    output_dir = args.output_root / "recognition_lora" / "training"
    run_stage(
        "train B1 recognition LoRA on MathWriting train pool",
        training_command(
            args,
            output_dir=output_dir,
            adapter=None,
            recognition_split="train_pool",
            epochs=args.recognition_epochs,
        ),
        marker=output_dir / ".complete",
    )
    return output_dir / "adapter"


def evaluate_variant(args: argparse.Namespace, *, variant: str, adapter: Path) -> None:
    for year, manifest in crohme_manifests(args).items():
        output_dir = args.output_root / variant / f"crohme_{year}"
        run_stage(
            f"{variant} recognition: CROHME {year}",
            inference_command(
                args,
                split="test",
                output_dir=output_dir,
                adapter=adapter,
                metadata=manifest,
            ),
            marker=output_dir / ".complete",
        )
        report_dir = args.report_root / variant / f"crohme_{year}"
        run_stage(
            f"{variant} paired report: CROHME {year}",
            comparison_command(
                args,
                predictions=output_dir / "predictions.jsonl",
                baseline_predictions=args.output_root
                / "frozen"
                / f"crohme_{year}"
                / "predictions.jsonl",
                output_dir=report_dir,
            ),
            marker=report_dir / ".complete",
        )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.dataset_path = args.dataset_path.resolve()
    args.crohme_root = args.crohme_root.resolve()
    args.split_dir = args.split_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.corpus_root = args.corpus_root.resolve()
    args.report_root = args.report_root.resolve()
    args.state_path = args.output_root / "dynamic_error" / "experiment_state.json"
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Downloaded MathWriting dataset not found: {args.dataset_path}")
    missing_manifests = [path for path in crohme_manifests(args).values() if not path.exists()]
    if missing_manifests:
        raise FileNotFoundError(
            f"CROHME manifest not found: {missing_manifests[0]}. "
            "Run `dec-prepare-crohme --download` first."
        )
    args.cdm_evaluator = args.cdm_evaluator.resolve()
    if not args.cdm_evaluator.exists():
        raise FileNotFoundError(f"CDM evaluator not found: {args.cdm_evaluator}")

    prepare_fixed_split(args)
    run_frozen(args)
    run_frozen_reports(args)

    d0 = args.corpus_root / "D0"
    run_stage(
        "build static source corpus D0",
        corpus_command(
            args,
            cycle=0,
            predictions=args.output_root / "frozen" / "train_pool" / "predictions.jsonl",
            output_dir=d0,
        ),
        marker=d0 / ".complete",
    )

    dynamic = run_dynamic_error(args)
    recognition_adapter = run_recognition_lora(args)
    static_adapter = run_static_error(args, cycles=int(dynamic["best_cycle"]))
    dynamic_adapter = Path(dynamic["adapter"])

    evaluate_variant(args, variant="recognition_lora", adapter=recognition_adapter)
    evaluate_variant(args, variant="static_error", adapter=static_adapter)
    evaluate_variant(args, variant="dynamic_error", adapter=dynamic_adapter)
    comparison = aggregate_reports(args.report_root)

    summary = {
        "model": args.model,
        "dataset_path": str(args.dataset_path),
        "crohme_manifests": {
            str(year): str(path) for year, path in crohme_manifests(args).items()
        },
        "split_dir": str(args.split_dir),
        "corpus_root": str(args.corpus_root),
        "output_root": str(args.output_root),
        "report_root": str(args.report_root),
        "seed": args.seed,
        "variants": {
            "B0": {"key": "frozen", "adapter": None},
            "B1": {"key": "recognition_lora", "adapter": str(recognition_adapter)},
            "B2": {
                "key": "static_error",
                "cycles": dynamic["best_cycle"],
                "corpus": str(d0),
                "adapter": str(static_adapter),
            },
            "B3": {
                "key": "dynamic_error",
                "best_cycle": dynamic["best_cycle"],
                "last_cycle": dynamic["last_cycle"],
                "best_validation_value": dynamic["state"]["best_value"],
                "early_stop_metric": dynamic["state"]["metric"],
                "adapter": str(dynamic_adapter),
            },
        },
        "comparison": str(args.report_root / "comparison.json"),
        "comparison_models": len(comparison["models"]),
        "cdm_enabled": True,
        "cdm_docker_image": args.cdm_docker_image or "",
    }
    summary_path = args.output_root / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--crohme-root", type=Path, default=Path("data/crohme"))
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--split-dir", type=Path, default=Path("data/mathwriting/split"))
    parser.add_argument("--corpus-root", type=Path, default=Path("data/mathwriting/corpora"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/mathwriting"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/mathwriting"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--backend", choices=["transformers"], default="transformers")
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", default="net_fixed_count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--epochs-per-cycle", type=float, default=1.0)
    parser.add_argument("--recognition-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--generation-eval-batch-size", type=int, default=1)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--flush-csv-every", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--device-map")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--vision-lora-target-modules", default="qkv")
    parser.add_argument("--error-fraction", type=float, default=0.50)
    parser.add_argument("--keep-fraction", type=float, default=0.35)
    parser.add_argument("--replay-fraction-of-errors", type=float, default=0.20)
    parser.add_argument(
        "--error-task-mode",
        choices=["alternate", "detection", "correction"],
        default="alternate",
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--wandb-project", default="dynamic-error-corpus-unimumer")
    parser.add_argument("--cdm-evaluator", type=Path, required=True)
    parser.add_argument("--cdm-python")
    parser.add_argument("--cdm-docker-image")
    parser.add_argument("--cdm-pools", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_cycles < 1:
        raise ValueError("--max-cycles must be at least 1.")
    summary = run_experiment(args)
    print("\nExperiment complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
