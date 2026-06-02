"""Compare baseline and current recognition predictions, optionally with CDM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dec_unimumer.latex.metrics import compute_metrics
from dec_unimumer.latex.normalize import normalize_latex
from dec_unimumer.latex.validation_metrics import correction_validation_metrics, run_official_cdm


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(args.input)
        if str(row.get(args.status_field) or "ok") == "ok"
    ]
    baseline_by_id: dict[str, dict[str, Any]] = {}
    if args.baseline_input is not None:
        baseline_by_id = {
            str(row.get(args.baseline_input_id_field) or ""): row
            for row in read_jsonl(args.baseline_input)
            if str(row.get(args.baseline_input_id_field) or "").strip()
            and str(row.get(args.baseline_input_status_field) or "ok") == "ok"
        }

    metric_rows: list[dict[str, Any]] = []
    evaluated_rows: list[dict[str, Any]] = []
    missing_baseline: list[str] = []
    for row in rows:
        sample_id = str(row.get(args.id_field) or "")
        ground_truth = str(row.get(args.ground_truth_field) or "")
        if baseline_by_id:
            baseline_row = baseline_by_id.get(sample_id)
            if baseline_row is None:
                missing_baseline.append(sample_id)
                continue
            baseline = str(
                baseline_row.get(args.baseline_input_prediction_field) or ""
            )
            baseline_ground_truth = str(
                baseline_row.get(args.baseline_input_ground_truth_field) or ""
            )
            if (
                baseline_ground_truth
                and normalize_latex(baseline_ground_truth)
                != normalize_latex(ground_truth)
            ):
                raise ValueError(
                    f"Ground truth mismatch between runs for id {sample_id!r}."
                )
        else:
            baseline = str(row.get(args.baseline_field) or "")
        prediction = str(row.get(args.prediction_field) or "")
        baseline_metrics = compute_metrics(ground_truth, baseline)
        current_metrics = compute_metrics(ground_truth, prediction)
        baseline_correct = (
            baseline_metrics.normalized_exact_match
            if args.normalize
            else baseline_metrics.raw_exact_match
        )
        current_correct = (
            current_metrics.normalized_exact_match
            if args.normalize
            else current_metrics.raw_exact_match
        )
        metric_rows.append(
            {
                "id": sample_id,
                "ground_truth": ground_truth,
                "baseline_prediction": baseline,
                "prediction": prediction,
            }
        )
        evaluated_rows.append(
            {
                **row,
                "baseline_correct": bool(baseline_correct),
                "current_correct": bool(current_correct),
                "transition": (
                    "wrong_to_correct"
                    if not baseline_correct and current_correct
                    else "correct_to_wrong"
                    if baseline_correct and not current_correct
                    else "correct_to_correct"
                    if baseline_correct
                    else "wrong_to_wrong"
                ),
                "baseline_cer": (
                    baseline_metrics.normalized_cer
                    if args.normalize
                    else baseline_metrics.cer
                ),
                "current_cer": (
                    current_metrics.normalized_cer
                    if args.normalize
                    else current_metrics.cer
                ),
            }
        )

    if missing_baseline:
        raise ValueError(
            f"Baseline input does not cover {len(missing_baseline)} current ids; "
            f"examples: {missing_baseline[:5]}"
        )

    summary = {
        "input": str(args.input),
        "baseline_input": str(args.baseline_input or ""),
        "num_examples": len(metric_rows),
        **correction_validation_metrics(
            metric_rows,
            clean_predictions=True,
            normalize=args.normalize,
        ),
    }
    if args.cdm_evaluator:
        summary.update(
            run_official_cdm(
                metric_rows,
                evaluator=args.cdm_evaluator,
                output_dir=args.output_dir / "cdm",
                pools=args.cdm_pools,
                python_executable=args.cdm_python,
                docker_image=args.cdm_docker_image,
                timeout_sec=args.cdm_timeout_sec,
            )
        )
    else:
        summary["cdm"] = None
        summary["cdm_exprate"] = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "evaluated.jsonl").open("w", encoding="utf-8") as file:
        for row in evaluated_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--ground-truth-field", default="ground_truth")
    parser.add_argument("--baseline-field", default="baseline_prediction")
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--status-field", default="status")
    parser.add_argument("--baseline-input", type=Path)
    parser.add_argument("--baseline-input-id-field", default="id")
    parser.add_argument(
        "--baseline-input-prediction-field",
        default="prediction",
    )
    parser.add_argument(
        "--baseline-input-ground-truth-field",
        default="ground_truth",
    )
    parser.add_argument("--baseline-input-status-field", default="status")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cdm-evaluator", type=Path)
    parser.add_argument("--cdm-python")
    parser.add_argument("--cdm-docker-image")
    parser.add_argument("--cdm-pools", type=int, default=1)
    parser.add_argument("--cdm-timeout-sec", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
