"""Generation metrics for recognition and self-correction validation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .metrics import compute_metrics, corpus_bleu
from .normalize import clean_model_prediction, normalize_latex


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metric_text(text: str, *, clean: bool, normalize: bool) -> str:
    value = clean_model_prediction(text) if clean else (text or "").strip()
    return normalize_latex(value) if normalize else value


def correction_validation_metrics(
    rows: list[dict[str, Any]],
    *,
    clean_predictions: bool = True,
    normalize: bool = True,
) -> dict[str, Any]:
    """Aggregate recognition metrics and baseline-to-current transitions."""
    if not rows:
        return {
            "exprate": 0.0,
            "cer": 0.0,
            "bleu": 0.0,
            "raw_exprate": 0.0,
            "raw_cer": 0.0,
            "normalized_exprate": 0.0,
            "normalized_cer": 0.0,
            "baseline_examples": 0,
            "baseline_correct_count": 0,
            "baseline_wrong_count": 0,
            "fixed_count": 0,
            "spoiled_count": 0,
            "fixed_rate": 0.0,
            "spoiled_rate": 0.0,
            "net_fixed_count": 0,
        }

    references: list[str] = []
    predictions: list[str] = []
    raw_matches: list[float] = []
    raw_cers: list[float] = []
    normalized_matches: list[float] = []
    normalized_cers: list[float] = []
    baseline_examples = 0
    baseline_wrong = 0
    baseline_correct = 0
    fixed_count = 0
    spoiled_count = 0

    for row in rows:
        ground_truth = str(row.get("ground_truth") or "")
        prediction = metric_text(
            str(row.get("prediction") or row.get("prediction_for_eval") or ""),
            clean=clean_predictions,
            normalize=False,
        )
        metrics = compute_metrics(ground_truth, prediction)
        raw_matches.append(float(metrics.raw_exact_match))
        raw_cers.append(float(metrics.cer))
        normalized_matches.append(float(metrics.normalized_exact_match))
        normalized_cers.append(float(metrics.normalized_cer))
        references.append(metric_text(ground_truth, clean=False, normalize=normalize))
        predictions.append(metric_text(prediction, clean=False, normalize=normalize))

        baseline_value = row.get("baseline_prediction")
        if baseline_value is None:
            continue
        baseline_prediction = metric_text(
            str(baseline_value),
            clean=clean_predictions,
            normalize=False,
        )
        baseline_metrics = compute_metrics(ground_truth, baseline_prediction)
        baseline_is_correct = (
            baseline_metrics.normalized_exact_match if normalize else baseline_metrics.raw_exact_match
        )
        current_is_correct = metrics.normalized_exact_match if normalize else metrics.raw_exact_match
        baseline_examples += 1
        baseline_correct += int(baseline_is_correct)
        baseline_wrong += int(not baseline_is_correct)
        fixed_count += int(not baseline_is_correct and current_is_correct)
        spoiled_count += int(baseline_is_correct and not current_is_correct)

    selected_matches = normalized_matches if normalize else raw_matches
    selected_cers = normalized_cers if normalize else raw_cers
    return {
        "exprate": mean(selected_matches),
        "cer": mean(selected_cers),
        "bleu": corpus_bleu(references, predictions),
        "raw_exprate": mean(raw_matches),
        "raw_cer": mean(raw_cers),
        "normalized_exprate": mean(normalized_matches),
        "normalized_cer": mean(normalized_cers),
        "baseline_examples": baseline_examples,
        "baseline_correct_count": baseline_correct,
        "baseline_wrong_count": baseline_wrong,
        "fixed_count": fixed_count,
        "spoiled_count": spoiled_count,
        "fixed_rate": fixed_count / baseline_wrong if baseline_wrong else 0.0,
        "spoiled_rate": spoiled_count / baseline_correct if baseline_correct else 0.0,
        "net_fixed_count": fixed_count - spoiled_count,
    }


def run_official_cdm(
    rows: list[dict[str, Any]],
    *,
    evaluator: Path,
    output_dir: Path,
    pools: int = 1,
    python_executable: str | None = None,
    docker_image: str | None = None,
    timeout_sec: int = 0,
) -> dict[str, Any]:
    """Run the official UniMERNet CDM evaluator over generated validation rows."""
    evaluator = evaluator.resolve()
    if not evaluator.exists():
        raise FileNotFoundError(f"CDM evaluator not found: {evaluator}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "cdm_input.json"
    payload = [
        {
            "img_id": (
                f"{index:08d}_"
                + re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    str(row.get("id") or f"sample_{index}"),
                )[:120]
            ),
            "gt": str(row.get("ground_truth") or ""),
            "pred": str(row.get("prediction") or row.get("prediction_for_eval") or ""),
        }
        for index, row in enumerate(rows)
    ]
    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if docker_image:
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{evaluator.parent}:/code:ro",
            "-v",
            f"{output_dir.resolve()}:/work",
            "-w",
            "/code",
            docker_image,
            "python",
            "/code/evaluation.py",
            "--input",
            "/work/cdm_input.json",
            "--output",
            "/work",
            "--pools",
            str(max(1, pools)),
        ]
        command_cwd = None
    else:
        command = [
            python_executable or sys.executable,
            str(evaluator),
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--pools",
            str(max(1, pools)),
        ]
        command_cwd = evaluator.parent
    completed = subprocess.run(
        command,
        cwd=command_cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec if timeout_sec > 0 else None,
    )
    (output_dir / "cdm_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "cdm_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"CDM evaluator failed with exit code {completed.returncode}. "
            f"See {output_dir / 'cdm_stderr.log'}."
        )

    metrics_path = output_dir / input_path.stem / "metrics_res.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"CDM result was not created: {metrics_path}")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "cdm": float(result["mean_score"]),
        "cdm_exprate": float(result["exp_rate"]),
        "cdm_metrics_path": str(metrics_path),
    }
