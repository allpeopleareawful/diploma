"""Aggregate per-benchmark reports into the project comparison tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from dec_unimumer.crohme import CROHME_YEARS

VARIANTS = (
    ("B0", "frozen", "Frozen Uni-MuMER"),
    ("B1", "recognition_lora", "Vanilla LoRA"),
    ("B2", "static_error", "Static EDL"),
    ("B3", "dynamic_error", "Dynamic EDL"),
)


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _percent(value: float | None) -> float | None:
    return round(value * 100.0, 4) if value is not None else None


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_reports(
    report_root: Path,
    *,
    output_dir: Path | None = None,
    variants: tuple[tuple[str, str, str], ...] = VARIANTS,
    years: tuple[int, ...] = CROHME_YEARS,
) -> dict[str, Any]:
    output_dir = output_dir or report_root
    models: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for model_id, key, label in variants:
        reports = {
            year: load_report(report_root / key / f"crohme_{year}" / "summary.json")
            for year in years
        }
        benchmarks = {
            str(year): {
                metric: reports[year].get(metric)
                for metric in ("num_examples", "exprate", "cdm", "cer", "bleu")
            }
            for year in years
        }
        averages = {
            metric: _mean(reports[year].get(metric) for year in years)
            for metric in ("exprate", "cdm", "cer", "bleu")
        }
        if key == "frozen":
            fixed_count = spoiled_count = net_fixed_count = None
        else:
            fixed_count = sum(int(reports[year].get("fixed_count") or 0) for year in years)
            spoiled_count = sum(int(reports[year].get("spoiled_count") or 0) for year in years)
            net_fixed_count = fixed_count - spoiled_count
        model = {
            "model_id": model_id,
            "key": key,
            "label": label,
            "benchmarks": benchmarks,
            "average": averages,
            "fixed_count": fixed_count,
            "spoiled_count": spoiled_count,
            "net_fixed_count": net_fixed_count,
        }
        models.append(model)

        csv_row: dict[str, Any] = {
            "model_id": model_id,
            "model": label,
            "average_exprate_percent": _percent(averages["exprate"]),
            "average_cdm_percent": _percent(averages["cdm"]),
            "average_cer_percent": _percent(averages["cer"]),
            "average_bleu_percent": _percent(averages["bleu"]),
            "fixed": fixed_count,
            "spoiled": spoiled_count,
            "net_fixed": net_fixed_count,
        }
        for year in years:
            for metric in ("exprate", "cdm", "cer", "bleu"):
                csv_row[f"crohme_{year}_{metric}_percent"] = _percent(
                    reports[year].get(metric)
                )
        csv_rows.append(csv_row)

    summary = {
        "benchmarks": [f"CROHME {year}" for year in years],
        "rate_scale": "JSON rates are in [0, 1]; CSV percentage columns are in [0, 100].",
        "models": models,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if csv_rows:
        with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    return summary
