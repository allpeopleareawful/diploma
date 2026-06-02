"""Run batched Uni-MuMER inference and save restartable prediction files.

Typical first run:

    dec-infer \
      --source hf_disk \
      --dataset-path data/mathwriting/split/dataset \
      --split validation \
      --backend transformers \
      --model phxember/Uni-MuMER-Qwen2.5-VL-3B

The script writes JSONL incrementally, so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import re
from pathlib import Path
from typing import Any, Iterable

from dec_unimumer.data import (
    DatasetSample,
    append_jsonl,
    image_to_pil,
    iter_csv_samples,
    iter_hf_disk_samples,
    iter_hf_samples,
    iter_jsonl_samples,
)
from dec_unimumer.latex.metrics import compute_metrics
from dec_unimumer.latex.normalize import normalize_latex, strip_model_wrappers
from dec_unimumer.latex.risk import compute_risk
from dec_unimumer.paths import PROJECT_ROOT
from dec_unimumer.backends import (
    DEFAULT_PROMPT,
    GenerationResult,
    build_backend,
)


DEFAULT_MODEL = "phxember/Uni-MuMER-Qwen2.5-VL-3B"


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line).get("id") or ""))
            except json.JSONDecodeError:
                continue
    completed.discard("")
    return completed


def training_quality_flag(row: dict[str, Any]) -> bool:
    if row["status"] != "ok":
        return False
    if not row["prediction_normalized"]:
        return False
    if not bool(row["valid_latex"]):
        return False
    if float(row["risk_score"]) >= 45:
        return False
    if row.get("normalized_cer") not in ("", None) and float(row["normalized_cer"]) > 0.35:
        return False
    return True


def safe_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return value[:180] or "sample"


def row_for_sample(
    *,
    sample: DatasetSample,
    prediction: str,
    elapsed_sec: float,
    status: str,
    error: str = "",
    saved_image_path: str = "",
    avg_logprob: float | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    cleaned_prediction = strip_model_wrappers(prediction)
    normalized_prediction = normalize_latex(cleaned_prediction)
    normalized_gt = normalize_latex(sample.ground_truth)
    risk = compute_risk(cleaned_prediction)

    metrics = None
    if sample.ground_truth:
        metrics = compute_metrics(sample.ground_truth, cleaned_prediction)

    row: dict[str, Any] = {
        "id": sample.sample_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "source": sample.source,
        "image_path": saved_image_path,
        "ground_truth": sample.ground_truth,
        "ground_truth_normalized": normalized_gt,
        "model_id": "",
        "adapter_path": "",
        "prediction": cleaned_prediction,
        "prediction_normalized": normalized_prediction,
        "raw_output": prediction,
        "status": status,
        "error": error,
        "elapsed_sec": round(elapsed_sec, 4),
        "avg_logprob": "" if avg_logprob is None else round(avg_logprob, 6),
        "confidence": "" if confidence is None else round(confidence, 6),
        "valid_latex": risk.validation.valid,
        "risk_score": risk.score,
        "risk_level": risk.level,
        "risk_reasons": "; ".join(risk.reasons),
        "num_tokens": risk.validation.num_tokens,
        "num_commands": risk.validation.num_commands,
        "num_frac": risk.validation.num_frac,
        "num_sqrt": risk.validation.num_sqrt,
        "num_subscripts": risk.validation.num_subscripts,
        "num_superscripts": risk.validation.num_superscripts,
        "max_brace_depth": risk.validation.max_brace_depth,
        "unknown_commands": " ".join(risk.validation.unknown_commands),
        "raw_exact_match": "",
        "normalized_exact_match": "",
        "cer": "",
        "normalized_cer": "",
    }

    if metrics is not None:
        row.update(
            {
                "raw_exact_match": metrics.raw_exact_match,
                "normalized_exact_match": metrics.normalized_exact_match,
                "cer": round(metrics.cer, 6),
                "normalized_cer": round(metrics.normalized_cer, 6),
            }
        )

    row["accepted_for_training"] = training_quality_flag(row)
    return row


def save_sample_image(sample: DatasetSample, output_dir: Path) -> tuple[Any, str]:
    image = image_to_pil(sample.image)
    image_dir = output_dir / "images" / sample.split
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{safe_filename(sample.sample_id)}.png"
    image.save(image_path)
    return image, str(image_path.relative_to(output_dir)).replace("\\", "/")


def iter_samples(args: argparse.Namespace) -> Iterable[DatasetSample]:
    if args.source == "hf":
        return iter_hf_samples(
            dataset_id=args.dataset_id,
            split=args.split,
            image_field=args.image_field,
            label_field=args.label_field,
            id_field=args.id_field,
            prompt_field=args.prompt_field,
            conversation_field=args.conversation_field,
            task_field=args.task_field,
            streaming=not args.no_streaming,
            trust_remote_code=args.trust_remote_code,
        )

    if args.source == "hf_disk":
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required with --source hf_disk")
        return iter_hf_disk_samples(
            dataset_path=args.dataset_path,
            split=args.split,
            image_field=args.image_field,
            label_field=args.label_field,
            id_field=args.id_field,
            prompt_field=args.prompt_field,
            conversation_field=args.conversation_field,
            task_field=args.task_field,
        )

    if args.source == "csv":
        if args.metadata is None:
            raise ValueError("--metadata is required with --source csv")
        return iter_csv_samples(
            metadata_path=args.metadata,
            root_dir=args.root_dir,
            image_field=args.image_field,
            label_field=args.label_field,
            id_field=args.id_field,
            prompt_field=args.prompt_field,
            conversation_field=args.conversation_field,
            task_field=args.task_field,
            split=args.split if args.split else None,
            dataset_name=args.dataset_name or args.metadata.parent.name,
        )

    if args.source == "jsonl":
        if args.metadata is None:
            raise ValueError("--metadata is required with --source jsonl")
        return iter_jsonl_samples(
            metadata_path=args.metadata,
            root_dir=args.root_dir,
            image_field=args.image_field,
            label_field=args.label_field,
            id_field=args.id_field,
            prompt_field=args.prompt_field,
            conversation_field=args.conversation_field,
            task_field=args.task_field,
            split=args.split if args.split else None,
            dataset_name=args.dataset_name or args.metadata.parent.name,
        )

    raise ValueError(f"Unsupported source: {args.source}")


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf", "hf_disk", "csv", "jsonl"], default="hf_disk")
    parser.add_argument("--dataset-id")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/raw/unimumer_mathwriting_hf"),
    )
    parser.add_argument("--dataset-name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--root-dir", type=Path)
    parser.add_argument("--image-field")
    parser.add_argument("--label-field")
    parser.add_argument("--id-field")
    parser.add_argument("--prompt-field")
    parser.add_argument("--conversation-field")
    parser.add_argument("--task-field", default="task")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")

    parser.add_argument("--backend", choices=["vllm", "transformers", "mock"], default="transformers")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=4,
        help="Images per generation call. Failed batches are split automatically.",
    )

    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "unimumer_predictions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--flush-csv-every", type=int, default=100)
    return parser.parse_args()


def predict_batch_with_fallback(
    backend: Any,
    images: list[Any],
    prompts: list[str],
) -> list[GenerationResult | Exception]:
    """Predict a batch and recursively isolate OOM or per-example failures."""
    if not images:
        return []
    try:
        batch_method = getattr(backend, "predict_batch_with_metadata", None)
        if batch_method is None:
            return [
                backend.predict_with_metadata(image, prompt)
                for image, prompt in zip(images, prompts, strict=True)
            ]
        predictions = batch_method(images, prompts)
        if len(predictions) != len(images):
            raise RuntimeError(
                f"Backend returned {len(predictions)} predictions for {len(images)} images."
            )
        return list(predictions)
    except Exception as exc:
        if len(images) == 1:
            return [exc]
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        middle = len(images) // 2
        return [
            *predict_batch_with_fallback(backend, images[:middle], prompts[:middle]),
            *predict_batch_with_fallback(backend, images[middle:], prompts[middle:]),
        ]


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.inference_batch_size < 1:
        raise ValueError("--inference-batch-size must be at least 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.output_dir / "predictions.jsonl"
    csv_path = args.output_dir / "predictions.csv"
    if not args.resume:
        jsonl_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
    completed = read_completed_ids(jsonl_path) if args.resume else set()

    backend = build_backend(
        args.backend,
        model_name=args.model,
        adapter=args.adapter,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    processed = 0
    skipped = 0
    pending: list[tuple[DatasetSample, Any, str]] = []

    def write_row(row: dict[str, Any]) -> None:
        nonlocal processed
        row["model_id"] = args.model
        row["adapter_path"] = args.adapter or ""
        append_jsonl(jsonl_path, row)
        processed += 1
        if args.flush_csv_every > 0 and processed % args.flush_csv_every == 0:
            write_csv_from_jsonl(jsonl_path, csv_path)
            print(
                f"processed={processed} skipped={skipped} last_id={row['id']}"
            )

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        batch = pending
        pending = []
        started = time.perf_counter()
        results = predict_batch_with_fallback(
            backend,
            [image for _sample, image, _path in batch],
            [sample.prompt or args.prompt for sample, _image, _path in batch],
        )
        elapsed_per_sample = (time.perf_counter() - started) / len(batch)
        for (sample, _image, saved_image_path), result in zip(
            batch,
            results,
            strict=True,
        ):
            if isinstance(result, Exception):
                row = row_for_sample(
                    sample=sample,
                    prediction="",
                    elapsed_sec=elapsed_per_sample,
                    status="error",
                    error=repr(result),
                    saved_image_path=saved_image_path,
                )
            else:
                row = row_for_sample(
                    sample=sample,
                    prediction=result.text,
                    elapsed_sec=elapsed_per_sample,
                    status="ok",
                    saved_image_path=saved_image_path,
                    avg_logprob=result.avg_logprob,
                    confidence=result.confidence,
                )
            write_row(row)

    for index, sample in enumerate(iter_samples(args)):
        if index < args.start:
            continue
        if args.limit is not None and processed + skipped + len(pending) >= args.limit:
            break
        if sample.sample_id in completed:
            skipped += 1
            continue

        saved_image_path = ""
        try:
            image = image_to_pil(sample.image)
            if args.save_images:
                image, saved_image_path = save_sample_image(sample, args.output_dir)
            pending.append((sample, image, saved_image_path))
            if len(pending) >= args.inference_batch_size:
                flush_pending()
        except Exception as exc:
            flush_pending()
            row = row_for_sample(
                sample=sample,
                prediction="",
                elapsed_sec=0.0,
                status="error",
                error=repr(exc),
                saved_image_path=saved_image_path,
            )
            write_row(row)

    flush_pending()
    write_csv_from_jsonl(jsonl_path, csv_path)
    print(f"Done. processed={processed} skipped={skipped}")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")
    return jsonl_path, csv_path


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
