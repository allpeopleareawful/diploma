"""Language-and-vision LoRA SFT for Uni-MuMER dynamic error corpora."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dec_unimumer.data import (
    DatasetSample,
    image_to_pil,
    iter_csv_samples,
    iter_hf_disk_samples,
    iter_hf_samples,
    iter_jsonl_samples,
)
from dec_unimumer.latex.normalize import clean_model_prediction
from dec_unimumer.latex.validation_metrics import (
    correction_validation_metrics,
    run_official_cdm,
)
from dec_unimumer.model_utils import resolve_model_ref, torch_dtype_from_name
from dec_unimumer.paths import PROJECT_ROOT
from dec_unimumer.prompts import RECOGNITION_PROMPT


DEFAULT_MODEL = "phxember/Uni-MuMER-Qwen2.5-VL-3B"
DEFAULT_PROMPT = RECOGNITION_PROMPT
DEFAULT_LANGUAGE_LORA_TARGETS = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)
DEFAULT_VISION_LORA_TARGETS = "qkv"


@dataclass(frozen=True)
class TrainSample:
    sample_id: str
    image: Any
    label: str
    prompt: str | None = None
    task: str | None = None
    baseline_prediction: str | None = None


def qwen_image_content(
    image: Any,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "type": "image",
        "image": image_to_pil(image).convert("RGB"),
    }
    if min_pixels and min_pixels > 0:
        content["min_pixels"] = min_pixels
    if max_pixels and max_pixels > 0:
        content["max_pixels"] = max_pixels
    return content


class ListDataset:
    def __init__(self, samples: list[TrainSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainSample:
        return self.samples[index]


def truncate_qwen_sequence_inputs(inputs: Any, max_length: int) -> Any:
    """Truncate every processor field aligned with the language sequence."""
    if max_length <= 0:
        return inputs

    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.shape[-1] <= max_length:
        return inputs

    sequence_length = input_ids.shape[-1]
    for key in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "mm_token_type_ids",
    ):
        value = inputs.get(key)
        if (
            value is not None
            and getattr(value, "ndim", 0) >= 2
            and value.shape[-1] == sequence_length
        ):
            inputs[key] = value[..., :max_length]
    return inputs


class QwenVLDataCollator:
    def __init__(
        self,
        *,
        processor: Any,
        prompt: str,
        max_length: int,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> None:
        self.processor = processor
        self.prompt = prompt
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = processor.tokenizer.eos_token_id
        self.processor.tokenizer.padding_side = "right"

    def _messages(self, image: Any, prompt: str, answer: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    qwen_image_content(
                        image,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    ),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
        return messages

    def _chat_texts(
        self,
        batch: list[TrainSample],
        *,
        add_answers: bool,
        add_generation_prompt: bool,
    ) -> tuple[list[str], list[list[dict[str, Any]]]]:
        messages_batch: list[list[dict[str, Any]]] = []
        texts: list[str] = []
        for sample in batch:
            image = image_to_pil(sample.image)
            prompt = sample.prompt or self.prompt
            answer = sample.label if add_answers else None
            messages = self._messages(image, prompt, answer)
            messages_batch.append(messages)
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        return texts, messages_batch

    def __call__(self, batch: list[TrainSample]) -> dict[str, Any]:
        import torch
        from qwen_vl_utils import process_vision_info

        full_texts, full_messages = self._chat_texts(
            batch,
            add_answers=True,
            add_generation_prompt=False,
        )
        prompt_texts, prompt_messages = self._chat_texts(
            batch,
            add_answers=False,
            add_generation_prompt=True,
        )

        full_images, full_videos = process_vision_info(full_messages)
        prompt_images, prompt_videos = process_vision_info(prompt_messages)

        full_inputs = self.processor(
            text=full_texts,
            images=full_images,
            videos=full_videos,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=prompt_images,
            videos=prompt_videos,
            padding=True,
            return_tensors="pt",
        )

        full_inputs = truncate_qwen_sequence_inputs(full_inputs, self.max_length)

        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=-1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            prompt_len = min(int(prompt_length), labels.shape[-1])
            labels[row_index, :prompt_len] = -100
        labels[full_inputs["attention_mask"] == 0] = -100
        labels[labels == self.pad_token_id] = -100

        full_inputs["labels"] = labels.to(dtype=torch.long)
        return dict(full_inputs)


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


def collect_samples(args: argparse.Namespace) -> list[TrainSample]:
    samples: list[TrainSample] = []
    for index, sample in enumerate(iter_samples(args)):
        if index < args.start:
            continue
        if args.limit is not None and len(samples) >= args.limit:
            break
        if not sample.ground_truth.strip():
            continue
        samples.append(
            TrainSample(
                sample_id=sample.sample_id,
                image=sample.image,
                label=sample.ground_truth.strip(),
                prompt=sample.prompt.strip() if sample.prompt and sample.prompt.strip() else None,
                task=sample.task,
            )
        )
    if not samples:
        raise ValueError("No training samples were collected.")
    return samples


def collect_explicit_validation_samples(args: argparse.Namespace) -> list[TrainSample] | None:
    if (
        args.validation_metadata is None
        and args.validation_dataset_path is None
        and args.validation_dataset_id is None
    ):
        return None
    if args.validation_source == "hf" and args.validation_dataset_id is None:
        raise ValueError("--validation-dataset-id is required with --validation-source hf.")
    if args.validation_source == "hf_disk" and args.validation_dataset_path is None:
        raise ValueError("--validation-dataset-path is required with --validation-source hf_disk.")
    if args.validation_source in {"csv", "jsonl"} and args.validation_metadata is None:
        raise ValueError(
            f"--validation-metadata is required with --validation-source {args.validation_source}."
        )
    validation_args = argparse.Namespace(**vars(args))
    validation_args.source = args.validation_source
    validation_args.dataset_id = args.validation_dataset_id
    validation_args.dataset_path = args.validation_dataset_path
    validation_args.metadata = args.validation_metadata
    validation_args.root_dir = args.validation_root_dir
    validation_args.split = args.validation_split
    validation_args.image_field = args.validation_image_field
    validation_args.label_field = args.validation_label_field
    validation_args.id_field = args.validation_id_field
    validation_args.prompt_field = args.validation_prompt_field
    validation_args.conversation_field = args.validation_conversation_field
    validation_args.task_field = args.validation_task_field
    validation_args.start = args.validation_start
    validation_args.limit = args.validation_limit
    return collect_samples(validation_args)


def read_prediction_map(
    path: Path | None,
    *,
    id_field: str,
    prediction_field: str,
) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        raise ValueError(f"Unsupported baseline prediction format: {path.suffix}")
    return {
        str(row[id_field]): str(row.get(prediction_field) or "")
        for row in rows
        if row.get(id_field) not in (None, "")
    }


def attach_baseline_predictions(
    samples: list[TrainSample],
    predictions: dict[str, str],
) -> list[TrainSample]:
    if not predictions:
        return samples
    return [
        TrainSample(
            sample_id=sample.sample_id,
            image=sample.image,
            label=sample.label,
            prompt=sample.prompt,
            task=sample.task,
            baseline_prediction=predictions.get(sample.sample_id),
        )
        for sample in samples
    ]


def split_train_validation(
    samples: list[TrainSample],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[TrainSample], list[TrainSample]]:
    if validation_ratio <= 0:
        return samples, []
    if validation_ratio >= 1:
        raise ValueError("--validation-ratio must be < 1.0")
    if len(samples) < 2:
        return samples, []

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(samples) * validation_ratio))
    validation_count = min(validation_count, len(samples) - 1)
    validation_indices = set(indices[:validation_count])
    train_samples = [sample for index, sample in enumerate(samples) if index not in validation_indices]
    validation_samples = [sample for index, sample in enumerate(samples) if index in validation_indices]
    return train_samples, validation_samples


def comma_separated_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def merged_lora_target_modules(args: argparse.Namespace) -> str | list[str]:
    if args.lora_target_modules.strip() == "all-linear":
        return "all-linear"
    targets = comma_separated_names(args.lora_target_modules)
    if args.vision_lora:
        targets.extend(comma_separated_names(args.vision_lora_target_modules))
    return list(dict.fromkeys(targets))


def configure_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=merged_lora_target_modules(args),
    )
    return get_peft_model(model, config)


def is_vision_parameter_name(name: str) -> bool:
    lowered = name.lower()
    return "visual" in lowered or "vision" in lowered


def apply_vision_lora_policy(model: Any, *, enabled: bool) -> int:
    trainable_vision = 0
    for name, parameter in model.named_parameters():
        if not is_vision_parameter_name(name):
            continue
        if not enabled:
            parameter.requires_grad = False
        if parameter.requires_grad:
            trainable_vision += parameter.numel()
    if enabled and trainable_vision == 0:
        raise RuntimeError(
            "Vision LoRA is enabled, but no visual parameters are trainable. "
            "Start a fresh adapter with --vision-lora and visual target modules such as qkv."
        )
    return trainable_vision


def trainable_parameter_summary(model: Any) -> dict[str, int | float]:
    trainable = 0
    trainable_vision = 0
    trainable_text = 0
    total = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            if is_vision_parameter_name(name):
                trainable_vision += count
            else:
                trainable_text += count
    return {
        "trainable": trainable,
        "trainable_vision": trainable_vision,
        "trainable_text": trainable_text,
        "total": total,
        "trainable_percent": round(100 * trainable / total, 4) if total else 0.0,
    }


def save_training_manifest(samples: list[TrainSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(
                json.dumps(
                    {
                        "id": sample.sample_id,
                        "label": sample.label,
                        "prompt": sample.prompt,
                        "task": sample.task,
                        "baseline_prediction": sample.baseline_prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def build_training_arguments(training_arguments_class: Any, kwargs: dict[str, Any]) -> Any:
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    parameters = inspect.signature(training_arguments_class.__init__).parameters
    if "eval_strategy" not in parameters and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return training_arguments_class(**kwargs)


def limited_validation_samples(samples: list[TrainSample], limit: int) -> list[TrainSample]:
    if limit <= 0 or limit >= len(samples):
        return samples
    return samples[:limit]


def estimate_optimizer_steps(
    *,
    num_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    world_size: int = 1,
) -> int:
    if min(num_samples, batch_size, gradient_accumulation_steps, world_size) < 1:
        return 0
    micro_batches_per_epoch = math.ceil(num_samples / (batch_size * world_size))
    updates_per_epoch = math.ceil(
        micro_batches_per_epoch / gradient_accumulation_steps
    )
    return math.ceil(updates_per_epoch * num_train_epochs)


def generate_qwen_batch(
    *,
    model: Any,
    processor: Any,
    images: list[Any],
    prompts: list[str],
    max_new_tokens: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> list[str]:
    from qwen_vl_utils import process_vision_info

    messages_batch: list[list[dict[str, Any]]] = []
    texts: list[str] = []
    for image, prompt in zip(images, prompts, strict=True):
        messages = [
            {
                "role": "user",
                "content": [
                    qwen_image_content(
                        image,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    ),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        messages_batch.append(messages)
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    prompt_width = inputs["input_ids"].shape[-1]
    generated_trimmed = [output_ids[prompt_width:] for output_ids in generated_ids]
    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def run_generation_validation(
    *,
    model: Any,
    processor: Any,
    samples: list[TrainSample],
    prompt: str,
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    min_pixels: int | None,
    max_pixels: int | None,
    clean_predictions: bool,
    normalize: bool,
    cdm_evaluator: Path | None,
    cdm_python: str | None,
    cdm_docker_image: str | None,
    cdm_pools: int,
    cdm_timeout_sec: int,
    global_step: int,
) -> dict[str, Any]:
    import torch

    if batch_size < 1:
        raise ValueError("Generation validation batch size must be >= 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    was_training = bool(getattr(model, "training", False))
    model.eval()
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start : batch_start + batch_size]
            images = [sample.image for sample in batch]
            raw_predictions = generate_qwen_batch(
                model=model,
                processor=processor,
                images=images,
                prompts=[sample.prompt or prompt for sample in batch],
                max_new_tokens=max_new_tokens,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            predictions = [
                clean_model_prediction(output)
                if clean_predictions
                else output.strip()
                for output in raw_predictions
            ]

            for sample, raw_prediction, prediction in zip(
                batch,
                raw_predictions,
                predictions,
                strict=True,
            ):
                rows.append(
                    {
                        "id": sample.sample_id,
                        "task": sample.task or "",
                        "validation_mode": "recognition",
                        "ground_truth": sample.label,
                        "baseline_prediction": sample.baseline_prediction,
                        "raw_prediction": raw_prediction,
                        "prediction": prediction,
                    }
                )

    if was_training:
        model.train()

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "global_step": global_step,
        "num_examples": len(rows),
        "validation_mode": "recognition",
        "metric_mode": "normalized" if normalize else "raw",
        "elapsed_sec": round(time.perf_counter() - started, 4),
        **correction_validation_metrics(
            rows,
            clean_predictions=clean_predictions,
            normalize=normalize,
        ),
    }
    if cdm_evaluator is not None:
        summary.update(
            run_official_cdm(
                rows,
                evaluator=cdm_evaluator,
                output_dir=output_dir / "cdm",
                pools=cdm_pools,
                python_executable=cdm_python,
                docker_image=cdm_docker_image,
                timeout_sec=cdm_timeout_sec,
            )
        )
    else:
        summary["cdm"] = None
        summary["cdm_exprate"] = None

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


class GenerationValidationCallback:
    def __init__(
        self,
        *,
        processor: Any,
        samples: list[TrainSample],
        prompt: str,
        output_dir: Path,
        batch_size: int,
        max_new_tokens: int,
        min_pixels: int | None,
        max_pixels: int | None,
        clean_predictions: bool,
        normalize: bool,
        cdm_evaluator: Path | None,
        cdm_python: str | None,
        cdm_docker_image: str | None,
        cdm_pools: int,
        cdm_timeout_sec: int,
    ) -> None:
        self.processor = processor
        self.samples = samples
        self.prompt = prompt
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.clean_predictions = clean_predictions
        self.normalize = normalize
        self.cdm_evaluator = cdm_evaluator
        self.cdm_python = cdm_python
        self.cdm_docker_image = cdm_docker_image
        self.cdm_pools = cdm_pools
        self.cdm_timeout_sec = cdm_timeout_sec
        self.last_step = -1

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(getattr(state, "global_step", 0) or 0)
        if not self.samples or step == self.last_step:
            return control
        self.last_step = step
        summary = run_generation_validation(
            model=kwargs["model"],
            processor=self.processor,
            samples=self.samples,
            prompt=self.prompt,
            output_dir=self.output_dir / f"step-{step}",
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            clean_predictions=self.clean_predictions,
            normalize=self.normalize,
            cdm_evaluator=self.cdm_evaluator,
            cdm_python=self.cdm_python,
            cdm_docker_image=self.cdm_docker_image,
            cdm_pools=self.cdm_pools,
            cdm_timeout_sec=self.cdm_timeout_sec,
            global_step=step,
        )
        logged = {
            f"eval_generation/{key}": value
            for key, value in summary.items()
            if key
            in {
                "exprate",
                "cdm",
                "cdm_exprate",
                "cer",
                "bleu",
                "fixed_count",
                "spoiled_count",
                "fixed_rate",
                "spoiled_rate",
                "net_fixed_count",
                "baseline_examples",
            }
            and value is not None
            and (
                key
                not in {
                    "fixed_count",
                    "spoiled_count",
                    "fixed_rate",
                    "spoiled_rate",
                    "net_fixed_count",
                }
                or int(summary.get("baseline_examples") or 0) > 0
            )
        }
        metrics = kwargs.get("metrics")
        if isinstance(metrics, dict):
            metrics.update(logged)
        print(f"[validation step={step}] {json.dumps(logged, ensure_ascii=False)}", flush=True)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(logged, step=step)
        except Exception:
            pass
        return control


def parse_report_to(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "no", "false", "0"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_tracking(args: argparse.Namespace) -> list[str]:
    report_to = parse_report_to(args.report_to)
    if "wandb" not in {item.lower() for item in report_to}:
        return report_to

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_mode:
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    if args.wandb_tags:
        os.environ.setdefault("WANDB_TAGS", args.wandb_tags)
    if args.wandb_notes:
        os.environ.setdefault("WANDB_NOTES", args.wandb_notes)
    return report_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["hf", "hf_disk", "csv", "jsonl"], default="hf_disk")
    parser.add_argument("--dataset-id", default="phxember/Uni-MuMER-Data")
    parser.add_argument("--dataset-path", type=Path)
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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Existing LoRA adapter to continue training from the previous cycle.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional Transformers device_map. Use auto for quantized/large models.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the frozen base model with bitsandbytes NF4 quantization for QLoRA.",
    )
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=0,
        help="Qwen-VL min_pixels for image preprocessing. 0 keeps the processor default.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="Qwen-VL max_pixels for image preprocessing. 0 keeps the processor default.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)

    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "unimumer_lora_edl")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive value overrides --num-train-epochs. Default: train by epochs.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-ratio", type=float, default=0.0)
    parser.add_argument(
        "--validation-source",
        choices=["hf", "hf_disk", "csv", "jsonl"],
        default="jsonl",
        help="Source format for a fixed recognition validation manifest.",
    )
    parser.add_argument("--validation-dataset-id")
    parser.add_argument("--validation-dataset-path", type=Path)
    parser.add_argument(
        "--validation-metadata",
        type=Path,
        help="Fixed recognition validation manifest, kept separate from the training error corpus.",
    )
    parser.add_argument("--validation-root-dir", type=Path)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--validation-image-field")
    parser.add_argument("--validation-label-field")
    parser.add_argument("--validation-id-field")
    parser.add_argument("--validation-prompt-field")
    parser.add_argument("--validation-conversation-field")
    parser.add_argument("--validation-task-field", default="task")
    parser.add_argument("--validation-start", type=int, default=0)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument(
        "--validation-baseline-predictions",
        type=Path,
        help="Baseline JSONL/CSV for the same validation ids; required for fixed/spoiled counts.",
    )
    parser.add_argument("--validation-baseline-id-field", default="id")
    parser.add_argument("--validation-baseline-prediction-field", default="prediction")
    parser.add_argument("--generation-eval-limit", type=int, default=0)
    parser.add_argument("--generation-eval-batch-size", type=int, default=1)
    parser.add_argument("--generation-eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-clean-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cdm-evaluator",
        type=Path,
        help="Path to the official UniMERNet cdm/evaluation.py.",
    )
    parser.add_argument("--cdm-python")
    parser.add_argument("--cdm-docker-image")
    parser.add_argument("--cdm-pools", type=int, default=1)
    parser.add_argument("--cdm-timeout-sec", type=int, default=0)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--wandb-project", default="dynamic-error-corpus-unimumer")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-notes")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=DEFAULT_LANGUAGE_LORA_TARGETS,
    )
    parser.add_argument(
        "--vision-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train LoRA adapters in the visual encoder as well as the language model.",
    )
    parser.add_argument(
        "--vision-lora-target-modules",
        default=DEFAULT_VISION_LORA_TARGETS,
        help=(
            "Additional visual module suffixes. Qwen2.5-VL uses qkv in visual "
            "attention; gate/up/down projections are already shared with language targets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_ref = resolve_model_ref(str(args.model), DEFAULT_MODEL)
    processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=args.trust_remote_code)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype_from_name(args.dtype),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    elif args.load_in_4bit:
        model_kwargs["device_map"] = "auto"
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_ref, **model_kwargs)
    if args.device != "auto" and not args.load_in_4bit and not args.device_map:
        model.to(args.device)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = configure_lora(model, args)
    apply_vision_lora_policy(model, enabled=args.vision_lora)
    parameter_summary = trainable_parameter_summary(model)
    print(
        f"LoRA targets: {merged_lora_target_modules(args)}; "
        f"vision_lora={args.vision_lora}",
        flush=True,
    )
    print(f"Trainable parameters: {parameter_summary}", flush=True)

    samples = collect_samples(args)
    explicit_validation_samples = collect_explicit_validation_samples(args)
    if explicit_validation_samples is None:
        train_samples, validation_samples = split_train_validation(
            samples,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
    else:
        train_samples = samples
        validation_samples = explicit_validation_samples
    baseline_predictions = read_prediction_map(
        args.validation_baseline_predictions,
        id_field=args.validation_baseline_id_field,
        prediction_field=args.validation_baseline_prediction_field,
    )
    validation_samples = attach_baseline_predictions(validation_samples, baseline_predictions)
    generation_validation_samples = limited_validation_samples(
        validation_samples,
        args.generation_eval_limit,
    )
    save_training_manifest(train_samples, args.output_dir / "train_samples.jsonl")
    if validation_samples:
        save_training_manifest(validation_samples, args.output_dir / "validation_samples.jsonl")
    baseline_coverage = sum(
        sample.baseline_prediction is not None for sample in generation_validation_samples
    )
    if (
        args.validation_baseline_predictions is not None
        and baseline_coverage != len(generation_validation_samples)
    ):
        raise ValueError(
            "Baseline prediction ids do not fully cover generation validation: "
            f"{baseline_coverage}/{len(generation_validation_samples)} matched. "
            "Use the exact same validation manifest for baseline inference and training validation."
        )
    print(
        f"Dataset split: train={len(train_samples)} validation={len(validation_samples)} "
        f"generation_validation={len(generation_validation_samples)} "
        f"baseline_coverage={baseline_coverage}",
        flush=True,
    )
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    estimated_optimizer_steps = (
        args.max_steps
        if args.max_steps > 0
        else estimate_optimizer_steps(
            num_samples=len(train_samples),
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            world_size=world_size,
        )
    )
    print(
        "Cycle training schedule: "
        f"samples={len(train_samples)} epochs={args.num_train_epochs} "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps * world_size} "
        f"estimated_optimizer_steps={estimated_optimizer_steps}",
        flush=True,
    )

    train_dataset = ListDataset(train_samples)
    eval_dataset = ListDataset(validation_samples) if validation_samples else None
    collator = QwenVLDataCollator(
        processor=processor,
        prompt=args.prompt,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    report_to = configure_tracking(args)
    do_eval = eval_dataset is not None and args.eval_steps > 0
    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "trainer"),
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "no" if args.save_steps <= 0 else "steps",
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "steps" if do_eval else "no",
        "eval_steps": args.eval_steps if do_eval else None,
        "prediction_loss_only": True,
        "report_to": report_to,
        "run_name": (args.wandb_run_name or args.output_dir.name) if report_to else None,
        "remove_unused_columns": False,
        "fp16": args.dtype in ("auto", "float16") and torch.cuda.is_available(),
        "bf16": args.dtype == "bfloat16" and torch.cuda.is_available(),
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": 0,
        "seed": args.seed,
    }
    training_args = build_training_arguments(TrainingArguments, training_args_kwargs)

    class TrainerGenerationValidationCallback(GenerationValidationCallback, TrainerCallback):
        pass

    callbacks: list[TrainerCallback] = []
    if do_eval and generation_validation_samples:
        callbacks.append(
            TrainerGenerationValidationCallback(
                processor=processor,
                samples=generation_validation_samples,
                prompt=args.prompt,
                output_dir=args.output_dir / "validation",
                batch_size=args.generation_eval_batch_size,
                max_new_tokens=args.generation_eval_max_new_tokens,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                clean_predictions=args.eval_clean_predictions,
                normalize=args.eval_normalize,
                cdm_evaluator=args.cdm_evaluator,
                cdm_python=args.cdm_python,
                cdm_docker_image=args.cdm_docker_image,
                cdm_pools=args.cdm_pools,
                cdm_timeout_sec=args.cdm_timeout_sec,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    train_output = trainer.train()

    final_validation_metrics: dict[str, Any] | None = None
    if args.eval_at_end and generation_validation_samples:
        final_validation_metrics = run_generation_validation(
            model=model,
            processor=processor,
            samples=generation_validation_samples,
            prompt=args.prompt,
            output_dir=args.output_dir / "validation" / "final",
            batch_size=args.generation_eval_batch_size,
            max_new_tokens=args.generation_eval_max_new_tokens,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            clean_predictions=args.eval_clean_predictions,
            normalize=args.eval_normalize,
            cdm_evaluator=args.cdm_evaluator,
            cdm_python=args.cdm_python,
            cdm_docker_image=args.cdm_docker_image,
            cdm_pools=args.cdm_pools,
            cdm_timeout_sec=args.cdm_timeout_sec,
            global_step=int(getattr(trainer.state, "global_step", args.max_steps)),
        )

    model.save_pretrained(args.output_dir / "adapter")
    processor.save_pretrained(args.output_dir / "processor")

    metrics = {
        **train_output.metrics,
        "model": model_ref,
        "num_samples": len(samples),
        "num_train_samples": len(train_samples),
        "num_validation_samples": len(validation_samples),
        "estimated_optimizer_steps": estimated_optimizer_steps,
        "actual_optimizer_steps": int(getattr(trainer.state, "global_step", 0)),
        "final_validation": final_validation_metrics,
        "trainable_parameters": parameter_summary,
        "args": json_safe(vars(args)),
    }
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
