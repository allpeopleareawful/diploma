from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image


IMAGE_FIELDS = ("image", "img", "picture", "formula_image", "im")
LABEL_FIELDS = ("latex", "formula", "text", "label", "label_selected", "ground_truth", "caption")
ID_FIELDS = ("id", "image_id", "img_id", "name", "filename", "file_name")
PROMPT_FIELDS = ("prompt", "question", "instruction", "query")
CONVERSATION_FIELDS = ("conversations", "messages")
TASK_FIELDS = ("task", "source_config", "label_source")


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image: Any
    ground_truth: str
    split: str
    dataset: str
    source: str
    prompt: str | None = None
    task: str | None = None


@dataclass(frozen=True, slots=True)
class HFDatasetImageRef:
    dataset: Any
    index: int
    image_field: str


def infer_field(columns: Iterable[str], candidates: tuple[str, ...], kind: str) -> str:
    columns_list = list(columns)
    lower_to_original = {column.lower(): column for column in columns_list}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise ValueError(
        f"Cannot infer {kind} field. Available columns: {', '.join(columns_list)}. "
        f"Pass --{kind}-field explicitly."
    )


def stable_id(*parts: str) -> str:
    joined = "|".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def infer_optional_field(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    try:
        return infer_field(columns, candidates, "optional")
    except ValueError:
        return ""


def infer_label_field(columns: Iterable[str], label_field: str | None, conversation_field: str | None) -> str:
    if label_field is not None:
        return label_field
    try:
        return infer_field(columns, LABEL_FIELDS, "label")
    except ValueError:
        if conversation_field:
            return ""
        raise


def text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "value", "content"):
            if key in value:
                return text_from_content(value[key])
        return ""
    if isinstance(value, list):
        parts = [text_from_content(item) for item in value]
        return "\n".join(part for part in parts if part.strip())
    return str(value)


def role_from_message(message: dict[str, Any]) -> str:
    role = str(
        message.get("role")
        or message.get("from")
        or message.get("speaker")
        or message.get("author")
        or ""
    ).strip().lower()
    if role in {"human", "user", "prompter"}:
        return "user"
    if role in {"gpt", "assistant", "model", "bot"}:
        return "assistant"
    return role


def message_text(message: dict[str, Any]) -> str:
    for key in ("content", "value", "text"):
        if key in message:
            return text_from_content(message[key])
    return ""


def clean_conversation_prompt(text: str) -> str:
    text = re.sub(r"<\s*image\s*/?\s*>", "", text, flags=re.IGNORECASE)
    text = text.replace("<|image|>", "")
    return text.strip()


def extract_conversation_prompt_answer(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None, None
    if not isinstance(value, list):
        return None, None

    user_messages: list[str] = []
    assistant_messages: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = role_from_message(item)
        text = message_text(item).strip()
        if not text:
            continue
        if role == "user":
            user_messages.append(clean_conversation_prompt(text))
        elif role == "assistant":
            assistant_messages.append(text)

    prompt = next((message for message in user_messages if message), None)
    answer = next((message for message in assistant_messages if message), None)
    return prompt, answer


def sample_text_fields(
    row: dict[str, Any],
    *,
    label_column: str,
    prompt_column: str,
    conversation_column: str,
) -> tuple[str, str | None]:
    prompt: str | None = None
    label = str(row.get(label_column) or "") if label_column else ""

    if prompt_column:
        prompt = str(row.get(prompt_column) or "").strip() or None

    if conversation_column and row.get(conversation_column) is not None:
        conversation_prompt, conversation_answer = extract_conversation_prompt_answer(row.get(conversation_column))
        if conversation_prompt:
            prompt = conversation_prompt
        if conversation_answer:
            label = conversation_answer

    return label, prompt


def image_to_pil(value: Any) -> Image.Image:
    if isinstance(value, HFDatasetImageRef):
        return image_to_pil(value.dataset[value.index][value.image_field])

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")

    if isinstance(value, dict):
        if "path" in value and value["path"]:
            return Image.open(value["path"]).convert("RGB")
        if "bytes" in value and value["bytes"]:
            from io import BytesIO

            return Image.open(BytesIO(value["bytes"])).convert("RGB")

    raise TypeError(f"Unsupported image value: {type(value)!r}")


def iter_hf_samples(
    *,
    dataset_id: str,
    split: str,
    image_field: str | None = None,
    label_field: str | None = None,
    id_field: str | None = None,
    prompt_field: str | None = None,
    conversation_field: str | None = None,
    task_field: str | None = None,
    streaming: bool = True,
    trust_remote_code: bool = False,
) -> Iterator[DatasetSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is required. Install it with: "
            "pip install datasets"
        ) from exc

    dataset = load_dataset(
        dataset_id,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )

    columns = getattr(dataset, "column_names", None)
    if not columns:
        iterator = iter(dataset)
        first = next(iterator)
        columns = list(first.keys())
        dataset = _chain_first(first, iterator)

    image_column = image_field or infer_field(columns, IMAGE_FIELDS, "image")
    conversation_column = conversation_field or ""
    label_column = infer_label_field(columns, label_field, conversation_column)
    prompt_column = prompt_field or infer_optional_field(columns, PROMPT_FIELDS)
    task_column = task_field or infer_optional_field(columns, TASK_FIELDS)
    id_column = id_field
    if id_column is None:
        try:
            id_column = infer_field(columns, ID_FIELDS, "id")
        except ValueError:
            id_column = ""

    for index, row in enumerate(dataset):
        source_id = str(row[id_column]) if id_column and row.get(id_column) is not None else str(index)
        label, prompt = sample_text_fields(
            row,
            label_column=label_column,
            prompt_column=prompt_column,
            conversation_column=conversation_column,
        )
        yield DatasetSample(
            sample_id=source_id or stable_id(dataset_id, split, str(index)),
            image=row[image_column],
            ground_truth=label,
            split=split,
            dataset=dataset_id,
            source="huggingface",
            prompt=prompt,
            task=str(row.get(task_column) or "") if task_column else None,
        )


def iter_hf_disk_samples(
    *,
    dataset_path: Path,
    split: str,
    image_field: str | None = None,
    label_field: str | None = None,
    id_field: str | None = None,
    prompt_field: str | None = None,
    conversation_field: str | None = None,
    task_field: str | None = None,
) -> Iterator[DatasetSample]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is required. Install it with: "
            "pip install datasets"
        ) from exc

    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            available = ", ".join(loaded.keys())
            raise ValueError(f"Split '{split}' not found in {dataset_path}. Available splits: {available}")
        dataset = loaded[split]
    else:
        dataset = loaded

    columns = list(dataset.column_names)
    image_column = image_field or infer_field(columns, IMAGE_FIELDS, "image")
    conversation_column = conversation_field or ""
    label_column = infer_label_field(columns, label_field, conversation_column)
    prompt_column = prompt_field or infer_optional_field(columns, PROMPT_FIELDS)
    task_column = task_field or infer_optional_field(columns, TASK_FIELDS)
    resolved_id_field = id_field
    if resolved_id_field is None:
        try:
            resolved_id_field = infer_field(columns, ID_FIELDS, "id")
        except ValueError:
            resolved_id_field = ""

    metadata_columns = list(
        dict.fromkeys(
            column
            for column in (
                resolved_id_field,
                label_column,
                prompt_column,
                conversation_column,
                task_column,
            )
            if column
        )
    )
    if hasattr(dataset, "select_columns"):
        metadata_dataset = dataset.select_columns(metadata_columns)
    else:
        metadata_dataset = dataset.remove_columns(
            [column for column in dataset.column_names if column not in metadata_columns]
        )
    for index, row in enumerate(metadata_dataset):
        source_id = (
            str(row[resolved_id_field])
            if resolved_id_field and row.get(resolved_id_field) is not None
            else str(index)
        )
        label, prompt = sample_text_fields(
            row,
            label_column=label_column,
            prompt_column=prompt_column,
            conversation_column=conversation_column,
        )
        yield DatasetSample(
            sample_id=source_id or stable_id(str(dataset_path), split, str(index)),
            image=HFDatasetImageRef(dataset=dataset, index=index, image_field=image_column),
            ground_truth=label,
            split=split,
            dataset=dataset_path.name,
            source="hf_disk",
            prompt=prompt,
            task=str(row.get(task_column) or "") if task_column else None,
        )


def _chain_first(first: dict[str, Any], dataset: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield first
    yield from dataset


def iter_csv_samples(
    *,
    metadata_path: Path,
    root_dir: Path | None = None,
    image_field: str | None = None,
    label_field: str | None = None,
    id_field: str | None = None,
    prompt_field: str | None = None,
    conversation_field: str | None = None,
    task_field: str | None = None,
    split_field: str = "split",
    split: str | None = None,
    dataset_name: str | None = None,
) -> Iterator[DatasetSample]:
    root_dir = root_dir or metadata_path.parent
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return

        image_column = image_field or infer_field(reader.fieldnames, ("image_path", *IMAGE_FIELDS), "image")
        conversation_column = conversation_field or ""
        label_column = infer_label_field(reader.fieldnames, label_field, conversation_column)
        prompt_column = prompt_field or infer_optional_field(reader.fieldnames, PROMPT_FIELDS)
        task_column = task_field or infer_optional_field(reader.fieldnames, TASK_FIELDS)
        id_column = id_field
        if id_column is None:
            try:
                id_column = infer_field(reader.fieldnames, ID_FIELDS, "id")
            except ValueError:
                id_column = ""

        for index, row in enumerate(reader):
            row_split = str(row.get(split_field) or "")
            if split and row_split != split:
                continue
            image_path = Path(str(row[image_column]))
            if not image_path.is_absolute():
                image_path = root_dir / image_path
            source_id = str(row[id_column]) if id_column and row.get(id_column) else str(index)
            label, prompt = sample_text_fields(
                row,
                label_column=label_column,
                prompt_column=prompt_column,
                conversation_column=conversation_column,
            )
            yield DatasetSample(
                sample_id=source_id or stable_id(str(metadata_path), str(index)),
                image=image_path,
                ground_truth=label,
                split=row_split or (split or "unknown"),
                dataset=dataset_name or metadata_path.parent.name,
                source="csv",
                prompt=prompt,
                task=str(row.get(task_column) or "") if task_column else None,
            )


def iter_jsonl_samples(
    *,
    metadata_path: Path,
    root_dir: Path | None = None,
    image_field: str | None = None,
    label_field: str | None = None,
    id_field: str | None = None,
    prompt_field: str | None = None,
    conversation_field: str | None = None,
    task_field: str | None = None,
    split_field: str = "split",
    split: str | None = None,
    dataset_name: str | None = None,
) -> Iterator[DatasetSample]:
    root_dir = root_dir or metadata_path.parent
    with metadata_path.open("r", encoding="utf-8") as file:
        first_row: dict[str, Any] | None = None
        for line in file:
            if not line.strip():
                continue
            first_row = json.loads(line)
            break

    if first_row is None:
        return

    fieldnames = list(first_row.keys())
    image_column = image_field or infer_field(fieldnames, ("image_path", *IMAGE_FIELDS), "image")
    conversation_column = conversation_field or ""
    label_column = infer_label_field(fieldnames, label_field, conversation_column)
    if (
        label_column in {"ground_truth", "prediction", "prediction_normalized"}
        and label_field is None
        and not conversation_column
    ):
        raise ValueError(
            f"Refusing to infer ambiguous JSONL label field '{label_column}'. "
            "Pass --label-field explicitly for prediction manifests."
        )
    prompt_column = prompt_field or infer_optional_field(fieldnames, PROMPT_FIELDS)
    task_column = task_field or infer_optional_field(fieldnames, TASK_FIELDS)
    resolved_id_field = id_field
    if resolved_id_field is None:
        try:
            resolved_id_field = infer_field(fieldnames, ID_FIELDS, "id")
        except ValueError:
            resolved_id_field = ""

    with metadata_path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip():
                continue
            row = json.loads(line)
            row_split = str(row.get(split_field) or "")
            if split and row_split and row_split != split:
                continue
            raw_image_path = str(row.get(image_column) or "")
            if not raw_image_path:
                continue
            image_path = Path(raw_image_path)
            if not image_path.is_absolute():
                image_path = root_dir / image_path
            source_id = str(row[resolved_id_field]) if resolved_id_field and row.get(resolved_id_field) else str(index)
            label, prompt = sample_text_fields(
                row,
                label_column=label_column,
                prompt_column=prompt_column,
                conversation_column=conversation_column,
            )
            yield DatasetSample(
                sample_id=source_id or stable_id(str(metadata_path), str(index)),
                image=image_path,
                ground_truth=label,
                split=row_split or (split or "unknown"),
                dataset=dataset_name or str(row.get("dataset") or metadata_path.parent.name),
                source="jsonl",
                prompt=prompt,
                task=str(row.get(task_column) or "") if task_column else None,
            )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
