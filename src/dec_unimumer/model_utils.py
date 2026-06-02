from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def is_local_path(value: str) -> bool:
    if value.startswith(("~", ".", "/", "\\")):
        return True
    drive, _tail = os.path.splitdrive(value)
    return bool(drive)


def resolve_model_ref(value: str, default_repo_id: str) -> str:
    if not is_local_path(value):
        return value

    path = Path(value).expanduser()
    if path.exists():
        return str(path)

    raise FileNotFoundError(
        f"Local model path does not exist: {path}\n"
        f"Pass the Hugging Face id `{default_repo_id}` or download it first."
    )


def torch_dtype_from_name(name: str) -> Any:
    import torch

    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

