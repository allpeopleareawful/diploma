"""Prepare the official CROHME test manifests released with Uni-MuMER."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from dec_unimumer.data import extract_conversation_prompt_answer
from dec_unimumer.latex.normalize import normalize_latex
from dec_unimumer.prompts import RECOGNITION_PROMPT


ARCHIVE_URL = "https://raw.githubusercontent.com/BFlameSwift/Uni-MuMER/main/data.zip"
CROHME_YEARS = (2014, 2016, 2019)


def _progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(blocks * block_size, total_size)
    percent = int(downloaded / total_size * 100)
    if percent == getattr(_progress, "last_percent", -1):
        return
    _progress.last_percent = percent
    print(f"\rDownloading CROHME data: {percent:3d}%", end="")


def download_archive(url: str, destination: Path, *, overwrite: bool) -> Path:
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _progress.last_percent = -1
    urllib.request.urlretrieve(url, destination, _progress)
    print()
    return destination


def extract_archive(archive: Path, destination: Path, *, overwrite: bool) -> Path:
    marker = destination / ".complete"
    if marker.exists() and not overwrite:
        return destination
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe archive member: {member.filename}") from exc
        source.extractall(destination)
    marker.write_text(str(archive.resolve()), encoding="utf-8")
    return destination


def find_prompt_file(data_root: Path, year: int) -> Path:
    matches = [
        path
        for path in data_root.rglob("*.json")
        if "prompts" in {part.lower() for part in path.parts}
        and "crohme" in path.stem.lower()
        and str(year) in path.stem
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one CROHME {year} prompt file under {data_root}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _strip_image_token(text: str) -> str:
    text = re.sub(r"<\s*image\s*/?\s*>", "", text or "", flags=re.IGNORECASE)
    return text.replace("<|image|>", "").strip()


def _image_value(item: dict[str, Any]) -> str:
    value = item.get("images") or item.get("image") or item.get("image_path")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").replace("\\", "/")


def resolve_image_path(item: dict[str, Any], data_root: Path) -> Path:
    raw = _image_value(item)
    if not raw:
        raise ValueError("CROHME record has no image path.")
    path = Path(raw)
    if path.is_absolute():
        return path
    parts = [part for part in path.parts if part not in {"", "."}]
    candidates = [data_root / Path(*parts)]
    if not parts or parts[0].lower() != "data":
        candidates.append(data_root / "data" / Path(*parts))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"CROHME image not found for path: {raw}")


def rows_from_prompt_file(
    prompt_file: Path,
    data_root: Path,
    *,
    year: int,
    limit: int = 0,
) -> list[dict[str, Any]]:
    records = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {prompt_file}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        if limit > 0 and len(rows) >= limit:
            break
        if not isinstance(item, dict):
            continue
        image_path = resolve_image_path(item, data_root)
        messages = item.get("messages") or item.get("conversations")
        prompt, answer = extract_conversation_prompt_answer(messages)
        answer = str(answer or "").strip()
        if not answer:
            raise ValueError(f"Missing ground truth at row {index} in {prompt_file}")
        rows.append(
            {
                "id": f"crohme_{year}__{image_path.stem}",
                "dataset": "CROHME",
                "benchmark": f"CROHME {year}",
                "split": "test",
                "source": "Uni-MuMER official release",
                "image_path": str(image_path),
                "ground_truth": answer,
                "ground_truth_normalized": normalize_latex(answer),
                "prompt": _strip_image_token(prompt or "") or RECOGNITION_PROMPT,
            }
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate sample ids in {prompt_file}")
    return rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    years = tuple(args.year or CROHME_YEARS)
    if args.data_root is not None:
        data_root = args.data_root.resolve()
    else:
        archive = args.archive
        if args.download:
            archive = download_archive(args.url, args.download_path, overwrite=args.overwrite)
        if archive is None:
            archive = args.download_path
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}. Pass --download or --archive.")
        data_root = extract_archive(archive, args.extract_dir, overwrite=args.overwrite).resolve()

    manifests = {year: args.output_dir / str(year) / "test.jsonl" for year in years}
    benchmark_summaries: dict[str, dict[str, Any]] = {}
    for year, output in manifests.items():
        prompt_file = find_prompt_file(data_root, year)
        if output.exists() and not args.overwrite:
            examples = sum(
                1 for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            benchmark_summaries[str(year)] = {
                "benchmark": f"CROHME {year}",
                "examples": examples,
                "prompt_file": str(prompt_file),
                "manifest": str(output),
                "reused": True,
            }
            continue
        rows = rows_from_prompt_file(prompt_file, data_root, year=year, limit=args.limit)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        benchmark_summaries[str(year)] = {
            "benchmark": f"CROHME {year}",
            "examples": len(rows),
            "prompt_file": str(prompt_file),
            "manifest": str(output),
            "reused": False,
        }

    summary = {
        "source": "Uni-MuMER official release",
        "data_root": str(data_root),
        "benchmarks": benchmark_summaries,
        "total_examples": sum(item["examples"] for item in benchmark_summaries.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument("--download-path", type=Path, default=Path("data/raw/unimumer_data.zip"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--extract-dir", type=Path, default=Path("data/raw/unimumer_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/crohme"))
    parser.add_argument("--year", type=int, choices=CROHME_YEARS, action="append")
    parser.add_argument("--limit", type=int, default=0, help="Maximum examples per benchmark.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = prepare(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
