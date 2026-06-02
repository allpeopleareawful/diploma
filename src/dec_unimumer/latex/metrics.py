"""Recognition metrics for image-to-LaTeX experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from .normalize import normalize_latex
from .tokenizer import tokenize_latex


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance using O(min(n, m)) memory."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    """Return CER with a stable convention for empty strings."""
    reference = reference or ""
    prediction = prediction or ""
    if not reference:
        return 0.0 if not prediction else 1.0
    return levenshtein_distance(reference, prediction) / len(reference)


@dataclass(frozen=True)
class RecognitionMetrics:
    raw_exact_match: bool
    normalized_exact_match: bool
    cer: float
    normalized_cer: float


def ngram_counts(tokens: list[str], order: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if order <= 0 or len(tokens) < order:
        return counts
    for index in range(len(tokens) - order + 1):
        ngram = tuple(tokens[index : index + order])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def corpus_bleu(
    references: list[str],
    predictions: list[str],
    *,
    max_order: int = 4,
    smooth: float = 1.0,
) -> float:
    """Compute a compact smoothed corpus BLEU score over LaTeX tokens."""
    if not references or not predictions:
        return 0.0

    clipped_matches = [0.0 for _ in range(max_order)]
    total_predicted = [0.0 for _ in range(max_order)]
    reference_length = 0
    prediction_length = 0

    for reference, prediction in zip(references, predictions):
        ref_tokens = tokenize_latex(reference)
        pred_tokens = tokenize_latex(prediction)
        reference_length += len(ref_tokens)
        prediction_length += len(pred_tokens)

        for order in range(1, max_order + 1):
            ref_counts = ngram_counts(ref_tokens, order)
            pred_counts = ngram_counts(pred_tokens, order)
            total_predicted[order - 1] += max(len(pred_tokens) - order + 1, 0)
            for ngram, count in pred_counts.items():
                clipped_matches[order - 1] += min(count, ref_counts.get(ngram, 0))

    if prediction_length == 0:
        return 0.0

    precisions = [
        (clipped_matches[index] + smooth) / (total_predicted[index] + smooth)
        for index in range(max_order)
    ]
    brevity_penalty = 1.0
    if prediction_length < reference_length:
        brevity_penalty = exp(1.0 - reference_length / prediction_length)

    return brevity_penalty * exp(sum(log(precision) for precision in precisions) / max_order)


def compute_metrics(ground_truth: str, prediction: str) -> RecognitionMetrics:
    """Compute raw and normalized metrics for one recognition result."""
    gt_raw = (ground_truth or "").strip()
    pred_raw = (prediction or "").strip()
    gt_norm = normalize_latex(gt_raw)
    pred_norm = normalize_latex(pred_raw)

    return RecognitionMetrics(
        raw_exact_match=gt_raw == pred_raw,
        normalized_exact_match=gt_norm == pred_norm,
        cer=character_error_rate(gt_raw, pred_raw),
        normalized_cer=character_error_rate(gt_norm, pred_norm),
    )
