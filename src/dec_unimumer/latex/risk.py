"""Interpretable rule-based risk score for LaTeX predictions."""

from __future__ import annotations

from dataclasses import dataclass

from .validator import ValidationResult, validate_latex


@dataclass(frozen=True)
class RiskResult:
    score: float
    level: str
    reasons: list[str]
    validation: ValidationResult


def risk_level(score: float) -> str:
    if score < 25:
        return "low"
    if score < 55:
        return "medium"
    return "high"


def compute_risk(text: str) -> RiskResult:
    """Compute an interpretable 0..100 risk score from structural features."""
    validation = validate_latex(text)
    score = 0.0
    reasons: list[str] = []

    warning_weights = {
        "empty output": 45,
        "closing brace without opening brace": 25,
        "unclosed brace": 25,
        "unknown command": 12,
        "dangling _": 15,
        "dangling ^": 15,
        "possibly broken frac": 25,
        "possibly broken sqrt": 15,
    }

    for warning in validation.warnings:
        weight = warning_weights.get(warning, 8)
        score += weight
        reasons.append(warning)

    if validation.num_tokens > 80:
        score += 12
        reasons.append("long expression")
    elif validation.num_tokens > 45:
        score += 6
        reasons.append("medium-length expression")

    if validation.max_brace_depth > 4:
        score += 12
        reasons.append("high nesting depth")
    elif validation.max_brace_depth > 2:
        score += 6
        reasons.append("nested structure")

    structural_load = (
        validation.num_frac * 4
        + validation.num_sqrt * 3
        + validation.num_subscripts * 2
        + validation.num_superscripts * 2
    )
    if structural_load:
        score += min(18, structural_load)
        reasons.append("structural commands")

    score = min(100.0, round(score, 2))
    return RiskResult(
        score=score,
        level=risk_level(score),
        reasons=reasons,
        validation=validation,
    )

