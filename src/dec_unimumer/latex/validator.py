"""Lightweight structural validation for LaTeX outputs."""

from __future__ import annotations

from dataclasses import dataclass

from .tokenizer import tokenize_latex


KNOWN_COMMANDS = {
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\varepsilon",
    r"\theta",
    r"\lambda",
    r"\mu",
    r"\pi",
    r"\sigma",
    r"\phi",
    r"\varphi",
    r"\omega",
    r"\frac",
    r"\sqrt",
    r"\sum",
    r"\prod",
    r"\int",
    r"\lim",
    r"\sin",
    r"\cos",
    r"\tan",
    r"\log",
    r"\ln",
    r"\exp",
    r"\le",
    r"\leq",
    r"\ge",
    r"\geq",
    r"\neq",
    r"\ne",
    r"\times",
    r"\div",
    r"\cdot",
    r"\pm",
    r"\mp",
    r"\rightarrow",
    r"\to",
    r"\left",
    r"\right",
    r"\angle",
    r"\triangle",
    r"\cap",
    r"\cup",
    r"\in",
    r"\notin",
    r"\subset",
    r"\subseteq",
    r"\supset",
    r"\supseteq",
    r"\bot",
    r"\perp",
    r"\parallel",
    r"\prime",
    r"\circ",
    r"\infty",
    r"\downarrow",
    r"\uparrow",
    r"\quad",
    r"\textcircled",
    r"\mathbb",
    r"\mathrm",
    r"\operatorname",
    r"\begin",
    r"\end",
    r"\\",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    warnings: list[str]
    num_tokens: int
    num_commands: int
    num_frac: int
    num_sqrt: int
    num_subscripts: int
    num_superscripts: int
    max_brace_depth: int
    unknown_commands: list[str]


def _brace_depth(tokens: list[str]) -> tuple[int, list[str]]:
    depth = 0
    max_depth = 0
    warnings: list[str] = []
    for token in tokens:
        if token == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif token == "}":
            depth -= 1
            if depth < 0:
                warnings.append("closing brace without opening brace")
                depth = 0
    if depth:
        warnings.append("unclosed brace")
    return max_depth, warnings


def _has_argument(tokens: list[str], index: int) -> bool:
    if index + 1 >= len(tokens):
        return False
    next_token = tokens[index + 1]
    return bool(next_token and next_token not in {"}", "_", "^"})


def validate_latex(text: str) -> ValidationResult:
    """Run basic structural checks over a LaTeX string."""
    tokens = tokenize_latex(text or "")
    warnings: list[str] = []

    if not tokens:
        warnings.append("empty output")

    max_depth, brace_warnings = _brace_depth(tokens)
    warnings.extend(brace_warnings)

    unknown_commands = sorted(
        {token for token in tokens if token.startswith("\\") and token not in KNOWN_COMMANDS}
    )
    if unknown_commands:
        warnings.append("unknown command")

    for index, token in enumerate(tokens):
        if token in {"_", "^"} and not _has_argument(tokens, index):
            warnings.append(f"dangling {token}")
        if token == r"\frac":
            remaining = tokens[index + 1 :]
            if remaining.count("{") < 2 or remaining.count("}") < 2:
                warnings.append("possibly broken frac")
        if token == r"\sqrt" and not _has_argument(tokens, index):
            warnings.append("possibly broken sqrt")

    return ValidationResult(
        valid=not warnings,
        warnings=warnings,
        num_tokens=len(tokens),
        num_commands=sum(1 for token in tokens if token.startswith("\\")),
        num_frac=tokens.count(r"\frac"),
        num_sqrt=tokens.count(r"\sqrt"),
        num_subscripts=tokens.count("_"),
        num_superscripts=tokens.count("^"),
        max_brace_depth=max_depth,
        unknown_commands=unknown_commands,
    )
