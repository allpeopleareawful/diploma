"""LaTeX-aware tokenization utilities."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(
    r"""
    \\[a-zA-Z]+        | # LaTeX command
    \\[^a-zA-Z\s]      | # escaped symbol, e.g. \{
    \d+\.\d+           | # decimal number
    \d+                | # integer
    [A-Za-z]           | # single latin letter
    [а-яА-ЯёЁ]         | # single cyrillic letter, useful for noisy outputs
    [{}()[\]_^]        | # structural symbols
    [+\-*/=<>.,;:|!&]  | # operators and punctuation
    \S                   # fallback non-space char
    """,
    re.VERBOSE,
)


def tokenize_latex(text: str) -> list[str]:
    """Split a LaTeX string into commands, symbols, variables and operators."""
    if not text:
        return []
    return TOKEN_PATTERN.findall(text)


def detokenize_latex(tokens: list[str]) -> str:
    """Join tokens into a stable normalized representation."""
    return " ".join(token for token in tokens if token.strip())

