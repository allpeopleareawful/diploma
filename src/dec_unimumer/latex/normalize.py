"""Normalization rules for LaTeX comparison."""

from __future__ import annotations

import re

from .tokenizer import detokenize_latex, tokenize_latex


COMMAND_ALIASES = {
    r"\le": r"\leq",
    r"\ge": r"\geq",
    r"\ne": r"\neq",
    r"\to": r"\rightarrow",
    r"\perp": r"\bot",
}


CHAT_TAIL_PATTERNS = (
    r"(?:\r?\n)+\s*(?:assistant|answer|final answer|result|latex)\s*:",
    r"(?:\r?\n)+\s*(?:the answer is|the final answer is|therefore|question analysis)\b",
    r"(?:\r?\n)+\s*(?:which of the following|note that|so the answer)\b",
    r"(?:\r?\n)+\s*(?:we can also|we can write|the expression can be|the expression represents)\b",
)


def strip_model_wrappers(text: str) -> str:
    """Remove wrappers commonly produced by chat models."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("```latex", "").replace("```", "")
    text = text.replace("$$", "").replace("$", "")
    text = text.strip()

    prefixes = ("latex:", "answer:", "final answer:", "result:")
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text


def truncate_chat_tail(text: str) -> str:
    """Keep the first answer span and drop later chat/VQA style continuations."""
    best = len(text)
    for pattern in CHAT_TAIL_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            best = min(best, match.start())
    return text[:best].strip()


def clean_model_prediction(text: str) -> str:
    """Normalize common chat-model wrappers without changing LaTeX token spacing."""
    text = strip_model_wrappers(text)
    text = re.sub(
        r"^\s*(assistant|answer|final answer|result|latex)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return truncate_chat_tail(text).strip()


def normalize_latex(text: str) -> str:
    """Normalize LaTeX for exact-match style comparison."""
    text = strip_model_wrappers(text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\s+", " ", text).strip()

    tokens = tokenize_latex(text)
    normalized_tokens = [COMMAND_ALIASES.get(token, token) for token in tokens]
    return detokenize_latex(normalized_tokens)
