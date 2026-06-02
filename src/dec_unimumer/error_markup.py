from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from dec_unimumer.latex.tokenizer import tokenize_latex


@dataclass(frozen=True)
class MarkedFormula:
    marked: str
    correction_log: str
    edit_count: int


def _text(tokens: list[str]) -> str:
    return " ".join(token for token in tokens if token.strip())


def _insert_anchor(tokens: list[str], index: int) -> str:
    return "<start>" if index <= 0 else tokens[index - 1]


def build_marked_formula(reference: str, candidate: str) -> MarkedFormula:
    reference_tokens = tokenize_latex(reference)
    candidate_tokens = tokenize_latex(candidate)
    matcher = SequenceMatcher(a=candidate_tokens, b=reference_tokens, autojunk=False)

    marked_tokens: list[str] = []
    correction_ops: list[str] = []
    edit_count = 0

    for tag, pred_start, pred_end, ref_start, ref_end in matcher.get_opcodes():
        predicted = candidate_tokens[pred_start:pred_end]
        expected = reference_tokens[ref_start:ref_end]
        if tag == "equal":
            marked_tokens.extend(predicted)
            continue

        edit_count += 1
        old_text = _text(predicted)
        new_text = _text(expected)

        if tag == "replace":
            marked_tokens.extend(["<error_start>", old_text, "<error_end>"] if old_text else ["<deleted>"])
            if old_text and new_text:
                correction_ops.append(f"REPLACE:{old_text} -> {new_text}")
            elif old_text:
                correction_ops.append(f"DELETE:{old_text}")
            elif new_text:
                correction_ops.append(f"INSERT:{new_text} after {_insert_anchor(candidate_tokens, pred_start)}")
        elif tag == "delete":
            if old_text:
                marked_tokens.extend(["<error_start>", old_text, "<error_end>"])
                correction_ops.append(f"DELETE:{old_text}")
        elif tag == "insert":
            marked_tokens.append("<deleted>")
            if new_text:
                correction_ops.append(f"INSERT:{new_text} after {_insert_anchor(candidate_tokens, pred_start)}")
        else:
            raise ValueError(f"Unexpected SequenceMatcher tag: {tag}")

    return MarkedFormula(
        marked=" ".join(marked_tokens).strip(),
        correction_log="\n".join(correction_ops).strip(),
        edit_count=edit_count,
    )


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe[:180].strip("._-") or "sample"
