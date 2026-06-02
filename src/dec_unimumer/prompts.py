"""Exact task prompts published in phxember/Uni-MuMER-Data."""

from __future__ import annotations


RECOGNITION_PROMPT = (
    "I have an image of a handwritten mathematical expression. Please write out "
    "the expression of the formula in the image using LaTeX format."
)

ERROR_DETECTION_INSTRUCTION = (
    "I have an image of a handwritten mathematical expression and its OCR "
    "recognition result. Please help me to detect possible errors in the "
    "recognition result and mark the places where errors occur with "
    "<error_start> <error_end> and <deleted>."
)

ERROR_CORRECTION_INSTRUCTION = (
    "I have an image of a handwritten mathematical expression and a predicted "
    "formula with error tags, correct the formula by modifying the parts marked "
    "with <error_start> and <error_end> and inserting content where <deleted> "
    "are present. Output the modifications in a single string with the following "
    "format: REPLACE:old -> new for errors to be replaced. INSERT:content after "
    "position for missing content. DELETE:to_delete for parts to be removed."
)

TASK_PROMPTS = {
    "recognition": RECOGNITION_PROMPT,
    "error_find": ERROR_DETECTION_INSTRUCTION,
    "error_fix": ERROR_CORRECTION_INSTRUCTION,
}


def error_detection_prompt(candidate: str) -> str:
    return (
        f"{ERROR_DETECTION_INSTRUCTION}\n\n"
        f"erroneous formula: {candidate}\n"
        "Marked formula: "
    )


def error_correction_prompt(marked_formula: str) -> str:
    return (
        f"{ERROR_CORRECTION_INSTRUCTION}\n\n"
        f"Marked formula: {marked_formula}\n"
        "Correction log: "
    )
