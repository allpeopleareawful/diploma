"""Update validation early-stopping state after one training cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested_value(payload: dict[str, Any], field: str) -> float:
    value: Any = payload
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Metric field not found: {field}")
        value = value[part]
    return float(value)


def update_state(
    state: dict[str, Any],
    *,
    cycle: int,
    metric: str,
    value: float,
    patience: int,
    min_delta: float,
) -> dict[str, Any]:
    state.setdefault("best_value", None)
    state.setdefault("best_cycle", None)
    state.setdefault("stale_cycles", 0)
    state.setdefault("should_stop", False)
    state.setdefault("history", [])
    previous_best = state["best_value"]
    improved = previous_best is None or value > float(previous_best) + min_delta
    if improved:
        state["best_value"] = value
        state["best_cycle"] = cycle
        state["stale_cycles"] = 0
    else:
        state["stale_cycles"] = int(state["stale_cycles"]) + 1
    state["should_stop"] = int(state["stale_cycles"]) >= patience
    state["metric"] = metric
    state["patience"] = patience
    state["min_delta"] = min_delta
    state["history"].append(
        {
            "cycle": cycle,
            "value": value,
            "improved": improved,
            "best_value": state["best_value"],
            "best_cycle": state["best_cycle"],
            "stale_cycles": state["stale_cycles"],
            "should_stop": state["should_stop"],
        }
    )
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--metric", default="net_fixed_count")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    state = (
        json.loads(args.state.read_text(encoding="utf-8"))
        if args.state.exists()
        else {}
    )
    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    value = nested_value(validation, args.metric)
    state = update_state(
        state,
        cycle=args.cycle,
        metric=args.metric,
        value=value,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    print("STOP" if state["should_stop"] else "CONTINUE")


if __name__ == "__main__":
    main()
