"""Command-line differentiable learning smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import train_synthetic_parametric_operator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn a small hidden parametric butterfly operator."
    )
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rhs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = train_synthetic_parametric_operator(
        n=args.n,
        steps=args.steps,
        rhs=args.rhs,
        seed=args.seed,
        threads=args.threads,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

