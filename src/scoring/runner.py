"""CLI entrypoint for model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from .compare_models import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined model-comparison artifacts.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        help="Directory containing eval_results_*.toml files for one dataset.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparisons for every dataset directory under ./results.",
    )
    parser.add_argument(
        "--pricing-file",
        type=Path,
        default=None,
        help=(
            "Pricing CSV with columns: model,input_per_1m,output_per_1m "
            "(default: reports/model_pricing.csv)."
        ),
    )
    args = parser.parse_args()

    try:
        outputs = run(
            args.results_dir,
            run_all=args.all,
            pricing_file=args.pricing_file,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if outputs:
        print("Generated combined reports:")
        for key, path in outputs.items():
            print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
