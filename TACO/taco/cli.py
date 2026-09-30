"""Command line."""

import argparse

from taco.evaluate import SPECS, run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a TACO benchmark.")
    parser.add_argument("--benchmark", required=True, choices=sorted(SPECS))
    parser.add_argument("--runs", type=int, default=3, help="Random initializations. The seed is not reset between them.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument("--output", default=None, help="Where to write the JSON summary.")
    parser.add_argument("--paths", default=None, help="YAML file of dataset and checkpoint paths.")
    args = parser.parse_args(argv)
    summary = run_benchmark(
        args.benchmark,
        runs=args.runs,
        limit=args.limit,
        paths=args.paths,
        output=args.output,
    )
    print("mean", summary["mean"])
    print("std", summary["std"])
    return summary


if __name__ == "__main__":
    main()
