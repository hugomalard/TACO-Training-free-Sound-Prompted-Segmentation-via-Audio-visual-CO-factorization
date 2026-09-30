"""Compare a benchmark JSON file with the stored reference numbers."""

import argparse
import json
from pathlib import Path

import yaml

from taco.paths import EXPECTED_RESULTS


def load_expected(path=None):
    path = Path(path) if path is not None else EXPECTED_RESULTS
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def compare_results(results, expected=None):
    """Return a list of human-readable failures. An empty list means the run matches."""
    expected = load_expected() if expected is None else expected
    name = results["benchmark"]
    if name not in expected:
        return [f"No reference numbers stored for benchmark {name}"]
    failures = []
    for metric, target in expected[name].items():
        if metric not in results["mean"]:
            failures.append(f"{name}.{metric} is missing from the result file")
            continue
        got = float(results["mean"][metric])
        tolerance = max(2 * float(target["std"]), 1.0)
        if abs(got - float(target["mean"])) > tolerance:
            failures.append(
                f"{name}.{metric}: got {got:.2f}, reference {target['mean']} ± {target['std']} "
                f"(tolerance {tolerance:.2f})"
            )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check TACO results against the stored reference numbers.")
    parser.add_argument("results", help="JSON written by python -m taco.eval")
    parser.add_argument("--expected", default=None, help="Optional expected-results YAML")
    args = parser.parse_args(argv)
    with open(args.results, "r") as handle:
        results = json.load(handle)
    expected = load_expected(args.expected) if args.expected else None
    failures = compare_results(results, expected)
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print(f"{results['benchmark']} matches the reference within tolerance.")


if __name__ == "__main__":
    main()
