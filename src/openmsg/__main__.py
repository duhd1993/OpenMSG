"""Command-line interface for OpenMSG."""

from __future__ import annotations

import argparse

from openmsg.config import result_to_json, run_config, write_result_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an OpenMSG homogenization input file.")
    parser.add_argument("input", help="Path to a JSON input file.")
    parser.add_argument("-o", "--output", help="Optional path for result JSON.")
    parser.add_argument("--include-internal", action="store_true", help="Include E, H, D0, G, and V0 in JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_config(args.input)
    if args.output:
        write_result_json(result, args.output, include_internal=args.include_internal)
    else:
        print(result_to_json(result, include_internal=args.include_internal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

