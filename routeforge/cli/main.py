"""RouteForge CLI entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .tape import add_tape_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routeforge")
    subparsers = parser.add_subparsers(dest="command")
    add_tape_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 3
    return int(args.func(args))
