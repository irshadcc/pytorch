#!/usr/bin/env python3
"""Snapshot and compare the exported stable C ABI with libabigail."""

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


LIBRARIES = ("libtorch_cpu.so", "libtorch_cuda.so")
STABLE_SYMBOL = re.compile(r"^(aoti_torch_|torch_)")
HERE = Path(__file__).resolve().parent


def run(command):
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "LIBABIGAIL_DEFAULT_SYSTEM_SUPPRESSION_FILE": os.devnull,
            "LIBABIGAIL_DEFAULT_USER_SUPPRESSION_FILE": os.devnull,
        },
    )


def validate_corpus(path):
    root = ET.parse(path).getroot()
    symbols = {
        item.attrib["name"]
        for item in root.findall("./elf-function-symbols/elf-symbol")
        if STABLE_SYMBOL.match(item.attrib.get("name", ""))
    }
    declarations = {
        item.attrib.get("mangled-name", item.attrib.get("name", ""))
        for item in root.iter("function-decl")
    }
    if not symbols or symbols - declarations:
        missing = sorted(symbols - declarations)
        raise RuntimeError(f"{path}: missing stable ABI type information: {missing}")


def snapshot(args):
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "libabigail": run(["abidw", "--version"]).stdout.strip(),
        "configuration": args.configuration,
        "libraries": [],
    }
    for library in args.libraries:
        binary = args.build / library
        if not binary.is_file():
            raise RuntimeError(f"Missing required library: {binary}")
        output = args.output / f"{library}.abi"
        result = run(
            [
                "abidw",
                "--no-corpus-path",
                "--no-show-locs",
                "--exported-interfaces-only",
                "--suppressions",
                str(HERE / "stable_surface.abignore"),
                "--out-file",
                str(output),
                str(binary),
            ]
        )
        if result.returncode:
            raise RuntimeError(result.stdout)
        validate_corpus(output)
        metadata["libraries"].append(library)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


def compare(args):
    report = [
        "# Torch stable ABI report",
        "",
        "New symbols are reported separately and do not fail the check.",
        "",
    ]
    status = 0
    try:
        old = json.loads((args.baseline / "metadata.json").read_text())
        new = json.loads((args.current / "metadata.json").read_text())
        if old != new:
            raise RuntimeError(
                f"Snapshot toolchain or library mismatch: {old} != {new}"
            )
        if not new["libraries"]:
            raise RuntimeError("Empty ABI snapshot")
        for library in new["libraries"]:
            before, after = (
                args.baseline / f"{library}.abi",
                args.current / f"{library}.abi",
            )
            validate_corpus(before)
            validate_corpus(after)
            command = [
                "abidiff",
                "--no-default-suppression",
                "--leaf-changes-only",
                "--suppressions",
                str(HERE / "stable_abi.abignore"),
            ]
            full = run([*command, str(before), str(after)])
            gate = run([*command, "--no-added-syms", str(before), str(after)])
            code = gate.returncode
            if code < 0 or code & ~12:
                status = 2
            elif code & 12 and status != 2:
                status = 1
            report.extend(
                [
                    f"## {library}",
                    "",
                    f"abidiff exit status: {code}",
                    "",
                    "```text",
                    full.stdout,
                    "```",
                    "",
                ]
            )
            if full.returncode < 0 or full.returncode & ~12:
                status = 2
    except (OSError, ValueError, ET.ParseError, RuntimeError) as error:
        report.extend(["## Infrastructure error", "", str(error), ""])
        status = 2
    args.report.write_text("\n".join(report))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--build", type=Path, required=True)
    snap.add_argument("--output", type=Path, required=True)
    snap.add_argument("--configuration", default="local")
    snap.add_argument(
        "--libraries", nargs="+", choices=LIBRARIES, default=list(LIBRARIES)
    )
    diff = commands.add_parser("compare")
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--current", type=Path, required=True)
    diff.add_argument("--report", type=Path, default=Path("abi-report.md"))
    args = parser.parse_args()
    try:
        return snapshot(args) if args.command == "snapshot" else compare(args)
    except (OSError, ValueError, ET.ParseError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
