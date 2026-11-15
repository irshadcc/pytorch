#!/usr/bin/env python3
"""Download the closest available ancestor snapshot from trusted main builds."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ref", required=True)
parser.add_argument("--repo", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
# An ancestor check prevents using a newer main snapshot for an older PR.
commits = subprocess.check_output(
    ["git", "rev-list", "--first-parent", args.ref],
    text=True,
).splitlines()
runs = json.loads(
    subprocess.check_output(
        [
            "gh",
            "api",
            f"repos/{args.repo}/actions/workflows/pull.yml/runs?branch=main&event=push&per_page=100",
        ],
        text=True,
    )
)["workflow_runs"]
by_sha = {}
for run in runs:
    if run["status"] == "completed":
        by_sha.setdefault(run["head_sha"], []).append(run["id"])
for sha in commits:
    for run_id in by_sha.get(sha, []):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    str(run_id),
                    "--repo",
                    args.repo,
                    "--name",
                    f"stable-abi-{sha}",
                    "--dir",
                    temporary,
                ]
            )
            if result.returncode == 0:
                shutil.copytree(temporary, args.output)
                print(f"Using stable ABI baseline from {sha} (run {run_id})")
                raise SystemExit(0)
raise SystemExit(
    "No ancestor baseline in the last 100 main push runs; "
    "seed one with a successful main ABI job."
)
