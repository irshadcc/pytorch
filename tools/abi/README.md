# Stable C ABI compatibility checks

The `stable-abi-check` job in `pull.yml` consumes the existing Linux x86-64
CUDA 13.2 / clang 20 build. `abi-debuginfo` disables wheel reuse and enables
`USE_ABI_DEBUGINFO`, which retains DWARF 4 type information for shim translation
units, including generated CPU and CUDA shims. Other builds default to OFF.
No additional PyTorch build is introduced.

The check runs in **shadow mode** (`continue-on-error`). It does not post PR
comments or require labels. ABI changes fail the check step and remain visible
in the job summary and `stable-abi-report` artifact, without gating merges.

## Scope

Exported `aoti_torch_*` and `torch_*` functions in `libtorch_cpu.so` and
`libtorch_cuda.so`, including types reachable through those functions, are
checked. Other functions and exported variables are excluded by a read-time
symbol suppression. Snapshots must have type declarations for every selected
exported symbol; empty or stripped snapshots are infrastructure errors.

Header-only types that are not reachable from an exported C function and inline
C++ wrappers are outside this check. Existing stable extension tests remain
necessary. Full-libtorch drift reports, header-only probes, release baselines,
owner approval labels, and blocking enforcement are follow-up rollout work.

## Baselines and reproducibility

Main push builds publish `stable-abi-<commit>` artifacts retained for 90 days.
PRs compare against their merge-base with the PR base commit; main compares
against its first parent. The fetcher chooses the closest ancestor with an ABI
artifact among the latest 100 main push runs. It only reads artifacts from this
repository's `pull.yml` main push runs. Missing baselines fail explicitly as
infrastructure errors, including during initial deployment before the first
main snapshot has been published. Skipped builds do not run the check.

The analysis runner is Ubuntu 24.04 with libabigail **2.4.0**, checked explicitly
after package installation. This initial implementation uses the packaged tool
rather than the RFC's proposed, not-yet-provisioned 2.10 image. Changing the
analysis version or canonical build configuration requires reseeding baselines;
metadata mismatches are infrastructure errors. User/system default suppressions
are disabled so local configuration cannot silently change results.

## Local use

On Linux with `abigail-tools` installed, build PyTorch through the normal
editable installation with `USE_ABI_DEBUGINFO=1`. Then:

```bash
python tools/abi/check_abi.py snapshot --build build/lib --output abi-current
python tools/abi/check_abi.py compare --baseline abi-baseline --current abi-current --report abi-report.md
```

Use `--libraries libtorch_cpu.so` for CPU-only builds, on both sides.
Use the same compiler, architecture, CUDA version, build flags and libabigail
version for both snapshots. Set `--configuration` to the same explicit identifier
on both snapshots when managing multiple configurations.

The comparison report includes additions, while the status ignores them:

- 0: no changes other than additions.
- 1: stable ABI changed (libabigail bits 4 or 8).
- 2: tool failure, missing inputs/type information, or metadata mismatch.

Exceptions belong in `stable_abi.abignore`; each entry must include a comment
citing the reviewing PR, approver and reason. Do not broaden
`stable_surface.abignore` to hide a compatibility change.
