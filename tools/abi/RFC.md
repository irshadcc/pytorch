# RFC: ABI Compatibility Checking for PyTorch with libabigail

Nov 9, 2026 · @Irshad

## Summary

This RFC proposes adding a libabigail-based ABI check to PyTorch CI that blocks any PR which breaks the LibTorch stable ABI, and reports (without blocking) ABI drift in the wider `libtorch`/`libc10` surface.

The check runs `abidw` on the PR's Linux x86-64 build to produce an ABI snapshot (abixml), then runs `abidiff` against a stored baseline from the last release and from the PR's merge-base on `main`. Exit codes from `abidiff` drive the job result; human-readable reports are posted to the PR. The first phase is Linux-only and non-blocking; blocking is enabled only after a clean false-positive soak period.

## Motivation

PyTorch now promises binary compatibility for a subset of LibTorch, but nothing in CI checks that promise at the binary level. A [limited stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html) (C shims, `torch/headeronly`, `torch/csrc/stable`) lets extensions build once and run on later torch versions.

The ecosystem already depends on it. The [2026 stable-ABI plan](https://dev-discuss.pytorch.org/t/libtorch-abi-stable-plans-2026/3380) lists Flash-Attn-3, xformers, torchao and torchaudio as already ABI-stable, with vLLM and torchvision migrating. A silent break now ships broken wheels to every one of these users, and is found only after release.

The same plan names "testing to prevent regressions" and hardening of `TORCH_TARGET_VERSION` versioning as open work. Today's guards are source-level: operator schema BC checks and builds of test extensions. Neither catches a changed struct layout, a reordered vtable, or a removed `extern "C"` symbol reached through a header the test does not exercise. libabigail checks the compiled binary and its debug info directly, which is where those breaks show up.

A secondary benefit: C++ users who link the full `libtorch` (which has no ABI guarantee) get a per-release report of what changed. This is informational, useful for release notes and for knowing when an extension must be rebuilt.

## Background: libabigail

[libabigail](https://sourceware.org/libabigail/) builds an ABI model (a "corpus") of an ELF binary from its exported symbols plus the types reachable from them, read from DWARF, CTF or BTF debug info. Without debug info it falls back to symbol add/remove only. The current upstream release is 2.10 (April 2026), packaged in Fedora, Arch and others; Ubuntu currently ships 2.9 as `abigail-tools`.

| Tool | What it does | Use in this RFC |
| --- | --- | --- |
| `abidw` | Serializes a binary's ABI corpus to abixml text | Produce baselines and PR snapshots |
| `abidiff` | Compares two binaries or abixml files, reports changes | Core PR check |
| `abipkgdiff` | Compares two packages (RPM, DEB, tar) with their debug info | Optional: whole-wheel release comparison |
| `abicompat` | Checks whether an application is still compatible with a newer library | Optional: check prebuilt extension `.so` files against a new libtorch |

Four `abidiff` features make this practical for PyTorch ([manual](https://sourceware.org/libabigail/manual/abidiff.html)):

- **Header filtering.** `--headers-dir1/--headers-dir2` limit reports to types defined in public headers; `--drop-private-types` also drops the rest from memory.
- **Symbol filtering.** `--keep-fn <regex>` restricts the corpus to matching functions, e.g. the `aoti_torch_*` / `torch_*` C shim.
- **Suppression files.** `--suppressions` takes a file of reviewed, known-benign changes, kept in-tree.
- **Machine-readable exit code.** A bit field: 1 = tool error, 2 = usage error, 4 = ABI changed (needs review), 8 = incompatible change (a removed exported symbol, or a changed vtable index). CI can branch on these bits without parsing the report.

Threading is controlled with `-j` / `ABIGAIL_THREAD_POOL_SIZE`, which matters for a library the size of `libtorch_cpu.so`.

## Worked example: libabigail catching an ABI break

### What libabigail is, in brief

libabigail ("ABI Generic Analysis and Instrumentation Library") is a Red Hat–led, Apache-2.0 project on sourceware.org. It reads a compiled ELF shared library plus its DWARF debug info and answers one question: can code built against the old version still run against the new one? It is the engine behind ABI checks in Fedora/RHEL package updates and the Android kernel's KMI checks.

It works on the binary, not the source. So it sees what the compiler actually produced: struct sizes and member offsets, function parameter types, vtable layout, and which symbols are exported. A source diff can look harmless and still break all of these.

### The two versions

A tiny library, `libshape`, ships version 1 (`baseline.c`). Version 2 (`feature.c`) makes three changes that are each a classic ABI break.

```c
/* baseline.c — version 1 of libshape */
struct point {
    int x;
    int y;
};

int point_dot(const struct point *a, const struct point *b)
{
    return a->x * b->x + a->y * b->y;
}

void point_scale(struct point *p, int factor)
{
    p->x *= factor;
    p->y *= factor;
}

void point_reset(struct point *p)
{
    p->x = 0;
    p->y = 0;
}
```

```c
/* feature.c — version 2 of libshape: adds a z coordinate */
struct point {
    int x;
    int y;
    int z;                       /* NEW: changes size and layout */
};

int point_dot(const struct point *a, const struct point *b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

void point_scale(struct point *p, double factor)   /* CHANGED: int -> double */
{
    p->x = (int)(p->x * factor);
    p->y = (int)(p->y * factor);
    p->z = (int)(p->z * factor);
}

/* point_reset() REMOVED */
```

An application compiled against version 1 allocates an 8-byte `struct point`; version 2 reads 12 bytes from it. It passes `factor` in an integer register; version 2 reads a floating-point register. And it calls `point_reset`, which no longer exists, so it fails to load.

### Compile both

Build each as a shared library with debug info (`-g`); libabigail needs DWARF to see types.

```bash
gcc -g -O0 -fPIC -shared -o libshape-baseline.so baseline.c
gcc -g -O0 -fPIC -shared -o libshape-feature.so  feature.c
```

### Compare with abidiff

```bash
abidiff libshape-baseline.so libshape-feature.so; echo "exit code: $?"
```

Actual output (abidiff 2.4.0, gcc 13.3, Ubuntu 24.04):

```text
Functions changes summary: 1 Removed, 2 Changed, 0 Added functions
Variables changes summary: 0 Removed, 0 Changed, 0 Added variable

1 Removed function:

  [D] 'function void point_reset(point*)'    {point_reset}

2 functions with some indirect sub-type change:

  [C] 'function int point_dot(const point*, const point*)' at baseline.c:7:1 has some indirect sub-type changes:
    parameter 1 of type 'const point*' has sub-type changes:
      in pointed to type 'const point':
        in unqualified underlying type 'struct point' at feature.c:2:1:
          type size changed from 64 to 96 (in bits)
          1 data member insertion:
            'int z', at offset 64 (in bits) at feature.c:5:1

  [C] 'function void point_scale(point*, int)' at baseline.c:12:1 has some indirect sub-type changes:
    parameter 2 of type 'int' changed:
      type name changed from 'int' to 'double'
      type size changed from 32 to 64 (in bits)

exit code: 12
```

All three breaks are found: the removed symbol (`[D]`), the struct growing from 64 to 96 bits, and the parameter type change. Note that `point_scale` has the same C symbol name in both versions, so a plain `nm -D` symbol diff would miss that change entirely.

### Reading the exit code

The exit code is what CI acts on. Running variants of the same example gives:

| Comparison | Exit code | Bits set | Meaning |
| --- | --- | --- | --- |
| baseline vs. baseline | 0 | none | ABI identical |
| baseline vs. feature, with `point_reset` kept | 4 | ABI\_CHANGE | Changed; needs human review |
| baseline vs. feature (as above) | 12 | ABI\_CHANGE + INCOMPATIBLE | A removed symbol: a definite break |

### The CI form: compare against a stored baseline

CI should not rebuild the old version. Instead it saves the baseline's ABI once as an abixml text file and diffs new builds against it. This returns the same exit code (12) as the binary-to-binary comparison:

```bash
# once, when version 1 is released
abidw --no-corpus-path --out-file baseline.abi libshape-baseline.so

# on every PR
abidiff baseline.abi libshape-feature.so
```

This is exactly the pattern the PyTorch job uses, with `libtorch_cpu.so` in place of `libshape` and release/merge-base abixml files as baselines.

## Goals and non-goals

**Goals**

1. Block PRs that remove or incompatibly change any symbol or type in the LibTorch stable ABI.
2. Require explicit owner sign-off for any other change to the stable ABI surface (exit bit 4).
3. Publish a non-blocking ABI drift report for the full `libtorch`/`libc10` on `main` and between releases. It never gates a PR or a release, because that surface has no ABI guarantee.
4. Keep added PR wall-clock under 10 minutes and add no new full build.

**Non-goals**

- Making all of `libtorch` ABI-stable. The stable-ABI project explicitly rules this out.
- Windows (MSVC/PDB) and macOS (Mach-O). libabigail reads ELF only; these platforms need a different tool.
- Python-level API or operator schema compatibility, already covered by existing BC checks.
- Replacing runtime tests that load old-version extension wheels against new torch. Those stay; this check is a faster, earlier signal.

## Proposed design

The check has two tiers with different scope, trigger and consequence.

| Tier | Scope | Runs on | Baselines | Failure means |
| --- | --- | --- | --- | --- |
| 1. Stable ABI | `extern "C"` shim symbols (`aoti_torch_*`, `torch_*`) in `libtorch_cpu.so` and `libtorch_cuda.so`; types under `torch/csrc/stable/c`, `torch/csrc/inductor/aoti_torch/c` and `torch/headeronly` | Every PR touching C++ (via path filter), plus `main` | Last release + merge-base nightly | Blocking after rollout |
| 2. Full libtorch (no ABI guarantee) | All exported symbols of `libc10.so`, `libtorch_cpu.so`, `libtorch_cuda.so`, `libc10_cuda.so` | Nightly on `main`; once per release | Previous nightly; previous release | Never fails; informational report only |

&#91;embedded content: Tier 1 check flow · build, snapshot, compare, gate\]

The PR snapshot is compared against both baselines; the merge-base result gates the PR, and every non-zero result posts the report to the PR.

### Debug info without a debug build

libabigail needs DWARF type info, but full `-g` on `libtorch_cpu.so` multiplies its size and build time. The design avoids this:

- **Tier 1:** add `-g` only to the translation units that define the shim (e.g. `torch/csrc/inductor/aoti_torch/shim_common.cpp`, `torch/csrc/stable/c/*.cpp` and generated `c_shim_*.cpp`). A new CMake option, `USE_ABI_DEBUGINFO=ON`, applies `set_source_files_properties(... COMPILE_OPTIONS -g)` to that list. Only the CI build used by this job sets it.
- **Tier 2:** do not rely on `-g1`, since `-g1` has no type info. Instead, use the split-debuginfo (`-gsplit-dwarf` or `objcopy --only-keep-debug`) artifacts from a `RelWithDebInfo` nightly, passed via `--debug-info-dir`. This is a nightly-only cost.
- Run `abidw` with `--fail-no-debug-info` so a build that silently lost its debug info errors instead of passing with a symbols-only comparison.

### Baselines

- **Release baselines** are abixml files produced once per release (`v2.N.0`) by the release workflow, uploaded to the PyTorch artifacts bucket and pinned by SHA-256 in a small manifest checked into the repo (`tools/abi/baselines.json`). Text abixml is compressed and small enough to fetch quickly.
- **Merge-base baselines** come from the `main` job: every commit on `main` uploads its abixml keyed by commit SHA. A PR fetches the one for its merge-base; if missing, it falls back to the newest ancestor that has one.
- Comparing against both answers two questions: "did this PR change the ABI?" (merge-base) and "is `main` still compatible with what users have installed?" (release).

### Suppressions and expected changes

Known-benign changes live in `tools/abi/stable_abi.abignore` (libabigail suppression-spec format). Every entry carries a comment linking the PR and approver. Additions to the stable surface are expected and do not fail the check (`--no-added-syms` for the gate, full report still posted). Additions must still sit behind the right `TORCH_TARGET_VERSION` guard; a lint script cross-checks new shim symbols against the versioned header list.

### Toolchain pinning

The job runs in a dedicated CI image with a pinned libabigail (2.10) and the same compiler as the manywheel build. Baselines record the libabigail and compiler versions; the job warns when they differ from the PR side, since DWARF produced by different compilers can show spurious changes.

## CI integration

The Tier 1 check is a new job in `.github/workflows/pull.yml` that consumes the artifacts of an existing Linux CPU+CUDA build job rather than building again. The YAML below is illustrative; job and input names will follow the conventions of the reusable `_linux-build.yml` workflow at implementation time.

### Build changes

- `_linux-build.yml` gains an input `abi-debuginfo: bool`. When true, it sets `USE_ABI_DEBUGINFO=1` and uploads `libc10.so`, `libtorch_cpu.so`, `libtorch_cuda.so` and the public header tree as an extra artifact.
- One existing PR build (e.g. the gcc CUDA build already in `pull.yml`) turns the input on. Cost: a handful of TUs compiled with `-g`, so seconds of build time and a few MB of extra artifact.

### New job

```yaml
  stable-abi-check:
    name: stable-abi-check
    needs: linux-jammy-cuda-py3-gcc11-build   # build job with abi-debuginfo: true
    if: github.repository_owner == 'pytorch'
    runs-on: linux.4xlarge                     # CPU-only runner
    container: ghcr.io/pytorch/ci-image:abi-check   # pinned libabigail 2.10
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - name: Download build artifacts
        uses: ./.github/actions/download-build-artifacts
        with: { name: abi-debuginfo }
      - name: Fetch baselines
        run: python tools/abi/fetch_baselines.py --release latest --merge-base origin/main -o baselines/
      - name: Run stable ABI check
        run: python tools/abi/check_abi.py --tier stable --build build/lib --baselines baselines/ --report abi-report.md
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with: { name: abi-report, path: abi-report.md }
```

### Check script

`tools/abi/check_abi.py` wraps libabigail so developers can run the same check locally. Its core for one library:

```bash
abidw --fail-no-debug-info --no-corpus-path \
  --headers-dir "$TORCH_INCLUDE" --drop-private-types \
  --out-file pr/libtorch_cpu.abi build/lib/libtorch_cpu.so

abidiff --no-added-syms --leaf-changes-only --impacted-interfaces \
  --suppressions tools/abi/stable_abi.abignore \
  --keep-fn '^(aoti_torch_|torch_)' \
  baselines/merge-base/libtorch_cpu.abi pr/libtorch_cpu.abi
rc=$?
```

The script maps the exit code to a result:

| `abidiff` exit bits | Meaning | Job result |
| --- | --- | --- |
| 0 | No change | Pass |
| 4 | ABI changed, compatibility unclear | Fail unless the PR carries the `abi-change-approved` label |
| 4 + 8 (12) | Incompatible: removed symbol or vtable change | Fail; label alone is not enough (see Policy) |
| 1 or 2 set | Tool or usage error | Fail as infra error, routed to the CI owners |

### Reporting

- The leaf-change report (with impacted interfaces) is written to the job summary and uploaded as an artifact.
- On any non-zero result, the PyTorch bot posts a short PR comment: libraries affected, change count, a link to the full report, and the exact local repro command.
- The release-baseline comparison runs in the same job; its result is reported but only the merge-base comparison gates the PR.

### Nightly and release jobs

- `periodic.yml` (or the nightly binary workflow) runs Tier 2 on `main` and posts the drift report to the HUD.
- When a release is cut, Tier 2 compares it to the previous release; the report is attached to the C++ release notes. It does not gate the release.
- The release workflow runs `abidw` on the final wheels' libraries and uploads the new release baselines, then opens a PR updating `tools/abi/baselines.json`.

### Local use

`python tools/abi/check_abi.py --tier stable --build build/lib` works on any Linux dev box with `abigail-tools` installed and a build made with `USE_ABI_DEBUGINFO=1`.

## Policy

A stable-ABI break never merges silently: every change is either reverted, re-done compatibly, or explicitly approved by the stable-ABI owners.

| Change found | Who decides | Path to merge |
| --- | --- | --- |
| New shim symbol | PR reviewer | Merges; lint confirms it is behind the right `TORCH_TARGET_VERSION` guard |
| Changed type or signature (exit 4) | Stable-ABI owners (new `CODEOWNERS` entry for `tools/abi/`) | Owner adds `abi-change-approved` label, or PR adds a reviewed suppression entry |
| Removed symbol / vtable change (exit 12) | Stable-ABI owners | Not mergeable to `main` while the symbol is inside the supported version window; deprecate first, remove after the window closes, with a suppression entry citing the deprecation |
| Any Tier 2 change (full libtorch) | No one | Never blocks, on `main` or `release/*`: this surface has no ABI guarantee. The report feeds C++ release notes |

Suppression entries are reviewed like code: each has a comment with the PR link, approver and reason. A quarterly cleanup removes entries that no longer match anything.

If the job itself is broken (exit bit 1 or 2, missing baseline), it is marked infra-failure and can be bypassed with the standard CI-flake process. It must not be bypassed for a real ABI diff.

## Rollout plan

Rollout runs in four phases, each gated on an exit criterion rather than a date.

1. **Prototype (offline).** Run `abidw`/`abidiff` by hand on the last two releases and on 20 recent `main` commits. Build the first suppression file.
   - Exit: Tier 1 runtime under 10 minutes; every reported change explained.
2. **Shadow mode on PRs.** Land `USE_ABI_DEBUGINFO`, the scripts, baselines upload on `main`, and the job as non-blocking (`continue-on-error`). Bot comments are off; results go to the HUD only.
   - Exit: 4 weeks with a false-positive rate below 1% of PRs and no missed known break.
3. **Blocking for Tier 1.** Turn on bot comments and make the job required for merge. Announce on dev-discuss with the local repro instructions.
   - Exit: one full release cycle without a reverted stable-ABI break.
4. **Tier 2 drift reports.** Turn on the nightly full-libtorch report and attach a per-release diff to the C++ release notes. It stays informational permanently.

A single revert of the `required` flag returns the project to phase 2 if the job proves noisy.

## Costs, risks and alternatives

### Costs

- **CI compute:** one CPU-only job per C++-touching PR, target under 10 minutes; no extra build. Tier 2 adds one `RelWithDebInfo` build per night.
- **Storage:** one compressed abixml set per `main` commit, expiring after 90 days; release baselines kept forever.
- **People:** a named stable-ABI owner group to review label requests and suppression entries.

### Risks

| Risk | Mitigation |
| --- | --- |
| False positives from compiler or libabigail differences between baseline and PR | Same pinned image for both sides; version mismatch warning; merge-base baseline as the gate |
| `abidw` slow or memory-hungry on `libtorch_cpu.so` | `-g` only on shim TUs for Tier 1; `--keep-fn`, `--drop-private-types` and `-j` |
| Header-only (`torch/headeronly`) changes have no symbols to check | Compile a small probe TU that instantiates each public header-only type and exports it, so its layout appears in DWARF |
| Breaks in inline C++ wrappers (`torch/csrc/stable`) are source, not binary, changes | Covered by existing extension build tests; out of scope for this check |
| Developers bypass via label | Label restricted to `CODEOWNERS` of `tools/abi/`; exit-12 changes cannot be labelled through |
| CUDA-specific symbols differ by CUDA version | Baselines keyed by CUDA version; check one canonical CUDA config |

### Alternatives considered

- **ABI Compliance Checker (abi-compliance-checker / abi-dumper).** Similar capability, but less actively maintained and slower on very large C++ binaries. libabigail is packaged by all major distros and has a stable exit-code contract.
- **Symbol-list diff only (`nm -D` against a checked-in list).** Cheap and catches removals, but misses type layout and signature changes, which are the subtle breaks.
- **Runtime-only testing** (build extensions against old torch, load on new). Kept as a complement, but it only covers APIs the test extensions call and runs late.
- **Header-diff tooling on the C shim.** Catches declaration edits but not changes in transitively used types or build-flag effects; libabigail checks what was actually compiled.

## Open questions

- [ ] Which build job in `pull.yml` should carry `abi-debuginfo`: a CPU-only build (cheaper, misses `libtorch_cuda.so`) or a CUDA build?
- [ ] How many releases back is the stable ABI promised? This sets the deprecation window in the Policy table.
- [ ] Should Tier 1 also run on aarch64 Linux, given ecosystem wheels ship for it?
- [ ] Who are the initial stable-ABI owners for `tools/abi/` in `CODEOWNERS`?
- [ ] Is a separate tool needed for Windows, or is Linux coverage enough given the shim is platform-neutral C?

## Sources

- [LibTorch Stable ABI — PyTorch docs](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
- [LibTorch ABI Stable Plans 2026 — dev-discuss](https://dev-discuss.pytorch.org/t/libtorch-abi-stable-plans-2026/3380)
- [abidiff manual — libabigail](https://sourceware.org/libabigail/manual/abidiff.html)
- [libabigail mirror (2.10 release)](https://forge.sourceware.org/libabigail/libabigail-mirror)
- [libabigail 2.10 Fedora package](https://packages.fedoraproject.org/pkgs/libabigail/libabigail-fedora/fedora-44-updates.html)
