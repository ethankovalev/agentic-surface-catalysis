# patches

One-shot edits to `src/`, kept as the record of what changed and why.
Each file's docstring explains the bug it fixed, with the evidence.

All of them have already been applied. Every one is idempotent: run from
the repository root, it reports "Already patched" and writes nothing.

    python3 patches/patch_example.py --check
