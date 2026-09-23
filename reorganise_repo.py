"""
Reorganise the repository root into folders.

    python3 reorganise_repo.py --dry-run     show what would move
    python3 reorganise_repo.py               do it

Run from the repo root, on the Mac, with a clean working tree. Uses
git mv, so every file keeps its history.

WHAT MOVES
----------
  patches/   every patch_*.py and wire_day1.py. One-shot edits, already
             applied, kept as the record of what changed and why. Each is
             idempotent: run from the repo root, it reports "Already
             patched" and writes nothing.
  analysis/  analyse_disagreement.py, plot_seeded_results.py,
             escalate.py, results_seeded_summary.md,
             seeded_d3_comparison.png
  scripts/   probe_robust_saddle.py, alongside the other runners

WHAT STAYS AT THE ROOT, AND WHY
-------------------------------
  config.py       imported by 15 files as "import config"
  invoke.py       the entry point every document tells you to run
  build_seeds.py  referenced by name in error messages and CLAUDE.md
  README.md, CLAUDE.md, NOTES.md, LICENSE, requirements*.txt

WHY THIS IS SAFE
----------------
Nothing imports any moved file. The patch scripts address src/tools.py
relative to the current directory, so from the repo root
"python3 patches/patch_x.py" behaves exactly as before. The one moved
file that computes paths from its own location, probe_robust_saddle.py,
has those lines rewritten here. README references to moved files are
rewritten too. Afterwards every Python file is compiled and every
document is searched for a reference that still points at the old
location.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()

MOVES = {}
for f in sorted(ROOT.glob("patch_*.py")):
    MOVES[f.name] = "patches"
MOVES["wire_day1.py"] = "patches"
for name in ("analyse_disagreement.py", "plot_seeded_results.py",
             "escalate.py", "results_seeded_summary.md",
             "seeded_d3_comparison.png"):
    MOVES[name] = "analysis"
MOVES["probe_robust_saddle.py"] = "scripts"

PROBE_FIX_OLD = '''sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))'''
PROBE_FIX_NEW = '''sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))'''

PATCHES_README = """# patches

One-shot edits to `src/`, kept as the record of what changed and why.
Each file's docstring explains the bug it fixed, with the evidence.

All of them have already been applied. Every one is idempotent: run from
the repository root, it reports "Already patched" and writes nothing.

    python3 patches/patch_example.py --check
"""


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def main():
    dry = "--dry-run" in sys.argv
    if not (ROOT / "config.py").exists() or not (ROOT / "src").is_dir():
        print("FAILED: run this from the repository root.")
        return 1
    if git("status", "--porcelain").stdout.strip() and not dry:
        print("FAILED: the working tree has uncommitted changes. Commit or "
              "stash them first, so this move is one clean commit.")
        return 1

    present = {name: d for name, d in MOVES.items() if (ROOT / name).exists()}
    missing = sorted(set(MOVES) - set(present))
    print(f"{len(present)} files to move:")
    for name, d in sorted(present.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"  {name:36s} -> {d}/")
    if missing:
        print(f"not found, skipped: {', '.join(missing)}")
    if dry:
        print("\nDry run. Nothing moved.")
        return 0

    for d in set(present.values()):
        (ROOT / d).mkdir(exist_ok=True)
    for name, d in present.items():
        r = git("mv", name, f"{d}/{name}")
        if r.returncode != 0:
            print(f"FAILED on {name}: {r.stderr.strip()}")
            return 1

    probe = ROOT / "scripts" / "probe_robust_saddle.py"
    if probe.exists():
        text = probe.read_text()
        if text.count(PROBE_FIX_OLD) == 1:
            probe.write_text(text.replace(PROBE_FIX_OLD, PROBE_FIX_NEW, 1)
                             .replace("python probe_robust_saddle.py",
                                      "python scripts/probe_robust_saddle.py"))
            print("fixed the import paths in scripts/probe_robust_saddle.py")

    # The plot script's docstring says it writes the figure "next to this
    # script", but its default was a bare filename, i.e. the current
    # directory. After the move that would put the figure back at the root.
    plot = ROOT / "analysis" / "plot_seeded_results.py"
    if plot.exists():
        text = plot.read_text()
        old_default = 'ap.add_argument("--out", default="seeded_d3_comparison.png")'
        new_default = ('ap.add_argument("--out", default=str('
                       'Path(__file__).resolve().parent / "seeded_d3_comparison.png"))')
        if text.count(old_default) == 1:
            plot.write_text(text.replace(old_default, new_default, 1))
            print("plot script now writes next to itself, in analysis/")

    readme = ROOT / "patches" / "README.md"
    if not readme.exists():
        readme.write_text(PATCHES_README)
        git("add", "patches/README.md")

    docs = [ROOT / "README.md", ROOT / "CLAUDE.md", ROOT / "NOTES.md"]
    for doc in docs:
        if not doc.exists():
            continue
        text = doc.read_text()
        before = text
        for name, d in present.items():
            text = re.sub(rf"(?<![\w/]){re.escape(name)}", f"{d}/{name}", text)
        if text != before:
            doc.write_text(text)
            print(f"updated references in {doc.name}")

    print("\nverifying...")
    bad = False
    for py in ROOT.rglob("*.py"):
        if ".venv" in py.parts:
            continue
        r = subprocess.run([sys.executable, "-m", "py_compile", str(py)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  does not compile: {py}")
            bad = True
    for doc in docs:
        if not doc.exists():
            continue
        text = doc.read_text()
        for name, d in present.items():
            for m in re.finditer(rf"(?<![\w/]){re.escape(name)}", text):
                print(f"  stale reference to {name} in {doc.name}")
                bad = True
    if bad:
        print("\nProblems above. Nothing has been committed; inspect with "
              "git status, or undo everything with: git reset --hard")
        return 1
    print("  every Python file compiles, no stale references")
    print("\nReady to commit:\n  git add -A && git commit -m \"Reorganise "
          "root into patches/, analysis/, scripts/\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
