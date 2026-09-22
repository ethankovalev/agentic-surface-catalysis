"""
Read an agent grid log and say, per reaction, what the agent actually did.

    python scripts/diagnose_run.py /workspace/agentic-surface-catalysis/logs/grid_v2_test.log

No GPU, no API, no project imports. Reads a text file.

WHY
---
A validation record says which checks failed. It does not say why the
agent stopped. Three different causes leave the same record:

  - the agent never called the recovery tool,
  - it called it and the call failed,
  - it ran out of supervisor turns (config.MAX_ATTEMPTS).

Each needs a different fix. This script tells them apart from the log,
which already contains every tool call in order.

It also flags sentences in the agent's own prose that cite things no
tool can produce, such as published benchmarks or literature trends.
Nothing in SIMULATION_TOOLS or VALIDATION_TOOLS retrieves literature, so
any such claim was invented. On 2026-09-22 a validated N2/Ru(0001)
report said its error was "consistent with published benchmarks for
this model class". The number was right; the sentence was not
supportable.
"""

import re
import sys
from pathlib import Path

HEADER = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)\s+(\S+)\s+d3=(\S+)\s+->\s+running")
TOOL = re.compile(r"^Name:\s*(\S+)")
GATE = re.compile(r"^\[gate\].*")
RESULT = re.compile(r"^\s+->\s+(.*)")
CHECK = re.compile(r"^(PASS|FAIL)\s+(\w+):\s*(.*)")

RECOVERY = {"refine_saddle", "refine_saddle_robust"}

# Phrases that point outside the tools. Deliberately narrow: each one has
# to be a claim about external knowledge, not ordinary wording.
UNSOURCED = [
    "published", "literature", "et al", "known systematic", "known tendency",
    "benchmarks for this model", "is well known", "widely reported",
    "consistent with previous", "previous studies", "reported values",
]


def split_sections(lines):
    """Yield (header_text, section_lines) per reaction."""
    current, body = None, []
    for line in lines:
        if HEADER.match(line):
            if current is not None:
                yield current, body
            current, body = line.strip(), []
        elif current is not None:
            body.append(line.rstrip("\n"))
    if current is not None:
        yield current, body


def analyse(header, body):
    tools = [m.group(1) for line in body if (m := TOOL.match(line))]
    gates = [line for line in body if GATE.match(line)]
    results = [m.group(1) for line in body if (m := RESULT.match(line))]
    checks = [(m.group(1), m.group(2), m.group(3))
              for line in body if (m := CHECK.match(line))]

    # when did path_resolved first fail, and was recovery called afterwards
    first_path_fail = None
    for i, line in enumerate(body):
        if line.startswith("FAIL") and "path_resolved" in line:
            first_path_fail = i
            break
    recovered_after = None
    if first_path_fail is not None:
        later_tools = [m.group(1) for line in body[first_path_fail:]
                       if (m := TOOL.match(line))]
        recovered_after = any(t in RECOVERY for t in later_tools)

    claims = []
    in_ai = False
    for line in body:
        if "Ai Message" in line:
            in_ai = True
            continue
        if "Tool Message" in line or "Human Message" in line:
            in_ai = False
            continue
        if in_ai:
            low = line.lower()
            for phrase in UNSOURCED:
                if phrase in low:
                    claims.append(line.strip()[:160])
                    break

    return {
        "tools": tools, "gates": gates, "results": results,
        "checks": checks, "first_path_fail": first_path_fail,
        "recovered_after": recovered_after, "claims": claims,
    }


def verdict(a):
    """One line: why the run ended the way it did."""
    gate = " ".join(a["gates"])
    called = set(a["tools"])
    if "attempt limit" in gate:
        if a["first_path_fail"] is not None and not a["recovered_after"]:
            return ("RAN OUT OF TURNS before recovering from a path_resolved "
                    "failure. The recovery rule was not the problem; the "
                    "turn budget was.")
        return "RAN OUT OF TURNS (config.MAX_ATTEMPTS)."
    if a["first_path_fail"] is not None and not a["recovered_after"]:
        return ("IGNORED A RULE: path_resolved failed and no refine_saddle or "
                "refine_saddle_robust call followed, with turns remaining.")
    if called & RECOVERY and "compute_zpe_correction" not in called:
        return ("RECOVERY CALLED BUT ZPE NEVER RAN: check whether the refined "
                "saddle was confirmed first order and connected.")
    if "all required checks passed" in gate:
        return "VALIDATED."
    return "UNCLEAR: read the section by hand."


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/diagnose_run.py <logfile>")
    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"{path} not found")
    lines = path.read_text(errors="replace").splitlines()

    sections = list(split_sections(lines))
    if not sections:
        raise SystemExit("no reaction headers found. Is this a run_grid.py log?")

    for header, body in sections:
        a = analyse(header, body)
        print("=" * 72)
        print(header)
        print("=" * 72)
        seq, last, n = [], None, 0
        for t in a["tools"] + [None]:
            if t == last:
                n += 1
                continue
            if last is not None:
                seq.append(f"{last} x{n}" if n > 1 else last)
            last, n = t, 1
        print("tool sequence:")
        for step in seq:
            print(f"  {step}")
        print(f"\ntool calls: {len(a['tools'])}")
        print(f"recovery called: {bool(set(a['tools']) & RECOVERY)}")
        print(f"compute_zpe_correction called: "
              f"{'compute_zpe_correction' in a['tools']}")
        if a["first_path_fail"] is not None:
            print(f"path_resolved failed; recovery afterwards: "
                  f"{a['recovered_after']}")
        failed = sorted({name for s, name, _ in a["checks"] if s == "FAIL"})
        if failed:
            print(f"checks that failed at some point: {', '.join(failed)}")
        for g in a["gates"]:
            print(g)
        for r in a["results"]:
            print(f"result: {r}")
        print(f"\nVERDICT: {verdict(a)}")
        if a["claims"]:
            print("\nUNSOURCED CLAIMS in the agent's prose (no tool can "
                  "produce these):")
            for c in a["claims"]:
                print(f"  - {c}")
        print()


if __name__ == "__main__":
    main()
