from pathlib import Path
import sys

p = Path("src/benchmark.py")
if not p.exists():
    sys.exit("ABORT: run from the repository root.")
text = p.read_text()

SCANNER = '''

# Reference leak detection

# Words that mark a number as recalled or looked up rather than computed.
_LEAK_CUES = (
    "sbh10", "reference", "experimental", "experiment", "literature",
    "known", "reported", "published", "expected", "should be", "accepted",
    "benchmark value", "true value", "actual value",
)
_NUM_eV = re.compile(r"(-?\\d+\\.\\d+|-?\\d+)\\s*(?:eV|ev)\\b")
_LEAK_TOL = 0.08          # absolute eV, or fractional, whichever is larger


def _message_text(message) -> str:
    """Flatten a LangChain message into plain text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def scan_for_reference_leak(messages, reference_eV):
    """Find the experimental value in what the agents said to each other.

    The reference table is unreachable from every tool, so the pipeline is
    blind by construction. The model is not. SBH10 is published, these
    barriers are in its training data, and on N2/Ru(0001) the structure agent
    volunteered "the SBH10 reference (~1.84 eV)" unprompted and passed it to
    the simulation agent.

    No amount of code can stop a model recalling a published number, so this
    measures it instead and records it beside the result. A flagged run is not
    blind, and its agreement with experiment means correspondingly less.

    Only numbers sitting near a word like "reference" or "known" are counted.
    A computed barrier that happens to land near the true value is the outcome
    being tested for, not evidence of a leak.

    Runs after the graph returns, so it cannot itself affect the calculation.
    """
    if reference_eV is None:
        return []
    tol = max(_LEAK_TOL, abs(reference_eV) * _LEAK_TOL)
    hits = []
    for i, message in enumerate(messages or []):
        body = _message_text(message)
        if not body:
            continue
        low = body.lower()
        for match in _NUM_eV.finditer(body):
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if abs(value - reference_eV) > tol:
                continue
            a = max(0, match.start() - 120)
            b = min(len(body), match.end() + 60)
            cue = next((c for c in _LEAK_CUES if c in low[a:b]), None)
            if cue is None:
                continue
            hits.append({
                "message_index": i,
                "speaker": getattr(message, "name", None)
                           or type(message).__name__,
                "value_eV": value,
                "cue": cue,
                "snippet": " ".join(body[a:b].split())[:200],
            })
    return hits

'''

anchor = "\ndef run_one(graph"
if "def scan_for_reference_leak" in text:
    print("  = scanner already present")
elif text.count(anchor) != 1:
    sys.exit("ABORT: could not locate run_one definition.")
else:
    text = text.replace(anchor, SCANNER + anchor)
    print("  + scanner")

if "\nimport re\n" in text:
    print("  = re already imported")
else:
    old_imp = "import json\nimport sys"
    if text.count(old_imp) != 1:
        sys.exit("ABORT: import block not found as expected.")
    text = text.replace(old_imp, "import json\nimport re\nimport sys")
    print("  + import re")

old_inv = """        graph.invoke(
            {
                "messages": [("user", task)],"""
new_inv = """        final_state = graph.invoke(
            {
                "messages": [("user", task)],"""
if "final_state = graph.invoke" in text:
    print("  = invoke already captured")
elif text.count(old_inv) != 1:
    sys.exit("ABORT: graph.invoke call not found as expected.")
else:
    text = text.replace(old_inv, new_inv)
    print("  + capture graph state")

old_after = """        computed = store.get("barrier_eV")
    except Exception as exc:"""
new_after = """        computed = store.get("barrier_eV")
        leaks = scan_for_reference_leak(
            (final_state or {}).get("messages"), spec.get("reference_eV"))
    except Exception as exc:"""
if "leaks = scan_for_reference_leak" in text:
    print("  = scan already wired")
elif text.count(old_after) != 1:
    sys.exit("ABORT: post-invoke block not found as expected.")
else:
    text = text.replace(old_after, new_after)
    print("  + run the scan")

old_init = "    computed, error_note = None, None"
new_init = "    computed, error_note, leaks = None, None, []"
if new_init in text:
    print("  = leaks initialised")
elif text.count(old_init) != 1:
    sys.exit("ABORT: initialisation line not found.")
else:
    text = text.replace(old_init, new_init)
    print("  + initialise leaks")

old_ret = '''        "computed_eV": computed,
        "reference_eV": ref,'''
new_ret = '''        "computed_eV": computed,
        "reference_eV": ref,
        "reference_leaked": bool(leaks),
        "reference_leaks": leaks,'''
if '"reference_leaked"' in text:
    print("  = result fields already present")
elif text.count(old_ret) != 1:
    sys.exit("ABORT: result dict not found as expected.")
else:
    text = text.replace(old_ret, new_ret)
    print("  + result fields")

p.write_text(text)
print("\nDone.")
