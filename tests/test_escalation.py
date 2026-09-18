"""Tests for escalation.assess, including real records from the project.

    python tests/test_escalation.py        (from the repo root)

Imports from src/ when run inside the repo, and falls back to a bare
import so the file also works standing alone in a scratch directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.escalation import assess, format_verdict
except ImportError:
    from escalation import assess, format_verdict

passed = failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


print("=== 1. real record: CH4_Ni111_step, the run the gate refused ===")
# from the actual store dump earlier in the project
ch4_step = {
    "saddle": {"first_order_saddle": False,
               "imaginary_modes_meV": [83.88, 10.89]},
    "saddle_connectivity": {"connects": None},
    "endpoint_modes": {"initial": {"is_minimum": True},
                       "final": {"is_minimum": True}},
    "zpe": {"ok": True, "dzpe_eV": -0.186},
    "barrier_zpe_eV": 0.4908,
    "neb": {"reaction_energy_eV": 0.4844},
}
v = assess(ch4_step)
print(format_verdict("CH4_Ni111_step", v))
check("blocked", v["level"] == "BLOCKED", v["level"])
check("no barrier reported", v["barrier_eV"] is None)
check("names the saddle problem",
      any("first order saddle" in b for b in v["blocking"]))
check("names the connectivity problem",
      any("connect" in b for b in v["blocking"]))

print("\n=== 2. real record: H2_Cu111 seeded, the project's best result ===")
h2_cu111 = {
    "saddle": {"first_order_saddle": True, "imaginary_modes_meV": [121.0]},
    "saddle_connectivity": {"connects": True},
    "endpoint_modes": {"initial": {"is_minimum": True},
                       "final": {"is_minimum": True}},
    "zpe": {"ok": True, "dzpe_eV": -0.027},
    "barrier_zpe_eV": 0.657,
    "neb": {"reaction_energy_eV": 0.30},
    "r_b_drift_A": -0.116,
}
# UMA 0.657 against MACE 0.616 on the same reaction
v = assess(h2_cu111, disagreement_eV=abs(0.657 - 0.616))
print(format_verdict("H2_Cu111", v))
check("accepted", v["level"] == "ACCEPTED", v["level"])
check("barrier reported", v["barrier_eV"] == 0.657)
check("agreement noted", any("agree to" in n for n in v["notes"]))

print("\n=== 3. real record: N2_Ru0001_terrace, worst disagreement ===")
n2_terrace = {
    "saddle": {"first_order_saddle": True, "imaginary_modes_meV": [95.0]},
    "saddle_connectivity": {"connects": True},
    "endpoint_modes": {"initial": {"is_minimum": True},
                       "final": {"is_minimum": True}},
    "zpe": {"ok": True, "dzpe_eV": -0.032},
    "barrier_zpe_eV": 1.090,
    "neb": {"reaction_energy_eV": 0.40},
    "r_b_drift_A": -0.006,
}
# UMA 1.339 against MACE 1.113 at reference geometry: 0.226 eV apart
v = assess(n2_terrace, disagreement_eV=0.226)
print(format_verdict("N2_Ru0001_terrace", v))
check("flagged, not blocked", v["level"] in ("CAUTION", "REVIEW"), v["level"])
check("barrier still reported", v["barrier_eV"] == 1.090)
check("disagreement raised", any("disagree by" in r for r in v["review"]))
check("threshold honesty stated",
      any("NOT a validated threshold" in r for r in v["review"]))

print("\n=== 4. physics blocks are independent of the advisory signal ===")
# perfect agreement must NOT rescue a run that failed physics
v = assess(ch4_step, disagreement_eV=0.001)
check("still blocked despite perfect agreement", v["level"] == "BLOCKED",
      v["level"])

print("\n=== 5. endpoint that is not a minimum ===")
bad_endpoint = dict(h2_cu111)
bad_endpoint["endpoint_modes"] = {"initial": {"is_minimum": True},
                                  "final": {"is_minimum": False}}
v = assess(bad_endpoint)
check("blocked", v["level"] == "BLOCKED", v["level"])
check("names which endpoint",
      any("final endpoint" in b for b in v["blocking"]))

print("\n=== 6. barrier below the model's own resolution ===")
tiny = {
    "saddle": {"first_order_saddle": True, "imaginary_modes_meV": [60.0]},
    "saddle_connectivity": {"connects": True},
    "endpoint_modes": {"initial": {"is_minimum": True},
                       "final": {"is_minimum": True}},
    "zpe": {"ok": True},
    "barrier_zpe_eV": 0.012,
    "neb": {"reaction_energy_eV": -0.2},
}
v = assess(tiny)
check("flagged for review", v["level"] in ("CAUTION", "REVIEW"), v["level"])
check("names the resolution floor",
      any("resolution" in r for r in v["review"]))

print("\n=== 7. endothermic barrier below its own reaction energy ===")
impossible = dict(h2_cu111)
impossible["barrier_zpe_eV"] = 0.20
impossible["neb"] = {"reaction_energy_eV": 0.50}
v = assess(impossible)
check("flagged", v["level"] in ("CAUTION", "REVIEW"), v["level"])
check("names the impossibility",
      any("not physically possible" in r for r in v["review"]))

print("\n=== 8. malformed record must not crash ===")
for bad in ({}, {"saddle": None}, {"saddle": {}, "zpe": None},
            {"barrier_zpe_eV": "not a number"}):
    try:
        v = assess(bad)
        check(f"handled {str(bad)[:34]}", v["level"] == "BLOCKED", v["level"])
    except Exception as exc:
        check(f"handled {str(bad)[:34]}", False, f"raised {type(exc).__name__}")

print("\n=== 9. no second model run: absence is stated, not hidden ===")
v = assess(h2_cu111)
check("absence noted",
      any("no second in-domain model" in n for n in v["notes"]))
check("still accepted on physics alone", v["level"] == "ACCEPTED", v["level"])

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
