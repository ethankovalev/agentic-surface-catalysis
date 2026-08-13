"""
A run-scoped scratchpad.

LangGraph tools return strings into the message history, which is fine
for the model to read but useless for the runner, which needs typed
numbers. Rather than parsing prose back out of messages, tools write
structured results here and the runner reads them.

One store per reaction. `reset()` at the start of each run or results
bleed between reactions.

This is deliberately simple. A production version would use LangGraph's
state-injection API; this is easier to debug and easier to understand
if you have not built an agent before.
"""

_RESULTS = {}


def reset(reaction_id: str = None):
    """Clear the store. Call once per reaction, before invoking the graph."""
    global _RESULTS
    _RESULTS = {"reaction_id": reaction_id}


def put(key: str, value):
    """Record a result. Tools call this alongside returning their string."""
    _RESULTS[key] = value


def get(key: str, default=None):
    return _RESULTS.get(key, default)


def snapshot() -> dict:
    """Everything recorded this run. The runner reads this after invoke()."""
    return dict(_RESULTS)


def record_check(name: str, passed: bool, detail: str = ""):
    """Log one validation outcome.

    Kept separate from put() because the exit gate reads only these -
    the model cannot talk its way past a failed check by writing a
    reassuring message.
    """
    checks = _RESULTS.setdefault("validation", {})
    checks[name] = bool(passed)
    details = _RESULTS.setdefault("validation_detail", {})
    details[name] = detail


def validation() -> dict:
    return dict(_RESULTS.get("validation", {}))


def all_checks_passed() -> bool:
    checks = validation()
    return bool(checks) and all(checks.values())
