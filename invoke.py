"""
Entry point.

  python invoke.py --single H2_Cu111    one reaction, verbose
  python invoke.py --all                the whole benchmark
  python invoke.py --task "..."         an ad-hoc task, unscored

Start with --single. Get one reaction working end to end before running
the set - if the agent gets H2 on Cu(111) badly wrong, you want to know
on day one, not after ten runs.
"""

import argparse
import json
import sys

import config
from src import store
from src.benchmark import SBH10, run_benchmark, run_one
from src.graph import create_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", metavar="REACTION_ID",
                        help=f"one of: {', '.join(SBH10)}")
    parser.add_argument("--all", action="store_true",
                        help="run the full SBH10 set")
    parser.add_argument("--task", metavar="TEXT",
                        help="ad-hoc natural language task, not scored")
    args = parser.parse_args()

    if not any([args.single, args.all, args.task]):
        parser.print_help()
        return

    config.check_checkpoint()
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    # Initialise torch.det's lazy LAPACK backend on the MAIN thread.
    # LangGraph runs tools in a thread pool, and this lazy init is not
    # thread-safe - UMA calls torch.det on the cell matrix during forward,
    # which raises "lazy wrapper should be called at most once" if it first
    # happens inside a worker thread.
    import torch
    if torch.cuda.is_available():
        torch.det(torch.eye(3, device="cuda"))
    torch.det(torch.eye(3))

    graph = create_graph()

    if args.task:
        store.reset("adhoc")
        graph.invoke(
            {"messages": [("user", args.task)], "next": "",
             "reaction_id": "adhoc", "attempts": 0},
            {"configurable": {"thread_id": "adhoc"},
             "recursion_limit": config.RECURSION_LIMIT},
        )
        print("\n" + json.dumps(store.snapshot(), indent=2, default=str))
        return

    if args.single:
        if args.single not in SBH10:
            print(f"Unknown reaction. Known: {', '.join(SBH10)}")
            sys.exit(1)
        result = run_one(graph, args.single, SBH10[args.single])
        print("\n" + json.dumps(result, indent=2, default=str))
        return

    run_benchmark(graph)


if __name__ == "__main__":
    main()
