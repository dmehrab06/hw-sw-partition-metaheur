#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.synthetic import (
    attach_to_TaskGraph,
    generate_squeezenet_like_taskgraph,
    save_config_yaml,
    save_taskgraph_dot,
)


def _default_dot_path(nodes: int) -> Path:
    return ROOT / "inputs" / "task_graph_topology" / "synthetic" / f"squeezenet_like_{nodes}.dot"


def _default_pickle_path(nodes: int) -> Path:
    return ROOT / "inputs" / "task_graph_complete" / f"taskgraph-squeezenet_like_{nodes}.pkl"


def _generate_one(args: argparse.Namespace, nodes: int) -> None:
    attrs = generate_squeezenet_like_taskgraph(
        N=nodes,
        seed=args.seed,
        k=args.k,
        l=args.l,
        mu=args.mu,
        A_max=args.a_max,
        template_dot=args.template_dot,
    )

    tg = attach_to_TaskGraph(attrs)
    if hasattr(tg, "area_constraint"):
        tg.area_constraint = float(args.area_constraint)
    attrs["area_constraint"] = float(args.area_constraint)
    attrs["params"]["area-constraint"] = float(args.area_constraint)

    out_dot = Path(args.out_dot) if args.out_dot else _default_dot_path(nodes)
    out_pkl = Path(args.out_pickle) if args.out_pickle else _default_pickle_path(nodes)

    save_taskgraph_dot(attrs, out_dot)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as handle:
        pickle.dump(tg, handle)

    if args.out_yaml:
        save_config_yaml(
            args.out_yaml,
            {
                "graph-file": str(out_dot),
                "taskgraph-pickle": str(out_pkl),
                "area-constraint": float(args.area_constraint),
                "hw-scale-factor": float(args.k),
                "hw-scale-variance": float(args.l),
                "comm-scale-factor": float(args.mu),
                "seed": int(args.seed),
                "template-dot": str(args.template_dot),
                "template-nodes": int(attrs["template_nodes"]),
                "template-edges": int(attrs["template_edges"]),
                "generated-nodes": int(attrs["graph"].number_of_nodes()),
                "generated-edges": int(attrs["graph"].number_of_edges()),
            },
        )

    print(f"template_dot={attrs['template_graph_file']}")
    print(f"template_nodes={attrs['template_nodes']}")
    print(f"template_edges={attrs['template_edges']}")
    print(f"generated_nodes={attrs['graph'].number_of_nodes()}")
    print(f"generated_edges={attrs['graph'].number_of_edges()}")
    print(f"saved_dot={out_dot}")
    print(f"saved_pickle={out_pkl}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a SqueezeNet-like synthetic task graph.")
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        required=True,
        help="One or more requested node counts N.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k", type=float, default=0.1, help="Hardware scale factor.")
    parser.add_argument("--l", type=float, default=0.5, help="Hardware scale variance.")
    parser.add_argument("--mu", type=float, default=1.0, help="Communication scale factor.")
    parser.add_argument("--a-max", type=float, default=100.0, help="Maximum node area cost.")
    parser.add_argument("--area-constraint", type=float, default=0.5, help="Area constraint ratio.")
    parser.add_argument(
        "--template-dot",
        type=str,
        default="inputs/task_graph_topology/soda-benchmark-graphs/pytorch-graphs/squeeze_net_tosa.dot",
        help="Template DOT used to extract the SqueezeNet topology profile.",
    )
    parser.add_argument("--out-dot", type=str, default=None, help="Output DOT file path.")
    parser.add_argument("--out-pickle", type=str, default=None, help="Output TaskGraph pickle path.")
    parser.add_argument("--out-yaml", type=str, default=None, help="Optional metadata YAML path.")
    args = parser.parse_args()

    if len(args.nodes) > 1 and (args.out_dot or args.out_pickle or args.out_yaml):
        parser.error("--out-dot, --out-pickle, and --out-yaml require exactly one node count.")

    for idx, nodes in enumerate(args.nodes):
        if idx > 0:
            print("---")
        _generate_one(args, int(nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
