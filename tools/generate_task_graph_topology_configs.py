#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import math
import shutil
from pathlib import Path

import pydot
import yaml
import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_ROOT = ROOT / "inputs" / "task_graph_topology"
OUTPUTS_ROOT = ROOT / "outputs"
DEFAULT_BASE_CONFIG = ROOT / "configs" / "config_mkspan_default_gnn.yaml"
DEFAULT_CONFIG_ROOT = ROOT / "inputs" / "task_graph_topology_config"

DEFAULT_AREA = 0.5
DEFAULT_PILOT_SEEDS = [42, 43, 44]
DEFAULT_FULL_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
DEFAULT_SQUEEZE_AREAS = [0.1, 0.3, 0.5, 0.7]
DEFAULT_FACTORS = {
    "hw-scale-factor": 0.1,
    "hw-scale-variance": 0.5,
    "comm-scale-factor": 1.0,
}
SPECIAL_TASKGRAPH_PICKLES = {
    "paper_fig3_11node": (
        Path("inputs")
        / "task_graph_complete"
        / "taskgraph-paper_fig3_11node-instance-config-config_fig3_taskgraph_gnn.pkl"
    ).as_posix(),
}
ALL_GNN_METHODS = [
    "random",
    "greedy",
    "diff_gnn",
    "diff_gnn_order",
    "gcps",
    "pso",
    "dbpso",
    "clpso",
    "ccpso",
    "esa",
    "shade",
    "jade",
    "gl25",
]
RUNTIME_METHOD_BLOCK_KEYS = [
    "random",
    "pso",
    "dbpso",
    "clpso",
    "ccpso",
    "gl25",
    "shade",
    "jade",
    "esa",
    "gcps",
    "non_diffgnn",
    "diffgnn",
    "diffgnn_order",
    "gcon",
]


def _repo_rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_topology(dot_path: Path) -> nx.DiGraph:
    pgraphs = pydot.graph_from_dot_file(str(dot_path))
    if not pgraphs:
        raise ValueError(f"Could not load DOT file: {dot_path}")

    graph = nx.DiGraph()

    def _walk(pg) -> None:
        for node in pg.get_nodes():
            name = node.get_name().strip('"')
            if name in {"node", "graph", "edge"}:
                continue
            graph.add_node(name)
        for edge in pg.get_edges():
            src = edge.get_source().strip('"')
            dst = edge.get_destination().strip('"')
            graph.add_edge(src, dst)
        for subgraph in pg.get_subgraphs():
            _walk(subgraph)

    _walk(pgraphs[0])
    return graph


def _collect_topologies() -> list[dict]:
    rows = []
    for dot_path in sorted(TOPOLOGY_ROOT.rglob("*.dot")):
        graph_name = dot_path.stem
        if dot_path.parent.name == "synthetic" and graph_name.startswith("squeezenet_like_"):
            try:
                nodes = int(graph_name.rsplit("_", 1)[1])
            except Exception:
                graph = _load_topology(dot_path)
                nodes = int(graph.number_of_nodes())
                edges = int(graph.number_of_edges())
            else:
                edges = int(round((234 / 179) * nodes))
        else:
            graph = _load_topology(dot_path)
            nodes = int(graph.number_of_nodes())
            edges = int(graph.number_of_edges())
        rows.append(
            {
                "graph_path": dot_path,
                "graph_rel": _repo_rel(dot_path),
                "graph_name": graph_name,
                "family": dot_path.parent.name,
                "nodes": nodes,
                "edges": edges,
            }
        )
    return rows


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _diff_gnn_order_report(row: dict) -> float | None:
    static = _safe_float(row.get("diff_gnn_order_lssp_makespan"))
    learned = _safe_float(row.get("diff_gnn_order_lssp_swprio_makespan"))
    values = [v for v in (static, learned) if v is not None]
    if not values:
        return None
    return min(values)


def _discover_best_params(area_target: float) -> dict[str, dict]:
    wanted = [
        "GraphName",
        "Area_Percentage",
        "HW_Scale_Factor",
        "HW_Scale_Var",
        "Comm_Scale_Var",
        "Config",
        "diff_gnn_order_lssp_makespan",
        "diff_gnn_order_lssp_swprio_makespan",
    ]
    best: dict[str, dict] = {}

    csv_candidates = sorted(OUTPUTS_ROOT.rglob("*result-summary-soda-graphs-config.csv"))
    for csv_path in csv_candidates:
        try:
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "GraphName" not in reader.fieldnames:
                    continue
                if (
                    "diff_gnn_order_lssp_makespan" not in reader.fieldnames
                    and "diff_gnn_order_lssp_swprio_makespan" not in reader.fieldnames
                ):
                    continue
                for raw_row in reader:
                    row = {key: raw_row.get(key, "") for key in wanted}
                    graph_name = Path(str(row.get("GraphName", ""))).stem
                    if not graph_name:
                        continue

                    report = _diff_gnn_order_report(row)
                    if report is None:
                        continue

                    area_value = _safe_float(row.get("Area_Percentage"))
                    quality = 0 if area_value is not None and abs(area_value - float(area_target)) <= 1e-9 else 1

                    candidate = {
                        "graph_name": graph_name,
                        "quality": quality,
                        "report": report,
                        "hw-scale-factor": _safe_float(row.get("HW_Scale_Factor")) or DEFAULT_FACTORS["hw-scale-factor"],
                        "hw-scale-variance": _safe_float(row.get("HW_Scale_Var")) or DEFAULT_FACTORS["hw-scale-variance"],
                        "comm-scale-factor": _safe_float(row.get("Comm_Scale_Var")) or DEFAULT_FACTORS["comm-scale-factor"],
                        "source_csv": _repo_rel(csv_path),
                        "source_config": str(row.get("Config", "")),
                    }

                    current = best.get(graph_name)
                    if current is None or (
                        candidate["quality"],
                        candidate["report"],
                        candidate["source_csv"],
                    ) < (
                        current["quality"],
                        current["report"],
                        current["source_csv"],
                    ):
                        best[graph_name] = candidate
        except Exception:
            continue

    return best


def _selected_factors(graph_name: str, best_params: dict[str, dict]) -> tuple[dict, str, str]:
    if graph_name in best_params:
        row = best_params[graph_name]
        return (
            {
                "hw-scale-factor": row["hw-scale-factor"],
                "hw-scale-variance": row["hw-scale-variance"],
                "comm-scale-factor": row["comm-scale-factor"],
            },
            "graph_best",
            row.get("source_csv", ""),
        )

    if graph_name.startswith("squeezenet_like_") and "squeeze_net_tosa" in best_params:
        row = best_params["squeeze_net_tosa"]
        return (
            {
                "hw-scale-factor": row["hw-scale-factor"],
                "hw-scale-variance": row["hw-scale-variance"],
                "comm-scale-factor": row["comm-scale-factor"],
            },
            "squeeze_fallback",
            row.get("source_csv", ""),
        )

    return dict(DEFAULT_FACTORS), "default", ""


def _suite_prefix(suite_name: str, profile: str) -> str:
    return f"{suite_name}_{profile}"


def _format_num(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _taskgraph_pickle_path(graph_name: str, area: float, factors: dict, seed: int) -> str:
    if graph_name in SPECIAL_TASKGRAPH_PICKLES:
        return SPECIAL_TASKGRAPH_PICKLES[graph_name]
    filename = (
        f"taskgraph-{graph_name}_"
        f"area-{_format_num(area)}_"
        f"hwscale-{_format_num(factors['hw-scale-factor'], 2)}_"
        f"hwvar-{_format_num(factors['hw-scale-variance'], 2)}_"
        f"comm-{_format_num(factors['comm-scale-factor'], 2)}_"
        f"seed-{seed}.pkl"
    )
    return (Path("inputs") / "task_graph_complete" / filename).as_posix()


def _apply_common_overrides(
    cfg: dict,
    graph_rel: str,
    graph_name: str,
    area: float,
    factors: dict,
    seed: int,
    suite_name: str,
    profile: str,
) -> dict:
    suite_output_root = Path("outputs") / suite_name / profile
    prefix = _suite_prefix(suite_name, profile)

    cfg["graph-file"] = graph_rel
    cfg["area-constraint"] = float(area)
    cfg["hw-scale-factor"] = float(factors["hw-scale-factor"])
    cfg["hw-scale-variance"] = float(factors["hw-scale-variance"])
    cfg["comm-scale-factor"] = float(factors["comm-scale-factor"])
    cfg["seed"] = int(seed)
    cfg["solver-tool"] = "cvxpy"
    cfg["taskgraph-dir"] = "inputs/task_graph_complete"
    cfg["taskgraph-pickle"] = _taskgraph_pickle_path(graph_name, area, factors, seed)
    cfg["solution-dir"] = (suite_output_root / "partitions").as_posix()
    cfg["output-dir"] = suite_output_root.as_posix()
    cfg["result-file-prefix"] = prefix
    cfg["opt-cost-type"] = "makespan"
    cfg["methods"] = list(ALL_GNN_METHODS)

    vis_cfg = dict(cfg.get("visualization", {}))
    vis_cfg["enabled"] = False
    vis_cfg["include_input_graph"] = False
    vis_cfg["include_output_schedule"] = False
    vis_cfg["out_dir"] = (suite_output_root / "visualizations").as_posix()
    vis_cfg["input_dir"] = (suite_output_root / "visualizations" / "input").as_posix()
    vis_cfg["schedule_dir"] = (suite_output_root / "visualizations" / "schedule").as_posix()
    cfg["visualization"] = vis_cfg

    cfg.setdefault("random", {})
    cfg.setdefault("pso", {})
    cfg.setdefault("dbpso", {})
    cfg.setdefault("clpso", {})
    cfg.setdefault("ccpso", {})
    cfg.setdefault("gl25", {})
    cfg.setdefault("shade", {})
    cfg.setdefault("jade", {})
    cfg.setdefault("esa", {})
    cfg.setdefault("gcps", {})
    cfg.setdefault("diffgnn", {})
    cfg.setdefault("diffgnn_order", {})

    diff_cfg = dict(cfg["diffgnn"])
    diff_cfg.setdefault("feature_profile", "default_plus_paper")
    diff_cfg.setdefault("edge_weight_mode", "paper2_cosine")
    diff_cfg.setdefault("edge_weight_learner", "mlp")
    diff_cfg.setdefault("sinkhorn_iters", 12)
    diff_cfg.setdefault("pairwise_temp", 0.35)
    diff_cfg.setdefault("dropout", 0.2)
    diff_cfg.setdefault("num_layers", 3)
    cfg["diffgnn"] = diff_cfg

    order_cfg = dict(cfg["diffgnn_order"])
    order_cfg["feature_profile"] = "default_plus_paper"
    order_cfg["edge_weight_mode"] = "paper2_cosine"
    order_cfg["edge_weight_learner"] = "mlp"
    order_cfg["sinkhorn_iters"] = 8
    order_cfg["order_refine_steps"] = 2
    order_cfg["pairwise_mode"] = "rank_sigmoid"
    order_cfg["pairwise_temp"] = 0.35
    order_cfg["gumbel_noise"] = False
    order_cfg["use_hw_ordering"] = False
    order_cfg["dropout"] = 0.5
    order_cfg.setdefault("hidden_dim", 64)
    order_cfg.setdefault("num_layers", 2)
    order_cfg["checkpoint_eval_when_final_only"] = False
    order_cfg["early_stop_enabled"] = True
    order_cfg["early_stop_min_epochs"] = 80
    order_cfg["early_stop_patience"] = 3
    order_cfg["early_stop_min_delta"] = 1.0e-3
    order_post_cfg = dict(order_cfg.get("postprocess", {}))
    order_post_cfg["max_iters"] = 24
    order_cfg["postprocess"] = order_post_cfg
    cfg["diffgnn_order"] = order_cfg

    return cfg


def _strip_runtime_method_configs(cfg: dict) -> dict:
    for key in RUNTIME_METHOD_BLOCK_KEYS:
        cfg.pop(key, None)
    return cfg


def _apply_pilot_profile(cfg: dict, nodes: int) -> dict:
    random_samples = 50
    pso_particles = 10
    pso_iters = 4
    single_point_iters = 60
    single_point_pop = 12
    diff_iter = 10
    diff_hidden = 64
    gcps_pretrain = 10
    gcps_iter = 40
    post_iters = 20

    if nodes >= 1000:
        random_samples = 20
        pso_particles = 6
        pso_iters = 3
        single_point_iters = 20
        single_point_pop = 8
        diff_iter = 3
        diff_hidden = 32
        gcps_pretrain = 4
        gcps_iter = 12
        post_iters = 8
    elif nodes >= 200:
        random_samples = 30
        pso_particles = 8
        pso_iters = 3
        single_point_iters = 30
        single_point_pop = 10
        diff_iter = 6
        diff_hidden = 48
        gcps_pretrain = 6
        gcps_iter = 20
        post_iters = 12

    cfg["random"]["num_samples"] = random_samples
    cfg["pso"].update({"n_particles": pso_particles, "iterations": pso_iters, "verbose": 0})
    cfg["dbpso"].update({"n_particles": pso_particles, "iterations": pso_iters, "verbose": 0})
    cfg["clpso"].update({"n_individuals": single_point_pop, "iterations": 15, "verbose": 0})
    cfg["ccpso"].update({"n_individuals": single_point_pop, "iterations": 15, "verbose": 0})
    cfg["gl25"].update({"iter": single_point_iters, "n_pop": single_point_pop, "verbose": 0})
    cfg["shade"].update({"iter": single_point_iters, "n_individuals": single_point_pop, "verbose": 0})
    cfg["jade"].update({"iter": single_point_iters, "n_individuals": single_point_pop, "verbose": 0})
    cfg["esa"].update({"iter": single_point_iters, "verbose": 0})
    cfg["gcps"].update(
        {
            "iter": gcps_iter,
            "pretrain_iter": gcps_pretrain,
            "schedule_eval": "lssp",
            "quick_search": True,
            "device": "cpu",
            "verbose": 0,
            "lr": 1e-3,
            "dropout": 0.2,
            "hidden_dim_1": 10,
            "hidden_dim_2": 5,
            "schedule_skip": 5,
            "sigma": 0.3,
        }
    )

    for key in ("diffgnn", "diffgnn_order"):
        cfg[key]["iter"] = diff_iter
        cfg[key]["hidden_dim"] = diff_hidden
        cfg[key]["num_layers"] = 3
        cfg[key]["dropout"] = 0.2
        cfg[key]["device"] = "cpu"
        cfg[key]["speed_patch"] = True
        cfg[key]["hard_eval_every"] = min(5, diff_iter)
        cfg[key]["hard_eval_only_final"] = True
        cfg[key]["selection_metric"] = "queue"
        cfg[key]["selection_metric_train"] = "queue"
        cfg[key]["selection_metric_final"] = "queue"
        cfg[key]["final_legacy_lp_if_mip"] = True
        cfg[key]["reuse_trained_final_cost"] = True
        cfg[key].setdefault("postprocess", {})
        cfg[key]["postprocess"].update(
            {
                "mode": "hybrid",
                "during_train": False,
                "eval_mode": "lssp",
                "max_iters": post_iters,
                "enable_area_fill": True,
                "fill_allow_worsen": 0.0,
                "enable_swap": True,
                "dls_steps": 2,
                "dls_flip_eta": 0.35,
                "dls_swap_eta": 0.18,
                "dls_score_temp": 0.70,
                "dls_comm_coeff": 0.02,
                "dls_area_proj_iters": 4,
                "dls_area_proj_strength": 6.0,
                "dls_fill_decode": True,
            }
        )

    cfg["diffgnn"].update(
        {
            "model": "default",
            "edge_mlp_hidden_dim": 16,
            "edge_weight_min_scale": 0.5,
            "edge_weight_max_scale": 1.5,
            "decode_speedup_weight": 0.45,
            "entropy_coeff": 0.0,
            "usage_balance_coeff": 0.0,
            "partition_cost_coeff": 0.0833333333,
        }
    )
    cfg["diffgnn_order"].update(
        {
            "fast_mode": True,
            "edge_mlp_hidden_dim": 16,
            "edge_weight_min_scale": 0.5,
            "edge_weight_max_scale": 1.5,
            "sinkhorn_iters": 8,
            "order_decode_weight": 0.45,
            "entropy_coeff": 0.0,
            "usage_balance_coeff": 0.0,
            "partition_cost_coeff": 0.0833333333,
            "perm_reg_coeff": 0.0,
            "perm_entropy_coeff": 0.0,
        }
    )

    return cfg


def _prepare_config(
    base_cfg: dict,
    topo_row: dict,
    area: float,
    factors: dict,
    seed: int,
    suite_name: str,
    profile: str,
    strip_method_configs: bool = False,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg = _apply_common_overrides(
        cfg=cfg,
        graph_rel=topo_row["graph_rel"],
        graph_name=topo_row["graph_name"],
        area=area,
        factors=factors,
        seed=seed,
        suite_name=suite_name,
        profile=profile,
    )
    if profile == "pilot":
        cfg = _apply_pilot_profile(cfg, topo_row["nodes"])

    ccpso_cfg = dict(cfg.get("ccpso", {}))
    raw_group_sizes = list(ccpso_cfg.get("group_sizes", [5, 10, 20]))
    valid_group_sizes = [int(size) for size in raw_group_sizes if int(size) <= int(topo_row["nodes"])]
    if not valid_group_sizes:
        valid_group_sizes = [max(1, min(int(topo_row["nodes"]), 5))]
    ccpso_cfg["group_sizes"] = valid_group_sizes
    cfg["ccpso"] = ccpso_cfg
    if strip_method_configs:
        cfg = _strip_runtime_method_configs(cfg)
    else:
        # diff_gnn and diff_gnn_order now rely on runtime Python defaults unless
        # a user adds an explicit block manually to a specific YAML afterwards.
        cfg.pop("diffgnn", None)
        cfg.pop("diffgnn_order", None)
    return cfg


def _write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _manifest_row(
    suite_name: str,
    profile: str,
    topo_row: dict,
    config_path: Path,
    area: float,
    factors: dict,
    seed: int,
    source_kind: str,
    source_csv: str,
) -> dict:
    return {
        "suite": suite_name,
        "profile": profile,
        "config_stem": config_path.stem,
        "config_path": _repo_rel(config_path),
        "graph_name": topo_row["graph_name"],
        "graph_file": topo_row["graph_rel"],
        "family": topo_row["family"],
        "nodes": topo_row["nodes"],
        "edges": topo_row["edges"],
        "area_constraint": float(area),
        "hw_scale_factor": float(factors["hw-scale-factor"]),
        "hw_scale_variance": float(factors["hw-scale-variance"]),
        "comm_scale_factor": float(factors["comm-scale-factor"]),
        "seed": int(seed),
        "taskgraph_pickle": _taskgraph_pickle_path(topo_row["graph_name"], area, factors, seed),
        "param_source": source_kind,
        "param_source_csv": source_csv,
    }


def _write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        manifest_path.write_text("")
        return
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_configs(
    profile: str,
    base_config_path: Path,
    config_root: Path,
    seeds: list[int],
    area_value: float,
    squeeze_areas: list[float],
    clean: bool,
    strip_method_configs: bool,
) -> None:
    with base_config_path.open() as handle:
        base_cfg = yaml.safe_load(handle)

    best_params = _discover_best_params(area_value)
    topologies = _collect_topologies()

    profile_root = config_root / profile
    if clean and profile_root.exists():
        shutil.rmtree(profile_root)
    profile_root.mkdir(parents=True, exist_ok=True)

    best_rows = []
    for topo_row in topologies:
        factors, source_kind, source_csv = _selected_factors(topo_row["graph_name"], best_params)
        best_rows.append(
            {
                "graph_name": topo_row["graph_name"],
                "family": topo_row["family"],
                "nodes": topo_row["nodes"],
                "edges": topo_row["edges"],
                "hw_scale_factor": factors["hw-scale-factor"],
                "hw_scale_variance": factors["hw-scale-variance"],
                "comm_scale_factor": factors["comm-scale-factor"],
                "param_source": source_kind,
                "param_source_csv": source_csv,
            }
        )

    _write_manifest(profile_root / "best_diff_gnn_order_area05.csv", best_rows)

    suite_rows: list[dict] = []
    suite_name = "task_graph_topology_suite_area05"
    suite_dir = profile_root / "graph_suite_area05"
    for topo_row in topologies:
        factors, source_kind, source_csv = _selected_factors(topo_row["graph_name"], best_params)
        graph_dir = suite_dir / topo_row["graph_name"]
        for seed in seeds:
            cfg = _prepare_config(
                base_cfg=base_cfg,
                topo_row=topo_row,
                area=area_value,
                factors=factors,
                seed=seed,
                suite_name=suite_name,
                profile=profile,
                strip_method_configs=strip_method_configs,
            )
            config_name = (
                f"{topo_row['graph_name']}_"
                f"area-{_format_num(area_value)}_"
                f"hwscale-{_format_num(factors['hw-scale-factor'])}_"
                f"hwvar-{_format_num(factors['hw-scale-variance'])}_"
                f"comm-{_format_num(factors['comm-scale-factor'])}_"
                f"seed-{seed}.yaml"
            )
            config_path = graph_dir / config_name
            _write_config(config_path, cfg)
            suite_rows.append(
                _manifest_row(
                    suite_name=suite_name,
                    profile=profile,
                    topo_row=topo_row,
                    config_path=config_path,
                    area=area_value,
                    factors=factors,
                    seed=seed,
                    source_kind=source_kind,
                    source_csv=source_csv,
                )
            )

    _write_manifest(suite_dir / "manifest.csv", suite_rows)

    squeeze_rows: list[dict] = []
    sweep_name = "squeezenet_area_sweep"
    squeeze_dir = profile_root / "squeeze_net_tosa_area_sweep" / "squeeze_net_tosa"
    squeeze_topo = next(row for row in topologies if row["graph_name"] == "squeeze_net_tosa")
    squeeze_factors, source_kind, source_csv = _selected_factors("squeeze_net_tosa", best_params)
    for area in squeeze_areas:
        for seed in seeds:
            cfg = _prepare_config(
                base_cfg=base_cfg,
                topo_row=squeeze_topo,
                area=area,
                factors=squeeze_factors,
                seed=seed,
                suite_name=sweep_name,
                profile=profile,
                strip_method_configs=strip_method_configs,
            )
            config_name = (
                f"squeeze_net_tosa_"
                f"area-{_format_num(area)}_"
                f"hwscale-{_format_num(squeeze_factors['hw-scale-factor'])}_"
                f"hwvar-{_format_num(squeeze_factors['hw-scale-variance'])}_"
                f"comm-{_format_num(squeeze_factors['comm-scale-factor'])}_"
                f"seed-{seed}.yaml"
            )
            config_path = squeeze_dir / config_name
            _write_config(config_path, cfg)
            squeeze_rows.append(
                _manifest_row(
                    suite_name=sweep_name,
                    profile=profile,
                    topo_row=squeeze_topo,
                    config_path=config_path,
                    area=area,
                    factors=squeeze_factors,
                    seed=seed,
                    source_kind=source_kind,
                    source_csv=source_csv,
                )
            )

    _write_manifest((profile_root / "squeeze_net_tosa_area_sweep" / "manifest.csv"), squeeze_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate batch experiment configs for all task-graph topologies.")
    parser.add_argument(
        "--profile",
        choices=["pilot", "full"],
        default="pilot",
        help="Generation profile. 'pilot' uses reduced budgets for validation runs.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base YAML config to clone.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
        help="Root output directory for generated topology configs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seed list for each generated suite. If omitted, pilot uses 3 seeds and full uses 10 seeds.",
    )
    parser.add_argument(
        "--area",
        type=float,
        default=DEFAULT_AREA,
        help="Area constraint used for the all-topology suite.",
    )
    parser.add_argument(
        "--squeeze-areas",
        type=float,
        nargs="+",
        default=list(DEFAULT_SQUEEZE_AREAS),
        help="Area settings used for the SqueezeNet area sweep.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the target profile directory before regenerating configs.",
    )
    parser.add_argument(
        "--strip-method-configs",
        action="store_true",
        help="Remove runtime method blocks from generated YAMLs so methods resolve settings from Python defaults.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seeds is None:
        seeds = list(DEFAULT_FULL_SEEDS if args.profile == "full" else DEFAULT_PILOT_SEEDS)
    else:
        seeds = [int(v) for v in args.seeds]
    generate_configs(
        profile=args.profile,
        base_config_path=args.base_config,
        config_root=args.config_root,
        seeds=seeds,
        area_value=float(args.area),
        squeeze_areas=[float(v) for v in args.squeeze_areas],
        clean=not args.no_clean,
        strip_method_configs=bool(args.strip_method_configs),
    )
    print(f"Generated configs under {args.config_root / args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
