from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "meta_heuristic" / "partition_schedule_evaluator.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC = importlib.util.spec_from_file_location("partition_schedule_evaluator_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from utils.scheduler_utils import compute_dag_makespan


class PartitionScheduleEvaluatorTest(unittest.TestCase):
    def _bus_fixture(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("T1", "T4"), ("T2", "T4"), ("T2", "T5"), ("T3", "T5")])
        return SimpleNamespace(
            graph=graph,
            hardware_costs={"T1": 2.0, "T2": 3.0, "T3": 6.0, "T4": 9.0, "T5": 7.0},
            software_costs={"T1": 5.0, "T2": 6.0, "T3": 4.0, "T4": 5.0, "T5": 3.0},
            hardware_area={"T1": 2.0, "T2": 2.0, "T3": 1.0, "T4": 3.0, "T5": 2.0},
            communication_costs={
                ("T1", "T4"): 1.0,
                ("T2", "T4"): 2.0,
                ("T2", "T5"): 2.0,
                ("T3", "T5"): 1.0,
            },
            area_constraint=0.6,
            total_area=10.0,
            violation_cost=1e9,
        )

    def test_make_partition_valid_repairs_area_violation_deterministically(self):
        tg = self._bus_fixture()
        invalid_partition = {"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1}

        repair = MODULE.make_partition_valid(tg, invalid_partition)

        self.assertTrue(repair["is_valid"])
        self.assertEqual(repair["repaired_nodes"], ["T3", "T5", "T4"])
        self.assertEqual(repair["partition"], {"T1": 1, "T2": 1, "T3": 0, "T4": 0, "T5": 0})

    def test_make_partition_valid_keeps_valid_partition(self):
        tg = self._bus_fixture()
        partition = {"T1": 1, "T2": 1, "T3": 0, "T4": 0, "T5": 0}

        repair = MODULE.make_partition_valid(tg, partition)

        self.assertTrue(repair["is_valid"])
        self.assertFalse(repair["was_repaired"])
        self.assertEqual(repair["partition"], partition)

    def test_synchronize_problem_with_config_overrides_loaded_taskgraph_constraint(self):
        tg = self._bus_fixture()
        tg.area_constraint = 0.5

        synced = MODULE.synchronize_problem_with_config(tg, {"area-constraint": 0.4})

        self.assertIs(synced, tg)
        self.assertEqual(tg.area_constraint, 0.4)

    def test_synchronize_problem_with_config_returns_updated_problem(self):
        tg = self._bus_fixture()
        problem = MODULE.build_problem(tg)

        synced = MODULE.synchronize_problem_with_config(problem, {"area-constraint": 0.4})

        self.assertIsNot(synced, problem)
        self.assertEqual(problem.area_constraint, 0.6)
        self.assertEqual(synced.area_constraint, 0.4)

    def test_lssp_all_software_serializes_on_one_processor(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C")])
        tg = SimpleNamespace(
            graph=graph,
            hardware_costs={"A": 2.0, "B": 2.0, "C": 2.0},
            software_costs={"A": 5.0, "B": 4.0, "C": 3.0},
            hardware_area={"A": 1.0, "B": 1.0, "C": 1.0},
            communication_costs={("A", "B"): 7.0, ("B", "C"): 9.0},
            area_constraint=1.0,
            total_area=3.0,
            violation_cost=1e9,
        )

        result = MODULE.evaluate_partition_lssp(tg, {"A": 0, "B": 0, "C": 0}, auto_repair=False)

        self.assertEqual(result["makespan"], 12.0)
        self.assertEqual(result["start_times"], {"A": 0.0, "B": 5.0, "C": 9.0})
        self.assertEqual(result["finish_times"], {"A": 5.0, "B": 9.0, "C": 12.0})
        self.assertEqual(result["bus_schedule"], [])

    def test_lssp_serializes_shared_bus_for_cross_context_edges(self):
        tg = self._bus_fixture()
        partition = {"T1": 1, "T2": 1, "T3": 0, "T4": 0, "T5": 0}

        result = MODULE.evaluate_partition_lssp(tg, partition, auto_repair=False)

        self.assertEqual(result["makespan"], 13.0)
        self.assertEqual(result["start_times"]["T4"], 5.0)
        self.assertEqual(result["start_times"]["T5"], 10.0)
        self.assertEqual(
            [(item["edge"], item["start_time"], item["finish_time"]) for item in result["bus_schedule"]],
            [
                (("T1", "T4"), 2.0, 3.0),
                (("T2", "T4"), 3.0, 5.0),
                (("T2", "T5"), 5.0, 7.0),
            ],
        )

    def test_lssp_same_context_communication_has_zero_bus_cost(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "C"), ("B", "C")])
        tg = SimpleNamespace(
            graph=graph,
            hardware_costs={"A": 1.0, "B": 1.0, "C": 1.0},
            software_costs={"A": 2.0, "B": 3.0, "C": 4.0},
            hardware_area={"A": 1.0, "B": 1.0, "C": 1.0},
            communication_costs={("A", "C"): 10.0, ("B", "C"): 11.0},
            area_constraint=1.0,
            total_area=3.0,
            violation_cost=1e9,
        )

        result = MODULE.evaluate_partition_lssp(tg, {"A": 0, "B": 0, "C": 0}, auto_repair=False)

        self.assertEqual(result["makespan"], 9.0)
        self.assertEqual(result["bus_schedule"], [])
        self.assertEqual(result["start_times"]["C"], max(result["finish_times"]["A"], result["finish_times"]["B"]))
        self.assertEqual(result["active_communication_edges"], [])

    def test_lssp_uses_software_priority_scores_when_provided(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(["A", "B"])
        tg = SimpleNamespace(
            graph=graph,
            hardware_costs={"A": 1.0, "B": 1.0},
            software_costs={"A": 5.0, "B": 3.0},
            hardware_area={"A": 1.0, "B": 1.0},
            communication_costs={},
            area_constraint=1.0,
            total_area=2.0,
            violation_cost=1e9,
        )
        partition = {"A": 0, "B": 0}

        static_result = MODULE.evaluate_partition_lssp(tg, partition, auto_repair=False)
        guided_result = MODULE.evaluate_partition_lssp(
            tg,
            partition,
            auto_repair=False,
            software_priority_scores={"A": 0.1, "B": 0.9},
        )

        self.assertEqual(static_result["start_times"]["A"], 0.0)
        self.assertEqual(static_result["start_times"]["B"], 5.0)
        self.assertEqual(guided_result["start_times"]["B"], 0.0)
        self.assertEqual(guided_result["start_times"]["A"], 3.0)
        self.assertFalse(static_result["software_priority_used"])
        self.assertTrue(guided_result["software_priority_used"])

    def test_dag_mode_matches_current_compute_dag_makespan(self):
        graph = nx.DiGraph()
        for node, hw_t, sw_t, area in [
            ("A", 2.0, 5.0, 2.0),
            ("B", 3.0, 6.0, 2.0),
            ("C", 7.0, 4.0, 1.0),
            ("D", 9.0, 5.0, 3.0),
        ]:
            graph.add_node(node, hardware_time=hw_t, software_time=sw_t, area_cost=area)
        graph.add_edge("A", "D", communication_cost=1.0)
        graph.add_edge("B", "D", communication_cost=2.0)
        tg = SimpleNamespace(
            graph=graph,
            hardware_costs={"A": 2.0, "B": 3.0, "C": 7.0, "D": 9.0},
            software_costs={"A": 5.0, "B": 6.0, "C": 4.0, "D": 5.0},
            hardware_area={"A": 2.0, "B": 2.0, "C": 1.0, "D": 3.0},
            communication_costs={("A", "D"): 1.0, ("B", "D"): 2.0},
            area_constraint=0.8,
            total_area=8.0,
            violation_cost=1e9,
        )
        partition = {"A": 1, "B": 1, "C": 0, "D": 0}

        result = MODULE.evaluate_partition_dag(tg, partition, auto_repair=False)
        expected_makespan, expected_start_times = compute_dag_makespan(graph, [0, 0, 1, 1])

        self.assertAlmostEqual(result["makespan"], expected_makespan, places=6)
        for node in graph.nodes():
            self.assertAlmostEqual(result["start_times"][node], expected_start_times[node], places=6)
            exec_time = tg.hardware_costs[node] if partition[node] == 1 else tg.software_costs[node]
            self.assertAlmostEqual(result["finish_times"][node], result["start_times"][node] + exec_time, places=6)


if __name__ == "__main__":
    unittest.main()
