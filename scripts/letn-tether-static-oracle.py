#!/usr/bin/env python3
"""Independent LETN-003 static tether reference calculator.

This script intentionally does not import Symthaea code. It mirrors only the
published v0.1 equilibrium contract so the Rust implementation can be checked
against a separately executable standard-library oracle.

Input JSON schema:
{
  "density_kg_m3": 1000.0,
  "tip_tension_n": 100.0,
  "nodes": [
    {"s_m": 0.0, "effective_acceleration_m_s2": 2.0,
     "area_m2": 0.01, "lumped_force_n": 0.0},
    ...
  ]
}

Run `--self-test` for analytic regression cases, or pass a JSON file path and
receive a JSON result on stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Node:
    s_m: float
    effective_acceleration_m_s2: float
    area_m2: float
    lumped_force_n: float = 0.0


def _finite(value: float) -> bool:
    return math.isfinite(value)


def parse_model(raw: dict[str, Any]) -> tuple[float, float, list[Node]]:
    density = float(raw["density_kg_m3"])
    tip = float(raw["tip_tension_n"])
    nodes = [
        Node(
            s_m=float(item["s_m"]),
            effective_acceleration_m_s2=float(item["effective_acceleration_m_s2"]),
            area_m2=float(item["area_m2"]),
            lumped_force_n=float(item.get("lumped_force_n", 0.0)),
        )
        for item in raw["nodes"]
    ]
    validate(density, tip, nodes)
    return density, tip, nodes


def validate(density: float, tip: float, nodes: list[Node]) -> None:
    if len(nodes) < 2:
        raise ValueError("at least two nodes are required")
    if not _finite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    if not _finite(tip) or tip < 0.0:
        raise ValueError("tip tension must be finite and non-negative")
    for index, node in enumerate(nodes):
        values = (
            node.s_m,
            node.effective_acceleration_m_s2,
            node.area_m2,
            node.lumped_force_n,
        )
        if not all(_finite(value) for value in values):
            raise ValueError("all node values must be finite")
        if node.area_m2 <= 0.0:
            raise ValueError("area must be positive")
        if index and node.s_m <= nodes[index - 1].s_m:
            raise ValueError("positions must be strictly increasing")


def solve(density: float, tip: float, nodes: list[Node]) -> dict[str, Any]:
    validate(density, tip, nodes)
    count = len(nodes)
    tension = [0.0] * count
    tension[-1] = tip + nodes[-1].lumped_force_n
    mass = 0.0
    distributed_force = 0.0

    for index in range(count - 2, -1, -1):
        left = nodes[index]
        right = nodes[index + 1]
        ds = right.s_m - left.s_m
        lambda_left = density * left.area_m2
        lambda_right = density * right.area_m2
        mass += 0.5 * (lambda_left + lambda_right) * ds
        force_left = lambda_left * left.effective_acceleration_m_s2
        force_right = lambda_right * right.effective_acceleration_m_s2
        segment_force = 0.5 * (force_left + force_right) * ds
        distributed_force += segment_force
        tension[index] = tension[index + 1] + segment_force + left.lumped_force_n

    return {
        "tension_n": tension,
        "distributed_mass_kg": mass,
        "integrated_signed_distributed_force_n": distributed_force,
        "integrated_signed_lumped_force_n": sum(node.lumped_force_n for node in nodes),
        "minimum_tension_n": min(tension),
        "maximum_tension_n": max(tension),
        "remains_in_tension": min(tension) >= 0.0,
    }


def minimum_tip_tension_for_tension_only(density: float, nodes: list[Node]) -> float:
    zero_tip = solve(density, 0.0, nodes)
    return max(0.0, -float(zero_tip["minimum_tension_n"]))


def _uniform_nodes(acceleration: float) -> list[Node]:
    return [
        Node(0.0, acceleration, 0.01),
        Node(5.0, acceleration, 0.01),
        Node(10.0, acceleration, 0.01),
    ]


def _assert_close(actual: float, expected: float, tol: float = 1e-12) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{actual} != {expected} within {tol}")


def self_test() -> None:
    uniform = solve(1000.0, 100.0, _uniform_nodes(2.0))
    assert uniform["tension_n"] == [300.0, 200.0, 100.0]
    _assert_close(uniform["distributed_mass_kg"], 100.0)
    _assert_close(uniform["integrated_signed_distributed_force_n"], 200.0)

    zero = solve(1000.0, 123.0, _uniform_nodes(0.0))
    assert zero["tension_n"] == [123.0, 123.0, 123.0]

    negative_nodes = _uniform_nodes(-2.0)
    required = minimum_tip_tension_for_tension_only(1000.0, negative_nodes)
    _assert_close(required, 200.0)
    tensioned = solve(1000.0, required, negative_nodes)
    assert tensioned["tension_n"] == [0.0, 100.0, 200.0]

    lumped_nodes = _uniform_nodes(0.0)
    lumped_nodes[1] = Node(5.0, 0.0, 0.01, 50.0)
    lumped = solve(1000.0, 100.0, lumped_nodes)
    assert lumped["tension_n"] == [150.0, 150.0, 100.0]

    sign_change = solve(
        1000.0,
        100.0,
        [Node(0.0, 2.0, 0.01), Node(5.0, 0.0, 0.01), Node(10.0, -2.0, 0.01)],
    )
    assert sign_change["tension_n"] == [100.0, 50.0, 100.0]

    fine = solve(
        1000.0,
        100.0,
        [Node(float(i), 2.0, 0.01) for i in range(11)],
    )
    _assert_close(fine["tension_n"][0], 300.0)
    _assert_close(fine["distributed_mass_kg"], 100.0)

    print("LETN-003 independent oracle self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.input is None:
        parser.error("provide a JSON input file or --self-test")

    raw = json.loads(args.input.read_text())
    density, tip, nodes = parse_model(raw)
    json.dump(solve(density, tip, nodes), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
