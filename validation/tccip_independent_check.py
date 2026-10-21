#!/usr/bin/env python3
"""Independent standard-library cross-check of TCCIP manifests.

This intentionally has a separate parser/evaluator from tccip_reference_oracle.py.
It reads the two TSV manifests, exhaustively recomputes every single-cut utility,
partition range, and listed joint-cut utility, and fails closed on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = ROOT / "validation/fixtures/tccip_toy_expectations.tsv"
PAIR_DEFAULT = ROOT / "validation/fixtures/tccip_pairwise_utility_interactions.tsv"
TOL = 1e-12


def rule(token: str) -> tuple:
    parts = token.split(":")
    if parts[0] in ("input", "copy") and len(parts) == 2:
        return (parts[0], int(parts[1]))
    if parts[0] in ("xor", "and") and len(parts) == 3:
        return (parts[0], int(parts[1]), int(parts[2]))
    if parts == ["constant", "0"]:
        return ("constant", False)
    if parts == ["constant", "1"]:
        return ("constant", True)
    raise ValueError(f"bad rule token: {token}")


def task(token: str) -> tuple:
    parts = token.split(":")
    if parts[0] in ("xor-inputs", "and-inputs") and len(parts) == 3:
        return ("xor" if parts[0] == "xor-inputs" else "and", int(parts[1]), int(parts[2]))
    if parts[0] == "copy-input" and len(parts) == 2:
        return ("input", int(parts[1]))
    if parts == ["constant", "0"]:
        return ("constant", False)
    if parts == ["constant", "1"]:
        return ("constant", True)
    raise ValueError(f"bad task token: {token}")


def edge_value(source: int, target: int, values: list[bool],
               cuts: tuple[tuple[int, int], ...], coin_mask: int) -> bool:
    for index, (cut_source, cut_target) in enumerate(cuts):
        if (cut_source, cut_target) == (source, target):
            return bool((coin_mask >> index) & 1)
    return values[source]


def output_value(case: dict, inputs: tuple[bool, ...],
                 cuts: tuple[tuple[int, int], ...] = (), coin_mask: int = 0) -> bool:
    values: list[bool] = []
    for target_index, current_rule in enumerate(case["rules"]):
        op = current_rule[0]
        if op == "input":
            value = inputs[current_rule[1]]
        elif op == "constant":
            value = current_rule[1]
        elif op == "copy":
            value = edge_value(current_rule[1], target_index, values, cuts, coin_mask)
        elif op == "xor":
            value = (
                edge_value(current_rule[1], target_index, values, cuts, coin_mask)
                ^ edge_value(current_rule[2], target_index, values, cuts, coin_mask)
            )
        elif op == "and":
            value = (
                edge_value(current_rule[1], target_index, values, cuts, coin_mask)
                & edge_value(current_rule[2], target_index, values, cuts, coin_mask)
            )
        else:
            raise ValueError(f"unknown operation {op}")
        values.append(bool(value))
    return values[case["output"]]


def target_value(target: tuple, inputs: tuple[bool, ...]) -> bool:
    if target[0] == "xor":
        return inputs[target[1]] ^ inputs[target[2]]
    if target[0] == "and":
        return inputs[target[1]] & inputs[target[2]]
    if target[0] == "input":
        return inputs[target[1]]
    if target[0] == "constant":
        return bool(target[1])
    raise ValueError(f"unknown task target {target}")


def utility(case: dict, cuts: tuple[tuple[int, int], ...] = ()) -> float:
    weights = case["weights"]
    denominator = sum(weights) * (1 << len(cuts))
    correct = 0
    for state, weight in enumerate(weights):
        inputs = tuple(bool((state >> bit) & 1) for bit in range(case["input_count"]))
        expected = target_value(case["task"], inputs)
        for coin_mask in range(1 << len(cuts)):
            if output_value(case, inputs, cuts, coin_mask) == expected:
                correct += weight
    return correct / denominator


def read_cases(path: Path) -> tuple[dict[str, dict], bytes]:
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if "# schema=tccip-toy-expectations-v3" not in lines:
        raise ValueError("wrong main manifest schema")
    cases: dict[str, dict] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 19:
            raise ValueError(f"model manifest line {line_number}: expected 19 columns")
        name = fields[0]
        if name in cases:
            raise ValueError(f"duplicate fixture {name}")
        node_count, input_count, output = int(fields[1]), int(fields[2]), int(fields[3])
        if not 1 <= node_count <= 12:
            raise ValueError(f"node count outside 1..=12 in {name}")
        if not 1 <= input_count <= 8:
            raise ValueError(f"input count outside 1..=8 in {name}")
        weights = tuple(map(int, fields[5].split(",")))
        rules = tuple(map(rule, fields[6].split(",")))
        components = tuple(0 if label == "A" else 1 if label == "B" else -1 for label in fields[7])
        interventions = []
        for token in fields[8].split(";"):
            parts = token.split(":")
            if len(parts) != 4:
                raise ValueError(f"bad intervention {token}")
            interventions.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
        if len(rules) != node_count or len(components) != node_count:
            raise ValueError(f"node/component count mismatch in {name}")
        if len(weights) != (1 << input_count) or not weights or min(weights) < 0 or sum(weights) == 0:
            raise ValueError(f"invalid input prior in {name}")
        if not (0 <= output < node_count) or any(label not in (0, 1) for label in components):
            raise ValueError(f"invalid output or partition in {name}")
        if sum(component << i for i, component in enumerate(components)) != int(fields[9]):
            raise ValueError(f"partition-mask mismatch in {name}")
        for target_index, current_rule in enumerate(rules):
            if current_rule[0] == "input" and not (0 <= current_rule[1] < input_count):
                raise ValueError(f"invalid external input in {name}")
            sources = (
                current_rule[1:3] if current_rule[0] in ("xor", "and")
                else current_rule[1:2] if current_rule[0] == "copy"
                else ()
            )
            if any(source < 0 or source >= target_index for source in sources):
                raise ValueError(f"network is not in topological order in {name}")
        edge_pairs = [(source, target) for _, source, target, _ in interventions]
        if len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError(f"duplicate intervention edge in {name}")
        for identifier, source, target, cost in interventions:
            if cost < 1 or source < 0 or target < 0 or source >= node_count or target >= node_count:
                raise ValueError(f"invalid intervention {identifier} in {name}")
            current_rule = rules[target]
            sources = (
                [current_rule[1]] if current_rule[0] == "copy"
                else [current_rule[1], current_rule[2]] if current_rule[0] in ("xor", "and")
                else []
            )
            if source not in sources:
                raise ValueError(f"intervention {identifier} is not a graph edge in {name}")
        parsed_task = task(fields[4])
        task_inputs = parsed_task[1:] if parsed_task[0] in ("xor", "and") else parsed_task[1:2] if parsed_task[0] == "input" else ()
        if any(index < 0 or index >= input_count for index in task_inputs):
            raise ValueError(f"task target references invalid external input in {name}")
        cases[name] = {
            "name": name, "node_count": node_count, "input_count": input_count,
            "output": output, "task": parsed_task, "weights": weights,
            "rules": rules, "components": components, "interventions": interventions,
            "mask": int(fields[9]), "expected": tuple(map(float, fields[10:14])),
            "part_counts": tuple(map(int, fields[14:17])),
            "part_range": tuple(map(float, fields[17:19])),
        }
    if len(cases) != 6:
        raise ValueError(f"expected six model cases, got {len(cases)}")
    return cases, raw


def check_case(case: dict) -> tuple[tuple[float, ...], tuple[int, ...], tuple[float, float]]:
    intact = utility(case)
    individual = [
        utility(case, ((source, target),))
        for _, source, target, _ in case["interventions"]
    ]
    losses = [intact - score for score in individual]
    cross = [
        loss for cut, loss in zip(case["interventions"], losses)
        if case["components"][cut[1]] != case["components"][cut[2]]
    ]
    internal = [
        loss for cut, loss in zip(case["interventions"], losses)
        if case["components"][cut[1]] == case["components"][cut[2]]
    ]
    if not cross or not internal:
        raise ValueError(f"missing comparison stratum: {case['name']}")
    cross_mean = sum(cross) / len(cross)
    internal_mean = sum(internal) / len(internal)
    actual = (intact, cross_mean, internal_mean, cross_mean - internal_mean)
    if not all(math.isclose(a, b, abs_tol=TOL, rel_tol=0) for a, b in zip(actual, case["expected"])):
        raise AssertionError((case["name"], actual, case["expected"]))

    profiles = []
    node_count = case["node_count"]
    total_partitions = (1 << (node_count - 1)) - 1
    for mask in range(1, 1 << node_count):
        if mask & 1:
            continue
        cut_losses, internal_losses = [], []
        for cut, loss in zip(case["interventions"], losses):
            _, source, target, _ = cut
            destination = cut_losses if ((mask >> source) & 1) != ((mask >> target) & 1) else internal_losses
            destination.append(loss)
        if cut_losses and internal_losses:
            profiles.append(sum(cut_losses) / len(cut_losses) - sum(internal_losses) / len(internal_losses))
    counts = (total_partitions, len(profiles), total_partitions - len(profiles))
    bounds = (min(profiles), max(profiles))
    if counts != case["part_counts"]:
        raise AssertionError((case["name"], counts, case["part_counts"]))
    if not all(math.isclose(a, b, abs_tol=TOL, rel_tol=0) for a, b in zip(bounds, case["part_range"])):
        raise AssertionError((case["name"], bounds, case["part_range"]))
    return actual, counts, bounds


def check_pair_matrix(cases: dict[str, dict], path: Path) -> tuple[int, bytes]:
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if "# schema=tccip-pairwise-utility-interaction-expectations-v1" not in lines:
        raise ValueError("wrong pairwise manifest schema")
    seen = set()
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 8:
            raise ValueError(f"pair manifest line {line_number}: expected 8 columns")
        name, first_id, second_id = fields[:3]
        key = (name, first_id, second_id)
        if key in seen:
            raise ValueError(f"duplicate pair {key}")
        seen.add(key)
        case = cases[name]
        by_id = {item[0]: item for item in case["interventions"]}
        first, second = by_id[first_id], by_id[second_id]
        edge_first, edge_second = (first[1], first[2]), (second[1], second[2])
        observed = (
            utility(case),
            utility(case, (edge_first,)),
            utility(case, (edge_second,)),
            utility(case, (edge_first, edge_second)),
        )
        delta = observed[0] - observed[1] - observed[2] + observed[3]
        actual = observed + (delta,)
        expected = tuple(map(float, fields[3:8]))
        if not all(math.isclose(a, b, abs_tol=TOL, rel_tol=0) for a, b in zip(actual, expected)):
            raise AssertionError((key, actual, expected))
    if len(seen) != 16:
        raise ValueError(f"expected 16 declared intervention pairs, got {len(seen)}")
    return len(seen), raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--pair-manifest", type=Path, default=PAIR_DEFAULT)
    parser.add_argument("--source-sha", default=os.getenv("SYMTHAEA_SOURCE_SHA", "UNBOUND"))
    args = parser.parse_args()
    try:
        cases, model_bytes = read_cases(args.model_manifest)
        results = {name: check_case(case) for name, case in cases.items()}
        pair_count, pair_bytes = check_pair_matrix(cases, args.pair_manifest)
        xor = cases["role_matched_xor_synergy_uniform_prior"]
        cuts = xor["interventions"]
        edge_first, edge_second = (cuts[0][1], cuts[0][2]), (cuts[1][1], cuts[1][2])
        values = (utility(xor), utility(xor, (edge_first,)), utility(xor, (edge_second,)),
                  utility(xor, (edge_first, edge_second)))
        delta = values[0] - values[1] - values[2] + values[3]
        if tuple(round(value, 12) for value in values + (delta,)) != (1.0, 0.5, 0.5, 0.5, 0.5):
            raise AssertionError(f"XOR second-order utility interaction mismatch: {values}, {delta}")
        print(f"EXACT_INDEPENDENT_CHECKS_PASSED: fixtures={len(cases)} joint_intervention_pairs={pair_count}; source_sha={args.source_sha}")
        for name, (summary, counts, bounds) in results.items():
            print(f"{name}: utility={summary[0]:.3f} TCCIP={summary[3]:+.3f} partitions={counts[1]}/{counts[2]} range=[{bounds[0]:+.3f},{bounds[1]:+.3f}]")
        print(f"XOR Δ²J={delta:+.3f}; model_manifest_sha256={hashlib.sha256(model_bytes).hexdigest()}; pair_manifest_sha256={hashlib.sha256(pair_bytes).hexdigest()}")
    except (OSError, UnicodeError, ValueError, AssertionError, KeyError) as error:
        print(f"NOT QUALIFIED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
