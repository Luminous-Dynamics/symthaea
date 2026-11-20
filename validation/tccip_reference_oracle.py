#!/usr/bin/env python3
"""Independent standard-library oracle for the TCCIP finite-state toy manifest.

This intentionally uses a separate implementation from the Rust example. It
recomputes task utility by exhaustive enumeration, validates manifest oracle
columns, and emits raw intervention plus all usable partition results. It is a
research validator, not production control software or a consciousness measure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "validation/fixtures/tccip_toy_expectations.tsv"
SCHEMA = "# schema=tccip-toy-expectations-v3"
TOLERANCE = 1e-12


@dataclass(frozen=True)
class Rule:
    op: str
    args: tuple[int | bool, ...]


@dataclass(frozen=True)
class Intervention:
    identifier: str
    source: int
    target: int
    cost: int


@dataclass(frozen=True)
class Fixture:
    name: str
    node_count: int
    input_count: int
    output_node: int
    target: tuple[str, tuple[int | bool, ...]]
    input_state_weights: tuple[int, ...]
    rules: tuple[Rule, ...]
    components: tuple[int, ...]
    interventions: tuple[Intervention, ...]
    declared_mask: int
    expected: tuple[float, float, float, float]
    expected_partition_counts: tuple[int, int, int]
    expected_partition_range: tuple[float, float]


@dataclass(frozen=True)
class InterventionScore:
    fixture: str
    identifier: str
    kind: str
    source: int
    target: int
    cost: int
    intact: float
    utility: float
    loss: float


@dataclass(frozen=True)
class PartitionScore:
    fixture: str
    mask: int
    labels: str
    cross_loss: float
    control_loss: float
    tccip: float
    n_cross: int
    n_control: int


def parse_int(raw: str, where: str, minimum: int = 0) -> int:
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"{where}: expected integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{where}: expected integer >= {minimum}, got {value}")
    return value


def parse_rule(raw: str) -> Rule:
    parts = raw.split(":")
    try:
        if len(parts) == 2 and parts[0] == "input":
            return Rule("input", (parse_int(parts[1], "input index"),))
        if len(parts) == 2 and parts[0] == "copy":
            return Rule("copy", (parse_int(parts[1], "copy source"),))
        if len(parts) == 3 and parts[0] in {"xor", "and"}:
            return Rule(parts[0], (parse_int(parts[1], "left source"), parse_int(parts[2], "right source")))
        if len(parts) == 2 and parts == ["constant", "0"]:
            return Rule("constant", (False,))
        if len(parts) == 2 and parts == ["constant", "1"]:
            return Rule("constant", (True,))
    except ValueError:
        raise
    raise ValueError(f"unsupported rule token: {raw!r}")


def canonical_rule(rule: Rule) -> str:
    if rule.op == "input":
        return f"input:{rule.args[0]}"
    if rule.op == "copy":
        return f"copy:{rule.args[0]}"
    if rule.op in {"xor", "and"}:
        return f"{rule.op}:{rule.args[0]}:{rule.args[1]}"
    if rule.op == "constant":
        return f"constant:{1 if rule.args[0] else 0}"
    raise ValueError(f"unsupported rule {rule!r}")


def parse_target(raw: str) -> tuple[str, tuple[int | bool, ...]]:
    parts = raw.split(":")
    if len(parts) == 3 and parts[0] in {"xor-inputs", "and-inputs"}:
        return (("xor" if parts[0] == "xor-inputs" else "and"),
                (parse_int(parts[1], "task left input"), parse_int(parts[2], "task right input")))
    if len(parts) == 2 and parts[0] == "copy-input":
        return ("input", (parse_int(parts[1], "task copied input"),))
    if parts == ["constant", "0"]:
        return ("constant", (False,))
    if parts == ["constant", "1"]:
        return ("constant", (True,))
    raise ValueError(f"unsupported task target: {raw!r}")


def canonical_target(target: tuple[str, tuple[int | bool, ...]]) -> str:
    op, args = target
    if op in {"xor", "and"}:
        return f"{op}-inputs:{args[0]}:{args[1]}"
    if op == "input":
        return f"copy-input:{args[0]}"
    if op == "constant":
        return f"constant:{1 if args[0] else 0}"
    raise ValueError(f"unsupported task target {target!r}")


def parse_interventions(raw: str) -> tuple[Intervention, ...]:
    parsed = []
    for token in raw.split(";"):
        fields = token.split(":")
        if len(fields) != 4 or not fields[0]:
            raise ValueError(f"intervention must be id:source:target:cost: {token!r}")
        parsed.append(Intervention(
            fields[0],
            parse_int(fields[1], "intervention source"),
            parse_int(fields[2], "intervention target"),
            parse_int(fields[3], "intervention cost", minimum=1),
        ))
    return tuple(parsed)


def canonical_interventions(interventions: Sequence[Intervention]) -> str:
    return ";".join(
        f"{item.identifier}:{item.source}:{item.target}:{item.cost}"
        for item in interventions
    )


def validate_fixture(fixture: Fixture) -> None:
    if not 1 <= fixture.node_count <= 12:
        raise ValueError(f"{fixture.name}: node count outside 1..=12")
    if not 1 <= fixture.input_count <= 8:
        raise ValueError(f"{fixture.name}: input count outside 1..=8")
    if len(fixture.input_state_weights) != (1 << fixture.input_count):
        raise ValueError(f"{fixture.name}: input-state weights must have exactly 2^input_count entries")
    if any(weight < 0 for weight in fixture.input_state_weights) or not any(fixture.input_state_weights):
        raise ValueError(f"{fixture.name}: input-state weights must be nonnegative and not all zero")
    if len(fixture.rules) != fixture.node_count:
        raise ValueError(f"{fixture.name}: node count differs from rule count")
    if len(fixture.components) != fixture.node_count:
        raise ValueError(f"{fixture.name}: component labels differ from node count")
    if fixture.output_node >= fixture.node_count:
        raise ValueError(f"{fixture.name}: output node is out of range")
    if any(component not in {0, 1} for component in fixture.components):
        raise ValueError(f"{fixture.name}: component labels must be A/B")
    if not fixture.interventions:
        raise ValueError(f"{fixture.name}: no interventions")
    if canonical_target(fixture.target) not in {
        f"xor-inputs:{fixture.target[1][0]}:{fixture.target[1][1]}" if fixture.target[0] == "xor" else "",
        f"and-inputs:{fixture.target[1][0]}:{fixture.target[1][1]}" if fixture.target[0] == "and" else "",
        f"copy-input:{fixture.target[1][0]}" if fixture.target[0] == "input" else "",
        f"constant:{1 if fixture.target[1][0] else 0}" if fixture.target[0] == "constant" else "",
    }:
        raise ValueError(f"{fixture.name}: invalid canonical task target")

    for target, rule in enumerate(fixture.rules):
        if rule.op == "input":
            if int(rule.args[0]) >= fixture.input_count:
                raise ValueError(f"{fixture.name}: external input index out of range")
            continue
        if rule.op == "constant":
            continue
        for source in map(int, rule.args):
            if source >= target:
                raise ValueError(f"{fixture.name}: graph is not in topological order")
    target_op, target_args = fixture.target
    input_args = [int(value) for value in target_args if not isinstance(value, bool)]
    if any(index >= fixture.input_count for index in input_args):
        raise ValueError(f"{fixture.name}: task target input index out of range")

    ids = set()
    edges = set()
    for item in fixture.interventions:
        if item.identifier in ids:
            raise ValueError(f"{fixture.name}: duplicate intervention id {item.identifier}")
        ids.add(item.identifier)
        edge = (item.source, item.target)
        if edge in edges:
            raise ValueError(f"{fixture.name}: duplicate intervention edge {item.source}->{item.target}")
        edges.add(edge)
        if item.source >= fixture.node_count or item.target >= fixture.node_count:
            raise ValueError(f"{fixture.name}: intervention node out of range")
        if item.source >= item.target:
            raise ValueError(f"{fixture.name}: intervention is not on a topological edge")
        sources = rule_sources(fixture.rules[item.target])
        if item.source not in sources:
            raise ValueError(f"{fixture.name}: intervention is not a real edge")
    mask = sum(component << index for index, component in enumerate(fixture.components))
    if mask != fixture.declared_mask:
        raise ValueError(f"{fixture.name}: partition mask mismatch")


def rule_sources(rule: Rule) -> tuple[int, ...]:
    if rule.op in {"copy"}:
        return (int(rule.args[0]),)
    if rule.op in {"xor", "and"}:
        return (int(rule.args[0]), int(rule.args[1]))
    return ()


def parse_manifest(path: Path) -> tuple[list[Fixture], bytes]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    lines = text.splitlines()
    if SCHEMA not in {line.strip() for line in lines}:
        raise ValueError(f"manifest must declare {SCHEMA}")
    fixtures = []
    names = set()
    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 19:
            raise ValueError(f"line {line_number}: expected 19 fields, got {len(fields)}")
        name = fields[0]
        if not name or name in names:
            raise ValueError(f"line {line_number}: empty/duplicate fixture name {name!r}")
        names.add(name)
        n = parse_int(fields[1], "node count", minimum=1)
        inputs = parse_int(fields[2], "input count", minimum=1)
        output = parse_int(fields[3], "output node")
        target = parse_target(fields[4])
        input_weights = tuple(parse_int(token, "input-state weight") for token in fields[5].split(","))
        rules = tuple(parse_rule(token) for token in fields[6].split(","))
        component_names = fields[7]
        if any(label not in "AB" for label in component_names):
            raise ValueError(f"{name}: invalid component labels {component_names!r}")
        components = tuple(0 if label == "A" else 1 for label in component_names)
        interventions = parse_interventions(fields[8])
        mask = parse_int(fields[9], "partition mask")
        fixture = Fixture(
            name=name, node_count=n, input_count=inputs, output_node=output,
            target=target, input_state_weights=input_weights, rules=rules, components=components,
            interventions=interventions, declared_mask=mask,
            expected=(float(fields[10]), float(fields[11]), float(fields[12]), float(fields[13])),
            expected_partition_counts=(parse_int(fields[14], "partition count"),
                                       parse_int(fields[15], "usable partition count"),
                                       parse_int(fields[16], "skipped partition count")),
            expected_partition_range=(float(fields[17]), float(fields[18])),
        )
        validate_fixture(fixture)
        actual_mask = sum(component << index for index, component in enumerate(components))
        if actual_mask != mask:
            raise ValueError(f"{name}: declared partition mask {mask} disagrees with component labels {actual_mask}")
        if canonical_target(target) != fields[4]:
            raise ValueError(f"{name}: task target is not canonical")
        if ",".join(str(weight) for weight in input_weights) != fields[5]:
            raise ValueError(f"{name}: input-state weights are not canonical")
        if ",".join(canonical_rule(rule) for rule in rules) != fields[6]:
            raise ValueError(f"{name}: rule list is not canonical")
        if canonical_interventions(interventions) != fields[8]:
            raise ValueError(f"{name}: intervention list is not canonical")
        if len(component_names) != n:
            raise ValueError(f"{name}: component string length differs from node count")
        fixtures.append(fixture)
    if not fixtures:
        raise ValueError("manifest contains no fixtures")
    return fixtures, data


def eval_network(fixture: Fixture, inputs: Sequence[bool],
                 intervention: Intervention | None = None, cut_coin: bool = False) -> bool:
    values: list[bool] = []
    for target_index, rule in enumerate(fixture.rules):
        if rule.op == "input":
            value = inputs[int(rule.args[0])]
        elif rule.op == "constant":
            value = bool(rule.args[0])
        elif rule.op == "copy":
            source = int(rule.args[0])
            value = cut_coin if intervention and (intervention.source, intervention.target) == (source, target_index) else values[source]
        elif rule.op == "xor":
            left, right = map(int, rule.args)
            left_value = cut_coin if intervention and (intervention.source, intervention.target) == (left, target_index) else values[left]
            right_value = cut_coin if intervention and (intervention.source, intervention.target) == (right, target_index) else values[right]
            value = left_value ^ right_value
        elif rule.op == "and":
            left, right = map(int, rule.args)
            left_value = cut_coin if intervention and (intervention.source, intervention.target) == (left, target_index) else values[left]
            right_value = cut_coin if intervention and (intervention.source, intervention.target) == (right, target_index) else values[right]
            value = left_value & right_value
        else:
            raise ValueError(f"unsupported operation {rule.op!r}")
        values.append(bool(value))
    return values[fixture.output_node]


def expected_task(fixture: Fixture, inputs: Sequence[bool]) -> bool:
    op, args = fixture.target
    if op == "xor":
        return inputs[int(args[0])] ^ inputs[int(args[1])]
    if op == "and":
        return inputs[int(args[0])] & inputs[int(args[1])]
    if op == "input":
        return inputs[int(args[0])]
    if op == "constant":
        return bool(args[0])
    raise ValueError(f"unsupported task operation {op!r}")


def eval_network_with_cuts(
    fixture: Fixture,
    inputs: Sequence[bool],
    interventions: Sequence[Intervention],
    cut_mask: int,
) -> bool:
    """Evaluate with independent replacement coins for several simultaneous edge cuts."""
    values: list[bool] = []

    def source_value(source: int, target: int) -> bool:
        for index, intervention in enumerate(interventions):
            if (intervention.source, intervention.target) == (source, target):
                return bool((cut_mask >> index) & 1)
        return values[source]

    for target_index, rule in enumerate(fixture.rules):
        if rule.op == "input":
            value = inputs[int(rule.args[0])]
        elif rule.op == "constant":
            value = bool(rule.args[0])
        elif rule.op == "copy":
            value = source_value(int(rule.args[0]), target_index)
        elif rule.op == "xor":
            left, right = map(int, rule.args)
            value = source_value(left, target_index) ^ source_value(right, target_index)
        elif rule.op == "and":
            left, right = map(int, rule.args)
            value = source_value(left, target_index) & source_value(right, target_index)
        else:
            raise ValueError(f"unsupported operation {rule.op!r}")
        values.append(bool(value))
    return values[fixture.output_node]


def task_utility_for_cuts(fixture: Fixture, interventions: Sequence[Intervention]) -> float:
    input_count = 1 << fixture.input_count
    cut_states = 1 << len(interventions)
    total_input_weight = sum(fixture.input_state_weights)
    correct_weight = 0
    for state in range(input_count):
        inputs = tuple(bool((state >> index) & 1) for index in range(fixture.input_count))
        expected = expected_task(fixture, inputs)
        weight = fixture.input_state_weights[state]
        for cut_mask in range(cut_states):
            if eval_network_with_cuts(fixture, inputs, interventions, cut_mask) == expected:
                correct_weight += weight
    return correct_weight / (total_input_weight * cut_states)


def pairwise_utility_interactions(fixture: Fixture) -> list[dict[str, object]]:
    """Second finite difference of task utility, not PID synergy or IIT Φ."""
    intact = task_utility_for_cuts(fixture, ())
    output: list[dict[str, object]] = []
    for first_index, first in enumerate(fixture.interventions):
        for second in fixture.interventions[first_index + 1:]:
            first_utility = task_utility_for_cuts(fixture, (first,))
            second_utility = task_utility_for_cuts(fixture, (second,))
            joint_utility = task_utility_for_cuts(fixture, (first, second))
            output.append({
                "first_intervention_id": first.identifier,
                "second_intervention_id": second.identifier,
                "intact_utility": intact,
                "first_cut_utility": first_utility,
                "second_cut_utility": second_utility,
                "joint_cut_utility": joint_utility,
                "utility_interaction_contrast": intact - first_utility - second_utility + joint_utility,
            })
    return output


def task_utility(fixture: Fixture, intervention: Intervention | None = None) -> float:
    input_assignments = [
        tuple(bool((state >> index) & 1) for index in range(fixture.input_count))
        for state in range(1 << fixture.input_count)
    ]
    coin_assignments = (False, True) if intervention else (False,)
    total_input_weight = sum(fixture.input_state_weights)
    correct_weight = 0
    for state, inputs in enumerate(input_assignments):
        weight = fixture.input_state_weights[state]
        target = expected_task(fixture, inputs)
        for coin in coin_assignments:
            if eval_network(fixture, inputs, intervention, coin) == target:
                correct_weight += weight
    return correct_weight / (total_input_weight * len(coin_assignments))


def entropy_bits_from_counts(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total)
                for count in counts if count > 0)


def input_information_baseline(fixture: Fixture) -> dict[str, object]:
    """Exact H(Y), I(X_i;Y), and I(all inputs;Y) under the manifest prior."""
    total_weight = sum(fixture.input_state_weights)
    target_counts = [0, 0]
    joint_counts = [[[0, 0] for _ in range(2)] for _ in range(fixture.input_count)]
    for state in range(1 << fixture.input_count):
        inputs = tuple(bool((state >> index) & 1) for index in range(fixture.input_count))
        target_value = int(expected_task(fixture, inputs))
        weight = fixture.input_state_weights[state]
        target_counts[target_value] += weight
        for index, input_value in enumerate(inputs):
            joint_counts[index][int(input_value)][target_value] += weight
    target_entropy = entropy_bits_from_counts(target_counts)
    individual_mi = []
    for counts in joint_counts:
        conditional_entropy = 0.0
        for row in counts:
            row_weight = sum(row)
            if row_weight:
                conditional_entropy += row_weight / total_weight * entropy_bits_from_counts(row)
        mi = target_entropy - conditional_entropy
        individual_mi.append(0.0 if abs(mi) < TOLERANCE else mi)
    return {
        "target_entropy_bits": target_entropy,
        "individual_input_mi_bits": individual_mi,
        "joint_input_mi_bits": target_entropy,
    }


def evaluate_fixture(fixture: Fixture) -> tuple[list[InterventionScore], list[PartitionScore], dict[str, float | int]]:
    intact = task_utility(fixture)
    scores = []
    for intervention in fixture.interventions:
        utility = task_utility(fixture, intervention)
        kind = "cross_cut" if fixture.components[intervention.source] != fixture.components[intervention.target] else "matched_control"
        scores.append(InterventionScore(
            fixture=fixture.name, identifier=intervention.identifier, kind=kind,
            source=intervention.source, target=intervention.target, cost=intervention.cost,
            intact=intact, utility=utility, loss=intact - utility,
        ))
    cross_losses = [score.loss for score in scores if score.kind == "cross_cut"]
    control_losses = [score.loss for score in scores if score.kind == "matched_control"]
    if not cross_losses or not control_losses:
        raise ValueError(f"{fixture.name}: missing cross-cut/control stratum")
    mean_cross = sum(cross_losses) / len(cross_losses)
    mean_control = sum(control_losses) / len(control_losses)
    tccip = mean_cross - mean_control

    total_partitions = (1 << (fixture.node_count - 1)) - 1
    partition_scores = []
    for mask in range(1, 1 << fixture.node_count):
        if mask & 1:
            continue
        cross = []
        controls = []
        for score in scores:
            cross_partition = ((mask >> score.source) & 1) != ((mask >> score.target) & 1)
            (cross if cross_partition else controls).append(score.loss)
        if not cross or not controls:
            continue
        mean_cut = sum(cross) / len(cross)
        mean_control = sum(controls) / len(controls)
        labels = "".join("B" if mask & (1 << index) else "A" for index in range(fixture.node_count))
        partition_scores.append(PartitionScore(
            fixture.name, mask, labels, mean_cut, mean_control, mean_cut - mean_control, len(cross), len(controls)
        ))
    counts = {"total": total_partitions, "usable": len(partition_scores),
              "skipped": total_partitions - len(partition_scores)}
    if counts["total"] != fixture.expected_partition_counts[0] or counts["usable"] != fixture.expected_partition_counts[1] or counts["skipped"] != fixture.expected_partition_counts[2]:
        raise AssertionError(f"{fixture.name}: partition counts {counts}, expected {fixture.expected_partition_counts}")
    if not math.isclose(min(s.tccip for s in partition_scores), fixture.expected_partition_range[0], abs_tol=TOLERANCE):
        raise AssertionError(f"{fixture.name}: minimum partition profile differs")
    if not math.isclose(max(s.tccip for s in partition_scores), fixture.expected_partition_range[1], abs_tol=TOLERANCE):
        raise AssertionError(f"{fixture.name}: maximum partition profile differs")
    for actual, expected, label in zip((intact, mean_cross, mean_control, tccip), fixture.expected,
                                       ("intact utility", "cross-cut loss", "control loss", "TCCIP")):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOLERANCE):
            raise AssertionError(f"{fixture.name}: {label} {actual} != oracle {expected}")
    return scores, partition_scores, {"intact": intact, "cross_loss": mean_cross,
                                      "control_loss": mean_control, "tccip": tccip, **counts}


def render_artifact(fixtures: Sequence[Fixture], manifest_bytes: bytes, source_sha: str) -> str:
    rows: list[list[str]] = []
    for fixture in fixtures:
        interventions, partitions, summary = evaluate_fixture(fixture)
        information = input_information_baseline(fixture)
        pair_interactions = pairwise_utility_interactions(fixture)
        prior = ",".join(map(str, fixture.input_state_weights))
        target = canonical_target(fixture.target)
        for result in interventions:
            rows.append([
                "intervention", result.fixture, prior, target, result.identifier, result.kind,
                str(result.source), str(result.target), str(result.cost),
                f"{result.intact:.12g}", f"{result.utility:.12g}", f"{result.loss:.12g}",
                "", "", "", "", "", "", "", "", "", "",
            ])
        for point in partitions:
            rows.append([
                "partition", point.fixture, prior, target, "", "", "", "", "",
                "", "", "", str(point.mask), point.labels,
                f"{point.cross_loss:.12g}", f"{point.control_loss:.12g}",
                f"{point.tccip:.12g}", str(point.n_cross), str(point.n_control), "", "", "",
            ])
        rows.append([
            "summary", fixture.name, prior, target, "", "", "", "", "",
            f"{summary['intact']:.12g}", "", "", "", "",
            f"{summary['cross_loss']:.12g}", f"{summary['control_loss']:.12g}",
            f"{summary['tccip']:.12g}", str(summary['usable']), str(summary['skipped']), "", "", "",
        ])
        rows.append([
            "information_baseline", fixture.name, prior, target, "", "", "", "", "",
            "", "", "", "", "", "", "", "", "", "",
            f"{information['target_entropy_bits']:.12g}",
            ",".join(f"{value:.12g}" for value in information["individual_input_mi_bits"]),
            f"{information['joint_input_mi_bits']:.12g}",
        ])
        for interaction in pair_interactions:
            rows.append([
                "utility_interaction", fixture.name, prior, target, "", "pairwise_utility_interaction",
                "", "", "", f"{interaction['intact_utility']:.12g}", "", "", "", "", "", "", "", "", "",
                "", "", "", interaction["first_intervention_id"], interaction["second_intervention_id"],
                f"{interaction['first_cut_utility']:.12g}",
                f"{interaction['second_cut_utility']:.12g}",
                f"{interaction['joint_cut_utility']:.12g}",
                f"{interaction['utility_interaction_contrast']:.12g}",
            ])
    output = io.StringIO()
    output.write("# schema=tccip-independent-reference-v2\n")
    output.write(f"# source_sha={source_sha or 'UNBOUND'}\n")
    output.write(f"# manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}\n")
    output.write("# oracle=independent Python standard-library implementation; this is not Rust execution output\n")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    header = ["record_type", "fixture", "input_state_weights", "task_target", "intervention_id",
              "kind", "source", "target", "cost", "intact_utility", "utility", "utility_loss",
              "partition_mask", "partition_labels", "cross_mean_loss", "control_mean_loss",
              "tccip", "n_cross", "n_control", "target_entropy_bits",
              "individual_input_mi_bits", "joint_input_mi_bits", "first_intervention_id",
              "second_intervention_id", "first_cut_utility", "second_cut_utility",
              "joint_cut_utility", "utility_interaction_contrast"]
    writer.writerow(header)
    for row in rows:
        if len(row) > len(header):
            raise ValueError(f"artifact row has {len(row)} fields; schema allows {len(header)}")
        writer.writerow(row + [""] * (len(header) - len(row)))
    return output.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-sha", default=os.environ.get("SYMTHAEA_SOURCE_SHA", "UNBOUND"))
    parser.add_argument("--output", type=Path, help="write artifact here instead of stdout")
    args = parser.parse_args(argv)
    try:
        fixtures, manifest_bytes = parse_manifest(args.manifest)
        artifact = render_artifact(fixtures, manifest_bytes, args.source_sha)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(artifact, encoding="utf-8")
            print(f"wrote {args.output} ({len(artifact.encode('utf-8'))} bytes)", file=sys.stderr)
        else:
            sys.stdout.write(artifact)
        print(f"REFERENCE_ORACLE_CHECKS_PASSED: {len(fixtures)} fixtures; source_sha={args.source_sha}", file=sys.stderr)
        return 0
    except (OSError, UnicodeError, ValueError, AssertionError) as error:
        print(f"NOT QUALIFIED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
