#!/usr/bin/env python3
"""Exact reference oracle for deterministic adaptive intervention-policy fixtures.

V1 uses a complete deterministic potential-outcome vector per biological-world /
coherent-nuisance state. It makes no probability or independence assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


FEASIBLE = 'FEASIBLE'
VALID_ACTION_KINDS = {'OPTOGENETIC', 'ELECTRICAL', 'PHARMACOLOGICAL', 'SENSORY', 'BEHAVIORAL', 'SIMULATED', 'OTHER'}


class SearchLimitExceeded(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_hex(value: Any, size: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf'[0-9a-f]{{{size}}}', value) is not None


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return hashlib.sha256(raw).hexdigest()


def validate_input(data: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict], dict[tuple[str, str], dict[str, str]]]:
    require(data.get('schema_version') == 1, 'schema_version must equal 1')
    require(data.get('claim_scope') == 'synthetic_known_truth', 'V1 reference oracle accepts synthetic known-truth fixtures only')
    require(bool(re.fullmatch(r'[a-z0-9][a-z0-9._:-]{0,127}', data.get('analysis_id', ''))), 'analysis_id malformed')
    require(valid_hex(data.get('dataset_revision_commit_sha'), 40), 'dataset revision commit SHA malformed')
    query = data['query']
    require(bool(re.fullmatch(r'[a-z0-9][a-z0-9._:-]{0,127}', query.get('query_id', ''))), 'query ID malformed')
    require(valid_hex(query.get('query_spec_sha256'), 64), 'query specification digest malformed')
    require(valid_hex(query.get('decision_rule_sha256'), 64), 'query decision-rule digest malformed')

    worlds = sorted(data['worlds'], key=lambda row: row['world_id'])
    nuisance = sorted(data['nuisance_models'], key=lambda row: row['nuisance_model_id'])
    actions = sorted(data['actions'], key=lambda row: row['action_id'])
    require(2 <= len(worlds) <= 12, 'world count outside V1 bounds')
    require(1 <= len(nuisance) <= 8, 'nuisance-model count outside V1 bounds')
    require(1 <= len(actions) <= 12, 'action count outside V1 bounds')

    world_ids = [row['world_id'] for row in worlds]
    nuisance_ids = [row['nuisance_model_id'] for row in nuisance]
    action_ids = [row['action_id'] for row in actions]
    require(len(world_ids) == len(set(world_ids)), 'duplicate world ID')
    require(len(nuisance_ids) == len(set(nuisance_ids)), 'duplicate nuisance-model ID')
    require(len(action_ids) == len(set(action_ids)), 'duplicate action ID')
    for row in worlds:
        require(bool(re.fullmatch(r'[a-z0-9][a-z0-9._:-]{0,63}', row['world_id'])), 'world ID malformed')
        require(row['biological_disposition'] == 'ADMISSIBLE', 'only admissible worlds may enter V1')
        require(valid_hex(row['graph_sha256'], 64), 'world graph digest malformed')
        require(valid_hex(row['source_evidence_sha256'], 64), 'world source-evidence digest malformed')
        require(isinstance(row['query_decision'], bool), 'world query decision must be boolean')
    for row in nuisance:
        require(bool(re.fullmatch(r'[a-z0-9][a-z0-9._:-]{0,63}', row['nuisance_model_id'])), 'nuisance-model ID malformed')
        require(valid_hex(row['model_sha256'], 64), 'nuisance-model digest malformed')
        require(isinstance(row['description'], str) and row['description'].strip(), 'nuisance description is empty')
    for row in actions:
        require(bool(re.fullmatch(r'[a-z0-9][a-z0-9._:-]{0,63}', row['action_id'])), 'action ID malformed')
        require(row['intervention_kind'] in VALID_ACTION_KINDS, 'unknown intervention kind')
        require(isinstance(row['effort_units'], int) and not isinstance(row['effort_units'], bool) and row['effort_units'] > 0, 'action effort must be a positive integer')
        require(row['feasibility_status'] in {'FEASIBLE', 'DISALLOWED', 'INACCESSIBLE', 'NOT_ASSESSED'}, 'unknown action feasibility status')
        require(valid_hex(row['feasibility_receipt_sha256'], 64), 'feasibility receipt malformed')
        require(valid_hex(row['safety_profile_sha256'], 64), 'safety-profile digest malformed')

    outcome_index: dict[tuple[str, str], dict[str, str]] = {}
    for row in data['potential_outcomes']:
        key = (row['world_id'], row['nuisance_model_id'])
        require(key not in outcome_index, f'duplicate world/nuisance potential-outcome row: {key}')
        require(key[0] in set(world_ids), f'potential outcome references unknown world: {key[0]}')
        require(key[1] in set(nuisance_ids), f'potential outcome references unknown nuisance model: {key[1]}')
        observed: dict[str, str] = {}
        for item in row['observations']:
            action_id = item['action_id']
            outcome_id = item['outcome_id']
            require(action_id in set(action_ids), f'potential outcome references unknown action: {action_id}')
            require(bool(re.fullmatch(r'[a-z0-9][a-z0-9._:-]{0,63}', outcome_id)), f'outcome ID malformed: {outcome_id}')
            require(action_id not in observed, f'duplicate action outcome in latent state: {key}/{action_id}')
            observed[action_id] = outcome_id
        require(set(observed) == set(action_ids), f'potential-outcome vector is incomplete for {key}')
        outcome_index[key] = observed
    expected_keys = {(world_id, nuisance_id) for world_id in world_ids for nuisance_id in nuisance_ids}
    require(set(outcome_index) == expected_keys, 'potential-outcome matrix is not a complete world × nuisance Cartesian product')

    limits = data['search_limits']
    max_memo_states = limits['max_memo_states']
    max_policy_depth = limits['max_policy_depth']
    require(isinstance(max_memo_states, int) and 1 <= max_memo_states <= 2_000_000, 'max_memo_states outside V1 bounds')
    require(isinstance(max_policy_depth, int) and 1 <= max_policy_depth <= len(actions), 'max_policy_depth outside V1 bounds')
    return worlds, nuisance, actions, outcome_index


def solve_adaptive(worlds: list[dict], nuisance: list[dict], actions: list[dict], outcomes: dict, max_memo_states: int, max_depth: int) -> tuple[dict | None, int]:
    world_by_id = {row['world_id']: row for row in worlds}
    action_by_id = {row['action_id']: row for row in actions}
    feasible_ids = tuple(sorted(row['action_id'] for row in actions if row['feasibility_status'] == FEASIBLE))
    initial_states = tuple(sorted((world_id, nuisance_row['nuisance_model_id']) for world_id in world_by_id for nuisance_row in nuisance))
    memo_states: set[tuple[tuple[tuple[str, str], ...], tuple[str, ...]]] = set()

    @lru_cache(maxsize=None)
    def solve(states: tuple[tuple[str, str], ...], used: tuple[str, ...]) -> tuple[int, dict] | None:
        key = (states, used)
        memo_states.add(key)
        if len(memo_states) > max_memo_states:
            raise SearchLimitExceeded('adaptive policy memo-state limit exceeded')
        decisions = {world_by_id[world_id]['query_decision'] for world_id, _nuisance_id in states}
        if len(decisions) == 1:
            return 0, {
                'node_kind': 'STOP',
                'query_decision': next(iter(decisions)),
                'remaining_world_ids': sorted({world_id for world_id, _nuisance_id in states}),
                'remaining_latent_state_count': len(states),
            }
        if len(used) >= max_depth:
            return None
        best: tuple[int, str, dict] | None = None
        used_set = set(used)
        for action_id in feasible_ids:
            if action_id in used_set:
                continue
            branches: dict[str, list[tuple[str, str]]] = {}
            for state in states:
                outcome_id = outcomes[state][action_id]
                branches.setdefault(outcome_id, []).append(state)
            child_rows = []
            child_costs = []
            valid_action = True
            next_used = tuple(sorted((*used, action_id)))
            for outcome_id in sorted(branches):
                child_states = tuple(sorted(branches[outcome_id]))
                child = solve(child_states, next_used)
                if child is None:
                    valid_action = False
                    break
                child_cost, child_node = child
                child_costs.append(child_cost)
                child_rows.append({'outcome_id': outcome_id, 'child': child_node})
            if not valid_action:
                continue
            worst_cost = action_by_id[action_id]['effort_units'] + max(child_costs, default=0)
            node = {
                'node_kind': 'INTERVENE',
                'action_id': action_id,
                'effort_units': action_by_id[action_id]['effort_units'],
                'worst_case_subtree_effort_units': worst_cost,
                'branches': child_rows,
            }
            candidate = (worst_cost, action_id, node)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            return None
        return best[0], best[2]

    result = solve(initial_states, tuple())
    return (None if result is None else {'worst_case_effort_units': result[0], 'policy_tree': result[1]}), len(memo_states)


def verify_policy_tree(node: dict, states: tuple[tuple[str, str], ...], used: tuple[str, ...], world_by_id: dict, action_by_id: dict, outcomes: dict) -> int:
    decisions = {world_by_id[world_id]["query_decision"] for world_id, _nuisance_id in states}
    require(node.get("node_kind") in {"STOP", "INTERVENE"}, "policy node has unknown kind")
    if node["node_kind"] == "STOP":
        require(len(decisions) == 1, "STOP leaf retains opposite-query biological worlds")
        require(node.get("query_decision") == next(iter(decisions)), "STOP leaf declares the wrong query decision")
        require(node.get("remaining_world_ids") == sorted({world_id for world_id, _nuisance_id in states}), "STOP leaf world set is incorrect")
        require(node.get("remaining_latent_state_count") == len(states), "STOP leaf latent-state count is incorrect")
        return 0

    action_id = node.get("action_id")
    require(action_id in action_by_id, "policy references an unknown action")
    action = action_by_id[action_id]
    require(action["feasibility_status"] == FEASIBLE, "policy uses an infeasible action")
    require(action_id not in used, "policy repeats an intervention on one path")
    expected_branches: dict[str, list[tuple[str, str]]] = {}
    for state in states:
        expected_branches.setdefault(outcomes[state][action_id], []).append(state)
    actual_branches = node.get("branches")
    require(isinstance(actual_branches, list), "intervention node branches are not an array")
    actual_by_outcome = {row.get("outcome_id"): row.get("child") for row in actual_branches}
    require(len(actual_by_outcome) == len(actual_branches), "policy repeats an outcome branch")
    require(set(actual_by_outcome) == set(expected_branches), "policy has missing or unsupported outcome branches")
    next_used = tuple(sorted((*used, action_id)))
    child_costs = []
    for outcome_id in sorted(expected_branches):
        child_states = tuple(sorted(expected_branches[outcome_id]))
        child_costs.append(
            verify_policy_tree(actual_by_outcome[outcome_id], child_states, next_used, world_by_id, action_by_id, outcomes)
        )
    exact_cost = action["effort_units"] + max(child_costs, default=0)
    require(node.get("effort_units") == action["effort_units"], "policy action cost differs from the frozen action set")
    require(node.get("worst_case_subtree_effort_units") == exact_cost, "policy subtree cost is incorrect")
    return exact_cost


def solve_fixed(worlds: list[dict], nuisance: list[dict], actions: list[dict], outcomes: dict) -> dict | None:
    world_by_id = {row['world_id']: row for row in worlds}
    feasible = sorted((row for row in actions if row['feasibility_status'] == FEASIBLE), key=lambda row: row['action_id'])
    feasible_ids = [row['action_id'] for row in feasible]
    best: tuple[int, int, tuple[str, ...]] | None = None
    for subset_size in range(len(feasible_ids) + 1):
        for selected in itertools.combinations(feasible_ids, subset_size):
            cost = sum(next(row['effort_units'] for row in feasible if row['action_id'] == action_id) for action_id in selected)
            decision_by_tuple: dict[tuple[str, ...], set[bool]] = {}
            for world_id in sorted(world_by_id):
                for nuisance_row in nuisance:
                    state = (world_id, nuisance_row['nuisance_model_id'])
                    joint_outcome = tuple(outcomes[state][action_id] for action_id in selected)
                    decision_by_tuple.setdefault(joint_outcome, set()).add(world_by_id[world_id]['query_decision'])
            if any(len(decisions) > 1 for decisions in decision_by_tuple.values()):
                continue
            candidate = (cost, subset_size, selected)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    return {'minimum_effort_units': best[0], 'action_ids': list(best[2])}


def build_report(raw: bytes, data: dict) -> dict:
    worlds, nuisance, actions, outcomes = validate_input(data)
    limits = data['search_limits']
    try:
        adaptive, memo_state_count = solve_adaptive(
            worlds, nuisance, actions, outcomes, limits['max_memo_states'], limits['max_policy_depth']
        )
    except SearchLimitExceeded:
        return {
            'schema_version': 1, 'analysis_id': data['analysis_id'],
            'input_sha256': hashlib.sha256(raw).hexdigest(),
            'status': 'INCOMPLETE_SEARCH_BUDGET_EXCEEDED', 'search_complete': False,
            'claim_scope': 'synthetic_known_truth',
        }
    fixed = solve_fixed(worlds, nuisance, actions, outcomes)
    if adaptive is not None:
        state_set = tuple(sorted((world["world_id"], nuisance_row["nuisance_model_id"]) for world in worlds for nuisance_row in nuisance))
        world_by_id = {world["world_id"]: world for world in worlds}
        action_by_id = {action["action_id"]: action for action in actions}
        replayed_cost = verify_policy_tree(
            adaptive["policy_tree"], state_set, tuple(), world_by_id, action_by_id, outcomes
        )
        require(replayed_cost == adaptive["worst_case_effort_units"], "independent policy-tree replay cost differs from optimizer cost")
    adaptive_cost = None if adaptive is None else adaptive['worst_case_effort_units']
    fixed_cost = None if fixed is None else fixed['minimum_effort_units']
    savings = None if adaptive_cost is None or fixed_cost is None else fixed_cost - adaptive_cost
    require(savings is None or savings >= 0, 'adaptive minimax cost exceeded exact fixed-plan optimum')
    if (adaptive is None) != (fixed is None):
        raise ValueError('adaptive and fixed resolving-plan existence must agree for a complete deterministic potential-outcome table')

    world_rows = [{key: row[key] for key in ('world_id', 'graph_sha256', 'source_evidence_sha256', 'biological_disposition', 'query_decision')} for row in worlds]
    nuisance_rows = [{key: row[key] for key in ('nuisance_model_id', 'model_sha256')} for row in nuisance]
    action_rows = [{key: row[key] for key in ('action_id', 'intervention_kind', 'effort_units', 'feasibility_status', 'feasibility_receipt_sha256', 'safety_profile_sha256')} for row in actions]
    potential_rows = [
        {'world_id': world_id, 'nuisance_model_id': nuisance_id, 'observations': [
            {'action_id': action_id, 'outcome_id': outcomes[(world_id, nuisance_id)][action_id]}
            for action_id in sorted(outcomes[(world_id, nuisance_id)])
        ]}
        for world_id, nuisance_id in sorted(outcomes)
    ]
    report = {
        'schema_version': 1,
        'tool': 'connectome-adaptive-intervention-policy-reference-oracle',
        'analysis_id': data['analysis_id'],
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'dataset_revision_commit_sha': data['dataset_revision_commit_sha'],
        'query_id': data['query']['query_id'],
        'world_set_sha256': canonical_sha256(world_rows),
        'nuisance_set_sha256': canonical_sha256(nuisance_rows),
        'action_set_sha256': canonical_sha256(action_rows),
        'potential_outcome_table_sha256': canonical_sha256(potential_rows),
        'biological_world_count': len(worlds),
        'nuisance_model_count': len(nuisance),
        'latent_state_count': len(worlds) * len(nuisance),
        'action_count': len(actions),
        'feasible_action_count': sum(1 for row in actions if row['feasibility_status'] == FEASIBLE),
        'adaptive_status': 'EXACT_POLICY' if adaptive is not None else 'NO_GUARANTEED_QUERY_RESOLUTION',
        'adaptive_worst_case_effort_units': adaptive_cost,
        'adaptive_root_action_id': None if adaptive is None else adaptive['policy_tree'].get('action_id'),
        'adaptive_policy_tree': None if adaptive is None else adaptive['policy_tree'],
        'adaptive_memo_state_count': memo_state_count,
        'fixed_status': 'EXACT_FIXED_PLAN' if fixed is not None else 'NO_GUARANTEED_FIXED_PLAN',
        'fixed_minimum_effort_units': fixed_cost,
        'fixed_action_ids': [] if fixed is None else fixed['action_ids'],
        'adaptivity_savings_units': savings,
        'adaptivity_savings_basis_points': None if savings is None or fixed_cost == 0 else (savings * 10000) // fixed_cost,
        'adaptive_not_worse_than_fixed': savings is None or savings >= 0,
        'search_complete': True,
        'status': 'SYNTHETIC_EXACT_POLICY_VALIDATED' if adaptive is not None else 'SYNTHETIC_NO_GUARANTEED_RESOLUTION',
        'probability_claim': 'NONE_DETERMINISTIC_POTENTIAL_OUTCOMES_ONLY',
        'biological_truth_disposition': 'NOT_ESTABLISHED_BY_THIS_ORACLE',
        'hypothesis_space_completeness': 'NOT_ESTABLISHED_BY_THIS_ORACLE',
    }
    expected = data['expected']
    observed_expected = {
        'adaptive_status': report['adaptive_status'],
        'adaptive_worst_case_effort_units': report['adaptive_worst_case_effort_units'],
        'adaptive_root_action_id': report['adaptive_root_action_id'],
        'fixed_status': report['fixed_status'],
        'fixed_minimum_effort_units': report['fixed_minimum_effort_units'],
        'fixed_action_ids': report['fixed_action_ids'],
        'adaptivity_savings_units': report['adaptivity_savings_units'],
    }
    require(observed_expected == expected, f'known-answer mismatch: expected {expected}, observed {observed_expected}')
    return report


def self_test(fixtures: list[Path]) -> None:
    require(len(fixtures) == 3, 'self-test expects the three checked-in fixtures')
    reports = []
    for path in fixtures:
        raw = path.read_bytes()
        reports.append(build_report(raw, json.loads(raw)))
    require(reports[0]['adaptive_worst_case_effort_units'] == 3, 'adaptive savings fixture must cost 3')
    require(reports[0]['fixed_minimum_effort_units'] == 4, 'fixed plan must cost 4')
    require(reports[0]['adaptivity_savings_basis_points'] == 2500, 'adaptive saving must be 25%')
    require(reports[1]['adaptive_worst_case_effort_units'] == 4, 'nuisance erasure must remove adaptive saving')
    require(reports[1]['adaptivity_savings_units'] == 0, 'nuisance erasure saving must be zero')
    require(reports[2]['adaptive_status'] == 'NO_GUARANTEED_QUERY_RESOLUTION', 'overlap control must have no policy')
    require(reports[2]['fixed_status'] == 'NO_GUARANTEED_FIXED_PLAN', 'overlap control must have no fixed plan')

    data = json.loads(fixtures[0].read_text())
    broken = json.loads(json.dumps(data))
    broken['potential_outcomes'].pop()
    try:
        build_report(json.dumps(broken).encode(), broken)
    except ValueError as exc:
        require('Cartesian product' in str(exc) or 'incomplete' in str(exc), 'wrong failure for missing potential-outcome cell')
    else:
        raise AssertionError('missing potential-outcome cell was not rejected')

    broken = json.loads(json.dumps(data))
    broken['actions'][0]['feasibility_status'] = 'DISALLOWED'
    # The frozen expected root is no longer valid; this catches accidental use of an infeasible action.
    try:
        build_report(json.dumps(broken).encode(), broken)
    except ValueError:
        pass
    else:
        raise AssertionError('infeasible screening action was incorrectly retained in the expected policy')

    # Row order must not change the canonical set digests or exact optima.
    permuted = json.loads(json.dumps(data))
    for field in ("worlds", "nuisance_models", "actions", "potential_outcomes"):
        permuted[field].reverse()
    permuted_report = build_report(json.dumps(permuted).encode(), permuted)
    for field in (
        "world_set_sha256",
        "nuisance_set_sha256",
        "action_set_sha256",
        "potential_outcome_table_sha256",
        "adaptive_worst_case_effort_units",
        "fixed_minimum_effort_units",
        "adaptive_root_action_id",
    ):
        require(permuted_report[field] == reports[0][field], f"row permutation changed {field}")

    # A cheap but disallowed perfect action must not enter either optimum.
    decoy = json.loads(json.dumps(data))
    decoy_action = {
        "action_id": "perfect-decoy",
        "intervention_kind": "SIMULATED",
        "effort_units": 1,
        "feasibility_status": "DISALLOWED",
        "feasibility_receipt_sha256": "d" * 64,
        "safety_profile_sha256": "e" * 64,
    }
    decoy["actions"].append(decoy_action)
    for potential in decoy["potential_outcomes"]:
        potential["observations"].append({
            "action_id": "perfect-decoy",
            "outcome_id": "decoy-" + potential["world_id"] + "-" + potential["nuisance_model_id"],
        })
    decoy_report = build_report(json.dumps(decoy).encode(), decoy)
    require(decoy_report["adaptive_worst_case_effort_units"] == 3, "disallowed action changed adaptive optimum")
    require(decoy_report["fixed_minimum_effort_units"] == 4, "disallowed action changed fixed optimum")
    require("perfect-decoy" not in decoy_report["fixed_action_ids"], "fixed plan used a disallowed action")

    # Increasing one action cost cannot reduce either exact optimum.
    more_expensive = json.loads(json.dumps(data))
    next(row for row in more_expensive["actions"] if row["action_id"] == "resolve-a")["effort_units"] += 3
    w2, n2, a2, o2 = validate_input(more_expensive)
    adaptive2, _count2 = solve_adaptive(
        w2, n2, a2, o2, more_expensive["search_limits"]["max_memo_states"], more_expensive["search_limits"]["max_policy_depth"]
    )
    fixed2 = solve_fixed(w2, n2, a2, o2)
    require(adaptive2 is not None and adaptive2["worst_case_effort_units"] >= reports[0]["adaptive_worst_case_effort_units"], "increasing an action cost lowered adaptive optimum")
    require(fixed2 is not None and fixed2["minimum_effort_units"] >= reports[0]["fixed_minimum_effort_units"], "increasing an action cost lowered fixed optimum")

    # A deliberately exhausted state budget is incomplete, never exact or no-policy.
    capped = json.loads(json.dumps(data))
    capped["search_limits"]["max_memo_states"] = 1
    capped_raw = json.dumps(capped).encode()
    capped_report = build_report(capped_raw, capped)
    require(capped_report["status"] == "INCOMPLETE_SEARCH_BUDGET_EXCEEDED", "exhausted search budget was not fail-closed")
    require(capped_report["search_complete"] is False, "exhausted search was marked complete")

    too_deep = json.loads(json.dumps(data))
    too_deep["search_limits"]["max_policy_depth"] = len(too_deep["actions"]) + 1
    try:
        validate_input(too_deep)
    except ValueError as exc:
        require("max_policy_depth" in str(exc), "wrong failure for a depth above the action universe")
    else:
        raise AssertionError("policy depth above the available action count was accepted")

    broken = json.loads(json.dumps(data))
    broken['potential_outcomes'][0]['observations'][1]['outcome_id'] = 'a-minus'
    try:
        build_report(json.dumps(broken).encode(), broken)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpected mutation escaped the frozen known-answer expectation')
    print('adaptive intervention policy oracle and mutation controls: PASS')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, action='append')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    fixtures = args.fixture or [
        root / 'docs/neuroscience/examples/CONNECTOME_ADAPTIVE_INTERVENTION_POLICY_ADAPTIVE_SAVINGS.example.json',
        root / 'docs/neuroscience/examples/CONNECTOME_ADAPTIVE_INTERVENTION_POLICY_NUISANCE_ERASURE.example.json',
        root / 'docs/neuroscience/examples/CONNECTOME_ADAPTIVE_INTERVENTION_POLICY_NO_RESOLUTION.example.json',
    ]
    try:
        if args.self_test:
            self_test(fixtures)
            return 0
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        for fixture in fixtures:
            raw = fixture.read_bytes()
            report = build_report(raw, json.loads(raw))
            rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
            if args.output_dir is not None:
                output = args.output_dir / f'{fixture.stem}.report.json'
                output.write_text(rendered, encoding='utf-8')
            print(rendered, end='')
        print('adaptive intervention policy reference oracle: PASS')
        return 0
    except Exception as exc:
        print(f'adaptive intervention policy oracle: FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())