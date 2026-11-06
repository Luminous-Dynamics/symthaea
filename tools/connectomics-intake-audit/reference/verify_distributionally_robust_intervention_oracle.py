#!/usr/bin/env python3
"""Exact finite common-policy oracle for distributionally robust intervention design.

Uses only the Python standard library. For each intervention it enumerates every
deterministic observed-outcome -> SUPPORT/REFUTE policy, minimizes worst-case
expected 0-1 loss across the supplied joint-distribution vertices, and computes
the separate clairvoyant lower bound. It never lets the policy inspect a model ID.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

DECISIONS = ('SUPPORT', 'REFUTE')
CRITICAL_FEASIBILITY = {'DISALLOWED', 'INACCESSIBLE', 'NOT_ASSESSED'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def entropy(probabilities: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def ensure_unique(values: list[str], label: str) -> None:
    require(all(isinstance(x, str) and x.strip() for x in values), f'{label} contains an empty identity')
    require(len(values) == len(set(values)), f'{label} contains duplicate identities')


def baseline_robust_loss(models: list[dict[str, Any]], worlds: dict[str, dict[str, Any]]) -> tuple[float, str, list[float]]:
    candidates = []
    for decision in DECISIONS:
        per_vertex = []
        for model in models:
            prior = {row['world_id']: row['probability'] for row in model['prior']}
            risk = sum(
                prior[world_id]
                for world_id, world in worlds.items()
                if (decision == 'SUPPORT') != world['query_decision']
            )
            per_vertex.append(risk)
        candidates.append((max(per_vertex), decision, per_vertex))
    best = min(candidates, key=lambda row: (row[0], DECISIONS.index(row[1])))
    return best


def policy_risk(
    model: dict[str, Any],
    worlds: dict[str, dict[str, Any]],
    prediction_index: dict[tuple[str, str, str], dict[str, float]],
    action_id: str,
    policy: dict[str, str],
) -> float:
    prior = {row['world_id']: row['probability'] for row in model['prior']}
    total = 0.0
    for world_id, world in worlds.items():
        distribution = prediction_index[(model['model_vertex_id'], action_id, world_id)]
        for outcome_id, probability in distribution.items():
            decision = policy[outcome_id]
            wrong = (decision == 'SUPPORT') != world['query_decision']
            total += prior[world_id] * probability * float(wrong)
    return total


def vertex_eig(
    model: dict[str, Any],
    worlds: dict[str, dict[str, Any]],
    prediction_index: dict[tuple[str, str, str], dict[str, float]],
    action_id: str,
) -> float:
    prior = {row['world_id']: row['probability'] for row in model['prior']}
    predictive: dict[str, float] = {}
    conditional_entropy = 0.0
    for world_id in worlds:
        distribution = prediction_index[(model['model_vertex_id'], action_id, world_id)]
        conditional_entropy += prior[world_id] * entropy(list(distribution.values()))
        for outcome_id, probability in distribution.items():
            predictive[outcome_id] = predictive.get(outcome_id, 0.0) + prior[world_id] * probability
    value = entropy(list(predictive.values())) - conditional_entropy
    return max(0.0, value)


def pairwise_midpoint_eig(
    left: dict[str, Any],
    right: dict[str, Any],
    worlds: dict[str, dict[str, Any]],
    prediction_index: dict[tuple[str, str, str], dict[str, float]],
    action_id: str,
) -> float:
    left_prior = {row['world_id']: row['probability'] for row in left['prior']}
    right_prior = {row['world_id']: row['probability'] for row in right['prior']}
    mixed_prior = {
        world_id: 0.5 * (left_prior[world_id] + right_prior[world_id])
        for world_id in worlds
    }
    mixed_conditional: dict[str, dict[str, float]] = {}
    predictive: dict[str, float] = {}
    conditional_entropy = 0.0
    for world_id in worlds:
        joint: dict[str, float] = {}
        for vertex, weight, prior in (
            (left, 0.5, left_prior[world_id]),
            (right, 0.5, right_prior[world_id]),
        ):
            distribution = prediction_index[(vertex['model_vertex_id'], action_id, world_id)]
            for outcome_id, probability in distribution.items():
                joint[outcome_id] = joint.get(outcome_id, 0.0) + weight * prior * probability
        conditional = {
            outcome_id: joint_probability / mixed_prior[world_id]
            for outcome_id, joint_probability in joint.items()
        }
        mixed_conditional[world_id] = conditional
        conditional_entropy += mixed_prior[world_id] * entropy(list(conditional.values()))
        for outcome_id, probability in joint.items():
            predictive[outcome_id] = predictive.get(outcome_id, 0.0) + probability
    return max(0.0, entropy(list(predictive.values())) - conditional_entropy)


def action_result(
    action: dict[str, Any],
    models: list[dict[str, Any]],
    worlds: dict[str, dict[str, Any]],
    prediction_index: dict[tuple[str, str, str], dict[str, float]],
    baseline_loss: float,
    certificate_tolerance: float,
) -> dict[str, Any]:
    action_id = action['intervention_id']
    outcome_ids = sorted({
        outcome_id
        for model in models
        for world_id in worlds
        for outcome_id in prediction_index[(model['model_vertex_id'], action_id, world_id)]
    })
    require(0 < len(outcome_ids) <= 8, f'{action_id}: outcome alphabet must contain 1..8 categories')

    policies = []
    for decisions in itertools.product(DECISIONS, repeat=len(outcome_ids)):
        policy = dict(zip(outcome_ids, decisions))
        vertex_risks = [
            policy_risk(model, worlds, prediction_index, action_id, policy)
            for model in models
        ]
        policies.append((max(vertex_risks), policy, vertex_risks))
    robust_loss, shared_policy, shared_vertex_risks = min(
        policies, key=lambda row: (row[0], tuple(DECISIONS.index(row[1][outcome]) for outcome in outcome_ids))
    )

    clairvoyant_vertex_optima = []
    for model in models:
        best_for_vertex = math.inf
        for decisions in itertools.product(DECISIONS, repeat=len(outcome_ids)):
            policy = dict(zip(outcome_ids, decisions))
            best_for_vertex = min(
                best_for_vertex,
                policy_risk(model, worlds, prediction_index, action_id, policy),
            )
        clairvoyant_vertex_optima.append(best_for_vertex)
    clairvoyant_lower_bound = max(clairvoyant_vertex_optima)
    uncertainty_penalty = max(0.0, robust_loss - clairvoyant_lower_bound)

    vertex_eigs = [
        vertex_eig(model, worlds, prediction_index, action_id)
        for model in models
    ]
    pairwise_midpoint_eigs = [
        pairwise_midpoint_eig(models[i], models[j], worlds, prediction_index, action_id)
        for i in range(len(models))
        for j in range(i + 1, len(models))
    ]
    value_per_cost = max(0.0, baseline_loss - robust_loss) / action['cost_units']
    is_feasible = action['feasibility_status'] == 'FEASIBLE'
    certificate = is_feasible and robust_loss <= certificate_tolerance + 1e-12

    return {
        'intervention_id': action_id,
        'feasibility_status': action['feasibility_status'],
        'optimization_status': 'OPTIMIZED',
        'outcome_ids': outcome_ids,
        'robust_policy_loss': robust_loss,
        'shared_policy': shared_policy,
        'shared_policy_vertex_risks': {
            model['model_vertex_id']: risk
            for model, risk in zip(models, shared_vertex_risks)
        },
        'clairvoyant_lower_bound': clairvoyant_lower_bound,
        'clairvoyant_vertex_optima': {
            model['model_vertex_id']: risk
            for model, risk in zip(models, clairvoyant_vertex_optima)
        },
        'model_uncertainty_penalty': uncertainty_penalty,
        'value_per_cost': value_per_cost,
        'vertex_eig_bits': {
            model['model_vertex_id']: eig
            for model, eig in zip(models, vertex_eigs)
        },
        'vertex_min_eig_bits': min(vertex_eigs),
        'vertex_max_eig_bits': max(vertex_eigs),
        'pairwise_midpoint_eig_min_bits': min(pairwise_midpoint_eigs) if pairwise_midpoint_eigs else None,
        'pairwise_midpoint_eig_max_bits': max(pairwise_midpoint_eigs) if pairwise_midpoint_eigs else None,
        'robust_certificate': certificate,
        'certificate_status': 'ROBUST_CERTIFICATE' if certificate else 'NOT_CERTIFIABLE',
    }


def ancestry_overlaps(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    ordered = sorted(models, key=lambda row: row['model_vertex_id'])
    for index, left in enumerate(ordered):
        left_groups = set(left['dependence_group_ids'])
        for right in ordered[index + 1:]:
            shared = sorted(left_groups & set(right['dependence_group_ids']))
            if shared:
                result.append({
                    'model_vertex_a': left['model_vertex_id'],
                    'model_vertex_b': right['model_vertex_id'],
                    'shared_group_ids': shared,
                })
    return result


def evaluate(data: dict[str, Any], input_sha256: str) -> dict[str, Any]:
    require(data['schema_version'] == 1, 'unsupported schema version')
    require(data['claim_scope'] == 'synthetic_known_truth', 'this oracle version accepts only the synthetic known-truth fixture')
    require(data['dataset_binding']['evidence_tier'] == 'SYNTHETIC', 'synthetic fixture must use SYNTHETIC evidence tier')
    require(data['ambiguity_set']['scope'] == 'CONVEX_HULL_OF_ENUMERATED_JOINT_DISTRIBUTION_VERTICES', 'unsupported ambiguity-set scope')
    require(data['ambiguity_set']['vertex_completeness_status'] == 'DECLARED_COMPLETE_FOR_SYNTHETIC_FIXTURE', 'synthetic vertex set must be declared complete for the fixture')
    require(data['ambiguity_set']['declared_vertex_count'] == len(data['model_vertices']), 'model vertex count does not match ambiguity-set declaration')
    for field in ('dataset_revision_sha256', 'source_access_receipt_sha256') :
        require(valid_sha(data['dataset_binding'][field]), f'{field} malformed')
    for field in ('query_spec_sha256', 'decision_rule_sha256') :
        require(valid_sha(data['query'][field]), f'{field} malformed')
    require(0.0 <= data['query']['certificate_max_expected_loss'] <= 1.0, 'certificate tolerance outside [0,1]')

    worlds_rows = data['worlds']
    actions = sorted(data['interventions'], key=lambda row: row['intervention_id'])
    models = sorted(data['model_vertices'], key=lambda row: row['model_vertex_id'])
    ensure_unique([row['world_id'] for row in worlds_rows], 'world IDs')
    ensure_unique([row['intervention_id'] for row in actions], 'intervention IDs')
    ensure_unique([row['model_vertex_id'] for row in models], 'model vertex IDs')
    worlds = {row['world_id']: row for row in worlds_rows}
    world_ids = set(worlds)
    action_ids = {row['intervention_id'] for row in actions}
    require(all(row['biological_disposition'] == 'ADMISSIBLE' for row in worlds_rows), 'inadmissible/undetermined world included in synthetic prior')
    for world in worlds_rows:
        require(valid_sha(world['graph_sha256']) and valid_sha(world['source_evidence_sha256']), 'world source digest malformed')

    prediction_index: dict[tuple[str, str, str], dict[str, float]] = {}
    for model in models:
        for field in ('causal_model_sha256', 'observation_model_sha256'):
            require(valid_sha(model[field]), f"{field} malformed for {model['model_vertex_id']}")
        ensure_unique(model['dependence_group_ids'], 'model dependence group IDs')
        prior_rows = model['prior']
        ensure_unique([row['world_id'] for row in prior_rows], f"{model['model_vertex_id']} prior world IDs")
        require({row['world_id'] for row in prior_rows} == world_ids, f"{model['model_vertex_id']} prior does not cover the exact world set")
        prior_total = sum(row['probability'] for row in prior_rows)
        require(close(prior_total, 1.0), f"{model['model_vertex_id']} prior probabilities do not sum to one")
        require(all(0.0 <= row['probability'] <= 1.0 for row in prior_rows), 'prior probability outside [0,1]')

        expected_pairs = {(action_id, world_id) for action_id in action_ids for world_id in world_ids}
        seen_pairs = set()
        for prediction in model['predictions']:
            action_id = prediction['intervention_id']
            world_id = prediction['world_id']
            pair = (action_id, world_id)
            require(pair in expected_pairs, f"{model['model_vertex_id']} prediction has unknown action/world pair {pair}")
            require(pair not in seen_pairs, f"{model['model_vertex_id']} prediction duplicates action/world pair {pair}")
            seen_pairs.add(pair)
            require(valid_sha(prediction['trajectory_sha256']), 'trajectory digest malformed')
            outcomes = prediction['outcomes']
            ensure_unique([row['outcome_id'] for row in outcomes], 'outcome IDs')
            total = sum(row['probability'] for row in outcomes)
            require(close(total, 1.0), f"{model['model_vertex_id']} {pair} outcome probabilities do not sum to one")
            require(all(0.0 <= row['probability'] <= 1.0 for row in outcomes), 'outcome probability outside [0,1]')
            prediction_index[(model['model_vertex_id'], action_id, world_id)] = {
                row['outcome_id']: row['probability'] for row in outcomes
            }
        require(seen_pairs == expected_pairs, f"{model['model_vertex_id']} prediction matrix is incomplete")

    split = data['split_manifest']
    require(valid_sha(split['split_sha256']), 'split digest malformed')
    train, validation, lockbox = set(split['train_units']), set(split['validation_units']), set(split['lockbox_units'])
    require(train and validation and lockbox, 'train, validation and lockbox must all be non-empty')
    require(not (train & validation or train & lockbox or validation & lockbox), 'split leakage across independent units')

    base_loss, baseline_decision, baseline_vertex_risks = baseline_robust_loss(models, worlds)
    per_action = {}
    for action in actions:
        action_id = action['intervention_id']
        if action['feasibility_status'] == 'FEASIBLE':
            per_action[action_id] = action_result(
                action, models, worlds, prediction_index, base_loss,
                data['query']['certificate_max_expected_loss'],
            )
        else:
            per_action[action_id] = {
                'intervention_id': action_id,
                'feasibility_status': action['feasibility_status'],
                'optimization_status': 'EXCLUDED_BEFORE_OPTIMIZATION',
                'robust_certificate': False,
                'certificate_status': 'NOT_ELIGIBLE',
            }
    feasible_ids = sorted(
        action_id for action_id, result in per_action.items()
        if result['feasibility_status'] == 'FEASIBLE'
    )
    require(feasible_ids, 'no feasible interventions supplied')
    selected_id = sorted(
        feasible_ids,
        key=lambda action_id: (-per_action[action_id]['value_per_cost'], action_id),
    )[0]
    eig_greedy_id = sorted(
        feasible_ids,
        key=lambda action_id: (-per_action[action_id]['vertex_min_eig_bits'], action_id),
    )[0]
    overlaps = ancestry_overlaps(models)

    expected = data['oracle_expectations']
    require(close(expected['baseline_robust_loss'], base_loss), 'baseline robust-loss known answer mismatch')
    for action_id, result in per_action.items():
        if result['optimization_status'] == 'EXCLUDED_BEFORE_OPTIMIZATION':
            for field in (
                'robust_loss_by_intervention',
                'clairvoyant_lower_bound_by_intervention',
                'uncertainty_penalty_by_intervention',
                'value_per_cost_by_intervention',
                'vertex_min_eig_bits_by_intervention',
                'vertex_max_eig_bits_by_intervention',
                'pairwise_midpoint_eig_min_bits_by_intervention',
                'pairwise_midpoint_eig_max_bits_by_intervention',
            ):
                require(expected[field].get(action_id) is None, f"excluded action {action_id} must not have an optimization result in {field}")
            require(expected['robust_certificate_by_intervention'].get(action_id) is False, f"excluded action {action_id} cannot be certified")
            continue
        checks = {
            'robust_loss_by_intervention': result['robust_policy_loss'],
            'clairvoyant_lower_bound_by_intervention': result['clairvoyant_lower_bound'],
            'uncertainty_penalty_by_intervention': result['model_uncertainty_penalty'],
            'value_per_cost_by_intervention': result['value_per_cost'],
            'vertex_min_eig_bits_by_intervention': result['vertex_min_eig_bits'],
            'vertex_max_eig_bits_by_intervention': result['vertex_max_eig_bits'],
            'pairwise_midpoint_eig_min_bits_by_intervention': result['pairwise_midpoint_eig_min_bits'],
            'pairwise_midpoint_eig_max_bits_by_intervention': result['pairwise_midpoint_eig_max_bits'],
            'robust_certificate_by_intervention': result['robust_certificate'],
        }
        for field, actual in checks.items():
            require(action_id in expected[field], f"known-answer field {field} missing {action_id}")
            if isinstance(actual, bool):
                require(expected[field][action_id] is actual, f"known-answer mismatch {field}/{action_id}")
            else:
                require(close(expected[field][action_id], actual), f"known-answer mismatch {field}/{action_id}: {actual}")
    require(selected_id == expected['expected_selected_intervention_id'], 'wrong robust value-per-cost action selected')
    require(eig_greedy_id == expected['expected_vertex_min_eig_greedy_intervention_id'], 'wrong diagnostic vertex-min-EIG action selected')
    require(per_action[selected_id]['feasibility_status'] == 'FEASIBLE', 'selected intervention is not feasible')
    require(selected_id == 'orthogonal-robust', 'synthetic known-answer robust intervention was not selected')
    require(per_action['reversal-high-eig']['clairvoyant_lower_bound'] == 0.0, 'reversal fixture lost its perfect clairvoyant lower bound')
    require(per_action['reversal-high-eig']['robust_policy_loss'] > per_action['reversal-high-eig']['clairvoyant_lower_bound'], 'reversal fixture failed to expose model-identity leakage')
    require(
        per_action['infeasible-perfect']['optimization_status'] == 'EXCLUDED_BEFORE_OPTIMIZATION',
        'disallowed intervention entered optimisation',
    )
    require('shared_policy' not in per_action['infeasible-perfect'], 'disallowed intervention received a decision policy')
    require(per_action['orthogonal-robust']['robust_certificate'], 'orthogonal intervention did not pass the robust certificate tolerance')

    return {
        'schema_version': 1,
        'tool': 'verify-distributionally-robust-connectome-intervention-design',
        'input_sha256': input_sha256,
        'run_id': data['run_id'],
        'claim_scope': data['claim_scope'],
        'ambiguity_set_scope': data['ambiguity_set']['scope'],
        'model_vertex_count': len(models),
        'world_count': len(worlds),
        'intervention_count': len(actions),
        'prediction_record_count': len(prediction_index),
        'baseline_robust_loss': base_loss,
        'baseline_shared_decision': baseline_decision,
        'baseline_vertex_risks': {model['model_vertex_id']: risk for model, risk in zip(models, baseline_vertex_risks)},
        'per_intervention': per_action,
        'selected_intervention_id': selected_id,
        'vertex_min_eig_greedy_intervention_id': eig_greedy_id,
        'model_ancestry_overlaps': overlaps,
        'model_independence_disposition': 'NOT_ESTABLISHED_BY_ANCESTRY_LABELS',
        'population_estimate_status': 'NOT_COMPUTED_BY_THIS_TOOL',
        'biological_truth_disposition': 'NOT_ESTABLISHED_BY_THIS_TOOL',
        'ambiguity_set_completeness_disposition': data['ambiguity_set']['vertex_completeness_status'],
        'status': 'SYNTHETIC_COMMON_POLICY_MINIMAX_VALIDATED',
        'scope_note': 'Exact common-policy minimax loss is conditional on the supplied complete finite vertex set and its convex hull. Vertex EIG minima are descriptive only, not guarantees over the convex hull. This oracle does not authenticate biological evidence or prove that the declared vertex set spans causal uncertainty.'
    }


def load_and_evaluate(raw: bytes) -> dict[str, Any]:
    data = json.loads(raw)
    return evaluate(data, sha256(raw))


def self_tests(data: dict[str, Any]) -> None:
    expected = evaluate(data, sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()))
    reversal = expected['per_intervention']['reversal-high-eig']
    require(close(reversal['clairvoyant_lower_bound'], 0.0), 'negative control: clairvoyant reversal lower bound is not zero')
    require(close(reversal['robust_policy_loss'], 0.15), 'negative control: reversal shared-policy risk is not 0.15')
    require(close(reversal['model_uncertainty_penalty'], 0.15), 'negative control: reversal uncertainty penalty is not 0.15')
    require(
        reversal['pairwise_midpoint_eig_min_bits'] < reversal['vertex_min_eig_bits'],
        'negative control: interior midpoint did not expose the non-guarantee of vertex-minimum EIG',
    )
    require(expected['selected_intervention_id'] == 'orthogonal-robust', 'negative control: robust action selection changed')
    require(expected['vertex_min_eig_greedy_intervention_id'] == 'reversal-high-eig', 'negative control: entropy comparator changed')

    missing = json.loads(json.dumps(data))
    missing['model_vertices'][0]['predictions'].pop()
    try:
        evaluate(missing, '0' * 64)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing prediction-matrix negative control was not rejected')

    unnormalized = json.loads(json.dumps(data))
    unnormalized['model_vertices'][0]['predictions'][0]['outcomes'][0]['probability'] = 0.5
    try:
        evaluate(unnormalized, '0' * 64)
    except AssertionError:
        pass
    else:
        raise AssertionError('unnormalized likelihood negative control was not rejected')

    duplicate = json.loads(json.dumps(data))
    duplicate['model_vertices'][1]['model_vertex_id'] = duplicate['model_vertices'][0]['model_vertex_id']
    try:
        evaluate(duplicate, '0' * 64)
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate model vertex negative control was not rejected')

    overlap_rows = expected['model_ancestry_overlaps']
    require(any('shared-connectome-revision-01' in row['shared_group_ids'] for row in overlap_rows), 'shared-ancestry overlap was not detected')
    require(expected['model_independence_disposition'] == 'NOT_ESTABLISHED_BY_ANCESTRY_LABELS', 'shared ancestry was promoted to independence')
    print('distributionally robust decision negative controls: PASS')


def main() -> int:
    if len(sys.argv) != 2:
        print('usage: verify_distributionally_robust_intervention_oracle.py FIXTURE.json', file=sys.stderr)
        return 2
    try:
        raw = Path(sys.argv[1]).read_bytes()
        data = json.loads(raw)
        report = evaluate(data, sha256(raw))
        self_tests(data)
        print(json.dumps(report, indent=2, sort_keys=True))
        print('distributionally robust common-policy oracle: PASS')
        return 0
    except Exception as exc:
        print(f'distributionally robust intervention oracle: FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())