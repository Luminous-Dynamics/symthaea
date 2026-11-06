#!/usr/bin/env python3
"""Independent standard-library oracle for animal-level connectome transport audits.

This is a synthetic/finite-sample contract verifier. It does not estimate
population coverage, infer biological truth, or validate a statistical model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None



def hash_field(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def canonical_split_sha256(
    train_ids: list[str],
    validation_ids: list[str],
    calibration_ids: list[str],
    lockbox_ids: list[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"connectome-cohort-transport-split-v1")
    for label, values in (
        ("TRAIN", train_ids),
        ("VALIDATION", validation_ids),
        ("CALIBRATION", calibration_ids),
        ("LOCKBOX", lockbox_ids),
    ):
        rows = sorted(values)
        digest.update(label.encode("ascii"))
        digest.update(len(rows).to_bytes(8, "big"))
        for value in rows:
            hash_field(digest, value)
    return digest.hexdigest()


def canonical_training_manifest_sha256(
    model_id: str,
    split_sha256: str,
    training_animal_ids: list[str],
    animals: dict[str, dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"connectome-cohort-training-manifest-v1")
    hash_field(digest, model_id)
    digest.update(split_sha256.encode("ascii"))
    animal_rows = sorted(training_animal_ids)
    digest.update(len(animal_rows).to_bytes(8, "big"))
    for animal_id in animal_rows:
        hash_field(digest, animal_id)
    conditions = sorted({animals[animal_id]["condition_id"] for animal_id in training_animal_ids})
    digest.update(len(conditions).to_bytes(8, "big"))
    for condition_id in conditions:
        hash_field(digest, condition_id)
    return digest.hexdigest()




def canonical_calibration_manifest_sha256(
    model_id: str,
    split_sha256: str,
    calibration_method: str,
    calibration_animal_ids: list[str],
    animals: dict[str, dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"connectome-cohort-calibration-manifest-v1")
    hash_field(digest, model_id)
    digest.update(split_sha256.encode("ascii"))
    hash_field(digest, calibration_method)
    animal_rows = sorted(calibration_animal_ids)
    digest.update(len(animal_rows).to_bytes(8, "big"))
    for animal_id in animal_rows:
        hash_field(digest, animal_id)
    conditions = sorted({animals[animal_id]["condition_id"] for animal_id in calibration_animal_ids})
    digest.update(len(conditions).to_bytes(8, "big"))
    for condition_id in conditions:
        hash_field(digest, condition_id)
    return digest.hexdigest()



def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def unique(values: list[str], label: str) -> None:
    require(all(isinstance(value, str) and value.strip() for value in values), f'{label} contains an empty identity')
    require(len(values) == len(set(values)), f'{label} contains duplicate identities')


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(data.get('schema_version') == 1, 'unsupported schema version')
    require(data.get('claim_scope') in {'synthetic_known_truth', 'retrospective_biological', 'prospective_biological'}, 'unsupported claim scope')
    dataset = data['dataset_binding']
    require(valid_sha(dataset['dataset_revision_sha256']), 'dataset revision digest malformed')
    require(valid_sha(dataset['source_access_receipt_sha256']), 'source-access receipt malformed')
    if data['claim_scope'] == 'synthetic_known_truth':
        require(dataset['evidence_tier'] == 'SYNTHETIC', 'synthetic claim scope requires SYNTHETIC evidence tier')

    split = data['split_manifest']
    require(split['track'] == 'NEW_ANIMAL_SAME_PAIR' and split['split_unit'] == 'ANIMAL', 'V1 supports only animal-level new-animal/same-pair transport')
    require(split['lockbox_frozen_before_prediction'] is True, 'lockbox must be frozen before prediction')
    require(valid_sha(split['split_sha256']), 'split digest malformed')
    train_ids = split['train_animal_ids']
    validation_ids = split['validation_animal_ids']
    calibration_ids = split['calibration_animal_ids']
    lockbox_ids = split['lockbox_animal_ids']
    unique(train_ids, 'train animal IDs')
    unique(validation_ids, 'validation animal IDs')
    unique(calibration_ids, 'calibration animal IDs')
    unique(lockbox_ids, 'lockbox animal IDs')
    partitions = [set(train_ids), set(validation_ids), set(calibration_ids), set(lockbox_ids)]
    require(all(not (partitions[i] & partitions[j]) for i in range(len(partitions)) for j in range(i + 1, len(partitions))), 'animal leakage across train/validation/calibration/lockbox')
    expected_split_sha = canonical_split_sha256(train_ids, validation_ids, calibration_ids, lockbox_ids)
    require(split['split_sha256'] == expected_split_sha, 'split digest does not bind the exact animal membership')

    animals_list = data['animals']
    animal_ids = [row['animal_id'] for row in animals_list]
    unique(animal_ids, 'animal IDs')
    animals = {row['animal_id']: row for row in animals_list}
    expected_roles = {
        **{animal_id: 'TRAIN' for animal_id in train_ids},
        **{animal_id: 'VALIDATION' for animal_id in validation_ids},
        **{animal_id: 'CALIBRATION' for animal_id in calibration_ids},
        **{animal_id: 'LOCKBOX' for animal_id in lockbox_ids},
    }
    require(set(animals) == set(expected_roles), 'animal records do not exactly match the frozen split manifest')
    for animal_id, animal in animals.items():
        require(animal['cohort_role'] == expected_roles[animal_id], f'animal role disagrees with split manifest: {animal_id}')
        require(bool(animal['condition_id'].strip()), f'animal has an empty condition: {animal_id}')

    pairs_list = data['pairs']
    pair_ids = [row['pair_id'] for row in pairs_list]
    unique(pair_ids, 'pair IDs')
    pairs = {row['pair_id']: row for row in pairs_list}
    for pair in pairs_list:
        require(valid_sha(pair['structure_evidence_sha256']), f"structure evidence digest malformed: {pair['pair_id']}")

    observations_list = data['observations']
    observation_ids = [row['observation_id'] for row in observations_list]
    unique(observation_ids, 'observation IDs')
    observations = {row['observation_id']: row for row in observations_list}
    animal_pair_keys: set[tuple[str, str]] = set()
    for row in observations_list:
        animal_id = row['animal_id']
        pair_id = row['pair_id']
        require(animal_id in animals, f"observation references unknown animal: {animal_id}")
        require(pair_id in pairs, f"observation references unknown pair: {pair_id}")
        require(row['condition_id'] == animals[animal_id]['condition_id'], f"observation condition disagrees with animal: {row['observation_id']}")
        key = (animal_id, pair_id)
        require(key not in animal_pair_keys, f'duplicate animal/pair summary; aggregate trials within animal first: {key}')
        animal_pair_keys.add(key)
        disposition = row['measurement_disposition']
        truth = row['functional_disposition']
        if disposition == 'OBSERVED':
            require(truth in {'FUNCTIONAL_CONNECTION', 'FUNCTIONAL_NON_CONNECTION', 'INDETERMINATE'}, f"observed row has invalid functional disposition: {row['observation_id']}")
            require(isinstance(row['trial_count'], int) and row['trial_count'] >= 1, f"observed row needs positive within-animal trial_count: {row['observation_id']}")
            require(finite_number(row['response_mean']), f"observed response mean is not finite: {row['observation_id']}")
            require(valid_sha(row['response_kernel_sha256']), f"observed response-kernel digest malformed: {row['observation_id']}")
            if truth == 'FUNCTIONAL_NON_CONNECTION':
                require(valid_sha(row['equivalence_protocol_sha256']), f"functional non-connection lacks equivalence protocol: {row['observation_id']}")
                require(valid_sha(row['equivalence_receipt_sha256']), f"functional non-connection lacks equivalence receipt: {row['observation_id']}")
            else:
                require(row['equivalence_protocol_sha256'] is None and row['equivalence_receipt_sha256'] is None, f"non-negative disposition carries an equivalence receipt: {row['observation_id']}")
        else:
            expected_truth = {'NOT_MEASURED': 'NOT_MEASURED', 'FAILED': 'FAILED', 'INACCESSIBLE': 'INACCESSIBLE'}[disposition]
            require(truth == expected_truth, f"measurement/functional disposition mismatch: {row['observation_id']}")
            require(row['trial_count'] is None and row['response_mean'] is None and row['response_kernel_sha256'] is None, f"non-observation contains fabricated response data: {row['observation_id']}")
            require(row['equivalence_protocol_sha256'] is None and row['equivalence_receipt_sha256'] is None, f"non-observation carries equivalence evidence: {row['observation_id']}")

    models_list = data['models']
    model_ids = [row['model_id'] for row in models_list]
    unique(model_ids, 'model IDs')
    models = {row['model_id']: row for row in models_list}
    for model in models_list:
        require(valid_sha(model['implementation_sha256']), f"model implementation digest malformed: {model['model_id']}")
        require(valid_sha(model['training_manifest_sha256']), f"model training-manifest digest malformed: {model['model_id']}")
        unique(model['training_animal_ids'], f"{model['model_id']} training animal IDs")
        require(set(model['training_animal_ids']).issubset(set(train_ids)), f"model {model['model_id']} uses validation or lockbox animals for training")
        require(set(model['training_animal_ids']), f"model {model['model_id']} has no training animals")
        expected_training_sha = canonical_training_manifest_sha256(model['model_id'], split['split_sha256'], model['training_animal_ids'], animals)
        require(model['training_manifest_sha256'] == expected_training_sha, f"model {model['model_id']} training manifest digest does not bind exact animal membership")
        unique(model['calibration_animal_ids'], f"{model['model_id']} calibration animal IDs")
        require(set(model['calibration_animal_ids']).issubset(set(calibration_ids)), f"model {model['model_id']} uses a non-calibration animal to fit its calibrator")
        expected_calibration_sha = canonical_calibration_manifest_sha256(
            model['model_id'], split['split_sha256'], model['calibration_method'],
            model['calibration_animal_ids'], animals
        )
        require(model['calibration_manifest_sha256'] == expected_calibration_sha, f"model {model['model_id']} calibration manifest digest does not bind its exact calibration animal membership")
        if model['calibration_method'] == 'NONE':
            require(not model['calibration_animal_ids'], f"model {model['model_id']} declares no calibration method but lists calibration animals")
            require(model['calibration_receipt_sha256'] is None, f"model {model['model_id']} declares no calibration method but carries a calibrator receipt")
        else:
            require(model['calibration_method'] in {'TEMPERATURE_SCALING', 'PLATT_SCALING', 'ISOTONIC', 'SPLIT_CONFORMAL', 'ISOTONIC_PLUS_SPLIT_CONFORMAL', 'OTHER'}, f"model {model['model_id']} has an unknown calibration method")
            require(model['calibration_animal_ids'], f"model {model['model_id']} has a calibration method but no calibration animals")
            require(valid_sha(model['calibration_receipt_sha256']), f"model {model['model_id']} calibrated model lacks a calibration artifact receipt")
        support_ids = [row['condition_id'] for row in model['supported_conditions']]
        unique(support_ids, f"{model['model_id']} supported condition IDs")
        for support in model['supported_conditions']:
            require(valid_sha(support['support_receipt_sha256']), f"condition-support receipt malformed: {model['model_id']}/{support['condition_id']}")
            if support['support_basis'] == 'TRAIN_OBSERVED':
                model_train_conditions = {animals[animal_id]['condition_id'] for animal_id in model['training_animal_ids']}
                require(support['condition_id'] in model_train_conditions, f"{model['model_id']} claims TRAIN_OBSERVED support without training animals in {support['condition_id']}")
            else:
                require(support['support_basis'] in {'MECHANISTIC_INVARIANCE_ASSUMPTION', 'EXTERNAL_VALIDATION'}, 'unsupported condition support basis')

    predictions_list = data['predictions']
    prediction_ids = [row['prediction_id'] for row in predictions_list]
    unique(prediction_ids, 'prediction IDs')
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    for prediction in predictions_list:
        model_id = prediction['model_id']
        observation_id = prediction['observation_id']
        require(model_id in models, f"prediction references unknown model: {model_id}")
        require(observation_id in observations, f"prediction references unknown observation: {observation_id}")
        key = (model_id, observation_id)
        require(key not in predictions, f'duplicate model/observation prediction: {key}')
        predictions[key] = prediction
        observation = observations[observation_id]
        model = models[model_id]
        status = prediction['prediction_disposition']
        supported_conditions = {row['condition_id'] for row in model['supported_conditions']}
        values = (prediction['probability_functional'], prediction['predicted_response_mean'], prediction['lower90'], prediction['upper90'])
        if status == 'PREDICTED':
            require(observation['measurement_disposition'] == 'OBSERVED', f"model predicts a non-observed row: {observation_id}")
            require(observation['condition_id'] in supported_conditions, f"model predicts outside declared condition support: {model_id}/{observation_id}")
            require(prediction['out_of_domain_receipt_sha256'] is None, f"ordinary prediction carries an out-of-domain receipt: {prediction['prediction_id']}")
            probability, response_mean, lower, upper = values
            require(finite_number(probability) and 0.0 <= probability <= 1.0, f"invalid functional probability: {prediction['prediction_id']}")
            require(all(finite_number(value) for value in (response_mean, lower, upper)), f"non-finite response interval: {prediction['prediction_id']}")
            require(lower <= response_mean <= upper, f"predicted response mean is outside its 90% interval: {prediction['prediction_id']}")
        elif status == 'ABSTAINED_OUT_OF_DOMAIN':
            require(observation['measurement_disposition'] == 'OBSERVED', f"OOD abstention attached to non-observation: {observation_id}")
            require(observation['condition_id'] not in supported_conditions, f"model abstains OOD in a declared supported condition: {model_id}/{observation_id}")
            require(valid_sha(prediction['out_of_domain_receipt_sha256']), f"OOD abstention lacks receipt: {prediction['prediction_id']}")
            require(all(value is None for value in values), f"OOD abstention contains a point prediction: {prediction['prediction_id']}")
        elif status == 'NOT_APPLICABLE':
            require(observation['measurement_disposition'] != 'OBSERVED', f"NOT_APPLICABLE used for an observed row: {observation_id}")
            require(prediction['out_of_domain_receipt_sha256'] is None and all(value is None for value in values), f"NOT_APPLICABLE row contains a prediction: {prediction['prediction_id']}")
        else:
            raise ValueError(f"unknown prediction disposition: {status}")

    expected_keys = {(model_id, observation_id) for model_id in models for observation_id in observations}
    require(set(predictions) == expected_keys, 'prediction matrix must contain exactly one record for every model × lockbox observation')
    require(all(animals[obs['animal_id']]['cohort_role'] == 'LOCKBOX' for obs in observations_list), 'V1 observations must describe only the frozen lockbox animals')

    truth_counts = {
        'observed': sum(row['measurement_disposition'] == 'OBSERVED' for row in observations_list),
        'not_measured': sum(row['measurement_disposition'] == 'NOT_MEASURED' for row in observations_list),
        'failed': sum(row['measurement_disposition'] == 'FAILED' for row in observations_list),
        'inaccessible': sum(row['measurement_disposition'] == 'INACCESSIBLE' for row in observations_list),
        'functional_connection': sum(row['functional_disposition'] == 'FUNCTIONAL_CONNECTION' for row in observations_list),
        'functional_non_connection': sum(row['functional_disposition'] == 'FUNCTIONAL_NON_CONNECTION' for row in observations_list),
        'indeterminate': sum(row['functional_disposition'] == 'INDETERMINATE' for row in observations_list),
    }
    determinate = [row for row in observations_list if row['functional_disposition'] in {'FUNCTIONAL_CONNECTION', 'FUNCTIONAL_NON_CONNECTION'}]
    labels = {row['observation_id']: 1.0 if row['functional_disposition'] == 'FUNCTIONAL_CONNECTION' else 0.0 for row in determinate}
    lockbox_animal_count = len(lockbox_ids)

    model_reports = []
    for model_id in sorted(models):
        animal_briers: dict[str, list[float]] = {}
        animal_interval: dict[str, list[int]] = {}
        absolute_errors: list[float] = []
        scored_labels = 0
        false_negatives = 0
        false_positives = 0
        predicted_observed = 0
        abstained = 0
        not_applicable = 0
        interval_covered = 0
        interval_count = 0
        for observation in observations_list:
            prediction = predictions[(model_id, observation['observation_id'])]
            status = prediction['prediction_disposition']
            animal_id = observation['animal_id']
            if status == 'ABSTAINED_OUT_OF_DOMAIN':
                abstained += 1
                continue
            if status == 'NOT_APPLICABLE':
                not_applicable += 1
                continue
            predicted_observed += 1
            response = observation['response_mean']
            prediction_mean = prediction['predicted_response_mean']
            absolute_errors.append(abs(response - prediction_mean))
            covered = int(prediction['lower90'] <= response <= prediction['upper90'])
            interval_count += 1
            interval_covered += covered
            animal_interval.setdefault(animal_id, []).append(covered)
            if observation['observation_id'] in labels:
                scored_labels += 1
                probability = prediction['probability_functional']
                truth = labels[observation['observation_id']]
                animal_briers.setdefault(animal_id, []).append((probability - truth) ** 2)
                if probability < 0.5 and truth == 1.0:
                    false_negatives += 1
                if probability >= 0.5 and truth == 0.0:
                    false_positives += 1

        brier_by_animal = {animal_id: mean(values) for animal_id, values in sorted(animal_briers.items())}
        interval_by_animal = {animal_id: mean([float(value) for value in values]) for animal_id, values in sorted(animal_interval.items())}
        macro_brier = mean([value for value in brier_by_animal.values() if value is not None])
        macro_interval = mean([value for value in interval_by_animal.values() if value is not None])
        determinate_count = len(determinate)
        model_reports.append({
            'model_id': model_id,
            'calibration_method': models[model_id]['calibration_method'],
            'calibration_animal_ids': sorted(models[model_id]['calibration_animal_ids']),
            'calibration_manifest_sha256': models[model_id]['calibration_manifest_sha256'],
            'lockbox_animal_count': lockbox_animal_count,
            'observed_prediction_count': predicted_observed,
            'out_of_domain_abstention_count': abstained,
            'not_applicable_count': not_applicable,
            'prediction_coverage_of_observed': predicted_observed / truth_counts['observed'] if truth_counts['observed'] else None,
            'determinate_label_count': determinate_count,
            'binary_scored_count': scored_labels,
            'binary_scoring_coverage': scored_labels / determinate_count if determinate_count else None,
            'animal_macro_brier_score': macro_brier,
            'brier_evaluable_animal_count': len(brier_by_animal),
            'brier_by_animal': brier_by_animal,
            'false_negative_count': false_negatives,
            'false_positive_count': false_positives,
            'response_absolute_error_mean': mean(absolute_errors),
            'response_interval_coverage_90': interval_covered / interval_count if interval_count else None,
            'animal_macro_interval_coverage_90': macro_interval,
            'interval_evaluable_animal_count': len(interval_by_animal),
            'interval_by_animal': interval_by_animal,
            'population_confidence_intervals': 'NOT_COMPUTED_BY_V1_ORACLE',
        })

    return {
        'schema_version': 1,
        'tool': 'verify-connectome-cohort-transport',
        'run_id': data['run_id'],
        'claim_scope': data['claim_scope'],
        'track': split['track'],
        'split_integrity': 'PASS_ANIMAL_IDS_DISJOINT',
        'train_animal_count': len(train_ids),
        'validation_animal_count': len(validation_ids),
        'calibration_animal_count': len(calibration_ids),
        'lockbox_animal_count': lockbox_animal_count,
        'lockbox_observation_count': len(observations_list),
        'truth_counts': truth_counts,
        'model_count': len(models),
        'model_reports': model_reports,
        'false_functional_zero_from_missing_or_indeterminate': 0,
        'population_confidence_intervals': 'NOT_COMPUTED_BY_V1_ORACLE',
        'biological_calibration': 'NOT_ESTABLISHED_BY_SYNTHETIC_FIXTURE',
        'status': 'SYNTHETIC_COHORT_TRANSPORT_CONTRACT_VALIDATED' if data['claim_scope'] == 'synthetic_known_truth' else 'SAMPLED_COHORT_TRANSPORT_AUDIT',
        'scope_note': 'Finite lockbox arithmetic only. No population uncertainty, source authentication, causal mechanism, biological calibration, or cross-species transport claim is established.',
    }


def run_negative_controls(data: dict[str, Any]) -> None:
    mutations = []
    leaked = copy.deepcopy(data)
    leaked['split_manifest']['lockbox_animal_ids'].append('WT-TRAIN-01')
    mutations.append(('train/lockbox animal leakage', leaked))
    false_zero = copy.deepcopy(data)
    false_zero['observations'][2]['functional_disposition'] = 'FUNCTIONAL_NON_CONNECTION'
    false_zero['observations'][2]['equivalence_protocol_sha256'] = None
    false_zero['observations'][2]['equivalence_receipt_sha256'] = None
    mutations.append(('missing observation promoted to functional zero', false_zero))
    no_receipt = copy.deepcopy(data)
    no_receipt['observations'][3]['equivalence_receipt_sha256'] = None
    mutations.append(('equivalence without receipt', no_receipt))
    wrong_domain = copy.deepcopy(data)
    pred = next(row for row in wrong_domain['predictions'] if row['model_id'] == 'state-aware' and row['observation_id'] == 'OBS-UNC31-RID-URX')
    pred['prediction_disposition'] = 'PREDICTED'
    pred['out_of_domain_receipt_sha256'] = None
    pred['probability_functional'] = 0.5
    pred['predicted_response_mean'] = 0.2
    pred['lower90'] = 0.0
    pred['upper90'] = 0.4
    mutations.append(('unsupported mutant prediction without domain support', wrong_domain))
    pair_duplicate = copy.deepcopy(data)
    pair_duplicate['observations'].append(copy.deepcopy(pair_duplicate['observations'][0]))
    pair_duplicate['observations'][-1]['observation_id'] = 'DUPLICATE-PAIR-OBS'
    mutations.append(('duplicate animal/pair summaries', pair_duplicate))
    missing_prediction = copy.deepcopy(data)
    missing_prediction['predictions'].pop()
    mutations.append(('incomplete model × lockbox prediction matrix', missing_prediction))
    calibration_leak = copy.deepcopy(data)
    calibration_leak['models'][1]['calibration_animal_ids'] = ['UNC31-LOCKBOX-01']
    mutations.append(('lockbox animal used for calibration', calibration_leak))
    stale_calibration = copy.deepcopy(data)
    stale_calibration['models'][1]['calibration_manifest_sha256'] = '0' * 64
    mutations.append(('stale calibration manifest digest', stale_calibration))
    stale_split = copy.deepcopy(data)
    stale_split['split_manifest']['split_sha256'] = '0' * 64
    mutations.append(('stale animal split digest', stale_split))
    stale_training = copy.deepcopy(data)
    stale_training['models'][0]['training_manifest_sha256'] = '0' * 64
    mutations.append(('stale model training manifest digest', stale_training))
    for label, mutated in mutations:
        try:
            verify(mutated)
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError(f'negative control was not rejected: {label}')
    print(f'cohort-transport negative controls: PASS ({len(mutations)})')



def run_known_answer_checks(report: dict[str, Any]) -> None:
    models = {row["model_id"]: row for row in report["model_reports"]}
    anatomy = models["anatomy-only"]
    state = models["state-aware"]
    require(report["truth_counts"]["observed"] == 4, "known-answer observed count mismatch")
    require(report["truth_counts"]["not_measured"] == 2, "known-answer not-measured count mismatch")
    require(report["truth_counts"]["functional_connection"] == 1, "known-answer positive count mismatch")
    require(report["truth_counts"]["functional_non_connection"] == 1, "known-answer equivalence-negative count mismatch")
    require(report["truth_counts"]["indeterminate"] == 2, "known-answer indeterminate count mismatch")
    require(abs(anatomy["animal_macro_brier_score"] - 0.41) <= 1e-9, "anatomy-only animal-macro Brier known answer mismatch")
    require(abs(anatomy["animal_macro_interval_coverage_90"] - 0.25) <= 1e-9, "anatomy-only animal-macro interval coverage known answer mismatch")
    require(anatomy["false_negative_count"] == 1, "anatomy-only extrasynaptic-style false negative not detected")
    require(abs(state["animal_macro_brier_score"] - 0.04) <= 1e-9, "state-aware animal-macro Brier known answer mismatch")
    require(abs(state["animal_macro_interval_coverage_90"] - 1.0) <= 1e-9, "state-aware animal-macro interval coverage known answer mismatch")
    require(state["out_of_domain_abstention_count"] == 2, "state-aware unsupported-mutant abstention count mismatch")
    require(abs(state["binary_scoring_coverage"] - 0.5) <= 1e-9, "state-aware binary selective coverage mismatch")
    print("cohort-transport known answers: PASS")



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding='utf-8'))
        report = verify(data)
        if args.self_test:
            run_negative_controls(data)
            run_known_answer_checks(report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print('independent connectome cohort transport oracle: PASS')
        return 0
    except Exception as exc:
        print(f'connectome cohort transport oracle: FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())