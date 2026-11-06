#!/usr/bin/env python3
"""Independent oracle for query-conditioned connectome stopping fixtures.

Separates global graph convergence from query-specific robust decision stability.
This validates finite synthetic contracts; it does not reproduce biological data.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def expected_query_status(evaluation: dict[str, Any], maximum_missed: float) -> str:
    if evaluation['observation_identifiability'] == 'NOT_IDENTIFIABLE':
        return 'QUERY_NOT_IDENTIFIABLE'
    if (
        evaluation['observation_identifiability'] != 'IDENTIFIABLE'
        or evaluation['world_frame_status'] == 'INCOMPLETE'
        or evaluation['boundary_status'] != 'CLOSED_WITH_DECLARED_MARGIN'
        or evaluation['missed_candidate_upper_bound'] > maximum_missed
    ):
        return 'INSUFFICIENT_COVERAGE'
    if (
        len(set(value for value in evaluation['query_decisions'] if value is not None)) > 1
        or evaluation['unresolved_query_decision_world_count'] > 0
    ):
        return 'QUERY_UNSTABLE'
    return 'QUERY_STABLE_WITHIN_DECLARED_SCOPE'


def validate_track(track: dict[str, Any], global_stop_rule: dict[str, Any]) -> dict[str, Any]:
    snapshots = track['snapshots']
    require(len(snapshots) >= 2, 'track requires at least two progressive snapshots')
    require(valid_sha(track['query_spec_sha256']), 'query specification digest malformed')
    require(0 <= track['maximum_missed_candidate_upper_bound'] <= global_stop_rule['maximum_missed_candidate_upper_bound'], 'track risk bound exceeds run-level bound')
    ids = [snapshot['checkpoint_id'] for snapshot in snapshots]
    require(len(ids) == len(set(ids)), 'duplicate checkpoint IDs in track')
    edits = [snapshot['cumulative_edits'] for snapshot in snapshots]
    require(all(right > left for left, right in zip(edits, edits[1:])), 'snapshot cumulative edit counts must strictly increase')

    global_stops = 0
    query_safe_checkpoints: list[dict[str, Any]] = []
    false_safe_global_stops: list[str] = []
    for index, snapshot in enumerate(snapshots):
        require(valid_sha(snapshot['snapshot_sha256']), 'snapshot digest malformed')
        require(valid_sha(snapshot['global_invariant_sha256']), 'global invariant digest malformed')
        if index > 0:
            previous = snapshots[index - 1]
            if snapshot['edit_strategy'] != previous['edit_strategy']:
                require(snapshot['global_forecast_status'] in {'RESET_REQUIRED', 'NOT_ESTIMABLE'}, 'edit-strategy change must invalidate/reset the global forecast')
                require(snapshot['global_status'] != 'GLOBAL_CONVERGED', 'global convergence cannot carry across an edit-strategy regime shift without a reset')

        evaluation = snapshot['query_evaluation']
        world_ids = evaluation['admissible_world_ids']
        decisions = evaluation['query_decisions']
        require(len(world_ids) == len(set(world_ids)), 'duplicate admissible world ID')
        require(len(world_ids) == len(decisions), 'one query decision (or null unresolved result) is required for every admissible world')
        require(
            evaluation['unresolved_query_decision_world_count'] == sum(value is None for value in decisions),
            'unresolved query-decision world count differs from null decision entries',
        )
        require(valid_sha(evaluation['certificate_sha256']) or evaluation['certificate_sha256'] is None, 'query certificate digest malformed')
        require(evaluation['missed_candidate_upper_bound'] <= 1.0, 'missed-candidate upper bound outside [0,1]')

        expected_status = expected_query_status(evaluation, track['maximum_missed_candidate_upper_bound'])
        require(evaluation['query_status'] == expected_status, f"{snapshot['checkpoint_id']}: query status must be recomputed as {expected_status}")
        safe = expected_status == 'QUERY_STABLE_WITHIN_DECLARED_SCOPE'
        require(evaluation['safe_to_stop_for_query'] == safe, f"{snapshot['checkpoint_id']}: safe-to-stop flag differs from query evidence")
        if safe:
            require(evaluation['certificate_sha256'] is not None, 'safe query stop requires a certificate digest')
            require(evaluation['certificate_generation'] >= 1, 'safe query stop requires a positive certificate generation')
            query_safe_checkpoints.append({
                'checkpoint_id': snapshot['checkpoint_id'],
                'cumulative_edits': snapshot['cumulative_edits'],
                'certificate_sha256': evaluation['certificate_sha256'],
            })
        else:
            require(evaluation['certificate_sha256'] is None, 'unstable/unidentifiable/insufficient query cannot carry an active certificate')

        global_stop = snapshot['global_status'] == 'GLOBAL_CONVERGED'
        if global_stop:
            global_stops += 1
            if not safe:
                false_safe_global_stops.append(snapshot['checkpoint_id'])

    # Any late query-changing alternative must invalidate the exact earlier certificate.
    seen_event_ids: set[str] = set()
    for event in track['certificate_invalidation_events']:
        event_id = event['event_id']
        require(event_id not in seen_event_ids, 'duplicate certificate invalidation event ID')
        seen_event_ids.add(event_id)
        require(event['changes_query_decision'] is True, 'this fixture event must be query-changing')
        require(event['prior_certificate_invalidated'] is True, 'late query-changing alternative must invalidate the prior certificate')
        event_index = ids.index(event['at_checkpoint_id']) if event['at_checkpoint_id'] in ids else -1
        require(event_index > 0, 'invalidation event must occur after an earlier snapshot')
        prior_snapshots = snapshots[:event_index]
        prior_matches = [
            row for row in prior_snapshots
            if row['query_evaluation']['certificate_sha256'] == event['prior_certificate_sha256']
            and row['query_evaluation']['safe_to_stop_for_query']
        ]
        require(len(prior_matches) == 1, 'prior certificate must match exactly one earlier safe checkpoint')
        current = snapshots[event_index]
        current_eval = current['query_evaluation']
        require(event['new_alternative_id'] in current_eval['admissible_world_ids'], 'new alternative is absent from the event snapshot')
        require(current_eval['query_status'] in {'QUERY_UNSTABLE', 'QUERY_NOT_IDENTIFIABLE', 'INSUFFICIENT_COVERAGE'}, 'late query-changing alternative did not remove the prior safe disposition')
        require(current_eval['safe_to_stop_for_query'] is False, 'late query-changing alternative left a stale safe-stop flag')

    earliest_query_safe = query_safe_checkpoints[0] if query_safe_checkpoints else None
    earliest_global_stop = next((
        {'checkpoint_id': row['checkpoint_id'], 'cumulative_edits': row['cumulative_edits']}
        for row in snapshots if row['global_status'] == 'GLOBAL_CONVERGED'
    ), None)
    return {
        'track_id': track['track_id'],
        'query_id': track['query_id'],
        'snapshot_count': len(snapshots),
        'global_stop_checkpoint_count': global_stops,
        'false_safe_global_stop_checkpoint_ids': false_safe_global_stops,
        'false_safe_global_stop_count': len(false_safe_global_stops),
        'earliest_query_safe_checkpoint': earliest_query_safe,
        'earliest_global_stop_checkpoint': earliest_global_stop,
        'query_safe_before_global_stop': bool(
            earliest_query_safe is not None
            and earliest_global_stop is not None
            and earliest_query_safe['cumulative_edits'] < earliest_global_stop['cumulative_edits']
        ),
        'certificate_invalidation_event_count': len(track['certificate_invalidation_events']),
    }


def validate(data: dict[str, Any]) -> dict[str, Any]:
    require(data['schema_version'] == 1, 'unsupported schema version')
    require(data['claim_scope'] == 'synthetic_known_answer', 'starter oracle accepts only synthetic fixtures')
    require(data['source_reproduction']['exact_data_reproduction'] is False, 'synthetic fixture cannot claim exact paper data reproduction')
    require(data['source_reproduction']['snapshot_pin_status'] == 'SYNTHETIC', 'synthetic fixture must not masquerade as source-pinned biological snapshots')
    require(data['global_baseline']['claim_scope'] == 'GLOBAL_STABILIZATION_PROXY_NOT_QUERY_CERTIFICATE', 'global baseline claim ceiling changed')
    require(data['global_baseline']['invariant_count'] == 15, 'starter comparator must retain the paper-reported 15-invariant baseline')
    stop_rule = data['query_stop_rule']
    require(stop_rule['requires_closed_boundary'] is True, 'closed boundary requirement cannot be disabled')
    require(stop_rule['requires_identifiable_observation_model'] is True, 'observation identifiability requirement cannot be disabled')
    require(stop_rule['world_frame_claim_ceiling'] == 'DECLARED_SCOPE_ONLY', 'world-frame claim ceiling changed')

    tracks = data['tracks']
    track_ids = [track['track_id'] for track in tracks]
    require(len(track_ids) == len(set(track_ids)), 'duplicate track IDs')
    reports = [validate_track(track, stop_rule) for track in tracks]
    expected_by_id = {report['track_id']: report for report in reports}
    rare = expected_by_id.get('rare-path-global-false-stop')
    early = expected_by_id.get('query-specific-early-stop-and-late-invalidation')
    require(rare is not None and rare['false_safe_global_stop_checkpoint_ids'] == ['A1'], 'rare-path counterexample must expose exactly one global-only false safe stop at A1')
    require(rare['earliest_query_safe_checkpoint']['checkpoint_id'] == 'A2', 'query-conditioned rare-path stop must wait until A2')
    require(early is not None and early['earliest_query_safe_checkpoint']['checkpoint_id'] == 'B0', 'query-specific policy must be able to stop at B0 before global convergence')
    early_track = next(track for track in data['tracks'] if track['track_id'] == 'query-specific-early-stop-and-late-invalidation')
    b0 = early_track['snapshots'][0]['query_evaluation']
    require(b0['unresolved_graph_world_count'] == 1 and b0['unresolved_query_decision_world_count'] == 0, 'query-invariant graph ambiguity known answer is missing')
    require(b0['safe_to_stop_for_query'] is True, 'graph identity ambiguity with a known common query decision should not block query-safe stopping')
    require(early['certificate_invalidation_event_count'] == 1, 'late query-changing alternative invalidation control is missing')
    require(early['false_safe_global_stop_checkpoint_ids'] == ['B3'], 'late query-changing alternative must expose the stale global-only stop at B3')

    return {
        'status': 'SYNTHETIC_QUERY_CONDITIONED_STOPPING_CONTRACT_VALIDATED',
        'claim_scope': data['claim_scope'],
        'track_count': len(reports),
        'snapshot_count': sum(report['snapshot_count'] for report in reports),
        'false_safe_global_stop_count': sum(report['false_safe_global_stop_count'] for report in reports),
        'tracks': reports,
        'global_convergence_is_query_certificate': False,
        'biological_stopping_claim': 'NOT_ESTABLISHED',
    }


def self_test(data: dict[str, Any]) -> None:
    expected = validate(copy.deepcopy(data))
    require(expected['false_safe_global_stop_count'] == 2, 'known-answer global-only false stop count changed')
    mutated = copy.deepcopy(data)
    rare = next(track for track in mutated['tracks'] if track['track_id'] == 'rare-path-global-false-stop')
    rare['snapshots'][1]['query_evaluation']['safe_to_stop_for_query'] = True
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('false safe-stop flag with opposing world was not rejected')
    mutated = copy.deepcopy(data)
    early = next(track for track in mutated['tracks'] if track['track_id'] == 'query-specific-early-stop-and-late-invalidation')
    early['snapshots'][0]['query_evaluation']['boundary_status'] = 'OPEN_OR_UNVERIFIED'
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('open boundary with safe-stop flag was not rejected')
    mutated = copy.deepcopy(data)
    early = next(track for track in mutated['tracks'] if track['track_id'] == 'query-specific-early-stop-and-late-invalidation')
    early['snapshots'][0]['query_evaluation']['query_decisions'] = [False, None]
    early['snapshots'][0]['query_evaluation']['unresolved_query_decision_world_count'] = 1
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('unresolved query consequence with safe-stop flag was not rejected')
    mutated = copy.deepcopy(data)
    early = next(track for track in mutated['tracks'] if track['track_id'] == 'query-specific-early-stop-and-late-invalidation')
    early['snapshots'][1]['edit_strategy'] = 'SPLIT_FOCUSED'
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('edit-strategy regime shift without forecast reset was not rejected')
    mutated = copy.deepcopy(data)
    early = next(track for track in mutated['tracks'] if track['track_id'] == 'query-specific-early-stop-and-late-invalidation')
    early['certificate_invalidation_events'][0]['prior_certificate_invalidated'] = False
    try:
        validate(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError('late query-changing alternative without certificate invalidation was not rejected')
    print('query-conditioned stopping adversarial controls: PASS')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_bytes())
        report = validate(data)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.self_test:
            self_test(data)
        print('query-conditioned proofreading stopping oracle: PASS')
        return 0
    except Exception as exc:
        print(f'query-conditioned stopping oracle: FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())