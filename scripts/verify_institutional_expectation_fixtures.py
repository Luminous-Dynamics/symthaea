#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path

class OracleError(Exception):
    pass

def run(case):
    records = {}
    current = {}
    observations = {}
    for event in case.get('events', [case.get('event')]):
        if event is None:
            raise OracleError(f"{case['id']}: missing event")
        kind = event['type']
        if kind == 'Expectation':
            if event['information_set_as_of'] > event['observation_time']:
                return 'InformationSetAfterObservation', {}
            if event['expectation_id'] in records:
                return 'DuplicateExpectation', {}
            records[event['expectation_id']] = event.copy()
            current[event['subject_id']] = event['expectation_id']
        elif kind == 'Update':
            if event['update_id'] in observations:
                return 'DuplicateUpdate', {}
            prior = records.get(event['superseded_expectation_id'])
            if prior is None:
                return 'UnknownExpectation', {}
            if event['expectation_id'] in records:
                return 'DuplicateExpectation', {}
            if event['update_time'] < prior['observation_time']:
                return 'UpdateBeforeObservation', {}
            if event['information_set_as_of'] > event['update_time']:
                return 'InformationSetAfterObservation', {}
            updated = prior.copy()
            updated['expectation_id'] = event['expectation_id']
            updated['observation_time'] = event['update_time']
            updated['information_set_as_of'] = event['information_set_as_of']
            updated['belief'] = event['belief']
            records[event['superseded_expectation_id']]['current'] = False
            records[event['superseded_expectation_id']]['superseded_by'] = event['expectation_id']
            updated['current'] = True
            records[event['expectation_id']] = updated
            observations[event['update_id']] = event
            current[updated['subject_id']] = event['expectation_id']
        elif kind == 'ActualObservation':
            if event['observation_id'] in observations:
                return 'DuplicateUpdate', {}
            observations[event['observation_id']] = event
        else:
            return f'UnknownEvent:{kind}', {}
    current_id = next(iter(current.values()), None)
    result = {'current_expectation': current_id}
    if current_id:
        result['belief'] = records[current_id]['belief']
    for rid, r in records.items():
        if rid != current_id and not r.get('current', True):
            result['prior_belief'] = r['belief']
            break
    for obs in observations.values():
        if obs.get('type') == 'ActualObservation':
            result['actual_institution'] = obs['institution_hash']
            break
    if current_id and any(r.get('current') is False for r in records.values()):
        current_record = records[current_id]
        previous = next((r for r in records.values() if r.get('superseded_by') == current_id), None)
        if previous:
            result['prior_belief'] = previous['belief']
        result['current_belief'] = current_record['belief']
    return 'Valid', result

def main():
    if len(sys.argv) != 2:
        print(f'usage: {Path(sys.argv[0]).name} FIXTURES.json', file=sys.stderr)
        return 2
    p = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    passed = 0
    for case in p['valid']:
        disposition, observed = run(case)
        if disposition != 'Valid':
            raise OracleError(f"{case['id']}: unexpected {disposition}")
        for key, value in case['expected'].items():
            if observed.get(key) != value:
                raise OracleError(f"{case['id']}: {key}: {observed.get(key)!r} != {value!r}")
        passed += 1
    for case in p['rejected']:
        disposition, _ = run(case)
        if disposition != case['expected_disposition']:
            raise OracleError(f"{case['id']}: {disposition} != {case['expected_disposition']}")
        passed += 1
    print(f'verified {passed} institutional expectation fixtures')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
