#!/usr/bin/env python3
import hashlib
import json
from collections import Counter
from pathlib import Path

PATH = Path('docs/release/evidence/eng-design-001b-thread-integrity-reference-v1.json')
EXPECTED_SHA256 = '9d2aaa3637dcfef8ecf8c0b50bc202a0876b786661a84d713776f0e5dfd157c1'
EXPECTED_SCHEMA = 'eng-design-001b-thread-integrity-reference-v1'
EXPECTED_AUTHORITY = 'repository_process_integrity_only_no_physical_execution_authority'
EXPECTED_PARENT = {
    'issue': 6005,
    'source_pr': 6006,
    'source_head': '0cc3a61871192e011299290b64514b4b341e5f1f',
}
EXPECTED_PREFIXES = ['NEED','REQ','CON','ASM','IFC','DEC','RISK','VER','VAL','CFG','EVD','CHG']
EXPECTED_CHANGE = [
    'UnaffectedCurrent','HistoricallyValidNotCurrent','RequalificationRequired',
    'BlockedByAssumptionDrift','BlockedByInterfaceDrift','BlockedByConfigurationDrift',
    'ApplicabilityReviewRequired',
]
EXPECTED_IDS = [
    'C01_complete_minimal_trace','C02_orphan_requirement','C03_requirement_without_verification',
    'C04_validation_detached_from_need','C05_duplicate_identifier','C06_unresolved_reference',
    'C07_illegal_dependency_cycle','C08_stale_assumption','C09_changed_interface',
    'C10_config_generation_changed','C11_bounded_irrelevant_change','C12_repair_requires_requalification',
    'C13_component_substitution','C14_model_for_field_requirement','C15_derived_artifact_witness_inflation',
    'C16_history_deleted','C17_cross_owner_laundering','C18_physical_execution_authority',
]
FORBIDDEN_KEYS = {'readiness_score','priority_score','self_sufficiency_score','closure_score','maturity_score'}


def fail(msg):
    raise SystemExit(f'FAIL_ENG_DESIGN_001B: {msg}')


def walk_keys(v):
    if isinstance(v, dict):
        for k, x in v.items():
            yield k
            yield from walk_keys(x)
    elif isinstance(v, list):
        for x in v:
            yield from walk_keys(x)


def derive(c):
    # Identity/graph integrity must fail before downstream currentness claims.
    if not c['ids_unique'] or not c['refs_resolve'] or not c['acyclic']:
        return 'GraphIntegrityBlocked'
    # Traceability must exist before a release claim can consume evidence.
    if not c['need'] or not c['req_trace'] or not c['verification_route'] or not c['validation_need_trace']:
        return 'TraceabilityBlocked'
    # History and authority boundaries are non-negotiable integrity constraints.
    if not c['history_retained']:
        return 'HistoryIntegrityBlocked'
    if c['cross_owner_laundering'] or c['physical_authority_requested']:
        return 'AuthorityBoundaryBlocked'
    # Required evidence planes are preserved.
    if c['required_plane'] == 'FIELD' and c['actual_plane'] != 'FIELD':
        return 'EvidencePlaneBlocked'
    # One source witness cannot become many through derivation.
    if c['derived_artifacts'] > 1 and c['physical_witnesses'] <= 1:
        return 'WitnessMultiplicityBlocked'
    # Currentness blockers remain distinct for diagnosis and requalification.
    if not c['assumptions_current']:
        return 'BlockedByAssumptionDrift'
    if not c['interfaces_current']:
        return 'BlockedByInterfaceDrift'
    if not c['cfg_current']:
        return 'BlockedByConfigurationDrift'
    if c['repair'] and c['change_relevant']:
        return 'RequalificationRequired'
    if c['substitution'] and c['change_relevant']:
        return 'ApplicabilityReviewRequired'
    return 'UnaffectedCurrent'


def main():
    raw = PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        fail('corpus digest drift')
    data = json.loads(raw)
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':')).encode() + b'\n'
    if raw != canonical:
        fail('corpus is not canonical compact sorted-key JSON + final newline')
    if data.get('schema') != EXPECTED_SCHEMA:
        fail('schema drift')
    if data.get('authority') != EXPECTED_AUTHORITY:
        fail('authority drift')
    if data.get('parent') != EXPECTED_PARENT:
        fail('parent/source binding drift')
    if data.get('id_prefixes') != EXPECTED_PREFIXES:
        fail('typed ID vocabulary drift')
    if data.get('change_dispositions') != EXPECTED_CHANGE:
        fail('change disposition vocabulary drift')
    if FORBIDDEN_KEYS.intersection(walk_keys(data)):
        fail('forbidden scalar score key present')
    cases = data.get('cases')
    if not isinstance(cases, list) or len(cases) != 18:
        fail('case count drift')
    ids = [c.get('id') for c in cases]
    if ids != EXPECTED_IDS:
        fail('case identity/order drift')
    required = {
        'id','need','req_trace','verification_route','validation_need_trace','ids_unique','refs_resolve',
        'acyclic','assumptions_current','interfaces_current','cfg_current','change_relevant','repair',
        'substitution','required_plane','actual_plane','physical_witnesses','derived_artifacts',
        'history_retained','cross_owner_laundering','physical_authority_requested','expected',
    }
    derived = []
    for c in cases:
        if set(c) != required:
            fail(f"case shape drift: {c.get('id')}")
        if c['required_plane'] not in {'MODEL','FIELD'} or c['actual_plane'] not in {'MODEL','FIELD'}:
            fail(f"unknown evidence plane: {c['id']}")
        if not isinstance(c['physical_witnesses'], int) or c['physical_witnesses'] < 0:
            fail(f"invalid witness count: {c['id']}")
        if not isinstance(c['derived_artifacts'], int) or c['derived_artifacts'] < 0:
            fail(f"invalid derived artifact count: {c['id']}")
        got = derive(c)
        if got != c['expected']:
            fail(f"oracle mismatch {c['id']}: derived={got} expected={c['expected']}")
        derived.append(got)
    census = Counter(derived)
    expected_census = Counter({
        'UnaffectedCurrent':2,
        'TraceabilityBlocked':3,
        'GraphIntegrityBlocked':3,
        'BlockedByAssumptionDrift':1,
        'BlockedByInterfaceDrift':1,
        'BlockedByConfigurationDrift':1,
        'RequalificationRequired':1,
        'ApplicabilityReviewRequired':1,
        'EvidencePlaneBlocked':1,
        'WitnessMultiplicityBlocked':1,
        'HistoryIntegrityBlocked':1,
        'AuthorityBoundaryBlocked':2,
    })
    if census != expected_census:
        fail(f'disposition census drift: {dict(census)}')
    print('PASS_ENG_DESIGN_001B_REFERENCE', EXPECTED_SHA256, 'cases=18')


if __name__ == '__main__':
    main()
