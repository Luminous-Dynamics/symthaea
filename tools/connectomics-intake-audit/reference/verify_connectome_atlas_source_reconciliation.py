#!/usr/bin/env python3
"""Independent source-closure and reproduction-tier auditor for the C. elegans atlas.

This tool validates source inventory evidence and computes the strongest
reproduction tier supported by the submitted manifest. It never treats remote
metadata, third-party audit leads, or a paper table as raw trial-level controls.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


TIERS = (
    'METADATA_ONLY',
    'PARTIAL_RESPONSE_REANALYSIS',
    'PAPER_SOURCE_DATA_REPRODUCED',
    'FULL_RAW_REPRODUCED',
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf'[0-9a-f]{{{length}}}', value) is not None


def unique(values: list[str], label: str) -> None:
    require(all(isinstance(value, str) and value.strip() for value in values), f'{label} has an empty identity')
    require(len(values) == len(set(values)), f'{label} has duplicate identities')


def component_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = data['evidence_components']
    ids = [row['component_id'] for row in rows]
    unique(ids, 'evidence component IDs')
    return {row['component_id']: row for row in rows}


def infer_reproduction_tier(data: dict[str, Any]) -> str:
    components = component_map(data)
    reproduction = data['reproduction_evidence']
    materialized = {
        component_id
        for component_id, row in components.items()
        if row['availability'] == 'MATERIALIZED_AND_HASHED' and valid_sha(row['content_sha256'])
    }
    raw_stimulated = 'STIMULATED_RAW_RECORDINGS' in materialized
    raw_controls = 'UNSTIMULATED_NULL_CONTROLS' in materialized
    code_pinned = valid_sha(reproduction['analysis_code_commit_sha'], 40)
    full = (
        raw_stimulated
        and raw_controls
        and 'EVENT_EXCLUSION_FLAGS' in materialized
        and 'NEURON_IDENTITY_MAP' in materialized
        and reproduction['event_exclusion_lineage_complete'] is True
        and reproduction['neuron_identity_lineage_complete'] is True
        and reproduction['q_method_reproduced'] is True
        and reproduction['qeq_method_reproduced'] is True
        and reproduction['qeq_epsilon_sigma'] is not None
        and abs(float(reproduction['qeq_epsilon_sigma']) - 1.2) <= 1e-12
        and code_pinned
    )
    if full:
        return 'FULL_RAW_REPRODUCED'
    paper_tables = (
        reproduction['paper_source_tables_verified'] is True
        and valid_sha(reproduction['paper_source_tables_sha256'])
    )
    if paper_tables:
        return 'PAPER_SOURCE_DATA_REPRODUCED'
    if raw_stimulated:
        return 'PARTIAL_RESPONSE_REANALYSIS'
    dandi_remote = data['dandi_release']['inventory_state'] == 'REMOTE_METADATA_ONLY'
    osf_remote = data['osf_release']['inventory_state'] in {'NOT_YET_ASSESSED', 'REMOTE_METADATA_ONLY'}
    any_materialized = bool(materialized) or any(
        artifact['availability'] == 'MATERIALIZED_HASHED'
        for artifact in data['osf_release']['artifact_records']
    )
    if not any_materialized and dandi_remote and osf_remote:
        return 'METADATA_ONLY'
    if not any_materialized and not paper_tables:
        return 'NOT_REPRODUCIBLE'
    return 'METADATA_ONLY'


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(data.get('schema_version') == 1, 'unsupported schema version')
    require(data['paper']['doi'] == '10.1038/s41586-023-06683-4', 'unexpected paper DOI')
    dandi = data['dandi_release']
    require(dandi['dandiset_id'] == '001075', 'unexpected DANDI dataset ID')
    require(dandi['version'] == '0.240930.1859', 'DANDI version is not the pinned release')
    require(dandi['doi'] == '10.48324/dandi.001075/0.240930.1859', 'DANDI DOI does not match the pinned release')
    require(dandi['declared_asset_count'] == 223 and dandi['subject_count'] == 113, 'DANDI declared metadata count mismatch')
    require(dandi['data_standard'] == 'NWB' and dandi['license'] == 'CC-BY-4.0' and dandi['open_access'] is True, 'DANDI metadata fields differ from the pinned release')
    if dandi['inventory_state'] == 'REMOTE_METADATA_ONLY':
        require(dandi['inventory_manifest_sha256'] is None and dandi['materialized_asset_count'] == 0, 'remote-only DANDI inventory cannot claim a materialized manifest')
    else:
        require(valid_sha(dandi['inventory_manifest_sha256']), 'materialized DANDI inventory lacks manifest digest')
        require(dandi['materialized_asset_count'] >= 1, 'materialized DANDI inventory has zero assets')
        if dandi['inventory_state'] == 'VERIFIED_EXACT':
            require(dandi['materialized_asset_count'] == dandi['declared_asset_count'], 'VERIFIED_EXACT requires all declared DANDI assets')

    osf = data['osf_release']
    require(osf['doi'] == '10.17605/OSF.IO/E2SYT' and osf['project_id'] == 'E2SYT', 'OSF project identity mismatch')
    if osf['inventory_state'] in {'NOT_YET_ASSESSED', 'REMOTE_METADATA_ONLY'}:
        require(osf['inventory_manifest_sha256'] is None and not osf['artifact_records'], 'unmaterialized OSF inventory carries artifact records')
    else:
        require(valid_sha(osf['inventory_manifest_sha256']), 'materialized OSF inventory lacks manifest digest')
        require(osf['artifact_records'], 'materialized OSF inventory has no artifact records')
    artifact_ids = [row['artifact_id'] for row in osf['artifact_records']]
    unique(artifact_ids, 'OSF artifact IDs')
    for artifact in osf['artifact_records']:
        if artifact['availability'] == 'MATERIALIZED_HASHED':
            require(valid_sha(artifact['sha256']), f"OSF artifact lacks SHA-256: {artifact['artifact_id']}")
        else:
            require(artifact['sha256'] is None, f"unmaterialized OSF artifact has a digest: {artifact['artifact_id']}")

    repo_ids = [row['repository_id'] for row in data['code_repositories']]
    unique(repo_ids, 'code repository IDs')
    for code_repo in data['code_repositories']:
        if code_repo['revision_status'] == 'COMMIT_PINNED':
            require(valid_sha(code_repo['commit_sha'], 40), f"code repo commit pin is malformed: {code_repo['repository_id']}")
        else:
            require(code_repo['commit_sha'] is None, f"un-pinned code repo carries a commit SHA: {code_repo['repository_id']}")

    components = component_map(data)
    for row in data['evidence_components']:
        if row['availability'] == 'MATERIALIZED_AND_HASHED':
            require(valid_sha(row['content_sha256']), f"materialized component lacks content hash: {row['component_id']}")
        elif row['availability'] in {'NOT_YET_ASSESSED', 'REMOTE_METADATA_ONLY', 'NOT_FOUND_AFTER_DECLARED_SEARCH', 'NOT_PUBLICLY_ACCESSIBLE'}:
            require(row['content_sha256'] is None, f"unmaterialized component carries content hash: {row['component_id']}")

    reproduction = data['reproduction_evidence']
    if reproduction['paper_source_tables_verified']:
        require(valid_sha(reproduction['paper_source_tables_sha256']), 'verified paper tables lack SHA-256')
    else:
        require(reproduction['paper_source_tables_sha256'] is None, 'unverified paper tables carry a verification digest')
    if reproduction['q_method_reproduced']:
        require('STIMULATED_RAW_RECORDINGS' in components and components['STIMULATED_RAW_RECORDINGS']['availability'] == 'MATERIALIZED_AND_HASHED', 'q method cannot be reproduced without materialized stimulated recordings')
        require('UNSTIMULATED_NULL_CONTROLS' in components and components['UNSTIMULATED_NULL_CONTROLS']['availability'] == 'MATERIALIZED_AND_HASHED', 'q method cannot be reproduced without materialized unstimulated null controls')
    if reproduction['qeq_method_reproduced']:
        require('UNSTIMULATED_NULL_CONTROLS' in components and components['UNSTIMULATED_NULL_CONTROLS']['availability'] == 'MATERIALIZED_AND_HASHED', 'q_eq method cannot be reproduced without materialized unstimulated null controls')
        require(reproduction['qeq_epsilon_sigma'] is not None and abs(float(reproduction['qeq_epsilon_sigma']) - 1.2) <= 1e-12, 'q_eq reproduction must pin the paper epsilon=1.2 sigma')
    if reproduction['analysis_code_commit_sha'] is not None:
        require(valid_sha(reproduction['analysis_code_commit_sha'], 40), 'analysis code commit SHA malformed')

    leads = data['independent_audit_leads']
    lead_ids = [row['lead_id'] for row in leads]
    unique(lead_ids, 'independent audit lead IDs')
    for lead in leads:
        if lead['status'] == 'UNVERIFIED_LEAD':
            require(lead['verification_receipt_sha256'] is None, f"unverified lead carries a verification receipt: {lead['lead_id']}")
        else:
            require(valid_sha(lead['verification_receipt_sha256']), f"verified/disputed lead lacks a verification receipt: {lead['lead_id']}")

    inferred_tier = infer_reproduction_tier(data)
    tier_order = {tier: index for index, tier in enumerate(TIERS)}
    require(data['declared_reproduction_tier'] == inferred_tier, f"declared reproduction tier {data['declared_reproduction_tier']} exceeds or differs from evidence-supported tier {inferred_tier}")
    require(inferred_tier != 'FULL_RAW_REPRODUCED' or (
        reproduction['q_method_reproduced']
        and reproduction['qeq_method_reproduced']
        and reproduction['event_exclusion_lineage_complete']
        and reproduction['neuron_identity_lineage_complete']
        and reproduction['analysis_code_commit_sha'] is not None
    ), 'full tier requires independent q/q_eq reproduction, complete source lineage and a pinned analysis commit')
    require(
        data['claim_scope'] != 'metadata_inventory' or inferred_tier == 'METADATA_ONLY',
        'metadata_inventory claim scope cannot promote a response-reanalysis or reproduction tier'
    )

    return {
        'schema_version': 1,
        'tool': 'verify-connectome-atlas-source-reconciliation',
        'audit_id': data['audit_id'],
        'claim_scope': data['claim_scope'],
        'dandi_doi': dandi['doi'],
        'dandi_asset_count': dandi['declared_asset_count'],
        'dandi_materialized_asset_count': dandi['materialized_asset_count'],
        'dandi_inventory_state': dandi['inventory_state'],
        'osf_doi': osf['doi'],
        'osf_inventory_state': osf['inventory_state'],
        'code_repository_count': len(repo_ids),
        'code_repositories_pinned_count': sum(row['revision_status'] == 'COMMIT_PINNED' for row in data['code_repositories']),
        'evidence_component_count': len(components),
        'materialized_component_count': sum(row['availability'] == 'MATERIALIZED_AND_HASHED' for row in components.values()),
        'unstimulated_null_controls_status': components['UNSTIMULATED_NULL_CONTROLS']['availability'],
        'q_method_reproduced': reproduction['q_method_reproduced'],
        'qeq_method_reproduced': reproduction['qeq_method_reproduced'],
        'independent_audit_lead_count': len(leads),
        'unverified_audit_lead_count': sum(row['status'] == 'UNVERIFIED_LEAD' for row in leads),
        'inferred_reproduction_tier': inferred_tier,
        'status': 'METADATA_INVENTORY_VALIDATED' if inferred_tier == 'METADATA_ONLY' else 'SOURCE_RECONCILIATION_VALIDATED',
        'claim_boundary': 'Archive metadata validation is not file completeness, OSF reconciliation, q/q_eq reproduction, biological calibration or source-data authenticity.',
    }


def run_negative_controls(data: dict[str, Any]) -> None:
    cases = []
    overclaim = copy.deepcopy(data)
    overclaim['declared_reproduction_tier'] = 'FULL_RAW_REPRODUCED'
    cases.append(('metadata-only overclaimed as full raw reproduction', overclaim))
    false_qeq = copy.deepcopy(data)
    false_qeq['reproduction_evidence']['qeq_method_reproduced'] = True
    false_qeq['reproduction_evidence']['qeq_epsilon_sigma'] = 1.2
    cases.append(('q_eq reproduced without materialized null controls', false_qeq))
    promoted_lead = copy.deepcopy(data)
    promoted_lead['independent_audit_leads'][0]['status'] = 'VERIFIED'
    cases.append(('independent audit lead promoted without receipt', promoted_lead))
    duplicate_component = copy.deepcopy(data)
    duplicate_component['evidence_components'].append(copy.deepcopy(duplicate_component['evidence_components'][0]))
    cases.append(('duplicate evidence component identity', duplicate_component))
    stale_inventory = copy.deepcopy(data)
    stale_inventory['dandi_release']['inventory_state'] = 'VERIFIED_EXACT'
    stale_inventory['dandi_release']['inventory_manifest_sha256'] = 'a' * 64
    cases.append(('verified exact DANDI inventory without all assets', stale_inventory))
    false_q = copy.deepcopy(data)
    false_q['reproduction_evidence']['q_method_reproduced'] = True
    cases.append(('q method claimed without materialized raw/null data', false_q))
    false_claim_scope = copy.deepcopy(data)
    false_claim_scope['claim_scope'] = 'full_raw_reproduction'
    cases.append(('metadata-only evidence with full-reproduction claim scope', false_claim_scope))
    for label, mutated in cases:
        try:
            verify(mutated)
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError(f'negative control was not rejected: {label}')
    print(f'atlas source-reconciliation negative controls: PASS ({len(cases)})')


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
        print(json.dumps(report, indent=2, sort_keys=True))
        print('independent connectome atlas source-reconciliation oracle: PASS')
        return 0
    except Exception as exc:
        print(f'connectome atlas source-reconciliation oracle: FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())