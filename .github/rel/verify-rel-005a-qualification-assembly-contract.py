#!/usr/bin/env python3
"""Static verifier for REL-005A QualificationAssemblyOnly v2."""

from __future__ import annotations

import json
import pathlib

CONTRACT = pathlib.Path('.github/rel/rel-005a-qualification-assembly-contract.json')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    c = json.loads(CONTRACT.read_text())
    require(c['schema'] == 'symthaea.rel.qualification-assembly-contract.v2', 'schema mismatch')
    require(c['authority'] == 'QualificationAssemblyContractOnly', 'authority mismatch')
    require(c['relation'] == 'REL-005A', 'relation mismatch')
    require(c['qualification_contract_head'] == '452f0324a561a2872c00b6197237d4db46189f8d', 'qualification head mismatch')

    p = c['projection_contract']
    require(p == {
        'subject_head': '07af7c4bdfdda377aa8175efe7a3233ce90e889e',
        'source_run_id': 35324485955,
        'workflow_name': 'REL-005A Qualification Projection Contract',
        'artifact_name': 'rel-005a-qualification-projection-contract-07af7c4bdfdda377aa8175efe7a3233ce90e889e',
    }, 'projection identity mismatch')

    x = c['comparison']
    require(x == {
        'subject_head': 'a47efccf80fa66bacbdfa9930f59c8439be7ee38',
        'source_run_id': 35349750595,
        'workflow_name': 'REL-005A ComparisonOnly V3 R2',
        'artifact_name': 'rel-005a-comparison-v3-a47efccf80fa66bacbdfa9930f59c8439be7ee38',
    }, 'comparison identity mismatch')

    require(c['required_predicate_count'] == 41, 'predicate count mismatch')
    require(c['allowed_comparison_results'] == ['ALL_PREDICATES_PASS', 'PREDICATE_FAILURES'], 'allowed outcomes mismatch')
    require(c['selected_comparison_result'] is None, 'outcome must remain unselected')

    tr = c['transport_resolution']
    require(tr == {
        'policy': 'resolve_current_unique_unexpired_exact_name_artifact_within_frozen_run',
        'artifact_id_is_scientific_authority': False,
        'artifact_zip_digest_is_scientific_authority': False,
        'resolved_transport_identity_must_be_recorded': True,
        'inner_content_commitments_are_authoritative': True,
    }, 'transport policy mismatch')

    expected = {
        'comparison_authoritative': {'subject_head', 'source_run_id', 'source_job_id', 'artifact_name', 'comparison_sha256'},
        'comparison_transport_record': {'artifact_id', 'artifact_digest'},
        'projection_authoritative': {'subject_head', 'source_run_id', 'source_job_id', 'artifact_name', 'projection_contract_receipt_sha256'},
        'projection_transport_record': {'artifact_id', 'artifact_digest'},
    }
    for key, fields in expected.items():
        require(set(c['future_source_binding_required_fields'][key]) == fields, f'{key} fields mismatch')

    topology = c['assembly_topology']
    require([s['stage'] for s in topology] == ['EvidenceProjectionOnly', 'artifact_boundary', 'QualificationOnly'], 'topology mismatch')
    projection, boundary, qualification = topology
    require(projection['may_read_detailed_comparison'] is True, 'projection cannot read comparison')
    require(projection['may_read_raw_observation'] is False, 'projection may read raw observation')
    require(projection['may_issue_qualification'] is False, 'projection exceeds authority')
    require(projection['must_validate_exact_source_metadata_before_projection'] is True, 'source metadata validation not required')
    require(projection['must_validate_inner_content_commitments'] is True, 'inner commitments not required')

    required_files = {
        'predicate-contract-receipt.json', 'execution-v3-receipt.json',
        'observation-seal-v3.json', 'comparison-only-qualification-receipt.json',
        'qualification-input-manifest.json', 'qualification-assembly-receipt.json',
    }
    require(set(boundary['required_files']) == required_files, 'file census mismatch')
    require('raw observation JSON' in boundary['forbidden_content_classes'], 'raw observation not forbidden')
    require('individual observed values' in boundary['forbidden_content_classes'], 'metric values not forbidden')
    require('scientific threshold table' in boundary['forbidden_content_classes'], 'threshold table not forbidden')

    require(qualification['fresh_job_required'] is True, 'fresh qualifier not required')
    require(qualification['may_download_detailed_comparison_artifact'] is False, 'qualifier may see detailed comparison')
    require(qualification['may_download_execution_artifact'] is False, 'qualifier may see execution artifact')
    require(qualification['may_download_projection_contract_artifact'] is False, 'qualifier may see projection artifact')
    require(qualification['may_read_only_sanitized_assembly_artifact'] is True, 'qualifier input not restricted')

    require(not any(c['claims'].values()), 'contract makes authority claims')

    # Outcome-dependent transport/content values must not be preregistered here.
    forbidden = {
        'comparison_artifact_id', 'comparison_artifact_digest', 'comparison_sha256',
        'projection_artifact_id', 'projection_artifact_digest', 'projection_contract_receipt_sha256',
        'qualification_result', 'scientific_result',
    }
    require(forbidden.isdisjoint(c), 'outcome-dependent value leaked into preregistration')

    print(json.dumps({
        'schema': 'symthaea.rel.qualification-assembly-contract-static-receipt.v2',
        'authority': 'QualificationAssemblyContractOnly',
        'comparison_run_preregistered': True,
        'projection_run_preregistered': True,
        'outcome_preselected': False,
        'transport_identity_scientific_authority': False,
        'inner_content_commitments_authoritative': True,
        'fresh_qualification_job_required': True,
        'metric_free_boundary_required': True,
        'claims': c['claims'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
