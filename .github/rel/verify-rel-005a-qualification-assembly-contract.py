#!/usr/bin/env python3
"""Static verifier for the preregistered REL-005A QualificationAssemblyOnly contract."""

from __future__ import annotations

import json
import pathlib

CONTRACT = pathlib.Path('.github/rel/rel-005a-qualification-assembly-contract.json')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    contract = json.loads(CONTRACT.read_text())
    require(contract['schema'] == 'symthaea.rel.qualification-assembly-contract.v1', 'schema mismatch')
    require(contract['authority'] == 'QualificationAssemblyContractOnly', 'authority mismatch')
    require(contract['relation'] == 'REL-005A', 'relation mismatch')
    require(contract['qualification_contract_head'] == '452f0324a561a2872c00b6197237d4db46189f8d', 'qualification head mismatch')
    require(contract['projection_contract_head'] == '07af7c4bdfdda377aa8175efe7a3233ce90e889e', 'projection head mismatch')
    require(contract['projection_contract_run_id'] == 35324485955, 'projection run mismatch')
    require(contract['comparison_subject_head'] == '9ed033eb5ad63baa90d9c25166ce9c14e68013cb', 'comparison head mismatch')
    require(contract['comparison_run_id'] == 35324217302, 'comparison run mismatch')
    require(contract['required_predicate_count'] == 41, 'predicate count mismatch')
    require(contract['allowed_comparison_results'] == ['ALL_PREDICATES_PASS', 'PREDICATE_FAILURES'], 'allowed outcomes mismatch')
    require(contract['selected_comparison_result'] is None, 'contract must not preselect scientific outcome')

    expected_binding = {
        'subject_head', 'source_run_id', 'source_job_id',
        'artifact_id', 'artifact_name', 'artifact_digest',
    }
    for key in ('comparison', 'projection_contract'):
        require(set(contract['future_source_binding_required_fields'][key]) == expected_binding, f'{key} source-binding field mismatch')

    topology = contract['assembly_topology']
    require([stage['stage'] for stage in topology] == ['EvidenceProjectionOnly', 'artifact_boundary', 'QualificationOnly'], 'assembly topology mismatch')
    projection, boundary, qualification = topology
    require(projection['may_read_detailed_comparison'] is True, 'projection must be allowed detailed comparison')
    require(projection['may_read_raw_observation'] is False, 'projection must not read raw observation')
    require(projection['may_issue_qualification'] is False, 'projection exceeds authority')
    require(projection['must_validate_exact_source_metadata_before_projection'] is True, 'source metadata validation not required')

    expected_files = {
        'predicate-contract-receipt.json',
        'execution-v3-receipt.json',
        'observation-seal-v3.json',
        'comparison-only-qualification-receipt.json',
        'qualification-input-manifest.json',
        'qualification-assembly-receipt.json',
    }
    require(set(boundary['required_files']) == expected_files, 'sanitized file census mismatch')
    forbidden = set(boundary['forbidden_content_classes'])
    require('raw observation JSON' in forbidden, 'raw observation not forbidden')
    require('individual observed values' in forbidden, 'observed values not forbidden')
    require('scientific threshold table' in forbidden, 'threshold table not forbidden')

    require(qualification['fresh_job_required'] is True, 'fresh QualificationOnly job not required')
    require(qualification['may_download_detailed_comparison_artifact'] is False, 'qualifier may see detailed comparison')
    require(qualification['may_download_execution_artifact'] is False, 'qualifier may see execution artifact')
    require(qualification['may_download_projection_contract_artifact'] is False, 'qualifier may see projection-contract artifact')
    require(qualification['may_read_only_sanitized_assembly_artifact'] is True, 'qualifier input not restricted')

    claims = contract['claims']
    require(not any(claims.values()), 'assembly contract must not make outcome claims')

    # Outcome-specific artifact identities cannot exist yet. Reject any accidental
    # attempt to freeze them into this preregistration after inspecting results.
    forbidden_top_level = {
        'comparison_artifact_id', 'comparison_artifact_digest',
        'projection_artifact_id', 'projection_artifact_digest',
        'qualification_result', 'scientific_result',
    }
    require(forbidden_top_level.isdisjoint(contract), 'outcome-dependent identity leaked into preregistration')

    print(json.dumps({
        'schema': 'symthaea.rel.qualification-assembly-contract-static-receipt.v1',
        'authority': 'QualificationAssemblyContractOnly',
        'comparison_run_preregistered': True,
        'projection_run_preregistered': True,
        'outcome_preselected': False,
        'fresh_qualification_job_required': True,
        'metric_free_boundary_required': True,
        'claims': claims,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
