#!/usr/bin/env python3
"""Static verifier for REL-005A QualificationAssemblyOnly v3."""

from __future__ import annotations

import json
import pathlib

CONTRACT = pathlib.Path('.github/rel/rel-005a-qualification-assembly-contract.json')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    c = json.loads(CONTRACT.read_text())
    require(c['schema'] == 'symthaea.rel.qualification-assembly-contract.v3', 'schema mismatch')
    require(c['authority'] == 'QualificationAssemblyContractOnly', 'authority mismatch')
    require(c['relation'] == 'REL-005A', 'relation mismatch')
    require(c['qualification_contract_head'] == '452f0324a561a2872c00b6197237d4db46189f8d', 'qualification head mismatch')

    require(c['projection_contract'] == {
        'subject_head': '07af7c4bdfdda377aa8175efe7a3233ce90e889e',
        'source_run_id': 35324485955,
        'workflow_name': 'REL-005A Qualification Projection Contract',
        'artifact_name': 'rel-005a-qualification-projection-contract-07af7c4bdfdda377aa8175efe7a3233ce90e889e',
    }, 'projection identity mismatch')
    require(c['comparison'] == {
        'subject_head': 'a47efccf80fa66bacbdfa9930f59c8439be7ee38',
        'source_run_id': 35349750595,
        'workflow_name': 'REL-005A ComparisonOnly V3 R2',
        'artifact_name': 'rel-005a-comparison-v3-a47efccf80fa66bacbdfa9930f59c8439be7ee38',
    }, 'comparison identity mismatch')

    p = c['predicate_source']
    require(p['subject_head'] == '43bf1d588447f602ce0f1549986bb558a839762c', 'predicate source head mismatch')
    require(p['source_run_id'] == 35278498517, 'predicate source run mismatch')
    s = c['seal_source']
    require(s['subject_head'] == '0a94ed976926fbdcd6b752f94e174df21f915fc8', 'seal head mismatch')
    require(s['source_run_id'] == 35284772522, 'seal run mismatch')
    require(s['source_job_id'] == 105414583527, 'seal job mismatch')
    require(s['observation_sha256'] == 'e271a5c2e6b51fda15cd6c3209e52859296f44ecd790b9c5e72f769d4aec1c4d', 'observation hash mismatch')
    require(s['chain_commitment_sha256'] == 'a3a288caa9221ec5859b5ac81fcc55f61b4ff43d5e52ad096021c99ea6bf31ab', 'seal chain mismatch')

    require(c['required_predicate_count'] == 41, 'predicate count mismatch')
    require(c['allowed_comparison_results'] == ['ALL_PREDICATES_PASS', 'PREDICATE_FAILURES'], 'allowed outcomes mismatch')
    require(c['selected_comparison_result'] is None, 'outcome preselected')
    require(c['transport_resolution'] == {
        'policy': 'resolve_current_unique_unexpired_exact_name_artifact_within_frozen_run',
        'artifact_id_is_scientific_authority': False,
        'artifact_zip_digest_is_scientific_authority': False,
        'resolved_transport_identity_must_be_recorded': True,
        'inner_content_commitments_are_authoritative': True,
    }, 'transport policy mismatch')

    stages = c['assembly_topology']
    require([x['stage'] for x in stages] == [
        'AuthorityReceiptExtractionOnly',
        'EvidenceProjectionOnly',
        'QualificationAssemblyOnly',
        'artifact_boundary',
        'QualificationOnly',
    ], 'topology mismatch')
    extraction, projection, assembly, boundary, qualification = stages

    require(extraction['may_download_predicate_artifact'] is True, 'extractor cannot read predicate artifact')
    require(extraction['may_download_sealed_execution_artifact'] is True, 'extractor cannot read seal artifact')
    require(extraction['may_receive_detailed_comparison'] is False, 'extractor may receive comparison')
    require(extraction['may_issue_adjudication'] is False, 'extractor may adjudicate')
    require(extraction['may_issue_qualification'] is False, 'extractor may qualify')
    require(extraction['source_capsule_contains_raw_scientific_payload'] is True, 'raw source presence not acknowledged')
    require(extraction['scientific_interpretation_permitted'] is False, 'extractor may interpret science')
    require(extraction['must_validate_inner_content_commitments'] is True, 'extractor need not verify content')
    require(set(extraction['output_files']) == {
        'predicate-contract-receipt.json', 'execution-v3-receipt.json', 'observation-seal-v3.json'
    }, 'extractor output census mismatch')
    require(extraction['raw_observation_output'] is False, 'extractor may emit raw observation')
    require(extraction['execution_logs_output'] is False, 'extractor may emit execution logs')

    require(projection['may_read_authority_receipts'] is True, 'projection cannot read receipts')
    require(projection['may_read_detailed_comparison'] is True, 'projection cannot read comparison')
    require(projection['may_read_raw_observation'] is False, 'projection may read raw observation')
    require(projection['may_read_execution_logs'] is False, 'projection may read execution logs')
    require(projection['may_issue_qualification'] is False, 'projection may qualify')
    require(projection['must_validate_exact_source_metadata_before_projection'] is True, 'projection provenance validation missing')
    require(projection['must_validate_inner_content_commitments'] is True, 'projection content validation missing')

    require(assembly['may_read_detailed_comparison'] is False, 'assembly may read detailed comparison')
    require(assembly['may_read_raw_observation'] is False, 'assembly may read raw observation')
    require(assembly['adds_only_provenance_receipt'] is True, 'assembly exceeds provenance authority')

    required_files = {
        'predicate-contract-receipt.json', 'execution-v3-receipt.json',
        'observation-seal-v3.json', 'comparison-only-qualification-receipt.json',
        'qualification-input-manifest.json', 'qualification-assembly-receipt.json',
    }
    require(set(boundary['required_files']) == required_files, 'boundary file census mismatch')
    for token in ('raw observation JSON', 'execution stdout/stderr', 'detailed predicate array',
                  'individual observed values', 'individual expected values', 'scientific threshold table'):
        require(token in boundary['forbidden_content_classes'], f'missing forbidden class: {token}')

    require(qualification['fresh_job_required'] is True, 'fresh qualification job not required')
    require(qualification['may_download_detailed_comparison_artifact'] is False, 'qualifier may read comparison artifact')
    require(qualification['may_download_execution_artifact'] is False, 'qualifier may read execution artifact')
    require(qualification['may_download_projection_contract_artifact'] is False, 'qualifier may read projection artifact')
    require(qualification['may_read_only_sanitized_assembly_artifact'] is True, 'qualifier input not restricted')

    require(not any(c['claims'].values()), 'contract makes outcome claims')
    forbidden = {
        'comparison_artifact_id', 'comparison_artifact_digest', 'comparison_sha256',
        'projection_artifact_id', 'projection_artifact_digest', 'projection_contract_receipt_sha256',
        'qualification_result', 'scientific_result',
    }
    require(forbidden.isdisjoint(c), 'outcome-dependent value leaked into preregistration')

    print(json.dumps({
        'schema': 'symthaea.rel.qualification-assembly-contract-static-receipt.v3',
        'authority': 'QualificationAssemblyContractOnly',
        'authority_receipt_extraction_isolated': True,
        'projection_raw_observation_visibility': False,
        'comparison_run_preregistered': True,
        'projection_run_preregistered': True,
        'outcome_preselected': False,
        'transport_identity_scientific_authority': False,
        'inner_content_commitments_authoritative': True,
        'fresh_qualification_job_required': True,
        'claims': c['claims'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
