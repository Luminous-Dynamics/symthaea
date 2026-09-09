import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hak_provider_response_envelope as hak

POLICY_PATH = ROOT / "docs/architecture/hak/policies/hak019a-github-rest-link-pagination-v1.json"
POLICY = json.loads(POLICY_PATH.read_text())
POLICY_REF = (
    "git:Luminous-Dynamics/symthaea@1111111111111111111111111111111111111111:"
    "docs/architecture/hak/policies/hak019a-github-rest-link-pagination-v1.json"
)
PARSER_REF = (
    "git:Luminous-Dynamics/symthaea@1111111111111111111111111111111111111111:"
    "scripts/hak_provider_response_envelope.py"
)
PARSER_BYTES = hak.current_parser_bytes()
BASE_URL = "https://api.github.com/x?per_page=100&page=1"


def make_env(headers=None, *, body=b'{"total_count":0,"jobs":[]}', status=200):
    return hak.make_envelope(
        resource_kind="WorkflowJobsObservation",
        request_ref="request:jobs:page-1",
        request_url=BASE_URL,
        response_ref="response:jobs:page-1",
        status=status,
        headers=headers or [],
        body_bytes=body,
        request_not_before="2026-09-09T20:00:00Z",
        response_not_after="2026-09-09T20:00:01Z",
    )


def project(env, *, policy=POLICY, policy_ref=POLICY_REF, parser_ref=PARSER_REF, parser_bytes=PARSER_BYTES):
    return hak.project_next_relation(
        env,
        copy.deepcopy(policy),
        policy_artifact_ref=policy_ref,
        parser_ref=parser_ref,
        parser_bytes=parser_bytes,
    )


def redigest_receipt(receipt):
    receipt["receipt_digest"] = hak.compute_projection_receipt_digest(receipt)
    return receipt


def redigest_envelope(envelope):
    envelope["envelope_digest"] = hak.compute_envelope_digest(envelope)
    return envelope


def test_envelope_retains_exact_body_bytes_and_header_values():
    body = b'{"total_count":1,"jobs":[{"id":7}]}'
    headers = [
        {"name": "Content-Type", "value": "application/json"},
        {"name": "Link", "value": '<https://api.github.com/x?page=2>; rel="next"'},
    ]
    env = make_env(headers, body=body)
    hak.validate_envelope(env)
    assert hak.retained_body_bytes(env) == body
    assert env["response"]["headers"] == headers
    assert env["retention"]["raw_http_wire_representation"] == "NotRetained"
    assert env["provider_authentication"] == "NotEstablished"


def test_absolute_next_relation_is_replayed_from_retained_link_value():
    env = make_env([{"name": "Link", "value": '<https://api.github.com/x?page=2>; rel="next"'}])
    receipt = project(env)
    assert receipt["next_relation"] == {
        "state": "Present",
        "target_ref": "https://api.github.com/x?page=2",
        "target": "https://api.github.com/x?page=2",
    }
    assert receipt["projection_verification"] == "VerifiedAgainstRetainedEnvelope"
    assert receipt["exhaustion_verification"] == "NotEstablished"
    hak.validate_projection_receipt(
        receipt, env, POLICY,
        policy_artifact_ref=POLICY_REF,
        parser_ref=PARSER_REF,
        parser_bytes=PARSER_BYTES,
    )


def test_relative_next_target_resolves_against_exact_request_url():
    env = make_env([{"name": "link", "value": '<?per_page=100&page=2>; rel=next'}])
    receipt = project(env)
    assert receipt["next_relation"]["state"] == "Present"
    assert receipt["next_relation"]["target"] == "https://api.github.com/x?per_page=100&page=2"


def test_multiple_link_field_values_are_parsed_without_posthoc_page_construction():
    env = make_env([
        {"name": "Link", "value": '<https://api.github.com/x?page=1>; rel="prev"'},
        {"name": "LINK", "value": '<https://api.github.com/x?page=3>; rel="next"'},
    ])
    receipt = project(env)
    assert receipt["link_header_field_count"] == 2
    assert len(receipt["parsed_links"]) == 2
    assert receipt["next_relation"]["target"] == "https://api.github.com/x?page=3"


def test_no_link_or_next_relation_does_not_become_exhaustion_evidence():
    env = make_env([{"name": "x-test", "value": "ok"}])
    receipt = project(env)
    assert receipt["link_header_field_count"] == 0
    assert receipt["next_relation"] == {"state": "Absent", "target_ref": None, "target": None}
    assert receipt["exhaustion_verification"] == "NotEstablished"


def test_comma_inside_uri_reference_is_not_treated_as_link_separator():
    env = make_env([{
        "name": "Link",
        "value": '<https://api.github.com/x?labels=a,b&page=2>; rel="next", <https://api.github.com/x?page=9>; rel="last"',
    }])
    receipt = project(env)
    assert len(receipt["parsed_links"]) == 2
    assert receipt["next_relation"]["target"] == "https://api.github.com/x?labels=a,b&page=2"


def test_commas_and_semicolons_inside_quoted_parameter_do_not_split_links():
    env = make_env([{
        "name": "Link",
        "value": '<https://api.github.com/x?page=2>; rel="next"; title="a,b;c"',
    }])
    receipt = project(env)
    assert len(receipt["parsed_links"]) == 1
    assert receipt["next_relation"]["state"] == "Present"


def test_sp_and_htab_ows_are_accepted_around_link_syntax():
    env = make_env([{
        "name": "Link",
        "value": '\t<https://api.github.com/x?page=2>\t;\trel\t=\t"next"\t',
    }])
    receipt = project(env)
    assert receipt["next_relation"]["state"] == "Present"


def test_tab_inside_relation_type_list_is_rejected():
    env = make_env([{
        "name": "Link",
        "value": '<https://api.github.com/x?page=2>; rel="next\tlast"',
    }])
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_unicode_whitespace_is_not_treated_as_http_ows():
    env = make_env([{
        "name": "Link",
        "value": '\u00a0<https://api.github.com/x?page=2>; rel="next"',
    }])
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_duplicate_rel_parameter_is_rejected_by_strict_profile():
    env = make_env([{
        "name": "Link",
        "value": '<https://api.github.com/x?page=2>; rel="next"; rel="last"',
    }])
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_duplicate_next_targets_are_rejected_as_ambiguous():
    env = make_env([{
        "name": "Link",
        "value": '<https://api.github.com/x?page=2>; rel="next", <https://api.github.com/x?page=3>; rel="next"',
    }])
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_cross_origin_next_target_is_rejected():
    env = make_env([{"name": "Link", "value": '<https://example.com/x?page=2>; rel="next"'}])
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_different_api_path_next_target_is_rejected():
    env = make_env([{"name": "Link", "value": '<https://api.github.com/y?page=2>; rel="next"'}])
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_fragment_bearing_next_target_is_rejected():
    env = make_env([{"name": "Link", "value": '<https://api.github.com/x?page=2#frag>; rel="next"'}])
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_malformed_link_header_is_rejected():
    env = make_env([{"name": "Link", "value": '<https://api.github.com/x?page=2; rel="next"'}])
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_non_200_response_cannot_supply_pagination_projection():
    env = make_env([{"name": "Link", "value": '<https://api.github.com/x?page=2>; rel="next"'}], status=404)
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env)


def test_crlf_header_value_is_rejected_at_envelope_boundary():
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        make_env([{"name": "Link", "value": '<https://api.github.com/x>; rel="next"\r\nX-Evil: yes'}])


def test_inverted_observation_window_is_rejected():
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        hak.make_envelope(
            resource_kind="WorkflowJobsObservation",
            request_ref="request:x",
            request_url=BASE_URL,
            response_ref="response:x",
            status=200,
            headers=[],
            body_bytes=b"{}",
            request_not_before="2026-09-09T20:00:02Z",
            response_not_after="2026-09-09T20:00:01Z",
        )


def test_envelope_provider_authentication_cannot_be_promoted_after_redigest():
    env = make_env()
    env["provider_authentication"] = "Verified"
    redigest_envelope(env)
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        hak.validate_envelope(env)


def test_envelope_raw_wire_representation_cannot_be_promoted_after_redigest():
    env = make_env()
    env["retention"]["raw_http_wire_representation"] = "Retained"
    redigest_envelope(env)
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        hak.validate_envelope(env)


def test_policy_substitution_is_rejected():
    env = make_env()
    altered = copy.deepcopy(POLICY)
    altered["relation"] = "last"
    altered["policy_digest"] = hak.compute_projection_policy_digest(altered)
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env, policy=altered)


def test_parser_byte_substitution_is_rejected():
    env = make_env()
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        project(env, parser_bytes=PARSER_BYTES + b"\n# substituted")


def test_coherently_redigested_next_target_forgery_is_rejected_by_replay():
    env = make_env([{"name": "Link", "value": '<https://api.github.com/x?page=2>; rel="next"'}])
    receipt = project(env)
    receipt["next_relation"]["target_ref"] = "https://api.github.com/x?page=999"
    receipt["next_relation"]["target"] = "https://api.github.com/x?page=999"
    redigest_receipt(receipt)
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        hak.validate_projection_receipt(
            receipt, env, POLICY,
            policy_artifact_ref=POLICY_REF,
            parser_ref=PARSER_REF,
            parser_bytes=PARSER_BYTES,
        )


def test_absent_next_cannot_be_promoted_to_verified_exhaustion_after_redigest():
    env = make_env()
    receipt = project(env)
    receipt["exhaustion_verification"] = "Verified"
    redigest_receipt(receipt)
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        hak.validate_projection_receipt(
            receipt, env, POLICY,
            policy_artifact_ref=POLICY_REF,
            parser_ref=PARSER_REF,
            parser_bytes=PARSER_BYTES,
        )


def test_projection_provider_authentication_cannot_be_promoted_after_redigest():
    env = make_env()
    receipt = project(env)
    receipt["provider_authentication"] = "Verified"
    redigest_receipt(receipt)
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        hak.validate_projection_receipt(
            receipt, env, POLICY,
            policy_artifact_ref=POLICY_REF,
            parser_ref=PARSER_REF,
            parser_bytes=PARSER_BYTES,
        )


def test_http_wire_verification_cannot_be_promoted_after_redigest():
    env = make_env()
    receipt = project(env)
    receipt["http_wire_verification"] = "Verified"
    redigest_receipt(receipt)
    with pytest.raises(hak.ProviderResponseEnvelopeError):
        hak.validate_projection_receipt(
            receipt, env, POLICY,
            policy_artifact_ref=POLICY_REF,
            parser_ref=PARSER_REF,
            parser_bytes=PARSER_BYTES,
        )
