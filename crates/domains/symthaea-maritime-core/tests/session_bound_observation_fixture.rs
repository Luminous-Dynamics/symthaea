// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::Deserialize;
use symthaea_maritime_core::{
    AuthenticatedMachineSession, MachineSessionContext, MachineSessionPolicy,
    MaritimeObservationKind, ObservationSourceGrantV1, ObservationSourcePolicyV1,
    bind_observation_to_trusted_session,
};

const XENIA_SCHEMA_V1: &str = "xenia-verified-machine-session-evidence-v1";
const XENIA_FIXTURE: &str =
    include_str!("../fixtures/xenia-verified-machine-session-evidence-v1.json");
const OBSERVATION_FIXTURE: &str =
    include_str!("../fixtures/session-bound-observation-v1.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct XeniaSessionFixtureV1 {
    schema: String,
    session_id: String,
    peer_identity_binding: String,
    authenticated_at_ms: u64,
    expires_at_ms: u64,
    authority_epoch: u64,
    evidence_binding: String,
    negotiated_context_binding: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationFixtureV1 {
    platform_id: String,
    observed_at_us: u64,
    kind: MaritimeObservationKind,
    payload_json: String,
    position_evidence_refs: Vec<String>,
    expected_binding: String,
}

#[test]
fn xenia_session_and_local_source_policy_pin_exact_observation_binding() {
    let xenia: XeniaSessionFixtureV1 = serde_json::from_str(XENIA_FIXTURE).unwrap();
    let observation: ObservationFixtureV1 = serde_json::from_str(OBSERVATION_FIXTURE).unwrap();

    assert_eq!(xenia.schema, XENIA_SCHEMA_V1);
    assert!(xenia.negotiated_context_binding.is_some());

    // This constructor mirrors the explicit handoff after provider-native verification.
    // Parsing the fixture alone is not authentication.
    let session = AuthenticatedMachineSession::from_verified_provider(
        xenia.schema.clone(),
        xenia.session_id,
        xenia.peer_identity_binding.clone(),
        xenia.authenticated_at_ms,
        xenia.expires_at_ms,
        xenia.authority_epoch,
        xenia.evidence_binding,
    )
    .unwrap();

    let context = MachineSessionContext::from_authority_provider(
        xenia.schema.clone(),
        xenia.peer_identity_binding.clone(),
        observation.observed_at_us / 1_000,
        xenia.authority_epoch,
        true,
        false,
    )
    .unwrap();
    let session_policy = MachineSessionPolicy {
        accepted_schemas: &[XENIA_SCHEMA_V1],
        max_validity_ms: 60_000,
    };
    let source_policy = ObservationSourcePolicyV1::new([ObservationSourceGrantV1::new(
        xenia.schema,
        xenia.peer_identity_binding,
        observation.platform_id.clone(),
    )
    .unwrap()])
    .unwrap();

    let bound = bind_observation_to_trusted_session(
        &session,
        &context,
        session_policy,
        &source_policy,
        observation.platform_id,
        observation.observed_at_us,
        observation.kind,
        &observation.payload_json,
        &observation.position_evidence_refs,
    )
    .unwrap();

    assert_eq!(bound.evidence_binding(), observation.expected_binding);
    assert_eq!(
        bound.evidence_binding(),
        "symthaea-maritime-session-bound-observation-v1:blake3-256:c258e96264cee15f3e9fd95b36e049884a6a42e67ae968a36a5b9868fc2e52d0"
    );
}
