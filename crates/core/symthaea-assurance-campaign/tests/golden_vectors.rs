// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ASSURE-002G cross-language `campaign.v1` golden-vector conformance.
//!
//! The JSON corpus is transport-only test data. Rust reconstructs typed
//! ASSURE-000/001/002A/002 objects and compares their identities with the
//! independently derived Python/std-lib constants in the same fixture.

use symthaea_assurance_campaign::{
    CampaignEvidenceLedgerV1, CampaignPlanV1, EvidenceRequirementV1, OrderingReceiptV1,
    PreregistrationReceiptV1, RegistrationStatementV1, RegistrationWithdrawalV1,
    ReproductionRequirementV1, SupportCriterionV1, WithdrawalStatementV1,
    evidence_commitment_statement_digest, resolve_terminal_registration_in_view,
};
use symthaea_assurance_core::{
    Claim, DigestSha256, EvidenceArtifact, EvidenceKind, EvidenceProvenance, StableId, SupportTier,
};
use symthaea_assurance_semantics::{DefinitionSchemaV1, SemanticCommitmentV1};
use symthaea_assurance_subject::{
    AiSubjectManifest, AiSurfaceKind, MaterialCommitment, SurfaceBinding, SurfaceLocator,
    SurfaceProfile, SurfaceState,
};

const VECTORS: &str = include_str!("../vectors/campaign_v1.json");

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(value: &str) -> DigestSha256 {
    DigestSha256::new(value).unwrap()
}

fn repeated(byte: char) -> DigestSha256 {
    digest(&std::iter::repeat_n(byte, 64).collect::<String>())
}

fn marker<'a>(source: &'a str, needle: &str) -> &'a str {
    let start = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing fixture marker: {needle}"));
    &source[start + needle.len()..]
}

fn json_string(source: &str, key: &str) -> String {
    let tail = marker(source, &format!("\"{key}\":\""));
    let mut out = String::new();
    let mut escaped = false;
    for ch in tail.chars() {
        if escaped {
            match ch {
                '"' | '\\' | '/' => out.push(ch),
                'b' => out.push('\u{0008}'),
                'f' => out.push('\u{000c}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                other => panic!("unsupported JSON escape in fixture: \\{other}"),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return out;
        } else {
            out.push(ch);
        }
    }
    panic!("unterminated JSON string for key {key}")
}

fn json_u64(source: &str, key: &str) -> u64 {
    let tail = marker(source, &format!("\"{key}\":"));
    let digits: String = tail.chars().take_while(|ch| ch.is_ascii_digit()).collect();
    assert!(!digits.is_empty(), "missing u64 digits for key {key}");
    digits.parse().unwrap()
}

fn balanced_object(source: &str, start: usize) -> &str {
    let bytes = source.as_bytes();
    assert_eq!(bytes[start], b'{');
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, byte) in bytes[start..].iter().copied().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated JSON object")
}

fn object_for_key<'a>(source: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\":{{");
    let key_start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("missing object key: {key}"));
    let object_start = key_start + needle.len() - 1;
    balanced_object(source, object_start)
}

fn vector_object<'a>(source: &'a str, name: &str) -> &'a str {
    let needle = format!("{{\"name\":\"{name}\"");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("missing vector: {name}"));
    balanced_object(source, start)
}

fn evidence_object<'a>(vector: &'a str, evidence_id: &str) -> &'a str {
    let needle = format!("{{\"evidence_id\":\"{evidence_id}\"");
    let start = vector
        .find(&needle)
        .unwrap_or_else(|| panic!("missing evidence vector: {evidence_id}"));
    balanced_object(vector, start)
}

fn semantic(key: &str) -> SemanticCommitmentV1 {
    let source = object_for_key(object_for_key(VECTORS, "semantics"), key);
    let value = SemanticCommitmentV1::new(
        id(&json_string(source, "semantic_id")),
        DefinitionSchemaV1::new(
            id(&json_string(source, "definition_schema_id")),
            digest(&json_string(
                source,
                "definition_schema_specification_digest",
            )),
        ),
        digest(&json_string(source, "definition_digest")),
    );
    assert_eq!(
        value.digest().as_str(),
        json_string(source, "expected_digest"),
        "upstream semantic digest drift for {key}"
    );
    value
}

fn subject() -> AiSubjectManifest {
    let source = object_for_key(object_for_key(VECTORS, "upstream"), "subject");
    let profile = SurfaceProfile::new(
        id(&json_string(source, "profile_id")),
        vec![AiSurfaceKind::Model],
    )
    .unwrap();
    assert_eq!(
        profile.digest().as_str(),
        json_string(source, "expected_profile_digest")
    );
    let manifest = AiSubjectManifest::new(
        id(&json_string(source, "subject_key")),
        profile,
        vec![
            SurfaceBinding::applicable(
                AiSurfaceKind::Model,
                SurfaceLocator::new(
                    Some(id(&json_string(source, "model_provider"))),
                    id(&json_string(source, "model_name")),
                    Some(id(&json_string(source, "model_version"))),
                ),
                SurfaceState::Known(MaterialCommitment::artifact_bytes(digest(&json_string(
                    source,
                    "model_artifact_digest",
                )))),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(
        manifest.manifest_id().as_str(),
        json_string(source, "expected_manifest_id")
    );
    assert_eq!(
        manifest.core_subject_id().unwrap().as_str(),
        json_string(source, "expected_core_subject_id")
    );
    manifest
}

fn claim(subject: &AiSubjectManifest) -> Claim {
    let source = object_for_key(object_for_key(VECTORS, "upstream"), "claim");
    let claim = Claim::new(
        id(&json_string(source, "claim_id")),
        subject.core_subject_id().unwrap(),
        json_string(source, "statement"),
        id(&json_string(source, "scope")),
    )
    .unwrap();
    assert_eq!(
        claim.digest().as_str(),
        json_string(source, "expected_digest")
    );
    claim
}

fn evidence_kind(name: &str) -> EvidenceKind {
    match name {
        "observation" => EvidenceKind::Observation,
        "controlled-intervention" => EvidenceKind::ControlledIntervention,
        other => panic!("unsupported campaign-v1 vector evidence kind: {other}"),
    }
}

fn plan(subject: &AiSubjectManifest, claim: &Claim, vector: &str) -> CampaignPlanV1 {
    let upstream = object_for_key(VECTORS, "upstream");
    let plan_source = object_for_key(upstream, "plan");
    let extra = semantic(&json_string(vector, "extra_control_semantic"));
    let plan = CampaignPlanV1::new(
        id(&json_string(plan_source, "plan_key")),
        id(&json_string(vector, "campaign_nonce")),
        claim,
        subject,
        SupportTier::CausallySupported,
        ReproductionRequirementV1::NotRequired,
        vec![
            EvidenceRequirementV1::builtin(EvidenceKind::ControlledIntervention).unwrap(),
            EvidenceRequirementV1::builtin(EvidenceKind::Observation).unwrap(),
        ],
        vec![
            SupportCriterionV1::new(
                SupportTier::Observed,
                semantic("observable-authority-decision"),
            ),
            SupportCriterionV1::new(
                SupportTier::CausallySupported,
                semantic("authority-boundary-causal-effect"),
            ),
        ],
        vec![semantic("baseline"), extra],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap();
    assert_eq!(
        plan.core_plan().digest().as_str(),
        json_string(plan_source, "expected_core_plan_digest")
    );
    plan
}

fn ordering(
    statement: DigestSha256,
    epoch: u64,
    sequence: u64,
    external_receipt: char,
) -> OrderingReceiptV1 {
    let ordering_source = object_for_key(object_for_key(VECTORS, "upstream"), "ordering");
    OrderingReceiptV1::new(
        id(&json_string(ordering_source, "source")),
        semantic(&json_string(ordering_source, "validation_semantic")),
        epoch,
        sequence,
        statement,
        repeated(external_receipt),
    )
    .unwrap()
}

fn evidence(
    subject: &AiSubjectManifest,
    claim: &Claim,
    vector: &str,
    evidence_id: &str,
) -> EvidenceArtifact {
    let source = evidence_object(vector, evidence_id);
    let artifact = EvidenceArtifact::new(
        id(&json_string(source, "evidence_id")),
        subject.core_subject_id().unwrap(),
        claim.digest(),
        evidence_kind(&json_string(source, "kind")),
        digest(&json_string(source, "artifact_digest")),
        EvidenceProvenance::new(id("producer"), id("executor"), Some(id("verifier")), None),
    );
    assert_eq!(
        artifact.digest().as_str(),
        json_string(source, "digest"),
        "ASSURE-000 evidence digest drift for {evidence_id}"
    );
    artifact
}

fn assert_vector(name: &str) -> String {
    let vector = vector_object(VECTORS, name);
    let expected = object_for_key(vector, "expected");
    let subject = subject();
    let claim = claim(&subject);
    let plan = plan(&subject, &claim, vector);
    assert_eq!(
        plan.digest().as_str(),
        json_string(expected, "campaign_plan_digest")
    );

    let epoch = json_u64(vector, "epoch");
    let registration_sequence = json_u64(vector, "registration_sequence");
    let registration_statement =
        RegistrationStatementV1::new(&plan, id("registrar-a"), None).unwrap();
    assert_eq!(
        registration_statement.empty_evidence_root().as_str(),
        json_string(expected, "empty_evidence_root")
    );
    assert_eq!(
        registration_statement.digest().as_str(),
        json_string(expected, "registration_statement_digest")
    );
    let registration_ordering = ordering(
        registration_statement.digest(),
        epoch,
        registration_sequence,
        'e',
    );
    assert_eq!(
        registration_ordering.digest().as_str(),
        json_string(expected, "registration_ordering_digest")
    );
    let registration =
        PreregistrationReceiptV1::new(registration_statement, registration_ordering.clone(), None)
            .unwrap();
    assert_eq!(
        registration.digest().as_str(),
        json_string(expected, "preregistration_receipt_digest")
    );

    let withdrawal_statement = WithdrawalStatementV1::new(&registration, id("operator-withdrawal"));
    assert_eq!(
        withdrawal_statement.digest().as_str(),
        json_string(expected, "withdrawal_statement_digest")
    );
    let withdrawal_ordering = ordering(
        withdrawal_statement.digest(),
        epoch,
        registration_sequence.checked_add(5).unwrap(),
        'f',
    );
    assert_eq!(
        withdrawal_ordering.digest().as_str(),
        json_string(expected, "withdrawal_ordering_digest")
    );
    let withdrawal =
        RegistrationWithdrawalV1::new(withdrawal_statement, withdrawal_ordering, &registration)
            .unwrap();
    assert_eq!(
        withdrawal.digest().as_str(),
        json_string(expected, "withdrawal_digest")
    );

    let current =
        resolve_terminal_registration_in_view(std::slice::from_ref(&registration), &[]).unwrap();
    let mut ledger = CampaignEvidenceLedgerV1::new(&current);
    assert_eq!(
        ledger.evidence_root().as_str(),
        json_string(expected, "empty_evidence_root")
    );

    let evidence_1_id = format!("{name}-observation");
    let evidence_1 = evidence(&subject, &claim, vector, &evidence_1_id);
    let commitment_statement_1 = evidence_commitment_statement_digest(&current, &evidence_1);
    assert_eq!(
        commitment_statement_1.as_str(),
        json_string(expected, "commitment_statement_1_digest")
    );
    let commitment_ordering_1 = ordering(
        commitment_statement_1,
        epoch,
        registration_sequence.checked_add(1).unwrap(),
        '1',
    );
    assert_eq!(
        commitment_ordering_1.digest().as_str(),
        json_string(expected, "commitment_ordering_1_digest")
    );
    let admission_statement_1 = ledger
        .admission_statement_digest(&current, &evidence_1, &commitment_ordering_1)
        .unwrap();
    assert_eq!(
        admission_statement_1.as_str(),
        json_string(expected, "admission_statement_1_digest")
    );
    let admission_ordering_1 = ordering(
        admission_statement_1,
        epoch,
        registration_sequence.checked_add(2).unwrap(),
        '2',
    );
    assert_eq!(
        admission_ordering_1.digest().as_str(),
        json_string(expected, "admission_ordering_1_digest")
    );
    let admitted_1 = ledger
        .admit_preregistered(
            &plan,
            &current,
            &evidence_1,
            &commitment_ordering_1,
            &admission_ordering_1,
        )
        .unwrap();
    assert_eq!(
        admitted_1.evidence_root().as_str(),
        json_string(expected, "evidence_root_1")
    );
    assert_eq!(
        admitted_1.digest().as_str(),
        json_string(expected, "evidence_admission_1_digest")
    );

    let evidence_2_id = format!("{name}-intervention");
    let evidence_2 = evidence(&subject, &claim, vector, &evidence_2_id);
    let commitment_statement_2 = evidence_commitment_statement_digest(&current, &evidence_2);
    assert_eq!(
        commitment_statement_2.as_str(),
        json_string(expected, "commitment_statement_2_digest")
    );
    let commitment_ordering_2 = ordering(
        commitment_statement_2,
        epoch,
        registration_sequence.checked_add(3).unwrap(),
        '3',
    );
    assert_eq!(
        commitment_ordering_2.digest().as_str(),
        json_string(expected, "commitment_ordering_2_digest")
    );
    let admission_statement_2 = ledger
        .admission_statement_digest(&current, &evidence_2, &commitment_ordering_2)
        .unwrap();
    assert_eq!(
        admission_statement_2.as_str(),
        json_string(expected, "admission_statement_2_digest")
    );
    let admission_ordering_2 = ordering(
        admission_statement_2,
        epoch,
        registration_sequence.checked_add(4).unwrap(),
        '4',
    );
    assert_eq!(
        admission_ordering_2.digest().as_str(),
        json_string(expected, "admission_ordering_2_digest")
    );
    let admitted_2 = ledger
        .admit_preregistered(
            &plan,
            &current,
            &evidence_2,
            &commitment_ordering_2,
            &admission_ordering_2,
        )
        .unwrap();
    assert_eq!(
        admitted_2.evidence_root().as_str(),
        json_string(expected, "evidence_root_2")
    );
    assert_eq!(
        admitted_2.digest().as_str(),
        json_string(expected, "evidence_admission_2_digest")
    );
    assert_eq!(ledger.admitted_count(), 2);

    plan.digest().as_str().to_owned()
}

#[test]
fn campaign_v1_matches_independent_golden_vectors() {
    assert_eq!(
        json_string(VECTORS, "qualified_input_head"),
        "9e236b9da1d15ce486d7d6c308ac5f748e8abe30"
    );

    let ascii = assert_vector("ascii-baseline");
    let nfc = assert_vector("unicode-nfc");
    let nfd = assert_vector("unicode-nfd");
    let integer = assert_vector("integer-boundary");

    assert_ne!(ascii, nfc);
    assert_ne!(nfc, nfd);
    let nfc_nonce = json_string(vector_object(VECTORS, "unicode-nfc"), "campaign_nonce");
    let nfd_nonce = json_string(vector_object(VECTORS, "unicode-nfd"), "campaign_nonce");
    assert_ne!(nfc_nonce.as_bytes(), nfd_nonce.as_bytes());
    assert_eq!(
        json_u64(vector_object(VECTORS, "integer-boundary"), "epoch"),
        u64::MAX
    );
    assert_eq!(
        json_u64(
            vector_object(VECTORS, "integer-boundary"),
            "registration_sequence"
        )
        .checked_add(5)
        .unwrap(),
        u64::MAX
    );
    assert_ne!(integer, ascii);
}
