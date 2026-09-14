// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical commitment for the complete TPM2 attestation-possession policy.

#![deny(unsafe_code)]

use symthaea_assurance_tpm2_attestation_possession::AttestationPossessionPolicy;

pub const POSSESSION_POLICY_COMMITMENT_SCHEMA_V1: &str =
    "symthaea.assurance.tpm2-attestation-possession-policy-commitment.v1";

const POSSESSION_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-attestation-possession-policy.digest.v1\0";

/// Return a canonical, domain-separated commitment to every current field in
/// `AttestationPossessionPolicy`.
///
/// The upstream policy's `policy_id` is a stable identifier, not a commitment.
/// This digest therefore makes same-ID semantic drift visible to downstream
/// composition gates.
pub fn canonical_possession_policy_digest(
    policy: &AttestationPossessionPolicy,
) -> Option<String> {
    if !policy.validate() {
        return None;
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(POSSESSION_POLICY_DIGEST_DOMAIN);
    push_field(&mut hasher, POSSESSION_POLICY_COMMITMENT_SCHEMA_V1);
    push_field(&mut hasher, &policy.schema_version);
    push_field(&mut hasher, &policy.policy_id);
    push_field(
        &mut hasher,
        &policy.expected_platform_qualification_digest,
    );
    push_field(&mut hasher, &policy.expected_ak_binding_digest);
    push_field(&mut hasher, &policy.expected_pcr_selection_digest);
    push_field(&mut hasher, &policy.expected_verifier_ref);
    push_field(&mut hasher, &policy.expected_verification_tool_digest);
    hasher.update(&policy.max_challenge_lifetime_ms.to_le_bytes());
    hasher.update(&policy.max_quote_to_verification_ms.to_le_bytes());
    hasher.update(&[u8::from(policy.require_fixed_tpm)]);
    hasher.update(&[u8::from(policy.require_restricted_signing)]);

    let mut evidence_refs = policy.evidence_refs.clone();
    evidence_refs.sort();
    hasher.update(&(evidence_refs.len() as u64).to_le_bytes());
    for reference in evidence_refs {
        push_field(&mut hasher, &reference);
    }

    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

pub const fn grants_physical_authority() -> bool {
    false
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> AttestationPossessionPolicy {
        AttestationPossessionPolicy {
            schema_version: "1".into(),
            policy_id: "policy:attestation:1".into(),
            expected_platform_qualification_digest: digest("platform"),
            expected_ak_binding_digest: digest("ak"),
            expected_pcr_selection_digest: digest("selection"),
            expected_verifier_ref: "verifier:remote-1".into(),
            expected_verification_tool_digest: digest("checkquote"),
            max_challenge_lifetime_ms: 2_000,
            max_quote_to_verification_ms: 500,
            require_fixed_tpm: true,
            require_restricted_signing: true,
            evidence_refs: vec!["review:policy".into(), "audit:policy".into()],
        }
    }

    #[test]
    fn equal_semantics_have_equal_commitments() {
        let left = policy();
        let right = policy();
        assert_eq!(
            canonical_possession_policy_digest(&left),
            canonical_possession_policy_digest(&right)
        );
    }

    #[test]
    fn evidence_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(
            canonical_possession_policy_digest(&left),
            canonical_possession_policy_digest(&right)
        );
    }

    #[test]
    fn same_id_semantic_drift_changes_commitment() {
        let base = policy();
        let baseline = canonical_possession_policy_digest(&base).unwrap();

        let mut variants = Vec::new();

        let mut changed = base.clone();
        changed.schema_version = "2".into();
        variants.push(changed);

        let mut changed = base.clone();
        changed.expected_platform_qualification_digest = digest("other-platform");
        variants.push(changed);

        let mut changed = base.clone();
        changed.expected_ak_binding_digest = digest("other-ak");
        variants.push(changed);

        let mut changed = base.clone();
        changed.expected_pcr_selection_digest = digest("other-selection");
        variants.push(changed);

        let mut changed = base.clone();
        changed.expected_verifier_ref = "verifier:remote-2".into();
        variants.push(changed);

        let mut changed = base.clone();
        changed.expected_verification_tool_digest = digest("other-tool");
        variants.push(changed);

        let mut changed = base.clone();
        changed.max_challenge_lifetime_ms += 1;
        variants.push(changed);

        let mut changed = base.clone();
        changed.max_quote_to_verification_ms += 1;
        variants.push(changed);

        let mut changed = base.clone();
        changed.require_fixed_tpm = false;
        variants.push(changed);

        let mut changed = base.clone();
        changed.require_restricted_signing = false;
        variants.push(changed);

        let mut changed = base.clone();
        changed.evidence_refs.push("review:additional".into());
        variants.push(changed);

        for changed in variants {
            assert_eq!(changed.policy_id, base.policy_id);
            assert_ne!(
                canonical_possession_policy_digest(&changed).unwrap(),
                baseline
            );
        }
    }

    #[test]
    fn policy_id_itself_is_committed() {
        let base = policy();
        let mut changed = base.clone();
        changed.policy_id = "policy:attestation:2".into();
        assert_ne!(
            canonical_possession_policy_digest(&base),
            canonical_possession_policy_digest(&changed)
        );
    }

    #[test]
    fn invalid_upstream_policy_has_no_commitment() {
        let mut invalid = policy();
        invalid.evidence_refs.clear();
        assert!(!invalid.validate());
        assert_eq!(canonical_possession_policy_digest(&invalid), None);
    }

    #[test]
    fn policy_commitment_never_grants_physical_authority() {
        assert!(!grants_physical_authority());
    }
}
