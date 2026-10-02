use ed25519_dalek::SigningKey;
use symthaea_observation_attestation::{
    ResolvedVerificationMethod, ResolvedVerificationMethodSnapshot,
    VerificationMethodResolutionError, VerificationMethodResolver, VerificationMethodStatus,
};

/// A resolver representing an authoritative mutable state.
///
/// The important property under test is that an implementation can override
/// resolve_with_snapshot() and return both observations from one authoritative
/// view, rather than pairing a later snapshot read with an earlier resolution.
struct AtomicTestResolver {
    signing_key: SigningKey,
}

impl AtomicTestResolver {
    fn new() -> Self {
        Self {
            signing_key: SigningKey::from_bytes(&[7u8; 32]),
        }
    }

    fn resolved(&self) -> ResolvedVerificationMethod {
        ResolvedVerificationMethod {
            verification_method: "did:example:resolver#key-1".into(),
            verifying_key: self.signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["assertionMethod".into()],
        }
    }
}

impl VerificationMethodResolver for AtomicTestResolver {
    fn resolve(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
        if verification_method != "did:example:resolver#key-1" {
            return Err(VerificationMethodResolutionError::NotFound);
        }
        Ok(self.resolved())
    }

    fn resolve_with_snapshot(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
        let resolved = self.resolve(verification_method)?;
        // A real remote/mutable implementation would obtain this durable
        // snapshot identifier from the same authoritative read as resolved.
        Ok(ResolvedVerificationMethodSnapshot {
            resolved,
            snapshot_fingerprint: Some("authoritative-snapshot-42".into()),
        })
    }

    fn snapshot_fingerprint_for(&self, verification_method: &str) -> Option<String> {
        (verification_method == "did:example:resolver#key-1")
            .then(|| "authoritative-snapshot-42".into())
    }
}

#[test]
fn resolver_override_binds_resolution_and_snapshot_to_one_authoritative_view() {
    let resolver = AtomicTestResolver::new();

    let paired = resolver
        .resolve_with_snapshot("did:example:resolver#key-1")
        .expect("resolver should return its authoritative view");

    assert_eq!(
        paired.resolved.verification_method,
        "did:example:resolver#key-1"
    );
    assert_eq!(
        paired.snapshot_fingerprint.as_deref(),
        Some("authoritative-snapshot-42")
    );
    assert_eq!(
        resolver.snapshot_fingerprint_for("did:example:resolver#key-1"),
        paired.snapshot_fingerprint
    );
}
