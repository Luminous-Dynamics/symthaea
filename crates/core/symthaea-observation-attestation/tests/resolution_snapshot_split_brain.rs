use std::cell::Cell;

use ed25519_dalek::SigningKey;
use symthaea_observation_attestation::{
    ResolvedVerificationMethod, ResolvedVerificationMethodSnapshot,
    VerificationMethodResolutionError, VerificationMethodResolver, VerificationMethodStatus,
};

/// Models a resolver whose state can change between independent observations.
///
/// The override is intentionally the only authoritative read. The separate
/// snapshot accessor represents a later state and must not be consulted by the
/// paired-resolution operation.
struct SplitBrainTestResolver {
    signing_key: SigningKey,
    snapshot_reads: Cell<u32>,
}

impl SplitBrainTestResolver {
    fn new() -> Self {
        Self {
            signing_key: SigningKey::from_bytes(&[9u8; 32]),
            snapshot_reads: Cell::new(0),
        }
    }

    fn resolved(&self) -> ResolvedVerificationMethod {
        ResolvedVerificationMethod {
            verification_method: "did:example:split-brain#key-1".into(),
            verifying_key: self.signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["assertionMethod".into()],
        }
    }
}

impl VerificationMethodResolver for SplitBrainTestResolver {
    fn resolve(
        &self,
        _verification_method: &str,
    ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
        Ok(self.resolved())
    }

    fn resolve_with_snapshot(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
        let resolved = self.resolve(verification_method)?;
        Ok(ResolvedVerificationMethodSnapshot {
            resolved,
            snapshot_fingerprint: Some("same-authoritative-read".into()),
        })
    }

    fn snapshot_fingerprint_for(&self, _verification_method: &str) -> Option<String> {
        self.snapshot_reads.set(self.snapshot_reads.get() + 1);
        Some("later-mutated-state".into())
    }
}

#[test]
fn paired_resolution_does_not_recombine_independent_snapshot_observations() {
    let resolver = SplitBrainTestResolver::new();

    let paired = resolver
        .resolve_with_snapshot("did:example:split-brain#key-1")
        .expect("paired resolution should succeed");

    assert_eq!(
        paired.snapshot_fingerprint.as_deref(),
        Some("same-authoritative-read")
    );
    assert_eq!(resolver.snapshot_reads.get(), 0);
}

#[test]
fn compatibility_snapshot_accessor_can_remain_a_distinct_later_observation() {
    let resolver = SplitBrainTestResolver::new();

    let later_snapshot = resolver.snapshot_fingerprint_for("did:example:split-brain#key-1");

    assert_eq!(later_snapshot.as_deref(), Some("later-mutated-state"));
    assert_eq!(resolver.snapshot_reads.get(), 1);
}


/// Exercises the trait's compatibility default rather than an atomic resolver override.
/// A resolver that implements only the legacy accessors must not have a later snapshot
/// silently paired with the earlier resolution result.
struct DefaultSplitBrainTestResolver {
    signing_key: SigningKey,
    snapshot_reads: Cell<u32>,
}

impl DefaultSplitBrainTestResolver {
    fn new() -> Self {
        Self {
            signing_key: SigningKey::from_bytes(&[11u8; 32]),
            snapshot_reads: Cell::new(0),
        }
    }
}

impl VerificationMethodResolver for DefaultSplitBrainTestResolver {
    fn resolve(
        &self,
        _verification_method: &str,
    ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
        Ok(ResolvedVerificationMethod {
            verification_method: "did:example:default-split-brain#key-1".into(),
            verifying_key: self.signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["assertionMethod".into()],
        })
    }

    fn snapshot_fingerprint_for(&self, _verification_method: &str) -> Option<String> {
        self.snapshot_reads.set(self.snapshot_reads.get() + 1);
        Some("later-state".into())
    }
}

#[test]
fn compatibility_default_returns_no_paired_snapshot() {
    let resolver = DefaultSplitBrainTestResolver::new();

    let paired = resolver
        .resolve_with_snapshot("did:example:default-split-brain#key-1")
        .expect("compatibility resolution should succeed");

    assert_eq!(paired.snapshot_fingerprint, None);
    assert_eq!(resolver.snapshot_reads.get(), 0);
}
