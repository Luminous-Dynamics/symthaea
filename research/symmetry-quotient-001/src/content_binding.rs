// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Canonical content binding for exact finite-group quotient certificates.
//!
//! The encodings are versioned, domain-separated binary records. All lengths
//! and integers use fixed-width little-endian fields; strings are UTF-8 with
//! u64 byte-length prefixes. No JSON whitespace or map iteration order enters
//! the identity. SHA-256 comes from the maintained RustCrypto sha2 crate.
//!
//! Content identity is not mathematical proof or publisher authentication.
//! The bound verifier always replays the independent quotient verifier.

use crate::{
    FiniteActionV1, PruningAuthorityV1, QualificationBudgetV1, QuotientCertificateV1,
    QuotientErrorV1, SearchProblemV1, VerificationReceiptV1, verify_certificate,
};
use sha2::{Digest, Sha256};
use std::fmt;

const BINDING_SCHEMA_VERSION_V1: u32 = 1;
const PROBLEM_DOMAIN_V1: &[u8] = b"LUMINOUS-SYMM-QUOTIENT\0PROBLEM\0V1";
const ACTION_DOMAIN_V1: &[u8] = b"LUMINOUS-SYMM-QUOTIENT\0ACTION\0V1";
const CERTIFICATE_DOMAIN_V1: &[u8] = b"LUMINOUS-SYMM-QUOTIENT\0CERTIFICATE\0V1";
const PACKAGE_DOMAIN_V1: &[u8] = b"LUMINOUS-SYMM-QUOTIENT\0BOUND-PACKAGE\0V1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentBindingBudgetV1 {
    /// Maximum size of any one canonical encoding, in bytes.
    pub maximum_encoded_bytes: usize,
}

impl Default for ContentBindingBudgetV1 {
    fn default() -> Self {
        Self {
            maximum_encoded_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBindingErrorV1 {
    Quotient(QuotientErrorV1),
    EncodingBudgetExceeded {
        required_bytes: usize,
        maximum_bytes: usize,
    },
    EncodingLengthOverflow,
    UnsupportedSchemaVersion {
        expected: u32,
        received: u32,
    },
    ProblemDigestMismatch,
    ActionDigestMismatch,
    CertificateDigestMismatch,
    ManifestDigestMismatch,
}

impl fmt::Display for ContentBindingErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ContentBindingErrorV1 {}

impl From<QuotientErrorV1> for ContentBindingErrorV1 {
    fn from(value: QuotientErrorV1) -> Self {
        Self::Quotient(value)
    }
}

/// Binds problem, action and certificate digests to one qualification manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundQuotientPackageV1 {
    pub schema_version: u32,
    pub qualification_budget: QualificationBudgetV1,
    pub encoding_budget: ContentBindingBudgetV1,
    pub problem_sha256: [u8; 32],
    pub action_sha256: [u8; 32],
    pub certificate_sha256: [u8; 32],
    pub manifest_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonicalEncoderV1 {
    bytes: Vec<u8>,
    maximum_bytes: usize,
}

impl CanonicalEncoderV1 {
    fn new(maximum_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum_bytes,
        }
    }

    fn raw(&mut self, value: &[u8]) -> Result<(), ContentBindingErrorV1> {
        let required = self
            .bytes
            .len()
            .checked_add(value.len())
            .ok_or(ContentBindingErrorV1::EncodingLengthOverflow)?;
        if required > self.maximum_bytes {
            return Err(ContentBindingErrorV1::EncodingBudgetExceeded {
                required_bytes: required,
                maximum_bytes: self.maximum_bytes,
            });
        }
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    fn u32(&mut self, value: u32) -> Result<(), ContentBindingErrorV1> {
        self.raw(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), ContentBindingErrorV1> {
        self.raw(&value.to_le_bytes())
    }

    fn usize(&mut self, value: usize) -> Result<(), ContentBindingErrorV1> {
        let value =
            u64::try_from(value).map_err(|_| ContentBindingErrorV1::EncodingLengthOverflow)?;
        self.u64(value)
    }

    fn i64(&mut self, value: i64) -> Result<(), ContentBindingErrorV1> {
        self.raw(&value.to_le_bytes())
    }

    fn boolean(&mut self, value: bool) -> Result<(), ContentBindingErrorV1> {
        self.raw(&[u8::from(value)])
    }

    fn string(&mut self, value: &str) -> Result<(), ContentBindingErrorV1> {
        self.usize(value.len())?;
        self.raw(value.as_bytes())
    }

    fn digest(&mut self, value: &[u8; 32]) -> Result<(), ContentBindingErrorV1> {
        self.raw(value)
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

fn encode_problem(
    problem: &SearchProblemV1,
    budget: ContentBindingBudgetV1,
) -> Result<Vec<u8>, ContentBindingErrorV1> {
    let mut encoder = CanonicalEncoderV1::new(budget.maximum_encoded_bytes);
    encoder.raw(PROBLEM_DOMAIN_V1)?;
    encoder.string(&problem.problem_id)?;
    encoder.usize(problem.candidate_ids.len())?;
    for candidate_id in &problem.candidate_ids {
        encoder.string(candidate_id)?;
    }
    encoder.usize(problem.semantics.len())?;
    for semantics in &problem.semantics {
        encoder.boolean(semantics.in_search_domain)?;
        encoder.boolean(semantics.admissible)?;
        encoder.boolean(semantics.feasible)?;
        encoder.i64(semantics.objective_cost)?;
        encoder.usize(semantics.constraint_signature.len())?;
        for value in &semantics.constraint_signature {
            encoder.i64(*value)?;
        }
        encoder.string(&semantics.boundary_condition_id)?;
        encoder.string(&semantics.output_equivalence_id)?;
    }
    Ok(encoder.finish())
}

fn encode_action(
    action: &FiniteActionV1,
    budget: ContentBindingBudgetV1,
) -> Result<Vec<u8>, ContentBindingErrorV1> {
    let mut encoder = CanonicalEncoderV1::new(budget.maximum_encoded_bytes);
    encoder.raw(ACTION_DOMAIN_V1)?;
    encoder.string(&action.action_id)?;
    encoder.string(&action.group.group_id)?;
    encoder.usize(action.group.identity)?;
    encoder.usize(action.group.multiplication.len())?;
    for row in &action.group.multiplication {
        encoder.usize(row.len())?;
        for product in row {
            encoder.usize(*product)?;
        }
    }
    encoder.usize(action.permutations.len())?;
    for permutation in &action.permutations {
        encoder.usize(permutation.len())?;
        for image in permutation {
            encoder.usize(*image)?;
        }
    }
    Ok(encoder.finish())
}

fn encode_certificate(
    certificate: &QuotientCertificateV1,
    budget: ContentBindingBudgetV1,
) -> Result<Vec<u8>, ContentBindingErrorV1> {
    let mut encoder = CanonicalEncoderV1::new(budget.maximum_encoded_bytes);
    encoder.raw(CERTIFICATE_DOMAIN_V1)?;
    encoder.u32(certificate.schema_version)?;
    encoder.string(&certificate.problem_id)?;
    encoder.string(&certificate.action_id)?;
    encoder.string(&certificate.group_id)?;
    encoder.usize(certificate.group_order)?;
    encoder.usize(certificate.candidate_count)?;
    match certificate.pruning_authority {
        PruningAuthorityV1::ExactFiniteGroupAndExactTaskSemanticPreservation => {
            encoder.u32(1)?;
        }
    }
    encoder.usize(certificate.orbits.len())?;
    for orbit in &certificate.orbits {
        encoder.usize(orbit.representative_index)?;
        encoder.string(&orbit.representative_id)?;
        encoder.usize(orbit.members.len())?;
        for member in &orbit.members {
            encoder.usize(member.candidate_index)?;
            encoder.string(&member.candidate_id)?;
            encoder.usize(member.group_element)?;
            encoder.usize(member.inverse_group_element)?;
        }
        encoder.usize(orbit.stabilizer_group_elements.len())?;
        for element in &orbit.stabilizer_group_elements {
            encoder.usize(*element)?;
        }
    }
    encoder.usize(certificate.evidence.group_axiom_checks)?;
    encoder.usize(certificate.evidence.action_law_checks)?;
    encoder.usize(certificate.evidence.task_semantic_checks)?;
    encoder.usize(certificate.evidence.orbit_member_witnesses)?;
    encoder.usize(certificate.evidence.stabilizer_membership_checks)?;
    encoder.usize(certificate.evidence.stabilizer_subgroup_closure_checks)?;
    encoder.usize(certificate.evidence.stabilizer_witnesses)?;
    encoder.usize(certificate.evidence.orbit_stabilizer_checks)?;
    encoder.boolean(certificate.evidence.complete_domain_cover)?;
    Ok(encoder.finish())
}

fn encode_manifest(
    package: &BoundQuotientPackageV1,
    budget: ContentBindingBudgetV1,
) -> Result<Vec<u8>, ContentBindingErrorV1> {
    let mut encoder = CanonicalEncoderV1::new(budget.maximum_encoded_bytes);
    encoder.raw(PACKAGE_DOMAIN_V1)?;
    encoder.u32(package.schema_version)?;
    encoder.string("SHA-256")?;
    encoder.string("luminous.symmetry_quotient.canonical_binary.v1")?;
    encoder.usize(package.qualification_budget.maximum_group_order)?;
    encoder.usize(package.qualification_budget.maximum_candidates)?;
    encoder.usize(package.qualification_budget.maximum_work_units)?;
    encoder.usize(package.encoding_budget.maximum_encoded_bytes)?;
    encoder.digest(&package.problem_sha256)?;
    encoder.digest(&package.action_sha256)?;
    encoder.digest(&package.certificate_sha256)?;
    Ok(encoder.finish())
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(bytes);
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    output
}

/// SHA-256 hex helper for deterministic logs and external receipts.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = sha256(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

/// Canonical bytes for the complete versioned task object.
pub fn canonical_problem_bytes_v1(
    problem: &SearchProblemV1,
    budget: ContentBindingBudgetV1,
) -> Result<Vec<u8>, ContentBindingErrorV1> {
    encode_problem(problem, budget)
}

/// Canonical bytes for the complete finite group/action object.
pub fn canonical_action_bytes_v1(
    action: &FiniteActionV1,
    budget: ContentBindingBudgetV1,
) -> Result<Vec<u8>, ContentBindingErrorV1> {
    encode_action(action, budget)
}

/// Canonical bytes for the complete quotient certificate.
pub fn canonical_certificate_bytes_v1(
    certificate: &QuotientCertificateV1,
    budget: ContentBindingBudgetV1,
) -> Result<Vec<u8>, ContentBindingErrorV1> {
    encode_certificate(certificate, budget)
}

/// Create content bindings only after the independent quotient verifier passes.
pub fn bind_quotient_certificate(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    certificate: &QuotientCertificateV1,
    qualification_budget: QualificationBudgetV1,
    encoding_budget: ContentBindingBudgetV1,
) -> Result<BoundQuotientPackageV1, ContentBindingErrorV1> {
    verify_certificate(problem, action, certificate, qualification_budget)?;

    let problem_sha256 = sha256(&encode_problem(problem, encoding_budget)?);
    let action_sha256 = sha256(&encode_action(action, encoding_budget)?);
    let certificate_sha256 = sha256(&encode_certificate(certificate, encoding_budget)?);

    let mut package = BoundQuotientPackageV1 {
        schema_version: BINDING_SCHEMA_VERSION_V1,
        qualification_budget,
        encoding_budget,
        problem_sha256,
        action_sha256,
        certificate_sha256,
        manifest_sha256: [0_u8; 32],
    };
    package.manifest_sha256 = sha256(&encode_manifest(&package, encoding_budget)?);
    Ok(package)
}

/// Verify every content digest and then replay the independent quotient proof.
pub fn verify_bound_certificate(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    certificate: &QuotientCertificateV1,
    package: &BoundQuotientPackageV1,
) -> Result<VerificationReceiptV1, ContentBindingErrorV1> {
    if package.schema_version != BINDING_SCHEMA_VERSION_V1 {
        return Err(ContentBindingErrorV1::UnsupportedSchemaVersion {
            expected: BINDING_SCHEMA_VERSION_V1,
            received: package.schema_version,
        });
    }

    if sha256(&encode_problem(problem, package.encoding_budget)?) != package.problem_sha256 {
        return Err(ContentBindingErrorV1::ProblemDigestMismatch);
    }
    if sha256(&encode_action(action, package.encoding_budget)?) != package.action_sha256 {
        return Err(ContentBindingErrorV1::ActionDigestMismatch);
    }
    if sha256(&encode_certificate(certificate, package.encoding_budget)?)
        != package.certificate_sha256
    {
        return Err(ContentBindingErrorV1::CertificateDigestMismatch);
    }
    if sha256(&encode_manifest(package, package.encoding_budget)?) != package.manifest_sha256 {
        return Err(ContentBindingErrorV1::ManifestDigestMismatch);
    }

    Ok(verify_certificate(
        problem,
        action,
        certificate,
        package.qualification_budget,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CandidateSemanticsV1, CompileOutcomeV1, FiniteGroupV1, compile_quotient};

    fn semantics(cost: i64, constraint: i64, boundary: &str, output: &str) -> CandidateSemanticsV1 {
        CandidateSemanticsV1 {
            in_search_domain: true,
            admissible: true,
            feasible: true,
            objective_cost: cost,
            constraint_signature: vec![constraint],
            boundary_condition_id: boundary.to_owned(),
            output_equivalence_id: output.to_owned(),
        }
    }

    fn fixture() -> (SearchProblemV1, FiniteActionV1) {
        let problem = SearchProblemV1 {
            problem_id: "bound-c2-fixture-v1".to_owned(),
            candidate_ids: vec![
                "A0".to_owned(),
                "A1".to_owned(),
                "B0".to_owned(),
                "B1".to_owned(),
            ],
            semantics: vec![
                semantics(10, 2, "boundary-a", "output-a"),
                semantics(10, 2, "boundary-a", "output-a"),
                semantics(20, 3, "boundary-b", "output-b"),
                semantics(20, 3, "boundary-b", "output-b"),
            ],
        };
        let action = FiniteActionV1 {
            action_id: "bound-c2-action-v1".to_owned(),
            group: FiniteGroupV1 {
                group_id: "C2-v1".to_owned(),
                identity: 0,
                multiplication: vec![vec![0, 1], vec![1, 0]],
            },
            permutations: vec![vec![0, 1, 2, 3], vec![1, 0, 3, 2]],
        };
        (problem, action)
    }

    fn authorized_fixture() -> (
        SearchProblemV1,
        FiniteActionV1,
        QuotientCertificateV1,
        BoundQuotientPackageV1,
    ) {
        let (problem, action) = fixture();
        let qualification_budget = QualificationBudgetV1::default();
        let encoding_budget = ContentBindingBudgetV1::default();
        let outcome = compile_quotient(&problem, &action, qualification_budget).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = outcome else {
            panic!("expected authorized finite quotient");
        };
        let package = bind_quotient_certificate(
            &problem,
            &action,
            &certificate,
            qualification_budget,
            encoding_budget,
        )
        .unwrap();
        (problem, action, certificate, package)
    }

    #[test]
    fn sha256_known_empty_and_abc_vectors_match() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn canonical_encodings_are_deterministic_and_domain_separated() {
        let (problem, action) = fixture();
        let budget = ContentBindingBudgetV1::default();
        let first = canonical_problem_bytes_v1(&problem, budget).unwrap();
        let second = canonical_problem_bytes_v1(&problem, budget).unwrap();
        let action_bytes = canonical_action_bytes_v1(&action, budget).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, action_bytes);
        assert_ne!(sha256_hex(&first), sha256_hex(&action_bytes));
    }

    #[test]
    fn bound_package_verifies_after_digests_and_independent_replay() {
        let (problem, action, certificate, package) = authorized_fixture();
        let receipt = verify_bound_certificate(&problem, &action, &certificate, &package).unwrap();
        assert_eq!(receipt.problem_id, problem.problem_id);
        assert_eq!(receipt.candidate_count, 4);
    }

    #[test]
    fn changed_boundary_and_output_semantics_are_rejected_by_problem_digest() {
        let (_problem, action, certificate, package) = authorized_fixture();
        let mut changed_boundary = fixture().0;
        changed_boundary.semantics[0]
            .boundary_condition_id
            .push_str("-changed");
        assert_eq!(
            verify_bound_certificate(&changed_boundary, &action, &certificate, &package),
            Err(ContentBindingErrorV1::ProblemDigestMismatch)
        );

        let mut changed_output = fixture().0;
        changed_output.semantics[0]
            .output_equivalence_id
            .push_str("-changed");
        assert_eq!(
            verify_bound_certificate(&changed_output, &action, &certificate, &package),
            Err(ContentBindingErrorV1::ProblemDigestMismatch)
        );
    }

    #[test]
    fn canonical_certificate_encoding_is_deterministic() {
        let (_problem, _action, certificate, _package) = authorized_fixture();
        let budget = ContentBindingBudgetV1::default();
        let first = canonical_certificate_bytes_v1(&certificate, budget).unwrap();
        let second = canonical_certificate_bytes_v1(&certificate, budget).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn changed_objective_is_rejected_by_problem_digest() {
        let (_problem, action, certificate, package) = authorized_fixture();
        let mut changed = fixture().0;
        changed.semantics[0].objective_cost += 1;
        assert_eq!(
            verify_bound_certificate(&changed, &action, &certificate, &package),
            Err(ContentBindingErrorV1::ProblemDigestMismatch)
        );
    }

    #[test]
    fn changed_group_action_is_rejected_by_action_digest() {
        let (problem, _action, certificate, package) = authorized_fixture();
        let mut changed = fixture().1;
        changed.action_id.push_str("-changed");
        assert_eq!(
            verify_bound_certificate(&problem, &changed, &certificate, &package),
            Err(ContentBindingErrorV1::ActionDigestMismatch)
        );
    }

    #[test]
    fn changed_orbit_witness_is_rejected_by_certificate_digest() {
        let (problem, action, mut certificate, package) = authorized_fixture();
        certificate.orbits[0].members[0].group_element = 1;
        assert_eq!(
            verify_bound_certificate(&problem, &action, &certificate, &package),
            Err(ContentBindingErrorV1::CertificateDigestMismatch)
        );
    }

    #[test]
    fn modified_manifest_and_schema_are_rejected() {
        let (problem, action, certificate, mut package) = authorized_fixture();
        package.qualification_budget.maximum_work_units -= 1;
        assert_eq!(
            verify_bound_certificate(&problem, &action, &certificate, &package),
            Err(ContentBindingErrorV1::ManifestDigestMismatch)
        );

        let (_problem, _action, certificate, mut package) = authorized_fixture();
        package.schema_version += 1;
        let (problem, action) = fixture();
        assert_eq!(
            verify_bound_certificate(&problem, &action, &certificate, &package),
            Err(ContentBindingErrorV1::UnsupportedSchemaVersion {
                expected: BINDING_SCHEMA_VERSION_V1,
                received: BINDING_SCHEMA_VERSION_V1 + 1,
            })
        );
    }

    #[test]
    fn canonical_encoding_budget_fails_before_unbounded_growth() {
        let (problem, _action) = fixture();
        assert!(matches!(
            canonical_problem_bytes_v1(
                &problem,
                ContentBindingBudgetV1 {
                    maximum_encoded_bytes: 16,
                }
            ),
            Err(ContentBindingErrorV1::EncodingBudgetExceeded { .. })
        ));
    }
}
