// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First bounded consumer of Symthaea's algorithm-discovery contracts.
//!
//! The laboratory targets exact Hamming distance over the existing 16,384-bit `BinaryHV`.
//! It deliberately discovers/evaluates no cryptographic primitive and has no repository or
//! production authority.

use symthaea_algorithms::discovery::{
    CandidateArtifact, CandidateArtifactKind, CandidateProposal, DiscoveryError, DiscoveryPolicy,
    DiscoveryRun, SearchBudget,
};
use symthaea_algorithms::evaluation::{
    CorrectnessVerdict, EvaluationContext, EvaluationError, EvaluationReceipt, ObjectiveDirection,
    ObjectiveMeasurement,
};
use symthaea_algorithms::{
    AlgorithmLineage, AlgorithmProvenance, AlgorithmRecord, ContentId, DeterminismRequirement,
    DiscoveryRisk, ImplementationRecord, ProblemSpec, RegistryError, SemanticGuarantee,
};
use symthaea_core::hdc::binary_hv::BinaryHV;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum HdcLabError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
    #[error("candidate failed exact correctness and cannot produce performance evidence")]
    CorrectnessFailed,
    #[error("implementation does not belong to the HDC Hamming problem")]
    ImplementationProblemMismatch,
}

/// Small, auditable initial search space. Later generators can emit additional implementations
/// through the same registry protocol without changing the correctness oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HammingCandidate {
    BytePopcount,
    U64Popcount,
    U64Unrolled4,
    NativeSimd,
}

impl HammingCandidate {
    pub const ALL: [Self; 4] = [
        Self::BytePopcount,
        Self::U64Popcount,
        Self::U64Unrolled4,
        Self::NativeSimd,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::BytePopcount => "byte-popcount",
            Self::U64Popcount => "u64-popcount",
            Self::U64Unrolled4 => "u64-unrolled-4",
            Self::NativeSimd => "native-simd",
        }
    }
}

pub fn hamming_problem() -> Result<ProblemSpec, RegistryError> {
    ProblemSpec::new(
        "binary-hdc-hamming-distance-16384",
        "Return the exact number of differing bits between two 16,384-bit BinaryHV values.",
        SemanticGuarantee::Exact,
        DeterminismRequirement::Required,
        vec![
            "distance is in 0..=16384".into(),
            "distance(a, a) == 0".into(),
            "distance(a, b) == distance(b, a)".into(),
            "distance(zero, ones) == 16384".into(),
        ],
        DiscoveryRisk::Ordinary,
    )
}

pub fn hamming_algorithm(problem: &ProblemSpec) -> Result<AlgorithmRecord, RegistryError> {
    AlgorithmRecord::new(
        problem.id.clone(),
        "xor-popcount",
        "XOR the two bit vectors and exactly count set bits in the difference.",
        AlgorithmProvenance::HumanAuthored,
    )
}

/// Deliberately simple bit-by-bit oracle. It is independent from the production SIMD reduction
/// and from the word-popcount candidates, making it useful as a correctness reference despite
/// being intentionally slow.
pub fn oracle_distance(a: &BinaryHV, b: &BinaryHV) -> u32 {
    let mut total = 0u32;
    for (&left, &right) in a.0.iter().zip(b.0.iter()) {
        let difference = left ^ right;
        for bit in 0..8 {
            total += u32::from((difference >> bit) & 1);
        }
    }
    total
}

pub fn candidate_distance(candidate: HammingCandidate, a: &BinaryHV, b: &BinaryHV) -> u32 {
    match candidate {
        HammingCandidate::BytePopcount => byte_popcount(a, b),
        HammingCandidate::U64Popcount => u64_popcount(a, b),
        HammingCandidate::U64Unrolled4 => u64_unrolled_4(a, b),
        HammingCandidate::NativeSimd => a.hamming_distance(b),
    }
}

fn byte_popcount(a: &BinaryHV, b: &BinaryHV) -> u32 {
    a.0.iter()
        .zip(b.0.iter())
        .map(|(&left, &right)| (left ^ right).count_ones())
        .sum()
}

fn u64_popcount(a: &BinaryHV, b: &BinaryHV) -> u32 {
    a.0.chunks_exact(8)
        .zip(b.0.chunks_exact(8))
        .map(|(left, right)| {
            let mut left_word = [0u8; 8];
            let mut right_word = [0u8; 8];
            left_word.copy_from_slice(left);
            right_word.copy_from_slice(right);
            (u64::from_le_bytes(left_word) ^ u64::from_le_bytes(right_word)).count_ones()
        })
        .sum()
}

fn u64_unrolled_4(a: &BinaryHV, b: &BinaryHV) -> u32 {
    let mut total = 0u32;
    for base in (0..BinaryHV::BYTES).step_by(32) {
        for offset in [0usize, 8, 16, 24] {
            let start = base + offset;
            let end = start + 8;
            let mut left_word = [0u8; 8];
            let mut right_word = [0u8; 8];
            left_word.copy_from_slice(&a.0[start..end]);
            right_word.copy_from_slice(&b.0[start..end]);
            total += (u64::from_le_bytes(left_word) ^ u64::from_le_bytes(right_word)).count_ones();
        }
    }
    total
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HammingMismatch {
    pub case_label: String,
    pub expected: u32,
    pub observed: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HammingCorrectnessEvidence {
    pub id: ContentId,
    pub candidate: HammingCandidate,
    pub seeds: Vec<u64>,
    pub cases_checked: u64,
    pub passed: bool,
    pub first_mismatch: Option<HammingMismatch>,
}

impl HammingCorrectnessEvidence {
    pub fn verdict(&self) -> CorrectnessVerdict {
        if self.passed {
            CorrectnessVerdict::Passed
        } else {
            CorrectnessVerdict::Failed
        }
    }
}

pub fn verify_candidate(
    candidate: HammingCandidate,
    seeds: &[u64],
) -> HammingCorrectnessEvidence {
    let mut canonical_seeds = seeds.to_vec();
    canonical_seeds.sort_unstable();
    canonical_seeds.dedup();

    let mut cases_checked = 0u64;
    let mut first_mismatch = None;

    let edge_cases = [
        ("zero-zero", BinaryHV::zero(), BinaryHV::zero()),
        ("ones-ones", BinaryHV::ones(), BinaryHV::ones()),
        ("zero-ones", BinaryHV::zero(), BinaryHV::ones()),
        ("ones-zero", BinaryHV::ones(), BinaryHV::zero()),
    ];

    for (label, left, right) in edge_cases {
        cases_checked += 1;
        let expected = oracle_distance(&left, &right);
        let observed = candidate_distance(candidate, &left, &right);
        if expected != observed && first_mismatch.is_none() {
            first_mismatch = Some(HammingMismatch {
                case_label: label.into(),
                expected,
                observed,
            });
        }
    }

    for &seed in &canonical_seeds {
        let left = BinaryHV::random(seed);
        let right = BinaryHV::random(seed ^ 0x9e37_79b9_7f4a_7c15);
        cases_checked += 2;
        check_pair(
            candidate,
            &format!("seed-{seed}-forward"),
            &left,
            &right,
            &mut first_mismatch,
        );
        check_pair(
            candidate,
            &format!("seed-{seed}-reverse"),
            &right,
            &left,
            &mut first_mismatch,
        );
    }

    let passed = first_mismatch.is_none();
    let candidate_name = candidate.name();
    let cases = cases_checked.to_be_bytes();
    let verdict = if passed { b"pass".as_slice() } else { b"fail".as_slice() };
    let mut owned = vec![
        candidate_name.as_bytes().to_vec(),
        cases.as_slice().to_vec(),
        verdict.to_vec(),
    ];
    owned.extend(canonical_seeds.iter().map(|seed| seed.to_be_bytes().to_vec()));
    if let Some(mismatch) = &first_mismatch {
        owned.push(mismatch.case_label.as_bytes().to_vec());
        owned.push(mismatch.expected.to_be_bytes().to_vec());
        owned.push(mismatch.observed.to_be_bytes().to_vec());
    }
    let id = ContentId::derive(
        "symthaea.hdc-hamming-correctness.v1",
        owned.iter().map(Vec::as_slice),
    );

    HammingCorrectnessEvidence {
        id,
        candidate,
        seeds: canonical_seeds,
        cases_checked,
        passed,
        first_mismatch,
    }
}

fn check_pair(
    candidate: HammingCandidate,
    label: &str,
    left: &BinaryHV,
    right: &BinaryHV,
    first_mismatch: &mut Option<HammingMismatch>,
) {
    let expected = oracle_distance(left, right);
    let observed = candidate_distance(candidate, left, right);
    if expected != observed && first_mismatch.is_none() {
        *first_mismatch = Some(HammingMismatch {
            case_label: label.into(),
            expected,
            observed,
        });
    }
}

/// Build an exact implementation record. The caller supplies a content identity for the exact
/// source artifact being evaluated; the enum name alone is deliberately not treated as source
/// identity.
pub fn implementation_record(
    candidate: HammingCandidate,
    source_artifact_id: ContentId,
) -> Result<ImplementationRecord, HdcLabError> {
    let problem = hamming_problem()?;
    let algorithm = hamming_algorithm(&problem)?;
    Ok(ImplementationRecord::new(
        problem.id,
        algorithm.id,
        format!("symthaea-algorithm-lab::{}", candidate.name()),
        source_artifact_id,
        None,
    )?)
}

pub fn pilot_run(
    baseline_revision: impl Into<String>,
    seed: u64,
) -> Result<DiscoveryRun, HdcLabError> {
    let problem = hamming_problem()?;
    Ok(DiscoveryRun::new(
        &problem,
        DiscoveryPolicy::default(),
        ContentId::derive("symthaea.generator.v1", [b"hdc-hamming-enumeration".as_slice()]),
        baseline_revision,
        SearchBudget::new(HammingCandidate::ALL.len() as u64, 1, HammingCandidate::ALL.len() as u64)?,
        seed,
    )?)
}

pub fn proposal_for(
    run: &DiscoveryRun,
    candidate: HammingCandidate,
    source_artifact_id: ContentId,
) -> Result<CandidateProposal, HdcLabError> {
    let implementation = implementation_record(candidate, source_artifact_id.clone())?;
    let lineage = AlgorithmLineage::new(implementation.id.clone(), vec![], vec![])?;
    let artifact = CandidateArtifact::new(
        CandidateArtifactKind::SourceTree,
        source_artifact_id,
        format!("source://symthaea-algorithm-lab/{}", candidate.name()),
    )?;
    Ok(CandidateProposal::new(run, implementation, lineage, artifact)?)
}

/// Turn an externally measured latency into a canonical evaluation receipt only after the exact
/// correctness suite passed. Criterion remains the measurement mechanism; this function does not
/// fabricate timings or infer them from unit-test duration.
#[allow(clippy::too_many_arguments)]
pub fn latency_receipt(
    implementation: &ImplementationRecord,
    correctness: &HammingCorrectnessEvidence,
    source_revision: impl Into<String>,
    toolchain_profile: impl Into<String>,
    target_profile: impl Into<String>,
    latency_ns_per_op: f64,
) -> Result<EvaluationReceipt, HdcLabError> {
    let problem = hamming_problem()?;
    if implementation.problem_id != problem.id {
        return Err(HdcLabError::ImplementationProblemMismatch);
    }
    implementation.validate()?;
    if !correctness.passed {
        return Err(HdcLabError::CorrectnessFailed);
    }

    let target_profile = target_profile.into();
    let context = EvaluationContext::new(
        ContentId::derive("symthaea.evaluator.v1", [b"criterion-hdc-hamming".as_slice()]),
        ContentId::derive("symthaea.oracle.v1", [b"bitwise-hamming-oracle".as_slice()]),
        correctness.id.clone(),
        ContentId::derive("symthaea.environment.v1", [target_profile.as_bytes()]),
        source_revision,
        toolchain_profile,
        target_profile,
        correctness.seeds.clone(),
    )?;

    Ok(EvaluationReceipt::new(
        problem.id,
        implementation.id.clone(),
        context,
        correctness.verdict(),
        correctness.id.clone(),
        vec![ObjectiveMeasurement::new(
            "latency",
            ObjectiveDirection::Minimize,
            latency_ns_per_op,
            "ns/op",
        )?],
        None,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_initial_candidates_match_independent_oracle() {
        let seeds: Vec<u64> = (0..128).collect();
        for candidate in HammingCandidate::ALL {
            let evidence = verify_candidate(candidate, &seeds);
            assert!(
                evidence.passed,
                "{} failed: {:?}",
                candidate.name(),
                evidence.first_mismatch
            );
            assert_eq!(evidence.cases_checked, 4 + seeds.len() as u64 * 2);
        }
    }

    #[test]
    fn production_native_simd_matches_extreme_case() {
        assert_eq!(
            candidate_distance(HammingCandidate::NativeSimd, &BinaryHV::zero(), &BinaryHV::ones()),
            BinaryHV::DIM as u32
        );
    }

    #[test]
    fn pilot_problem_is_exact_deterministic_and_ordinary_risk() {
        let problem = hamming_problem().unwrap();
        assert_eq!(problem.guarantee, SemanticGuarantee::Exact);
        assert_eq!(problem.determinism, DeterminismRequirement::Required);
        assert_eq!(problem.risk, DiscoveryRisk::Ordinary);
    }

    #[test]
    fn proposal_binds_exact_source_artifact() {
        let run = pilot_run("abc123", 42).unwrap();
        let artifact = ContentId::derive("source-blob", [b"u64-popcount-source".as_slice()]);
        let proposal = proposal_for(&run, HammingCandidate::U64Popcount, artifact.clone()).unwrap();
        assert_eq!(proposal.artifact.content_id, artifact);
        assert_eq!(proposal.implementation.artifact_id, artifact);
        assert_eq!(proposal.run_id, run.id);
    }

    #[test]
    fn latency_receipt_requires_passed_correctness() {
        let artifact = ContentId::derive("source-blob", [b"candidate-source".as_slice()]);
        let implementation = implementation_record(HammingCandidate::BytePopcount, artifact).unwrap();
        let mut evidence = verify_candidate(HammingCandidate::BytePopcount, &[1, 2, 3]);
        evidence.passed = false;
        assert_eq!(
            latency_receipt(
                &implementation,
                &evidence,
                "abc123",
                "rust-1.96.0",
                "x86_64-unknown-linux-gnu",
                10.0,
            )
            .unwrap_err(),
            HdcLabError::CorrectnessFailed
        );
    }
}
