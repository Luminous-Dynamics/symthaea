#![forbid(unsafe_code)]

//! Fail-closed bridge from CMAQ-036 common-cause blocks to CMAQ-031 rank input.
//!
//! The bridge admits one attested score per effective calibration block. Raw
//! episode count is diagnostic only and never becomes the monitor's sample size.
//!
//! It validates correspondence and digest syntax, not signatures, signer
//! authority, canonical serialization, dependency completeness or physical
//! independence. Those duties remain with the upstream evidence authority.

use std::collections::{BTreeMap, HashSet};

use cmaq_031_anytime_calibration_monitor::{
    AnytimeCalibrationMonitor, CalibrationRankReference, CalibrationRankScore,
    DeploymentRankObservation, MonitorError, MonitorUpdate,
};
use cmaq_036_common_cause_closure::{
    ClosureManifest, EvidenceSplit, IndependentBlock, KeyReceiptBinding, SplitCount,
};

const SUPPORTED_CLOSURE_ALGORITHM: &str = "typed-common-cause-union-find-max-score-v0.1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureAuthorityBinding {
    pub closure_manifest_sha256: String,
    pub closure_authority_receipt_sha256: String,
    pub source_manifest_sha256: String,
    pub closure_algorithm: String,
    pub calibration_profile_sha256: String,
    pub calibration_manifest_sha256: String,
    pub scope_sha256: String,
    pub score_definition_sha256: String,
}

/// One evaluator attestation for one calibration closure block.
///
/// Arrays must use the canonical ordering emitted by CMAQ-036. The aggregate
/// score receipt must bind the block ID, all member IDs and score receipts,
/// every dependence-key receipt, the maximum dangerous score, score definition,
/// block manifest, and closure digest. The tie receipt binds the randomized key
/// to that same effective block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosedBlockScoreAttestation {
    pub closure_manifest_sha256: String,
    pub block_id: String,
    pub split: EvidenceSplit,
    pub member_unit_ids: Vec<String>,
    pub maximum_dangerous_score_units: u64,
    pub score_receipts_sha256: Vec<String>,
    pub dependence_key_receipts: Vec<KeyReceiptBinding>,
    pub score_definition_sha256: String,
    pub block_manifest_sha256: String,
    pub aggregate_score_receipt_sha256: String,
    pub tie_break_key: u64,
    pub tie_break_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureAdmissionDiagnostics {
    pub closure_manifest_sha256: String,
    pub source_manifest_sha256: String,
    pub closure_algorithm: String,
    pub raw_unit_count: usize,
    pub total_independent_block_count: usize,
    pub calibration_raw_unit_count: usize,
    pub calibration_effective_block_count: usize,
    pub pseudoreplication_factor_numerator: usize,
    pub pseudoreplication_factor_denominator: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureQualifiedRankReference {
    reference: CalibrationRankReference,
    diagnostics: ClosureAdmissionDiagnostics,
}

impl ClosureQualifiedRankReference {
    /// Read-only diagnostics; the reference itself remains owned by this wrapper.
    pub fn diagnostics(&self) -> &ClosureAdmissionDiagnostics {
        &self.diagnostics
    }

    /// Read-only view of the admitted effective-block calibration scores.
    pub fn calibration_scores(&self) -> &[CalibrationRankScore] {
        &self.reference.scores
    }

    /// Consume the closure-qualified reference directly into the CMAQ-031 monitor.
    ///
    /// The raw reference cannot be moved out of this wrapper, preventing callers
    /// from accidentally discarding the closure provenance during normal use.
    pub fn into_monitor(
        self,
        alpha_numerator: u32,
        alpha_denominator: u32,
    ) -> Result<ClosureQualifiedMonitor, MonitorError> {
        let monitor =
            AnytimeCalibrationMonitor::new(self.reference, alpha_numerator, alpha_denominator)?;
        Ok(ClosureQualifiedMonitor {
            monitor,
            diagnostics: self.diagnostics,
        })
    }
}

/// Runtime monitor that keeps closure provenance attached to CMAQ-031 state.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosureQualifiedMonitor {
    monitor: AnytimeCalibrationMonitor,
    diagnostics: ClosureAdmissionDiagnostics,
}

impl ClosureQualifiedMonitor {
    pub fn diagnostics(&self) -> &ClosureAdmissionDiagnostics {
        &self.diagnostics
    }

    pub fn observe(
        &mut self,
        observation: DeploymentRankObservation,
    ) -> Result<MonitorUpdate, MonitorError> {
        self.monitor.observe(observation)
    }

    pub fn is_quarantined(&self) -> bool {
        self.monitor.is_quarantined()
    }

    pub fn observations_seen(&self) -> u64 {
        self.monitor.observations_seen()
    }

    pub fn calibration_reference_count(&self) -> usize {
        self.monitor.calibration_reference_count()
    }

    pub fn log_mixture_wealth(&self) -> Result<f64, MonitorError> {
        self.monitor.log_mixture_wealth()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClosureBridgeError {
    InvalidSha256(&'static str),
    UnsupportedClosureAlgorithm,
    SourceManifestMismatch,
    ClosureBindingMismatch(&'static str),
    EmptyClosure,
    DuplicateBlockId,
    DuplicateMemberUnit,
    NonCanonicalBlockOrder,
    NonCanonicalMemberOrder,
    NonCanonicalScoreReceiptOrder,
    NonCanonicalDependenceReceiptOrder,
    InvalidBlockIdentity,
    InvalidBlockInventory,
    SplitCountMismatch,
    EffectiveCountMismatch,
    PseudoreplicationFactorMismatch,
    MissingCalibrationBlocks,
    AttestationInventoryMismatch,
    DuplicateAttestation,
    MissingAttestation,
    UnexpectedAttestation,
    WrongAttestationSplit,
    AttestationClosureMismatch,
    AttestationScoreDefinitionMismatch,
    AttestationMemberMismatch,
    AttestationScoreReceiptMismatch,
    AttestationDependenceReceiptMismatch,
    AttestationMaximumScoreMismatch,
    DuplicateTieBreakKey,
}

/// Builds a frozen CMAQ-031 rank reference from one attestation per effective
/// CMAQ-036 calibration block. The output score count is exactly the declared
/// effective block count; raw episodes never become separate statistical votes.
///
/// The caller must obtain the binding from an independent evidence authority
/// that authenticates the exact closure serialization, score/dependence receipts
/// and randomized tie records. Matching fields and digest syntax alone do not
/// establish these facts.
pub fn build_closure_qualified_rank_reference(
    closure: &ClosureManifest,
    binding: &ClosureAuthorityBinding,
    attestations: &[ClosedBlockScoreAttestation],
) -> Result<ClosureQualifiedRankReference, ClosureBridgeError> {
    validate_binding(closure, binding)?;
    validate_closure_structure(closure)?;

    let calibration_blocks: Vec<&IndependentBlock> = closure
        .blocks
        .iter()
        .filter(|block| block.split == EvidenceSplit::Calibration)
        .collect();

    if calibration_blocks.is_empty() {
        return Err(ClosureBridgeError::MissingCalibrationBlocks);
    }
    if attestations.len() != calibration_blocks.len() {
        return Err(ClosureBridgeError::AttestationInventoryMismatch);
    }

    let mut by_block = BTreeMap::new();
    for attestation in attestations {
        if attestation.block_id.trim().is_empty() {
            return Err(ClosureBridgeError::InvalidBlockIdentity);
        }
        if by_block
            .insert(attestation.block_id.as_str(), attestation)
            .is_some()
        {
            return Err(ClosureBridgeError::DuplicateAttestation);
        }
    }

    let mut tie_break_keys = HashSet::with_capacity(attestations.len());
    let mut scores = Vec::with_capacity(calibration_blocks.len());

    for block in calibration_blocks {
        let attestation = by_block
            .remove(block.block_id.as_str())
            .ok_or(ClosureBridgeError::MissingAttestation)?;
        validate_attestation(block, binding, attestation)?;

        if !tie_break_keys.insert(attestation.tie_break_key) {
            return Err(ClosureBridgeError::DuplicateTieBreakKey);
        }

        scores.push(CalibrationRankScore {
            independent_cluster_id: block.block_id.clone(),
            score_units: block.maximum_dangerous_score_units,
            tie_break_key: attestation.tie_break_key,
            block_manifest_sha256: attestation.block_manifest_sha256.clone(),
            aggregate_score_receipt_sha256: attestation.aggregate_score_receipt_sha256.clone(),
            tie_break_receipt_sha256: attestation.tie_break_receipt_sha256.clone(),
        });
    }

    if !by_block.is_empty() {
        return Err(ClosureBridgeError::UnexpectedAttestation);
    }
    if scores.len() != closure.calibration_effective_block_count {
        return Err(ClosureBridgeError::EffectiveCountMismatch);
    }

    Ok(ClosureQualifiedRankReference {
        reference: CalibrationRankReference {
            calibration_profile_sha256: binding.calibration_profile_sha256.clone(),
            calibration_manifest_sha256: binding.calibration_manifest_sha256.clone(),
            scope_sha256: binding.scope_sha256.clone(),
            scores,
        },
        diagnostics: ClosureAdmissionDiagnostics {
            closure_manifest_sha256: binding.closure_manifest_sha256.clone(),
            source_manifest_sha256: closure.source_manifest_sha256.clone(),
            closure_algorithm: closure.closure_algorithm.clone(),
            raw_unit_count: closure.raw_unit_count,
            total_independent_block_count: closure.total_independent_block_count,
            calibration_raw_unit_count: closure.calibration_raw_unit_count,
            calibration_effective_block_count: closure.calibration_effective_block_count,
            pseudoreplication_factor_numerator: closure
                .calibration_pseudoreplication_factor_numerator,
            pseudoreplication_factor_denominator: closure
                .calibration_pseudoreplication_factor_denominator,
        },
    })
}

fn validate_binding(
    closure: &ClosureManifest,
    binding: &ClosureAuthorityBinding,
) -> Result<(), ClosureBridgeError> {
    for (value, name) in [
        (&binding.closure_manifest_sha256, "closure_manifest_sha256"),
        (
            &binding.closure_authority_receipt_sha256,
            "closure_authority_receipt_sha256",
        ),
        (&binding.source_manifest_sha256, "source_manifest_sha256"),
        (
            &binding.calibration_profile_sha256,
            "calibration_profile_sha256",
        ),
        (
            &binding.calibration_manifest_sha256,
            "calibration_manifest_sha256",
        ),
        (&binding.scope_sha256, "scope_sha256"),
        (&binding.score_definition_sha256, "score_definition_sha256"),
    ] {
        validate_sha256(value, name)?;
    }

    if binding.closure_algorithm != SUPPORTED_CLOSURE_ALGORITHM
        || closure.closure_algorithm != SUPPORTED_CLOSURE_ALGORITHM
    {
        return Err(ClosureBridgeError::UnsupportedClosureAlgorithm);
    }
    if closure.source_manifest_sha256 != binding.source_manifest_sha256 {
        return Err(ClosureBridgeError::SourceManifestMismatch);
    }
    if binding.closure_algorithm != closure.closure_algorithm {
        return Err(ClosureBridgeError::ClosureBindingMismatch(
            "closure_algorithm",
        ));
    }
    Ok(())
}

fn validate_closure_structure(closure: &ClosureManifest) -> Result<(), ClosureBridgeError> {
    if closure.blocks.is_empty() {
        return Err(ClosureBridgeError::EmptyClosure);
    }
    if closure.total_independent_block_count != closure.blocks.len() {
        return Err(ClosureBridgeError::EffectiveCountMismatch);
    }

    let mut block_ids = HashSet::new();
    let mut member_ids = HashSet::new();
    let mut split_counts = BTreeMap::<EvidenceSplit, (usize, usize)>::new();
    let mut calibration_raw = 0usize;
    let mut calibration_effective = 0usize;
    let mut previous_block: Option<(EvidenceSplit, &str)> = None;

    for block in &closure.blocks {
        if !block_ids.insert(block.block_id.as_str()) {
            return Err(ClosureBridgeError::DuplicateBlockId);
        }
        if block.block_id.trim().is_empty()
            || block.member_unit_ids.is_empty()
            || block.member_unit_ids.first().map(String::as_str) != Some(block.block_id.as_str())
            || block
                .member_unit_ids
                .iter()
                .any(|member| member.trim().is_empty())
        {
            return Err(ClosureBridgeError::InvalidBlockIdentity);
        }
        if !strictly_sorted_strings(&block.member_unit_ids) {
            return Err(ClosureBridgeError::NonCanonicalMemberOrder);
        }
        if block.score_receipts_sha256.is_empty() {
            return Err(ClosureBridgeError::NonCanonicalScoreReceiptOrder);
        }
        for receipt in &block.score_receipts_sha256 {
            validate_sha256(receipt, "block.score_receipts_sha256")?;
        }
        if !strictly_sorted_strings(&block.score_receipts_sha256) {
            return Err(ClosureBridgeError::NonCanonicalScoreReceiptOrder);
        }
        for key in &block.dependence_key_receipts {
            if key.value.trim().is_empty() {
                return Err(ClosureBridgeError::InvalidBlockIdentity);
            }
            validate_sha256(
                &key.receipt_sha256,
                "block.dependence_key_receipts.receipt_sha256",
            )?;
        }
        if !strictly_sorted(&block.dependence_key_receipts) {
            return Err(ClosureBridgeError::NonCanonicalDependenceReceiptOrder);
        }

        let order_key = (block.split, block.block_id.as_str());
        if let Some(previous) = previous_block
            && previous > order_key
        {
            return Err(ClosureBridgeError::NonCanonicalBlockOrder);
        }
        previous_block = Some(order_key);

        for member_id in &block.member_unit_ids {
            if !member_ids.insert(member_id.as_str()) {
                return Err(ClosureBridgeError::DuplicateMemberUnit);
            }
        }

        let counts = split_counts.entry(block.split).or_insert((0, 0));
        counts.0 += block.member_unit_ids.len();
        counts.1 += 1;
        if block.split == EvidenceSplit::Calibration {
            calibration_raw += block.member_unit_ids.len();
            calibration_effective += 1;
        }
    }

    if member_ids.len() != closure.raw_unit_count {
        return Err(ClosureBridgeError::InvalidBlockInventory);
    }

    let expected_split_counts: Vec<SplitCount> = EvidenceSplit::ALL
        .into_iter()
        .filter_map(|split| {
            split_counts.get(&split).map(|(raw, effective)| SplitCount {
                split,
                raw_unit_count: *raw,
                independent_block_count: *effective,
            })
        })
        .collect();
    if expected_split_counts != closure.split_counts {
        return Err(ClosureBridgeError::SplitCountMismatch);
    }
    if calibration_raw != closure.calibration_raw_unit_count
        || calibration_effective != closure.calibration_effective_block_count
    {
        return Err(ClosureBridgeError::EffectiveCountMismatch);
    }
    if calibration_effective == 0 {
        return Err(ClosureBridgeError::MissingCalibrationBlocks);
    }
    if closure.calibration_pseudoreplication_factor_numerator != calibration_raw
        || closure.calibration_pseudoreplication_factor_denominator != calibration_effective
    {
        return Err(ClosureBridgeError::PseudoreplicationFactorMismatch);
    }
    Ok(())
}

fn validate_attestation(
    block: &IndependentBlock,
    binding: &ClosureAuthorityBinding,
    attestation: &ClosedBlockScoreAttestation,
) -> Result<(), ClosureBridgeError> {
    if attestation.closure_manifest_sha256 != binding.closure_manifest_sha256 {
        return Err(ClosureBridgeError::AttestationClosureMismatch);
    }
    if attestation.split != EvidenceSplit::Calibration {
        return Err(ClosureBridgeError::WrongAttestationSplit);
    }
    if attestation.score_definition_sha256 != binding.score_definition_sha256 {
        return Err(ClosureBridgeError::AttestationScoreDefinitionMismatch);
    }
    if attestation.member_unit_ids != block.member_unit_ids {
        return Err(ClosureBridgeError::AttestationMemberMismatch);
    }
    if attestation.score_receipts_sha256 != block.score_receipts_sha256 {
        return Err(ClosureBridgeError::AttestationScoreReceiptMismatch);
    }
    if attestation.dependence_key_receipts != block.dependence_key_receipts {
        return Err(ClosureBridgeError::AttestationDependenceReceiptMismatch);
    }
    if attestation.maximum_dangerous_score_units != block.maximum_dangerous_score_units {
        return Err(ClosureBridgeError::AttestationMaximumScoreMismatch);
    }
    validate_sha256(&attestation.block_manifest_sha256, "block_manifest_sha256")?;
    validate_sha256(
        &attestation.aggregate_score_receipt_sha256,
        "aggregate_score_receipt_sha256",
    )?;
    validate_sha256(
        &attestation.tie_break_receipt_sha256,
        "tie_break_receipt_sha256",
    )?;
    Ok(())
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), ClosureBridgeError> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(ClosureBridgeError::InvalidSha256(field))
    }
}

fn strictly_sorted_strings(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn strictly_sorted<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_031_anytime_calibration_monitor::AnytimeCalibrationMonitor;
    use cmaq_036_common_cause_closure::{
        DependenceKey, DependenceKind, EvidenceUnit, build_common_cause_closure,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn unit(
        unit_id: &str,
        cluster_id: &str,
        score: u64,
        cluster_receipt: char,
        score_receipt: char,
        keys: Vec<DependenceKey>,
    ) -> EvidenceUnit {
        EvidenceUnit {
            unit_id: unit_id.into(),
            independent_cluster_id: cluster_id.into(),
            cluster_identity_receipt_sha256: digest(cluster_receipt),
            split: EvidenceSplit::Calibration,
            dangerous_score_units: score,
            score_receipt_sha256: digest(score_receipt),
            dependence_keys: keys,
        }
    }

    fn source_units() -> Vec<EvidenceUnit> {
        vec![
            unit(
                "episode-a",
                "cluster-a",
                10,
                '1',
                'a',
                vec![DependenceKey {
                    kind: DependenceKind::PhysicalAsset,
                    value: "asset-1".into(),
                    receipt_sha256: digest('2'),
                }],
            ),
            unit(
                "episode-b",
                "cluster-b",
                90,
                '3',
                'b',
                vec![
                    DependenceKey {
                        kind: DependenceKind::PhysicalAsset,
                        value: "asset-1".into(),
                        receipt_sha256: digest('2'),
                    },
                    DependenceKey {
                        kind: DependenceKind::CollectionSession,
                        value: "session-7".into(),
                        receipt_sha256: digest('4'),
                    },
                ],
            ),
            unit(
                "episode-c",
                "cluster-c",
                30,
                '5',
                'c',
                vec![DependenceKey {
                    kind: DependenceKind::CollectionSession,
                    value: "session-7".into(),
                    receipt_sha256: digest('4'),
                }],
            ),
            unit(
                "episode-d",
                "cluster-d",
                20,
                '6',
                'd',
                vec![DependenceKey {
                    kind: DependenceKind::ScenarioSeed,
                    value: "seed-11".into(),
                    receipt_sha256: digest('7'),
                }],
            ),
        ]
    }

    fn fixture() -> (
        ClosureManifest,
        ClosureAuthorityBinding,
        Vec<ClosedBlockScoreAttestation>,
    ) {
        let closure = build_common_cause_closure(&digest('f'), &source_units()).unwrap();
        let binding = ClosureAuthorityBinding {
            closure_manifest_sha256: digest('a'),
            closure_authority_receipt_sha256: digest('b'),
            source_manifest_sha256: digest('f'),
            closure_algorithm: SUPPORTED_CLOSURE_ALGORITHM.into(),
            calibration_profile_sha256: digest('c'),
            calibration_manifest_sha256: digest('d'),
            scope_sha256: digest('e'),
            score_definition_sha256: digest('9'),
        };
        let attestations = closure
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.split == EvidenceSplit::Calibration)
            .map(|(index, block)| ClosedBlockScoreAttestation {
                closure_manifest_sha256: binding.closure_manifest_sha256.clone(),
                block_id: block.block_id.clone(),
                split: block.split,
                member_unit_ids: block.member_unit_ids.clone(),
                maximum_dangerous_score_units: block.maximum_dangerous_score_units,
                score_receipts_sha256: block.score_receipts_sha256.clone(),
                dependence_key_receipts: block.dependence_key_receipts.clone(),
                score_definition_sha256: binding.score_definition_sha256.clone(),
                block_manifest_sha256: digest(char::from(b'0' + index as u8)),
                aggregate_score_receipt_sha256: digest(char::from(b'5' + index as u8)),
                tie_break_key: 100 + index as u64,
                tie_break_receipt_sha256: digest(char::from(b'7' + index as u8)),
            })
            .collect();
        (closure, binding, attestations)
    }

    #[test]
    fn admits_one_rank_score_per_effective_closure_block() {
        let (closure, binding, attestations) = fixture();
        let result =
            build_closure_qualified_rank_reference(&closure, &binding, &attestations).unwrap();

        assert_eq!(closure.calibration_raw_unit_count, 4);
        assert_eq!(closure.calibration_effective_block_count, 2);
        assert_eq!(result.calibration_scores().len(), 2);
        assert_eq!(
            result
                .calibration_scores()
                .iter()
                .map(|score| score.score_units)
                .collect::<Vec<_>>(),
            vec![90, 20]
        );
        assert_eq!(
            result
                .reference
                .scores
                .iter()
                .map(|score| score.independent_cluster_id.as_str())
                .collect::<Vec<_>>(),
            vec!["episode-a", "episode-d"]
        );
        assert_eq!(result.diagnostics.pseudoreplication_factor_numerator, 4);
        assert_eq!(result.diagnostics.pseudoreplication_factor_denominator, 2);
    }

    #[test]
    fn produced_reference_is_accepted_by_cmaq031_monitor() {
        let (closure, binding, attestations) = fixture();
        let qualified =
            build_closure_qualified_rank_reference(&closure, &binding, &attestations).unwrap();
        let monitor = qualified.into_monitor(1, 20).unwrap();
        assert_eq!(monitor.calibration_reference_count(), 2);
        assert_eq!(monitor.diagnostics().calibration_effective_block_count, 2);
        assert_eq!(monitor.diagnostics().closure_manifest_sha256, digest('a'));
    }

    #[test]
    fn missing_attestation_is_rejected() {
        let (closure, binding, mut attestations) = fixture();
        attestations.pop();
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::AttestationInventoryMismatch)
        );
    }

    #[test]
    fn duplicate_attestation_is_rejected() {
        let (closure, binding, mut attestations) = fixture();
        attestations[1] = attestations[0].clone();
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::DuplicateAttestation)
        );
    }

    #[test]
    fn wrong_closure_digest_is_rejected() {
        let (closure, binding, mut attestations) = fixture();
        attestations[0].closure_manifest_sha256 = digest('8');
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::AttestationClosureMismatch)
        );
    }

    #[test]
    fn aggregate_score_must_equal_the_closure_maximum() {
        let (closure, binding, mut attestations) = fixture();
        attestations[0].maximum_dangerous_score_units -= 1;
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::AttestationMaximumScoreMismatch)
        );
    }

    #[test]
    fn member_and_receipt_order_is_canonical() {
        let (closure, binding, mut attestations) = fixture();
        attestations[0].member_unit_ids.reverse();
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::AttestationMemberMismatch)
        );

        let (closure, binding, mut attestations) = fixture();
        attestations[0].score_receipts_sha256.reverse();
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::AttestationScoreReceiptMismatch)
        );
    }

    #[test]
    fn duplicate_tie_break_keys_are_rejected() {
        let (closure, binding, mut attestations) = fixture();
        attestations[1].tie_break_key = attestations[0].tie_break_key;
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::DuplicateTieBreakKey)
        );
    }

    #[test]
    fn closure_effective_count_tampering_fails_closed() {
        let (mut closure, binding, attestations) = fixture();
        closure.calibration_effective_block_count += 1;
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::EffectiveCountMismatch)
        );
    }

    #[test]
    fn source_manifest_mismatch_is_rejected() {
        let (closure, mut binding, attestations) = fixture();
        binding.source_manifest_sha256 = digest('8');
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::SourceManifestMismatch)
        );
    }

    #[test]
    fn malformed_member_score_receipt_is_rejected() {
        let (mut closure, binding, attestations) = fixture();
        closure.blocks[0].score_receipts_sha256[0] = "invalid".into();
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::InvalidSha256(
                "block.score_receipts_sha256"
            ))
        );
    }

    #[test]
    fn malformed_dependence_receipt_is_rejected() {
        let (mut closure, binding, attestations) = fixture();
        closure.blocks[0].dependence_key_receipts[0].receipt_sha256 = "invalid".into();
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::InvalidSha256(
                "block.dependence_key_receipts.receipt_sha256"
            ))
        );
    }

    #[test]
    fn duplicate_member_identity_is_rejected() {
        let (mut closure, binding, attestations) = fixture();
        closure.blocks[0].member_unit_ids.push("episode-d".into());
        assert_eq!(
            build_closure_qualified_rank_reference(&closure, &binding, &attestations),
            Err(ClosureBridgeError::DuplicateMemberUnit)
        );
    }
}
