//! Bounded reference preflight evaluator for MUSE-MERGE-001.
//!
//! This crate deliberately has no runtime dependencies and is its own Cargo
//! workspace. It evaluates declared patch footprints only. It does not apply
//! patches, calculate a canonical score graph, authenticate actors, persist
//! commits, render audio, or authorize a merge.
//!
//! A SafeAutomaticMerge outcome means only that this preflight model found
//! no conflict in the declared footprints. commit_permitted is therefore
//! always false; a real implementation must still apply both operations,
//! compare both orders, check musical invariants, persist the result, and
//! independently replay the committed graph.

use std::collections::BTreeSet;

/// Whether the actor has already been qualified for this exact operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorityState {
    Verified,
    Missing,
    Revoked,
    Unknown,
}

/// Integrity state of the referenced source and derived evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactIntegrity {
    Verified,
    /// The bytes do not match the content identity; provenance is unresolved.
    DigestMismatch,
    /// The symbolic candidate may be valid, but an earlier renderer receipt is stale.
    StaleDerivedReceipt,
    Unknown,
}

/// Creative intention matters when two valid alternatives overlap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntentMode {
    Direct,
    Alternative,
}

/// Durable transaction observation. AlreadyCommittedSameKey is admissible
/// only when the durable store has matched both the idempotency key and the
/// exact prepared-operation fingerprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitState {
    Prepared,
    AlreadyCommittedSameKey,
    AcknowledgementLost,
}

/// Result of the conservative preflight evaluator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeOutcome {
    SafeAutomaticMerge,
    MergeWithRevalidation,
    HumanDecisionRequired,
    Conflict,
    Unknown,
    Rejected,
    InDoubt,
}

/// Stable, machine-readable reason codes. These are not free-form model explanations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ReasonCode {
    ExactIdempotentReplay,
    CommitAcknowledgementUncertain,
    StaleBaseOrBranchHead,
    MissingAuthority,
    RevokedAuthority,
    AuthorityUnknown,
    SourceDigestMismatch,
    EvidenceUnknown,
    DerivedReceiptStale,
    UnknownMusicalDependency,
    HardConstraintViolation,
    WriteWriteOverlap,
    ReadWriteDependency,
    SharedMusicalConstraint,
    SameTargetWithoutCommutativityProof,
    QualifiedOperationPair,
    RevalidationRequired,
    RevalidationUnavailable,
    DeclaredFootprintsIndependent,
}

/// A typed, conservative summary of one patch's declared dependency footprint.
///
/// Keys are stable semantic identifiers, not arbitrary JSON paths. Callers
/// must include cross-part relationships and non-local musical obligations
/// in musical_dependencies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchFootprint {
    pub patch_id: String,
    pub operation: String,
    pub targets: BTreeSet<String>,
    pub reads: BTreeSet<String>,
    pub writes: BTreeSet<String>,
    pub musical_dependencies: BTreeSet<String>,
    pub authority: AuthorityState,
    pub artifact_integrity: ArtifactIntegrity,
    pub intent_mode: IntentMode,
    pub hard_constraints_satisfied: bool,
    pub unknown_dependencies: bool,
    pub requires_revalidation: bool,
}

impl PatchFootprint {
    /// Construct a minimal direct patch. Production adapters should populate
    /// all semantic reads, writes, relations and trust state from verified
    /// source artifacts rather than taking caller assertions as proof.
    pub fn direct(
        patch_id: impl Into<String>,
        operation: impl Into<String>,
        targets: impl IntoIterator<Item = impl Into<String>>,
        reads: impl IntoIterator<Item = impl Into<String>>,
        writes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            patch_id: patch_id.into(),
            operation: operation.into(),
            targets: targets.into_iter().map(Into::into).collect(),
            reads: reads.into_iter().map(Into::into).collect(),
            writes: writes.into_iter().map(Into::into).collect(),
            musical_dependencies: BTreeSet::new(),
            authority: AuthorityState::Verified,
            artifact_integrity: ArtifactIntegrity::Verified,
            intent_mode: IntentMode::Direct,
            hard_constraints_satisfied: true,
            unknown_dependencies: false,
            requires_revalidation: false,
        }
    }
}

/// Facts that must be derived from the actual branch/transaction store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MergeContext {
    /// Both patches were created from the exact expected common base and
    /// both branch heads still match the versions used for preview.
    pub base_and_branch_heads_match: bool,
    /// A qualified revalidation path exists for the affected constraints.
    /// This permits constructing a candidate, not committing it.
    pub revalidation_available: bool,
    pub commit_state: CommitState,
}

/// Pairwise operation rules explicitly qualified by a separate proof/test suite.
///
/// Registration is deliberately explicit. The default registry has no rules,
/// so same-target edits with otherwise-disjoint fields return Unknown.
#[derive(Clone, Debug, Default)]
pub struct CommutativityRegistry {
    pairs: BTreeSet<(String, String)>,
}

impl CommutativityRegistry {
    /// Register a symmetric operation pair after independent qualification.
    /// This API records a rule; it does not itself prove that rule.
    pub fn register_qualified_pair(
        &mut self,
        left_operation: impl Into<String>,
        right_operation: impl Into<String>,
    ) {
        self.pairs.insert(canonical_pair(
            left_operation.into(),
            right_operation.into(),
        ));
    }

    fn qualifies(&self, left: &str, right: &str) -> bool {
        self.pairs
            .contains(&canonical_pair(left.to_owned(), right.to_owned()))
    }
}

/// Preflight decision. No outcome emitted by this crate authorizes commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergeDecision {
    pub outcome: MergeOutcome,
    pub reasons: Vec<ReasonCode>,
    /// True means a candidate graph may be constructed for further checks.
    pub candidate_may_be_built: bool,
    /// True means the durable result already exists and should be returned,
    /// not recreated.
    pub idempotent_replay: bool,
    /// Always false in this preflight-only crate.
    pub commit_permitted: bool,
}

impl MergeDecision {
    fn new(
        outcome: MergeOutcome,
        mut reasons: Vec<ReasonCode>,
        candidate_may_be_built: bool,
        idempotent_replay: bool,
    ) -> Self {
        reasons.sort();
        reasons.dedup();
        Self {
            outcome,
            reasons,
            candidate_may_be_built,
            idempotent_replay,
            commit_permitted: false,
        }
    }
}

/// Evaluate two patches against one exact common base.
///
/// The evaluator is symmetric. It is intentionally conservative:
/// - missing/revoked authority is rejected;
/// - digest mismatch or unknown provenance yields Unknown;
/// - cross-patch read/write overlap is a conflict unless revalidation is
///   explicitly available and required;
/// - same-target edits need an explicitly qualified operation-pair rule;
/// - shared musical dependencies are never inferred away from track/time
///   disjointness;
/// - alternatives that both satisfy hard constraints require a human choice.
///
/// This function is a preflight only: no graph mutation or durable commit.
pub fn analyze_merge(
    context: MergeContext,
    left: &PatchFootprint,
    right: &PatchFootprint,
    registry: &CommutativityRegistry,
) -> MergeDecision {
    if context.commit_state == CommitState::AlreadyCommittedSameKey {
        return MergeDecision::new(
            MergeOutcome::SafeAutomaticMerge,
            vec![ReasonCode::ExactIdempotentReplay],
            false,
            true,
        );
    }

    if context.commit_state == CommitState::AcknowledgementLost {
        return MergeDecision::new(
            MergeOutcome::InDoubt,
            vec![ReasonCode::CommitAcknowledgementUncertain],
            false,
            false,
        );
    }

    if !context.base_and_branch_heads_match {
        return MergeDecision::new(
            MergeOutcome::Rejected,
            vec![ReasonCode::StaleBaseOrBranchHead],
            false,
            false,
        );
    }

    let authority = [left.authority, right.authority];
    if authority.contains(&AuthorityState::Missing) {
        return MergeDecision::new(
            MergeOutcome::Rejected,
            vec![ReasonCode::MissingAuthority],
            false,
            false,
        );
    }
    if authority.contains(&AuthorityState::Revoked) {
        return MergeDecision::new(
            MergeOutcome::Rejected,
            vec![ReasonCode::RevokedAuthority],
            false,
            false,
        );
    }
    if authority.contains(&AuthorityState::Unknown) {
        return MergeDecision::new(
            MergeOutcome::Unknown,
            vec![ReasonCode::AuthorityUnknown],
            false,
            false,
        );
    }

    let integrity = [left.artifact_integrity, right.artifact_integrity];
    if integrity.contains(&ArtifactIntegrity::DigestMismatch) {
        return MergeDecision::new(
            MergeOutcome::Unknown,
            vec![ReasonCode::SourceDigestMismatch],
            false,
            false,
        );
    }
    if integrity.contains(&ArtifactIntegrity::Unknown) {
        return MergeDecision::new(
            MergeOutcome::Unknown,
            vec![ReasonCode::EvidenceUnknown],
            false,
            false,
        );
    }

    if left.unknown_dependencies || right.unknown_dependencies {
        return MergeDecision::new(
            MergeOutcome::Unknown,
            vec![ReasonCode::UnknownMusicalDependency],
            false,
            false,
        );
    }

    if !left.hard_constraints_satisfied || !right.hard_constraints_satisfied {
        return MergeDecision::new(
            MergeOutcome::Conflict,
            vec![ReasonCode::HardConstraintViolation],
            false,
            false,
        );
    }

    let write_write = intersects(&left.writes, &right.writes);
    if !write_write.is_empty() {
        if both_are_valid_alternatives(left, right) {
            return MergeDecision::new(
                MergeOutcome::HumanDecisionRequired,
                vec![ReasonCode::WriteWriteOverlap],
                false,
                false,
            );
        }
        return MergeDecision::new(
            MergeOutcome::Conflict,
            vec![ReasonCode::WriteWriteOverlap],
            false,
            false,
        );
    }

    let read_write = intersects(&left.writes, &right.reads)
        || intersects(&right.writes, &left.reads);
    if read_write {
        if either_requires_revalidation(left, right) {
            return revalidation_decision(context);
        }
        return MergeDecision::new(
            MergeOutcome::Conflict,
            vec![ReasonCode::ReadWriteDependency],
            false,
            false,
        );
    }

    if intersects(&left.musical_dependencies, &right.musical_dependencies) {
        if both_are_valid_alternatives(left, right) {
            return MergeDecision::new(
                MergeOutcome::HumanDecisionRequired,
                vec![ReasonCode::SharedMusicalConstraint],
                false,
                false,
            );
        }
        if either_requires_revalidation(left, right) {
            return revalidation_decision(context);
        }
        return MergeDecision::new(
            MergeOutcome::Conflict,
            vec![ReasonCode::SharedMusicalConstraint],
            false,
            false,
        );
    }

    let same_target = intersects(&left.targets, &right.targets);
    if same_target {
        if registry.qualifies(&left.operation, &right.operation) {
            return finish_independent_or_revalidate(
                context,
                left,
                right,
                vec![ReasonCode::QualifiedOperationPair],
            );
        }
        return MergeDecision::new(
            MergeOutcome::Unknown,
            vec![ReasonCode::SameTargetWithoutCommutativityProof],
            false,
            false,
        );
    }

    finish_independent_or_revalidate(
        context,
        left,
        right,
        vec![ReasonCode::DeclaredFootprintsIndependent],
    )
}

fn revalidation_decision(context: MergeContext) -> MergeDecision {
    if context.revalidation_available {
        MergeDecision::new(
            MergeOutcome::MergeWithRevalidation,
            vec![ReasonCode::RevalidationRequired],
            true,
            false,
        )
    } else {
        MergeDecision::new(
            MergeOutcome::Unknown,
            vec![
                ReasonCode::RevalidationRequired,
                ReasonCode::RevalidationUnavailable,
            ],
            false,
            false,
        )
    }
}

fn finish_independent_or_revalidate(
    context: MergeContext,
    left: &PatchFootprint,
    right: &PatchFootprint,
    reasons: Vec<ReasonCode>,
) -> MergeDecision {
    let stale_receipt = left.artifact_integrity == ArtifactIntegrity::StaleDerivedReceipt
        || right.artifact_integrity == ArtifactIntegrity::StaleDerivedReceipt;
    if stale_receipt || either_requires_revalidation(left, right) {
        return revalidation_decision(context);
    }

    MergeDecision::new(MergeOutcome::SafeAutomaticMerge, reasons, true, false)
}

fn both_are_valid_alternatives(left: &PatchFootprint, right: &PatchFootprint) -> bool {
    left.intent_mode == IntentMode::Alternative
        && right.intent_mode == IntentMode::Alternative
        && left.hard_constraints_satisfied
        && right.hard_constraints_satisfied
}

fn either_requires_revalidation(left: &PatchFootprint, right: &PatchFootprint) -> bool {
    left.requires_revalidation
        || right.requires_revalidation
        || left.artifact_integrity == ArtifactIntegrity::StaleDerivedReceipt
        || right.artifact_integrity == ArtifactIntegrity::StaleDerivedReceipt
}

fn intersects(left: &BTreeSet<String>, right: &BTreeSet<String>) -> bool {
    left.iter().any(|item| right.contains(item))
}

fn canonical_pair(left: String, right: String) -> (String, String) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_MANIFEST: &str =
        include_str!("../../../docs/research/fixtures/muse-merge-001-cases-v1.json");

    fn context() -> MergeContext {
        MergeContext {
            base_and_branch_heads_match: true,
            revalidation_available: false,
            commit_state: CommitState::Prepared,
        }
    }

    fn patch(
        id: &str,
        op: &str,
        targets: &[&str],
        reads: &[&str],
        writes: &[&str],
    ) -> PatchFootprint {
        PatchFootprint::direct(id, op, targets, reads, writes)
    }

    fn assert_outcome(
        expected: MergeOutcome,
        ctx: MergeContext,
        left: &PatchFootprint,
        right: &PatchFootprint,
    ) -> MergeDecision {
        let decision = analyze_merge(ctx, left, right, &CommutativityRegistry::default());
        assert_eq!(decision.outcome, expected, "{decision:?}");
        assert!(!decision.commit_permitted, "preflight must never authorize commit");
        decision
    }

    #[test]
    fn m01_independent_parts_are_preflight_compatible() {
        let left = patch("A", "set_articulation", &["event:A1"], &["event:A1"], &["event:A1.articulation"]);
        let right = patch("B", "set_velocity", &["event:B1"], &["event:B1"], &["event:B1.velocity"]);
        let decision = assert_outcome(MergeOutcome::SafeAutomaticMerge, context(), &left, &right);
        assert!(decision.candidate_may_be_built);
    }

    #[test]
    fn m02_conflicting_writes_to_one_pitch_are_rejected() {
        let left = patch("A", "set_pitch", &["event:E1"], &["event:E1.pitch"], &["event:E1.pitch"]);
        let right = patch("B", "set_pitch", &["event:E1"], &["event:E1.pitch"], &["event:E1.pitch"]);
        assert_outcome(MergeOutcome::Conflict, context(), &left, &right);
    }

    #[test]
    fn m03_same_event_distinct_properties_need_a_qualified_rule() {
        let left = patch("A", "set_pitch", &["event:E1"], &["event:E1.pitch"], &["event:E1.pitch"]);
        let right = patch("B", "set_articulation", &["event:E1"], &["event:E1.articulation"], &["event:E1.articulation"]);
        assert_outcome(MergeOutcome::Unknown, context(), &left, &right);
    }

    #[test]
    fn m04_tempo_and_absolute_onset_dependency_conflicts() {
        let left = patch("A", "change_tempo_segment", &["tempo:S2"], &["time_map"], &["time_map"]);
        let right = patch("B", "set_absolute_onset", &["event:E8"], &["time_map", "event:E8.onset_seconds"], &["event:E8.onset_seconds"]);
        assert_outcome(MergeOutcome::Conflict, context(), &left, &right);
    }

    #[test]
    fn m05_cross_part_harmony_violation_is_a_conflict() {
        let mut left = patch("A", "change_chord_function", &["chord:C1"], &["harmony:bar:4"], &["harmony:bar:4"]);
        let mut right = patch("B", "set_pitch", &["event:M1"], &["harmony:bar:4"], &["event:M1.pitch"]);
        left.musical_dependencies.insert("harmony:bar:4".into());
        right.musical_dependencies.insert("harmony:bar:4".into());
        right.hard_constraints_satisfied = false;
        assert_outcome(MergeOutcome::Conflict, context(), &left, &right);
    }

    #[test]
    fn m06_distinct_motif_branches_preserve_alternatives_for_human_choice() {
        let mut left = patch("A", "replace_motif_occurrence", &["event:P1a"], &["motif:P1"], &["event:P1a.sequence"]);
        let mut right = patch("B", "replace_motif_occurrence", &["event:P1b"], &["motif:P1"], &["event:P1b.sequence"]);
        left.musical_dependencies.insert("motif:P1:recurrence".into());
        right.musical_dependencies.insert("motif:P1:recurrence".into());
        left.intent_mode = IntentMode::Alternative;
        right.intent_mode = IntentMode::Alternative;
        assert_outcome(MergeOutcome::HumanDecisionRequired, context(), &left, &right);
    }

    #[test]
    fn m07_meter_change_requires_projection_revalidation() {
        let mut left = patch("A", "change_meter", &["meter:S2"], &["meter:S2"], &["beat_grid:S2"]);
        let right = patch("B", "set_beat_position", &["event:E22"], &["beat_grid:S2"], &["event:E22.beat"]);
        left.requires_revalidation = true;
        let ctx = MergeContext { revalidation_available: true, ..context() };
        let decision = assert_outcome(MergeOutcome::MergeWithRevalidation, ctx, &left, &right);
        assert!(decision.candidate_may_be_built);
    }

    #[test]
    fn m08_missing_scope_rejects_the_operation() {
        let mut left = patch("A", "replace_motif", &["motif:P1"], &["motif:P1"], &["motif:P1.content"]);
        let right = patch("B", "set_velocity", &["event:DR1"], &["event:DR1"], &["event:DR1.velocity"]);
        left.authority = AuthorityState::Missing;
        assert_outcome(MergeOutcome::Rejected, context(), &left, &right);
    }

    #[test]
    fn m09_stale_base_rejects_the_merge() {
        let left = patch("A", "set_pitch", &["event:E3"], &["event:E3.pitch"], &["event:E3.pitch"]);
        let right = patch("B", "set_articulation", &["event:E4"], &["event:E4.articulation"], &["event:E4.articulation"]);
        let ctx = MergeContext { base_and_branch_heads_match: false, ..context() };
        assert_outcome(MergeOutcome::Rejected, ctx, &left, &right);
    }

    #[test]
    fn m10_exact_retry_reuses_the_existing_commit() {
        let left = patch("A", "set_velocity", &["event:E9"], &["event:E9.velocity"], &["event:E9.velocity"]);
        let right = patch("B", "set_articulation", &["event:E10"], &["event:E10.articulation"], &["event:E10.articulation"]);
        let ctx = MergeContext { commit_state: CommitState::AlreadyCommittedSameKey, ..context() };
        let decision = assert_outcome(MergeOutcome::SafeAutomaticMerge, ctx, &left, &right);
        assert!(decision.idempotent_replay);
        assert!(!decision.candidate_may_be_built);
    }

    #[test]
    fn m11_valid_but_incompatible_harmonizations_need_human_choice() {
        let mut left = patch("A", "reharmonize_section", &["section:S3"], &["section:S3"], &["harmony:S3"]);
        let mut right = patch("B", "reharmonize_section", &["section:S3"], &["section:S3"], &["harmony:S3"]);
        left.intent_mode = IntentMode::Alternative;
        right.intent_mode = IntentMode::Alternative;
        assert_outcome(MergeOutcome::HumanDecisionRequired, context(), &left, &right);
    }

    #[test]
    fn m12_stale_renderer_receipt_requires_revalidation() {
        let left = patch("A", "set_pitch", &["event:E12"], &["event:E12.pitch"], &["event:E12.pitch"]);
        let mut right = patch("B", "set_articulation", &["event:E13"], &["event:E13.articulation"], &["event:E13.articulation"]);
        right.artifact_integrity = ArtifactIntegrity::StaleDerivedReceipt;
        let ctx = MergeContext { revalidation_available: true, ..context() };
        assert_outcome(MergeOutcome::MergeWithRevalidation, ctx, &left, &right);
    }

    #[test]
    fn m13_lost_commit_acknowledgement_is_in_doubt() {
        let left = patch("A", "set_pitch", &["event:E14"], &["event:E14.pitch"], &["event:E14.pitch"]);
        let right = patch("B", "set_velocity", &["event:E15"], &["event:E15.velocity"], &["event:E15.velocity"]);
        let ctx = MergeContext { commit_state: CommitState::AcknowledgementLost, ..context() };
        assert_outcome(MergeOutcome::InDoubt, ctx, &left, &right);
    }

    #[test]
    fn m14_substituted_parent_bytes_remain_unknown() {
        let mut left = patch("A", "set_pitch", &["event:E16"], &["event:E16.pitch"], &["event:E16.pitch"]);
        let right = patch("B", "set_velocity", &["event:E17"], &["event:E17.velocity"], &["event:E17.velocity"]);
        left.artifact_integrity = ArtifactIntegrity::DigestMismatch;
        assert_outcome(MergeOutcome::Unknown, context(), &left, &right);
    }

    #[test]
    fn m15_native_rhythm_and_tuning_representations_are_not_coerced() {
        let left = patch("A", "set_velocity", &["event:E18"], &["event:E18.velocity"], &["event:E18.velocity"]);
        let right = patch(
            "B",
            "add_non_isochronous_event",
            &["event:E19"],
            &["rhythm:cycle:C2", "pitch:non_equal_tempered:P2"],
            &["event:E19"],
        );
        let decision = assert_outcome(MergeOutcome::SafeAutomaticMerge, context(), &left, &right);
        assert!(decision.reasons.contains(&ReasonCode::DeclaredFootprintsIndependent));
        assert!(!decision.commit_permitted);
    }

    #[test]
    fn qualified_operation_pair_can_discharge_same_target_unknown() {
        let left = patch("A", "set_pitch", &["event:E1"], &["event:E1.pitch"], &["event:E1.pitch"]);
        let right = patch("B", "set_articulation", &["event:E1"], &["event:E1.articulation"], &["event:E1.articulation"]);
        let mut registry = CommutativityRegistry::default();
        registry.register_qualified_pair("set_pitch", "set_articulation");
        let decision = analyze_merge(context(), &left, &right, &registry);
        assert_eq!(decision.outcome, MergeOutcome::SafeAutomaticMerge);
        assert!(decision.reasons.contains(&ReasonCode::QualifiedOperationPair));
        assert!(!decision.commit_permitted);
    }

    #[test]
    fn analyzer_is_symmetric_for_all_frozen_cases() {
        let fixtures = [
            (patch("a", "x", &["A"], &["a"], &["a1"]), patch("b", "y", &["B"], &["b"], &["b1"])),
            (patch("a", "x", &["E"], &["p"], &["p"]), patch("b", "x", &["E"], &["p"], &["p"])),
            (patch("a", "x", &["E"], &["p"], &["p"]), patch("b", "y", &["E"], &["q"], &["q"])),
            (patch("a", "x", &["T"], &["time"], &["time"]), patch("b", "y", &["E"], &["time"], &["onset"])),
        ];
        for (left, right) in fixtures {
            let forward = analyze_merge(context(), &left, &right, &CommutativityRegistry::default());
            let reverse = analyze_merge(context(), &right, &left, &CommutativityRegistry::default());
            assert_eq!(forward.outcome, reverse.outcome);
            assert_eq!(forward.reasons, reverse.reasons);
        }
    }

    #[test]
    fn every_frozen_case_id_and_outcome_is_present_in_manifest() {
        let expected = [
            ("M01", "SafeAutomaticMerge"),
            ("M02", "Conflict"),
            ("M03", "Unknown"),
            ("M04", "Conflict"),
            ("M05", "Conflict"),
            ("M06", "HumanDecisionRequired"),
            ("M07", "MergeWithRevalidation"),
            ("M08", "Rejected"),
            ("M09", "Rejected"),
            ("M10", "SafeAutomaticMerge"),
            ("M11", "HumanDecisionRequired"),
            ("M12", "MergeWithRevalidation"),
            ("M13", "InDoubt"),
            ("M14", "Unknown"),
            ("M15", "SafeAutomaticMerge"),
        ];
        for (id, outcome) in expected {
            assert!(FIXTURE_MANIFEST.contains(&format!("\"id\": \"{id}\"")), "missing fixture {id}");
            let case = FIXTURE_MANIFEST
                .split(&format!("\"id\": \"{id}\""))
                .nth(1)
                .expect("fixture id present");
            assert!(
                case.split("\"expected_outcome\": \"")
                    .nth(1)
                    .is_some_and(|tail| tail.starts_with(outcome)),
                "wrong expected outcome for {id}"
            );
        }
    }

    #[test]
    fn missing_revalidation_path_never_promotes_to_safe() {
        let mut left = patch("A", "change_meter", &["meter:S2"], &["meter:S2"], &["beat_grid:S2"]);
        let right = patch("B", "set_beat_position", &["event:E22"], &["beat_grid:S2"], &["event:E22.beat"]);
        left.requires_revalidation = true;
        let decision = analyze_merge(context(), &left, &right, &CommutativityRegistry::default());
        assert_eq!(decision.outcome, MergeOutcome::Unknown);
        assert!(!decision.candidate_may_be_built);
    }

    #[test]
    fn unknown_authority_is_not_treated_as_denial_or_approval() {
        let mut left = patch("A", "set_pitch", &["event:E1"], &["event:E1.pitch"], &["event:E1.pitch"]);
        let right = patch("B", "set_velocity", &["event:E2"], &["event:E2.velocity"], &["event:E2.velocity"]);
        left.authority = AuthorityState::Unknown;
        assert_outcome(MergeOutcome::Unknown, context(), &left, &right);
    }
}
