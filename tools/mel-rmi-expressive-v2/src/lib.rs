#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA_VERSION: &str = "mel-rmi-expressive-v2-reference";
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ArticulationV2 {
    Detached = 0,
    Staccato = 1,
    Tenuto = 2,
    Legato = 3,
    Accent = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum PhraseGestureV2 {
    Neutral = 0,
    Pickup = 1,
    Sustain = 2,
    Release = 3,
    Pause = 4,
    Cadential = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetFeatureV2 {
    Velocity,
    Articulation,
    RubatoOffsetUs,
    Register,
    PhraseGesture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedRelationV2 {
    Follow,
    Contrast,
    AnyChange,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpressiveNoteV2 {
    pub event_id: u64,
    pub velocity: u8,
    pub articulation: ArticulationV2,
    pub rubato_offset_us: i32,
    pub register: u8,
    pub phrase_gesture: PhraseGestureV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceFrameV2 {
    pub events: Vec<ExpressiveNoteV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalArmV2 {
    pub input_control: PerformanceFrameV2,
    pub input_treatment: PerformanceFrameV2,
    pub response_control: PerformanceFrameV2,
    pub response_treatment: PerformanceFrameV2,
    pub expected_relation: ExpectedRelationV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RmiExpressiveTrialV2 {
    pub schema_version: String,
    pub trial_id: String,
    pub policy_id: String,
    pub policy_version: String,
    pub deterministic_seed: u64,
    pub source_score_sha256: String,
    pub target_feature: TargetFeatureV2,
    pub target_event_ids: Vec<u64>,
    pub partner_to_agent: CausalArmV2,
    pub agent_to_partner: CausalArmV2,
    pub trial_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssueV2 {
    pub code: String,
    pub direction: Option<String>,
    pub stage: Option<String>,
    pub event_id: Option<u64>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectionReportV2 {
    pub direction: String,
    pub input_target_delta: i64,
    pub response_target_delta: i64,
    pub input_target_changed_events: usize,
    pub response_target_changed_events: usize,
    pub input_collateral_mutations: usize,
    pub response_collateral_mutations: usize,
    pub target_sensitive: bool,
    pub relation_satisfied: bool,
    pub qualified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpressiveReportV2 {
    pub report_version: String,
    pub trial_id: String,
    pub valid: bool,
    pub target_feature: TargetFeatureV2,
    pub target_event_count: usize,
    pub partner_to_agent: DirectionReportV2,
    pub agent_to_partner: DirectionReportV2,
    pub issues: Vec<ValidationIssueV2>,
}

#[derive(Debug, Clone, Copy)]
struct PairMetricsV2 {
    target_delta: i64,
    target_changed_events: usize,
    collateral_mutations: usize,
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn trial_payload_sha256(trial: &RmiExpressiveTrialV2) -> Result<String, String> {
    let mut payload = trial.clone();
    payload.trial_sha256 = ZERO_SHA256.to_owned();
    digest_serializable(&payload)
}

/// Seals a trial after its complete inputs, policy identity and seed are frozen.
/// Sealing is an integrity operation, not proof that the named policy produced the outputs.
pub fn seal_trial(trial: &mut RmiExpressiveTrialV2) -> Result<(), String> {
    trial.trial_sha256 = ZERO_SHA256.to_owned();
    trial.trial_sha256 = trial_payload_sha256(trial)?;
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn note_value(note: &ExpressiveNoteV2, feature: TargetFeatureV2) -> i64 {
    match feature {
        TargetFeatureV2::Velocity => i64::from(note.velocity),
        TargetFeatureV2::Articulation => i64::from(note.articulation as u8),
        TargetFeatureV2::RubatoOffsetUs => i64::from(note.rubato_offset_us),
        TargetFeatureV2::Register => i64::from(note.register),
        TargetFeatureV2::PhraseGesture => i64::from(note.phrase_gesture as u8),
    }
}

fn non_target_fields_equal(
    left: &ExpressiveNoteV2,
    right: &ExpressiveNoteV2,
    target: TargetFeatureV2,
) -> bool {
    match target {
        TargetFeatureV2::Velocity => {
            left.articulation == right.articulation
                && left.rubato_offset_us == right.rubato_offset_us
                && left.register == right.register
                && left.phrase_gesture == right.phrase_gesture
        }
        TargetFeatureV2::Articulation => {
            left.velocity == right.velocity
                && left.rubato_offset_us == right.rubato_offset_us
                && left.register == right.register
                && left.phrase_gesture == right.phrase_gesture
        }
        TargetFeatureV2::RubatoOffsetUs => {
            left.velocity == right.velocity
                && left.articulation == right.articulation
                && left.register == right.register
                && left.phrase_gesture == right.phrase_gesture
        }
        TargetFeatureV2::Register => {
            left.velocity == right.velocity
                && left.articulation == right.articulation
                && left.rubato_offset_us == right.rubato_offset_us
                && left.phrase_gesture == right.phrase_gesture
        }
        TargetFeatureV2::PhraseGesture => {
            left.velocity == right.velocity
                && left.articulation == right.articulation
                && left.rubato_offset_us == right.rubato_offset_us
                && left.register == right.register
        }
    }
}

fn frame_map<'a>(
    frame: &'a PerformanceFrameV2,
    direction: &str,
    stage: &str,
    issues: &mut Vec<ValidationIssueV2>,
) -> BTreeMap<u64, &'a ExpressiveNoteV2> {
    let mut indexed = BTreeMap::new();
    for note in &frame.events {
        if note.velocity == 0 || note.velocity > 127 {
            push_issue(
                issues,
                "invalid_velocity",
                Some(direction),
                Some(stage),
                Some(note.event_id),
                "velocity must be in the inclusive range 1..=127 for a sounding note",
            );
        }
        if note.register > 127 {
            push_issue(
                issues,
                "invalid_register",
                Some(direction),
                Some(stage),
                Some(note.event_id),
                "register must be a MIDI pitch in the inclusive range 0..=127",
            );
        }
        if indexed.insert(note.event_id, note).is_some() {
            push_issue(
                issues,
                "duplicate_event_id",
                Some(direction),
                Some(stage),
                Some(note.event_id),
                "event identity must be unique within a performance frame",
            );
        }
    }
    indexed
}

fn push_issue(
    issues: &mut Vec<ValidationIssueV2>,
    code: &str,
    direction: Option<&str>,
    stage: Option<&str>,
    event_id: Option<u64>,
    detail: &str,
) {
    issues.push(ValidationIssueV2 {
        code: code.to_owned(),
        direction: direction.map(str::to_owned),
        stage: stage.map(str::to_owned),
        event_id,
        detail: detail.to_owned(),
    });
}

fn validate_frame_pair(
    control: &PerformanceFrameV2,
    treatment: &PerformanceFrameV2,
    target: TargetFeatureV2,
    scope: &BTreeSet<u64>,
    direction: &str,
    stage: &str,
    issues: &mut Vec<ValidationIssueV2>,
) -> PairMetricsV2 {
    if scope.is_empty() {
        push_issue(
            issues,
            "empty_target_scope",
            Some(direction),
            Some(stage),
            None,
            "target event scope must contain at least one event",
        );
    }

    let control_map = frame_map(control, direction, stage, issues);
    let treatment_map = frame_map(treatment, direction, stage, issues);
    if control_map.keys().ne(treatment_map.keys()) {
        push_issue(
            issues,
            "event_identity_mismatch",
            Some(direction),
            Some(stage),
            None,
            "control and treatment must contain exactly the same event identities",
        );
    }

    for event_id in scope {
        if !control_map.contains_key(event_id) || !treatment_map.contains_key(event_id) {
            push_issue(
                issues,
                "target_event_unresolved",
                Some(direction),
                Some(stage),
                Some(*event_id),
                "every target event must exist in control and treatment frames",
            );
        }
    }

    let mut target_delta = 0_i64;
    let mut target_changed_events = 0_usize;
    let mut collateral_mutations = 0_usize;

    for (event_id, control_note) in &control_map {
        let Some(treatment_note) = treatment_map.get(event_id) else {
            continue;
        };
        let in_scope = scope.contains(event_id);
        if in_scope {
            if !non_target_fields_equal(control_note, treatment_note, target) {
                collateral_mutations += 1;
                push_issue(
                    issues,
                    "collateral_feature_change",
                    Some(direction),
                    Some(stage),
                    Some(*event_id),
                    "non-target expressive dimensions must remain unchanged in a one-feature trial",
                );
            }
            let delta = note_value(treatment_note, target) - note_value(control_note, target);
            if delta != 0 {
                target_changed_events += 1;
                target_delta += delta;
            }
        } else if control_note != treatment_note {
            collateral_mutations += 1;
            push_issue(
                issues,
                "out_of_scope_event_mutation",
                Some(direction),
                Some(stage),
                Some(*event_id),
                "events outside the declared target scope must remain exactly unchanged",
            );
        }
    }

    PairMetricsV2 {
        target_delta,
        target_changed_events,
        collateral_mutations,
    }
}

fn relation_satisfied(
    relation: ExpectedRelationV2,
    target: TargetFeatureV2,
    input: PairMetricsV2,
    response: PairMetricsV2,
    direction: &str,
    issues: &mut Vec<ValidationIssueV2>,
) -> bool {
    if target_is_categorical(target) && relation != ExpectedRelationV2::AnyChange {
        push_issue(
            issues,
            "categorical_relation_unsupported",
            Some(direction),
            Some("response"),
            None,
            "categorical features require any_change; follow/contrast are only defined for ordered dimensions",
        );
        return false;
    }

    match relation {
        ExpectedRelationV2::AnyChange => response.target_changed_events > 0,
        ExpectedRelationV2::Follow => {
            input.target_delta != 0
                && response.target_delta != 0
                && input.target_delta.signum() == response.target_delta.signum()
        }
        ExpectedRelationV2::Contrast => {
            input.target_delta != 0
                && response.target_delta != 0
                && input.target_delta.signum() != response.target_delta.signum()
        }
    }
}

fn target_is_categorical(target: TargetFeatureV2) -> bool {
    matches!(
        target,
        TargetFeatureV2::Articulation | TargetFeatureV2::PhraseGesture
    )
}

fn evaluate_arm(
    arm: &CausalArmV2,
    target: TargetFeatureV2,
    scope: &BTreeSet<u64>,
    direction: &str,
    issues: &mut Vec<ValidationIssueV2>,
) -> DirectionReportV2 {
    let input = validate_frame_pair(
        &arm.input_control,
        &arm.input_treatment,
        target,
        scope,
        direction,
        "input",
        issues,
    );
    let response = validate_frame_pair(
        &arm.response_control,
        &arm.response_treatment,
        target,
        scope,
        direction,
        "response",
        issues,
    );

    if input.target_changed_events == 0 {
        push_issue(
            issues,
            "missing_target_intervention",
            Some(direction),
            Some("input"),
            None,
            "control and treatment must differ in the declared target feature",
        );
    }
    if response.target_changed_events == 0 {
        push_issue(
            issues,
            "no_target_response",
            Some(direction),
            Some("response"),
            None,
            "a target-sensitive response must change the declared expressive feature",
        );
    }

    let relation_ok = relation_satisfied(
        arm.expected_relation,
        target,
        input,
        response,
        direction,
        issues,
    );
    let target_sensitive = response.target_changed_events > 0;
    let qualified = target_sensitive
        && relation_ok
        && input.collateral_mutations == 0
        && response.collateral_mutations == 0
        && input.target_changed_events > 0;

    DirectionReportV2 {
        direction: direction.to_owned(),
        input_target_delta: input.target_delta,
        response_target_delta: response.target_delta,
        input_target_changed_events: input.target_changed_events,
        response_target_changed_events: response.target_changed_events,
        input_collateral_mutations: input.collateral_mutations,
        response_collateral_mutations: response.collateral_mutations,
        target_sensitive,
        relation_satisfied: relation_ok,
        qualified,
    }
}

/// Evaluates both causal directions independently. It does not run a live policy or
/// establish musical quality; the supplied control/treatment traces are the evidence.
pub fn validate_trial(trial: &RmiExpressiveTrialV2) -> ExpressiveReportV2 {
    let mut issues = Vec::new();
    if trial.schema_version != SCHEMA_VERSION {
        push_issue(
            &mut issues,
            "schema_version_mismatch",
            None,
            None,
            None,
            "unsupported RMI expressive v2 schema version",
        );
    }
    if trial.trial_id.trim().is_empty()
        || trial.policy_id.trim().is_empty()
        || trial.policy_version.trim().is_empty()
    {
        push_issue(
            &mut issues,
            "missing_trial_identity",
            None,
            None,
            None,
            "trial, policy and policy-version identifiers are required",
        );
    }
    if !valid_sha256(&trial.source_score_sha256) {
        push_issue(
            &mut issues,
            "invalid_source_score_digest",
            None,
            None,
            None,
            "source score digest must be 64 lowercase hexadecimal characters",
        );
    }
    match trial_payload_sha256(trial) {
        Ok(expected) if expected != trial.trial_sha256 => push_issue(
            &mut issues,
            "trial_digest_mismatch",
            None,
            None,
            None,
            "trial digest does not match its frozen policy, seed, intervention and response payload",
        ),
        Err(error) => push_issue(
            &mut issues,
            "trial_digest_failed",
            None,
            None,
            None,
            &error,
        ),
        _ => {}
    }

    let mut scope = BTreeSet::new();
    for event_id in &trial.target_event_ids {
        if !scope.insert(*event_id) {
            push_issue(
                &mut issues,
                "duplicate_target_event",
                None,
                None,
                Some(*event_id),
                "target event identifiers must be unique",
            );
        }
    }

    let partner_to_agent = evaluate_arm(
        &trial.partner_to_agent,
        trial.target_feature,
        &scope,
        "partner_to_agent",
        &mut issues,
    );
    let agent_to_partner = evaluate_arm(
        &trial.agent_to_partner,
        trial.target_feature,
        &scope,
        "agent_to_partner",
        &mut issues,
    );

    ExpressiveReportV2 {
        report_version: "mel-rmi-expressive-report-v2".to_owned(),
        trial_id: trial.trial_id.clone(),
        valid: issues.is_empty() && partner_to_agent.qualified && agent_to_partner.qualified,
        target_feature: trial.target_feature,
        target_event_count: scope.len(),
        partner_to_agent,
        agent_to_partner,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(event_id: u64, velocity: u8, rubato: i32) -> ExpressiveNoteV2 {
        ExpressiveNoteV2 {
            event_id,
            velocity,
            articulation: ArticulationV2::Tenuto,
            rubato_offset_us: rubato,
            register: 64,
            phrase_gesture: PhraseGestureV2::Neutral,
        }
    }

    fn frame(velocity_a: u8, velocity_b: u8, rubato_a: i32, rubato_b: i32) -> PerformanceFrameV2 {
        PerformanceFrameV2 {
            events: vec![
                note(101, velocity_a, rubato_a),
                note(102, velocity_b, rubato_b),
            ],
        }
    }

    fn positive_arm(feature: TargetFeatureV2) -> CausalArmV2 {
        match feature {
            TargetFeatureV2::Velocity => CausalArmV2 {
                input_control: frame(60, 60, 0, 0),
                input_treatment: frame(80, 60, 0, 0),
                response_control: frame(50, 50, 0, 0),
                response_treatment: frame(70, 50, 0, 0),
                expected_relation: ExpectedRelationV2::Follow,
            },
            TargetFeatureV2::RubatoOffsetUs => CausalArmV2 {
                input_control: frame(60, 60, 0, 0),
                input_treatment: frame(60, 60, 12_000, 0),
                response_control: frame(50, 50, 0, 0),
                response_treatment: frame(50, 50, 7_000, 0),
                expected_relation: ExpectedRelationV2::Follow,
            },
            TargetFeatureV2::Register => {
                let mut arm = CausalArmV2 {
                    input_control: frame(60, 60, 0, 0),
                    input_treatment: frame(60, 60, 0, 0),
                    response_control: frame(50, 50, 0, 0),
                    response_treatment: frame(50, 50, 0, 0),
                    expected_relation: ExpectedRelationV2::Follow,
                };
                arm.input_treatment.events[0].register = 72;
                arm.response_treatment.events[0].register = 69;
                arm
            }
            _ => unreachable!(),
        }
    }

    fn positive_trial() -> RmiExpressiveTrialV2 {
        let mut trial = RmiExpressiveTrialV2 {
            schema_version: SCHEMA_VERSION.to_owned(),
            trial_id: "rmi-exp-v2-001".to_owned(),
            policy_id: "synthetic-contingent-positive-control".to_owned(),
            policy_version: "1".to_owned(),
            deterministic_seed: 42,
            source_score_sha256: "a".repeat(64),
            target_feature: TargetFeatureV2::Velocity,
            target_event_ids: vec![101],
            partner_to_agent: positive_arm(TargetFeatureV2::Velocity),
            agent_to_partner: positive_arm(TargetFeatureV2::Velocity),
            trial_sha256: ZERO_SHA256.to_owned(),
        };
        seal_trial(&mut trial).unwrap();
        trial
    }

    #[test]
    fn velocity_positive_control_qualifies_both_directions() {
        let report = validate_trial(&positive_trial());
        assert!(report.valid, "{:?}", report.issues);
        assert!(report.partner_to_agent.qualified);
        assert!(report.agent_to_partner.qualified);
        assert_eq!(report.partner_to_agent.response_target_delta, 20);
        assert_eq!(report.agent_to_partner.response_target_delta, 20);
    }

    #[test]
    fn rubato_positive_control_qualifies_both_directions() {
        let mut trial = positive_trial();
        trial.target_feature = TargetFeatureV2::RubatoOffsetUs;
        trial.partner_to_agent = positive_arm(TargetFeatureV2::RubatoOffsetUs);
        trial.agent_to_partner = positive_arm(TargetFeatureV2::RubatoOffsetUs);
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.partner_to_agent.input_target_delta, 12_000);
        assert_eq!(report.partner_to_agent.response_target_delta, 7_000);
    }

    #[test]
    fn register_positive_control_qualifies_both_directions() {
        let mut trial = positive_trial();
        trial.target_feature = TargetFeatureV2::Register;
        trial.partner_to_agent = positive_arm(TargetFeatureV2::Register);
        trial.agent_to_partner = positive_arm(TargetFeatureV2::Register);
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.partner_to_agent.input_target_delta, 8);
        assert_eq!(report.partner_to_agent.response_target_delta, 5);
    }

    #[test]
    fn fixed_response_fails_target_sensitivity() {
        let mut trial = positive_trial();
        let CausalArmV2 {
            response_control,
            response_treatment,
            ..
        } = &mut trial.partner_to_agent;
        *response_treatment = response_control.clone();
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(!report.valid);
        assert!(!report.partner_to_agent.target_sensitive);
        assert!(report.issues.iter().any(|issue| issue.code == "no_target_response"));
    }

    #[test]
    fn wrong_feature_response_is_rejected_as_collateral_mutation() {
        let mut trial = positive_trial();
        if let Some(note) = trial.partner_to_agent.response_treatment.events.get_mut(0) {
            note.rubato_offset_us = 15_000;
        }
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "collateral_feature_change"));
    }

    #[test]
    fn out_of_scope_event_mutation_is_rejected() {
        let mut trial = positive_trial();
        if let Some(note) = trial.partner_to_agent.response_treatment.events.get_mut(1) {
            note.velocity = 75;
        }
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "out_of_scope_event_mutation"));
    }

    #[test]
    fn contrast_relation_requires_opposite_direction() {
        let mut trial = positive_trial();
        trial.partner_to_agent.expected_relation = ExpectedRelationV2::Contrast;
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(!report.valid);
        assert!(!report.partner_to_agent.relation_satisfied);
    }

    #[test]
    fn duplicate_and_missing_event_id_fail_closed() {
        let mut trial = positive_trial();
        trial.partner_to_agent.input_treatment.events[1].event_id = 101;
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "duplicate_event_id"));
    }

    #[test]
    fn articulation_any_change_relation_is_supported() {
        let mut trial = positive_trial();
        trial.target_feature = TargetFeatureV2::Articulation;
        for arm in [&mut trial.partner_to_agent, &mut trial.agent_to_partner] {
            arm.input_treatment.events[0].velocity = arm.input_control.events[0].velocity;
            arm.response_treatment.events[0].velocity = arm.response_control.events[0].velocity;
            arm.input_control.events[0].articulation = ArticulationV2::Detached;
            arm.input_treatment.events[0].articulation = ArticulationV2::Legato;
            arm.response_control.events[0].articulation = ArticulationV2::Staccato;
            arm.response_treatment.events[0].articulation = ArticulationV2::Accent;
            arm.expected_relation = ExpectedRelationV2::AnyChange;
        }
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(report.valid, "{:?}", report.issues);
    }

    #[test]
    fn trial_digest_tampering_is_detected() {
        let mut trial = positive_trial();
        trial.partner_to_agent.response_treatment.events[0].velocity = 95;
        let report = validate_trial(&trial);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "trial_digest_mismatch"));
    }

    #[test]
    fn categorical_feature_requires_any_change_relation() {
        let mut trial = positive_trial();
        trial.target_feature = TargetFeatureV2::Articulation;
        for arm in [&mut trial.partner_to_agent, &mut trial.agent_to_partner] {
            arm.input_control.events[0].articulation = ArticulationV2::Detached;
            arm.input_treatment.events[0].articulation = ArticulationV2::Legato;
            arm.response_control.events[0].articulation = ArticulationV2::Detached;
            arm.response_treatment.events[0].articulation = ArticulationV2::Legato;
            arm.expected_relation = ExpectedRelationV2::Follow;
        }
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|issue| issue.code == "categorical_relation_unsupported"));
    }

    #[test]
    fn categorical_any_change_relation_is_supported() {
        let mut trial = positive_trial();
        trial.target_feature = TargetFeatureV2::PhraseGesture;
        for arm in [&mut trial.partner_to_agent, &mut trial.agent_to_partner] {
            arm.input_treatment.events[0].velocity = arm.input_control.events[0].velocity;
            arm.response_treatment.events[0].velocity = arm.response_control.events[0].velocity;
            arm.input_control.events[0].phrase_gesture = PhraseGestureV2::Neutral;
            arm.input_treatment.events[0].phrase_gesture = PhraseGestureV2::Pickup;
            arm.response_control.events[0].phrase_gesture = PhraseGestureV2::Neutral;
            arm.response_treatment.events[0].phrase_gesture = PhraseGestureV2::Cadential;
            arm.expected_relation = ExpectedRelationV2::AnyChange;
        }
        seal_trial(&mut trial).unwrap();
        let report = validate_trial(&trial);
        assert!(report.valid, "{:?}", report.issues);
    }
}
