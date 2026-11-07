//! Finite path-support reference for SAM issue #10514.
//!
//! This crate is a deterministic conformance oracle, not an empirical support
//! estimator, causal estimator, simulator, or safety controller. Exact support
//! membership is distinct from practical evidence sufficiency.

use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ActorAction {
    pub actor_id: String,
    pub action_id: String,
    /// Digest of the exact normalized command payload, including units/schema.
    pub payload_digest: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DecisionStep {
    pub tick: u64,
    /// Digest of the full permitted history available at this decision point.
    pub history_digest: String,
    /// Joint action for every actor in the declared interaction set.
    pub joint_actions: Vec<ActorAction>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CanonicalStep {
    tick: u64,
    history_digest: String,
    joint_actions: Vec<ActorAction>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CanonicalPath(Vec<CanonicalStep>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SupportError {
    EmptyIdentity(&'static str),
    EmptyPath,
    NonIncreasingTicks,
    DuplicateActorAction(String),
    MissingJointAction,
    InvalidProbability,
    ArithmeticOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Disposition {
    /// Exact path exists in the declared finite support ledger. This is not a
    /// claim that the empirical sample is large enough for precise estimation.
    TrajectorySupported,
    /// The support universe is explicitly exhaustive for this finite fixture,
    /// and the queried joint path is absent.
    UnsupportedSequence,
    /// The evidence corpus is not exhaustive, so absence cannot establish zero support.
    SupportUnknown,
    /// Required evidence, path, receipt, or scope manifest is incomplete.
    Incomplete,
    /// Policy or context does not match the manifest's declared scope.
    OutOfScope,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Assessment {
    pub disposition: Disposition,
    pub decision_points: usize,
    pub matched_finite_path: bool,
}

pub struct AssessmentReceipt {
    // Private fields make assess_with_receipt the supported construction path.
    support_manifest_bytes: Vec<u8>,
    query_bytes: Vec<u8>,
    evidence_manifest_id: String,
    support_target_policy_digest: String,
    support_context_digest: String,
    path_universe_exhaustive: bool,
    query_target_policy_digest: String,
    query_context_digest: String,
    query_complete: bool,
    query_decision_points: usize,
    assessment: Assessment,
}

impl AssessmentReceipt {
    /// The assessment that was computed from the retained canonical inputs.
    pub fn assessment(&self) -> Assessment {
        self.assessment
    }

    /// One canonical artifact binding the exact support manifest, query and result.
    /// This is a content receipt, not authentication or a signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"sam.policy-trajectory-support.assessment-receipt.v0.1\0");
        push_bytes(&mut out, &self.support_manifest_bytes);
        push_bytes(&mut out, &self.query_bytes);
        push_string(&mut out, &self.evidence_manifest_id);
        push_string(&mut out, &self.support_target_policy_digest);
        push_string(&mut out, &self.support_context_digest);
        out.push(u8::from(self.path_universe_exhaustive));
        push_string(&mut out, &self.query_target_policy_digest);
        push_string(&mut out, &self.query_context_digest);
        out.push(u8::from(self.query_complete));
        push_len(&mut out, self.query_decision_points);
        push_bytes(&mut out, &self.assessment.canonical_bytes());
        out
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrajectoryQuery {
    pub target_policy_digest: String,
    pub context_digest: String,
    pub complete: bool,
    pub steps: Vec<DecisionStep>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SupportManifest {
    pub evidence_manifest_digest: String,
    pub target_policy_digest: String,
    pub context_digest: String,
    /// True only when the finite path universe is fully enumerated for this
    /// exact context and declared history/action abstraction.
    pub path_universe_exhaustive: bool,
    supported_paths: BTreeSet<CanonicalPath>,
}

impl SupportManifest {
    /// Canonical content bytes for the exact finite support ledger.
    /// Hash these bytes externally with the consumer's chosen digest algorithm.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"sam.policy-trajectory-support.manifest.v0.1\0");
        push_string(&mut out, &self.evidence_manifest_digest);
        push_string(&mut out, &self.target_policy_digest);
        push_string(&mut out, &self.context_digest);
        out.push(u8::from(self.path_universe_exhaustive));
        push_len(&mut out, self.supported_paths.len());
        for path in &self.supported_paths {
            push_len(&mut out, path.0.len());
            for step in &path.0 {
                push_step(&mut out, step);
            }
        }
        out
    }

    pub fn new(
        evidence_manifest_digest: impl Into<String>,
        target_policy_digest: impl Into<String>,
        context_digest: impl Into<String>,
        path_universe_exhaustive: bool,
        supported_paths: Vec<Vec<DecisionStep>>,
    ) -> Result<Self, SupportError> {
        let evidence_manifest_digest = evidence_manifest_digest.into();
        let target_policy_digest = target_policy_digest.into();
        let context_digest = context_digest.into();
        require_identity(&evidence_manifest_digest, "evidence_manifest_digest")?;
        require_identity(&target_policy_digest, "target_policy_digest")?;
        require_identity(&context_digest, "context_digest")?;

        let mut canonical_paths = BTreeSet::new();
        for path in supported_paths {
            canonical_paths.insert(canonicalize_path(&path)?);
        }

        Ok(Self {
            evidence_manifest_digest,
            target_policy_digest,
            context_digest,
            path_universe_exhaustive,
            supported_paths: canonical_paths,
        })
    }
}

/// Check exact membership in a finite joint-trajectory support ledger.
///
/// A missing path is UnsupportedSequence only when the declared finite
/// support universe is exhaustive. Otherwise the correct result is unknown.
impl TrajectoryQuery {
    /// Canonical content bytes for the exact query, including incomplete
    /// queries. Joint action order is normalized exactly as in path matching.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"sam.policy-trajectory-support.query.v0.1\0");
        push_string(&mut out, &self.target_policy_digest);
        push_string(&mut out, &self.context_digest);
        out.push(u8::from(self.complete));
        push_len(&mut out, self.steps.len());
        for step in &self.steps {
            let mut canonical = CanonicalStep {
                tick: step.tick,
                history_digest: step.history_digest.clone(),
                joint_actions: step.joint_actions.clone(),
            };
            canonical.joint_actions.sort();
            push_step(&mut out, &canonical);
        }
        out
    }
}

impl Assessment {
    /// Canonical content bytes for the exact finite membership result.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"sam.policy-trajectory-support.assessment.v0.1\0");
        out.push(match self.disposition {
            Disposition::TrajectorySupported => 1,
            Disposition::UnsupportedSequence => 2,
            Disposition::SupportUnknown => 3,
            Disposition::Incomplete => 4,
            Disposition::OutOfScope => 5,
        });
        push_len(&mut out, self.decision_points);
        out.push(u8::from(self.matched_finite_path));
        out
    }
}

/// Run the finite path assessment and retain the exact canonical inputs that
/// produced it. Consumers can export one artifact instead of independently
/// pairing three potentially stale byte strings.
pub fn assess_with_receipt(
    manifest: &SupportManifest,
    query: &TrajectoryQuery,
) -> Result<AssessmentReceipt, SupportError> {
    let support_manifest_bytes = manifest.canonical_bytes();
    let query_bytes = query.canonical_bytes();
    let assessment = assess(manifest, query)?;
    Ok(AssessmentReceipt {
        support_manifest_bytes,
        query_bytes,
        evidence_manifest_id: manifest.evidence_manifest_digest.clone(),
        support_target_policy_digest: manifest.target_policy_digest.clone(),
        support_context_digest: manifest.context_digest.clone(),
        path_universe_exhaustive: manifest.path_universe_exhaustive,
        query_target_policy_digest: query.target_policy_digest.clone(),
        query_context_digest: query.context_digest.clone(),
        query_complete: query.complete,
        query_decision_points: query.steps.len(),
        assessment,
    })
}

pub fn assess(
    manifest: &SupportManifest,
    query: &TrajectoryQuery,
) -> Result<Assessment, SupportError> {
    if !query.complete {
        return Ok(Assessment {
            disposition: Disposition::Incomplete,
            decision_points: query.steps.len(),
            matched_finite_path: false,
        });
    }
    if query.steps.is_empty() {
        return Err(SupportError::EmptyPath);
    }
    require_identity(&query.target_policy_digest, "target_policy_digest")?;
    require_identity(&query.context_digest, "context_digest")?;

    if query.target_policy_digest != manifest.target_policy_digest
        || query.context_digest != manifest.context_digest
    {
        return Ok(Assessment {
            disposition: Disposition::OutOfScope,
            decision_points: query.steps.len(),
            matched_finite_path: false,
        });
    }

    let path = canonicalize_path(&query.steps)?;
    let matched = manifest.supported_paths.contains(&path);
    let disposition = if matched {
        Disposition::TrajectorySupported
    } else if manifest.path_universe_exhaustive {
        Disposition::UnsupportedSequence
    } else {
        Disposition::SupportUnknown
    };

    Ok(Assessment {
        disposition,
        decision_points: query.steps.len(),
        matched_finite_path: matched,
    })
}

fn canonicalize_path(steps: &[DecisionStep]) -> Result<CanonicalPath, SupportError> {
    if steps.is_empty() {
        return Err(SupportError::EmptyPath);
    }

    let mut canonical = Vec::with_capacity(steps.len());
    let mut prior_tick = None;
    for step in steps {
        require_identity(&step.history_digest, "history_digest")?;
        if prior_tick.is_some_and(|prior| step.tick <= prior) {
            return Err(SupportError::NonIncreasingTicks);
        }
        prior_tick = Some(step.tick);

        if step.joint_actions.is_empty() {
            return Err(SupportError::MissingJointAction);
        }
        let mut actions = step.joint_actions.clone();
        for action in &actions {
            require_identity(&action.actor_id, "actor_id")?;
            require_identity(&action.action_id, "action_id")?;
            require_identity(&action.payload_digest, "payload_digest")?;
        }
        actions.sort();

        for pair in actions.windows(2) {
            if pair[0].actor_id == pair[1].actor_id {
                return Err(SupportError::DuplicateActorAction(pair[0].actor_id.clone()));
            }
        }

        canonical.push(CanonicalStep {
            tick: step.tick,
            history_digest: step.history_digest.clone(),
            joint_actions: actions,
        });
    }
    Ok(CanonicalPath(canonical))
}

fn push_len(out: &mut Vec<u8>, length: usize) {
    let length = u64::try_from(length).expect("collection length fits u64");
    out.extend_from_slice(&length.to_le_bytes());
}

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) {
    let length = u64::try_from(value.len()).expect("byte length fits u64");
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(value);
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    let length = u64::try_from(value.len()).expect("string length fits u64");
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn push_step(out: &mut Vec<u8>, step: &CanonicalStep) {
    out.extend_from_slice(&step.tick.to_le_bytes());
    push_string(out, &step.history_digest);
    push_len(out, step.joint_actions.len());
    for action in &step.joint_actions {
        push_string(out, &action.actor_id);
        push_string(out, &action.action_id);
        push_string(out, &action.payload_digest);
    }
}

fn require_identity(value: &str, field: &'static str) -> Result<(), SupportError> {
    if value.trim().is_empty() {
        Err(SupportError::EmptyIdentity(field))
    } else {
        Ok(())
    }
}

/// Exact nonnegative rational probability. Values are normalized on creation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rational {
    pub numerator: u128,
    pub denominator: u128,
}

impl Rational {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, SupportError> {
        if denominator == 0 || numerator > denominator {
            return Err(SupportError::InvalidProbability);
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn checked_mul(self, other: Self) -> Result<Self, SupportError> {
        // Cross-cancel before multiplication to reduce overflow risk.
        let a = gcd(self.numerator, other.denominator);
        let b = gcd(other.numerator, self.denominator);
        let left_num = self.numerator / a;
        let right_num = other.numerator / b;
        let left_den = self.denominator / b;
        let right_den = other.denominator / a;
        let numerator = left_num
            .checked_mul(right_num)
            .ok_or(SupportError::ArithmeticOverflow)?;
        let denominator = left_den
            .checked_mul(right_den)
            .ok_or(SupportError::ArithmeticOverflow)?;
        Self::new(numerator, denominator)
    }
}

/// Apply the chain rule to probabilities that are each conditional on the
/// full preceding joint history under the same declared estimand.
///
/// This is not valid for multiplying marginal state frequencies or separate
/// per-agent action propensities without an independently justified factorization.
pub fn joint_path_mass(
    full_history_conditional_probabilities: &[Rational],
) -> Result<Rational, SupportError> {
    if full_history_conditional_probabilities.is_empty() {
        return Err(SupportError::EmptyPath);
    }
    full_history_conditional_probabilities
        .iter()
        .try_fold(Rational::new(1, 1)?, |mass, conditional| {
            mass.checked_mul(*conditional)
        })
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(actor: &str, action_name: &str) -> ActorAction {
        ActorAction {
            actor_id: actor.to_owned(),
            action_id: action_name.to_owned(),
            payload_digest: format!("payload:{actor}:{action_name}"),
        }
    }

    fn step(tick: u64, history: &str, actions: Vec<ActorAction>) -> DecisionStep {
        DecisionStep {
            tick,
            history_digest: history.to_owned(),
            joint_actions: actions,
        }
    }

    fn path(actions: &[&str]) -> Vec<DecisionStep> {
        actions
            .iter()
            .enumerate()
            .map(|(index, action_name)| {
                step(
                    index as u64,
                    &format!("history-{index}"),
                    vec![action("ego", action_name)],
                )
            })
            .collect()
    }

    fn query(policy: &str, context: &str, steps: Vec<DecisionStep>) -> TrajectoryQuery {
        TrajectoryQuery {
            target_policy_digest: policy.to_owned(),
            context_digest: context.to_owned(),
            complete: true,
            steps,
        }
    }

    fn manifest(paths: Vec<Vec<DecisionStep>>, exhaustive: bool) -> SupportManifest {
        SupportManifest::new("evidence-v1", "policy-v1", "context-v1", exhaustive, paths)
            .expect("valid manifest")
    }

    #[test]
    fn support_manifest_canonical_bytes_are_order_independent() {
        let path_a = vec![step(
            0,
            "history",
            vec![action("ego", "continue"), action("actor-1", "yield")],
        )];
        let path_b = vec![step(
            0,
            "history",
            vec![action("actor-1", "yield"), action("ego", "continue")],
        )];
        let a = SupportManifest::new("evidence-v1", "policy-v1", "context-v1", true, vec![path_a])
            .unwrap();
        let b = SupportManifest::new("evidence-v1", "policy-v1", "context-v1", true, vec![path_b])
            .unwrap();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn query_and_assessment_have_canonical_content_exports() {
        let first = query(
            "policy-v1",
            "context-v1",
            vec![step(
                0,
                "history",
                vec![action("ego", "continue"), action("actor-1", "yield")],
            )],
        );
        let second = query(
            "policy-v1",
            "context-v1",
            vec![step(
                0,
                "history",
                vec![action("actor-1", "yield"), action("ego", "continue")],
            )],
        );
        assert_eq!(first.canonical_bytes(), second.canonical_bytes());

        let result = assess(&manifest(vec![first.steps.clone()], true), &first).unwrap();
        assert_eq!(
            result.canonical_bytes().len(),
            "sam.policy-trajectory-support.assessment.v0.1".len() + 1 + 1 + 8 + 1
        );
    }

    #[test]
    fn incomplete_query_canonical_bytes_preserve_completeness_and_partial_steps() {
        let complete = query("policy-v1", "context-v1", path(&["continue"]));
        let mut incomplete = complete.clone();
        incomplete.complete = false;
        assert_ne!(complete.canonical_bytes(), incomplete.canonical_bytes());
        incomplete.steps.clear();
        assert_ne!(complete.canonical_bytes(), incomplete.canonical_bytes());
        let mut mutated = complete.clone();
        mutated.steps[0].history_digest = "mutated-history".to_owned();
        assert_ne!(complete.canonical_bytes(), mutated.canonical_bytes());
    }

    fn canonical_vector(name: &str) -> String {
        include_str!("../fixtures/sam-canonical-artifact-vectors.v0.1.txt")
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name}=")))
            .expect("canonical vector fixture is present")
            .to_owned()
    }

    fn bytes_to_hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    #[test]
    fn canonical_artifact_exports_match_cross_repository_golden_vectors() {
        let support =
            SupportManifest::new("evidence-v1", "policy-v1", "context-v1", false, vec![]).unwrap();
        assert_eq!(
            bytes_to_hex(&support.canonical_bytes()),
            canonical_vector("manifest")
        );

        let query = query(
            "policy-v1",
            "context-v1",
            vec![step(
                7,
                "history-7",
                vec![
                    ActorAction {
                        actor_id: "ego".to_owned(),
                        action_id: "continue".to_owned(),
                        payload_digest: "payload-ego-continue".to_owned(),
                    },
                    ActorAction {
                        actor_id: "actor-1".to_owned(),
                        action_id: "yield".to_owned(),
                        payload_digest: "payload-actor-1-yield".to_owned(),
                    },
                ],
            )],
        );
        assert_eq!(
            bytes_to_hex(&query.canonical_bytes()),
            canonical_vector("query")
        );

        let assessment = assess(&support, &query).unwrap();
        assert_eq!(assessment.disposition, Disposition::SupportUnknown);
        assert_eq!(
            bytes_to_hex(&assessment.canonical_bytes()),
            canonical_vector("assessment")
        );

        let receipt = assess_with_receipt(&support, &query).unwrap();
        assert_eq!(receipt.assessment(), assessment);
        assert_eq!(
            bytes_to_hex(&receipt.canonical_bytes()),
            canonical_vector("assessment_receipt")
        );
    }

    #[test]
    fn marginal_actions_do_not_imply_path_support() {
        let corpus = vec![path(&["continue", "continue"]), path(&["yield", "yield"])];
        let target = path(&["continue", "yield"]);

        // Both target actions appear at each corresponding time index in the
        // corpus, so a marginal-only test would incorrectly accept the path.
        for (tick, target_step) in target.iter().enumerate() {
            assert!(corpus.iter().any(|observed| {
                observed[tick].joint_actions[0].action_id == target_step.joint_actions[0].action_id
            }));
        }

        let result = assess(
            &manifest(corpus, true),
            &query("policy-v1", "context-v1", target),
        )
        .expect("assessment");
        assert_eq!(result.disposition, Disposition::UnsupportedSequence);
        assert!(!result.matched_finite_path);
    }

    #[test]
    fn initial_action_support_does_not_imply_support_after_branch_feedback() {
        let observed_continue = vec![
            step(0, "crossing-start", vec![action("ego", "continue")]),
            step(
                1,
                "after-continue-delivery",
                vec![action("ego", "continue")],
            ),
        ];
        let observed_yield = vec![
            step(0, "crossing-start", vec![action("ego", "yield")]),
            step(1, "after-yield-delivery", vec![action("ego", "yield")]),
        ];
        let evidence = vec![observed_continue.clone(), observed_yield.clone()];

        // The initial Yield intervention is present in the evidence, and
        // Continue is also observed at the later decision point. The
        // post-intervention delivered history is nevertheless different.
        assert!(evidence.iter().any(|episode| {
            episode[0].history_digest == "crossing-start"
                && episode[0].joint_actions[0].action_id == "yield"
        }));
        assert!(
            evidence
                .iter()
                .any(|episode| { episode[1].joint_actions[0].action_id == "continue" })
        );

        let supported_branch = assess(
            &manifest(evidence.clone(), true),
            &query("policy-v1", "context-v1", observed_continue),
        )
        .expect("supported branch assessment");
        assert_eq!(
            supported_branch.disposition,
            Disposition::TrajectorySupported
        );

        let branch_after_feedback = vec![
            step(0, "crossing-start", vec![action("ego", "yield")]),
            step(
                1,
                "after-yield-new-actor-response",
                vec![action("ego", "continue")],
            ),
        ];
        let unsupported_branch = assess(
            &manifest(evidence, true),
            &query("policy-v1", "context-v1", branch_after_feedback),
        )
        .expect("branch feedback assessment");
        assert_eq!(
            unsupported_branch.disposition,
            Disposition::UnsupportedSequence
        );
    }

    #[test]
    fn per_agent_marginals_do_not_imply_joint_action_support() {
        let observed = vec![
            vec![step(
                0,
                "joint-history",
                vec![action("ego", "continue"), action("actor-1", "yield")],
            )],
            vec![step(
                0,
                "joint-history",
                vec![action("ego", "yield"), action("actor-1", "continue")],
            )],
        ];
        let target = vec![step(
            0,
            "joint-history",
            vec![action("ego", "continue"), action("actor-1", "continue")],
        )];

        let ego_has_continue = observed.iter().any(|p| {
            p[0].joint_actions
                .iter()
                .any(|a| a.actor_id == "ego" && a.action_id == "continue")
        });
        let actor_has_continue = observed.iter().any(|p| {
            p[0].joint_actions
                .iter()
                .any(|a| a.actor_id == "actor-1" && a.action_id == "continue")
        });
        assert!(ego_has_continue && actor_has_continue);

        let result = assess(
            &manifest(observed, true),
            &query("policy-v1", "context-v1", target),
        )
        .expect("assessment");
        assert_eq!(result.disposition, Disposition::UnsupportedSequence);
    }

    #[test]
    fn full_history_digest_prevents_compressed_state_aliasing() {
        let observed = vec![vec![step(
            0,
            "full-history-A",
            vec![action("ego", "yield")],
        )]];
        let target = vec![step(0, "full-history-B", vec![action("ego", "yield")])];

        let result = assess(
            &manifest(observed, true),
            &query("policy-v1", "context-v1", target),
        )
        .expect("assessment");
        assert_eq!(result.disposition, Disposition::UnsupportedSequence);
    }

    #[test]
    fn missing_path_in_non_exhaustive_evidence_is_unknown_not_unsupported() {
        let result = assess(
            &manifest(vec![path(&["continue"])], false),
            &query("policy-v1", "context-v1", path(&["yield"])),
        )
        .expect("assessment");
        assert_eq!(result.disposition, Disposition::SupportUnknown);
    }

    #[test]
    fn incomplete_or_censored_trajectory_never_passes() {
        let mut incomplete = query("policy-v1", "context-v1", path(&["continue"]));
        incomplete.complete = false;
        let result =
            assess(&manifest(vec![path(&["continue"])], true), &incomplete).expect("assessment");
        assert_eq!(result.disposition, Disposition::Incomplete);
    }

    #[test]
    fn support_is_scoped_to_policy_and_context_versions() {
        let policy_mismatch = assess(
            &manifest(vec![path(&["continue"])], true),
            &query("policy-v2", "context-v1", path(&["continue"])),
        )
        .expect("assessment");
        assert_eq!(policy_mismatch.disposition, Disposition::OutOfScope);

        let context_mismatch = assess(
            &manifest(vec![path(&["continue"])], true),
            &query("policy-v1", "context-v2", path(&["continue"])),
        )
        .expect("assessment");
        assert_eq!(context_mismatch.disposition, Disposition::OutOfScope);
    }

    #[test]
    fn joint_action_canonicalization_is_order_independent() {
        let a = step(
            0,
            "history",
            vec![action("ego", "continue"), action("actor-1", "yield")],
        );
        let b = step(
            0,
            "history",
            vec![action("actor-1", "yield"), action("ego", "continue")],
        );
        let result = assess(
            &manifest(vec![vec![a]], true),
            &query("policy-v1", "context-v1", vec![b]),
        )
        .expect("assessment");
        assert_eq!(result.disposition, Disposition::TrajectorySupported);
    }

    #[test]
    fn duplicate_actor_commands_are_rejected() {
        let invalid = vec![step(
            0,
            "history",
            vec![action("ego", "continue"), action("ego", "yield")],
        )];
        assert_eq!(
            canonicalize_path(&invalid),
            Err(SupportError::DuplicateActorAction("ego".to_owned()))
        );
    }

    #[test]
    fn low_positive_path_mass_is_not_structural_zero() {
        let conditional = vec![
            Rational::new(1, 2).expect("valid"),
            Rational::new(1, 2).expect("valid"),
            Rational::new(1, 2).expect("valid"),
            Rational::new(1, 2).expect("valid"),
        ];
        let mass = joint_path_mass(&conditional).expect("exact chain-rule mass");
        assert_eq!(
            mass,
            Rational {
                numerator: 1,
                denominator: 16
            }
        );
        assert_ne!(mass.numerator, 0);
    }

    #[test]
    fn probabilities_are_validated_and_multiplication_reduces_before_overflow() {
        assert_eq!(Rational::new(2, 1), Err(SupportError::InvalidProbability));
        assert_eq!(Rational::new(1, 0), Err(SupportError::InvalidProbability));

        let one = Rational {
            numerator: u128::MAX,
            denominator: u128::MAX,
        };
        assert_eq!(
            one.checked_mul(one).expect("reduces first"),
            Rational {
                numerator: 1,
                denominator: 1
            }
        );
    }

    #[test]
    fn identical_frozen_inputs_produce_identical_results() {
        let source = vec![path(&["continue", "yield"])];
        let manifest_a = manifest(source.clone(), true);
        let manifest_b = manifest(source, true);
        let query_a = query("policy-v1", "context-v1", path(&["continue", "yield"]));
        let query_b = query("policy-v1", "context-v1", path(&["continue", "yield"]));
        assert_eq!(
            assess(&manifest_a, &query_a).expect("assessment"),
            assess(&manifest_b, &query_b).expect("assessment")
        );
    }
}
