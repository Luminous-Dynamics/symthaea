//! Deterministic Track A oracle for RMI-Bench v1.
//!
//! This crate qualifies benchmark logic against known synthetic policies. It
//! does not evaluate Melothaea, infer human musical value, or establish
//! reciprocal interaction outside the controlled fixtures.

use std::collections::BTreeSet;
use std::fmt;

/// Version of the synthetic trial-manifest schema.
pub const RMI_SCHEMA_VERSION: u16 = 1;
/// Version of the separately reported causal-contrast metrics.
pub const RMI_METRIC_VERSION: &str = "rmi-causal-contrast-v1";

/// A musical feature family used by a symbolic Track A task.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Dimension {
    /// Motif or pitch-class feature.
    Motif,
    /// Rhythmic-cell feature.
    Rhythm,
    /// Tempo or beat-rate feature.
    Tempo,
    /// Musical role, such as lead or accompaniment.
    Role,
    /// Texture or density feature.
    Texture,
}

/// One discrete feature value with its declared musical dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Feature {
    /// Feature dimension.
    pub dimension: Dimension,
    /// Discrete task-specific feature code.
    pub value: i32,
}

/// A task-declared deterministic transformation from partner input to a
/// complementary response. This is a synthetic oracle rule, not a general
/// theory of musical relevance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResponseRule {
    /// Positive finite feature-domain modulus.
    pub modulus: i32,
    /// Signed modular offset applied to the partner feature.
    pub offset: i32,
}

impl ResponseRule {
    /// Apply the declared transformation in its finite feature domain.
    pub fn apply(self, value: i32) -> Result<i32, HarnessError> {
        if self.modulus <= 0 {
            return Err(HarnessError::InvalidTask);
        }
        Ok(
            ((i64::from(value) + i64::from(self.offset)).rem_euclid(i64::from(self.modulus)))
                as i32,
        )
    }

    /// Compute the shortest circular distance in the declared domain.
    pub fn distance(self, left: i32, right: i32) -> Result<u32, HarnessError> {
        if self.modulus <= 0 {
            return Err(HarnessError::InvalidTask);
        }
        let modulus = i64::from(self.modulus);
        let delta = (i64::from(left) - i64::from(right)).rem_euclid(modulus);
        Ok(delta.min(modulus - delta) as u32)
    }
}

/// Frozen task definition shared by all paired intervention conditions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskSpec {
    /// Immutable task identifier.
    pub task_id: String,
    /// Feature dimension targeted by the intervention.
    pub target_dimension: Dimension,
    /// Feature dimension changed only by the unrelated-feature control.
    pub unrelated_dimension: Dimension,
    /// Expected response relation used by this synthetic fixture.
    pub expected_response: ResponseRule,
}

impl TaskSpec {
    /// Validate task identity, feature separation, and response domain.
    pub fn validate(&self) -> Result<(), HarnessError> {
        if self.task_id.trim().is_empty()
            || self.target_dimension == self.unrelated_dimension
            || self.expected_response.modulus <= 0
        {
            return Err(HarnessError::InvalidTask);
        }
        Ok(())
    }
}

/// One immutable observation presented to a policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Observation {
    /// Event identity of this partner observation.
    pub event_id: String,
    /// Pseudonymous session identity.
    pub session_id: String,
    /// Frozen pre-intervention history identity.
    pub history_id: String,
    /// Task identity.
    pub task_id: String,
    /// Task seed shared by all paired conditions.
    pub seed: u64,
    /// Feature known before the intervention.
    pub initial_partner_feature: Feature,
    /// Current partner feature, possibly changed by the intervention.
    pub current_partner_feature: Feature,
    /// Feature used for unrelated-change negative control.
    pub unrelated_feature: Feature,
}

/// Order in which the four conditions were assigned for a paired fixture.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Condition {
    /// Untreated partner input.
    Control,
    /// Randomized targeted partner-input change.
    Targeted,
    /// Sham intervention with no target-feature change.
    Sham,
    /// Unrelated-feature intervention.
    Unrelated,
}

/// Produce a deterministic Fisher-Yates order for synthetic fixtures.
///
/// This function exists for replayable Track A fixtures only. It is not a
/// cryptographic RNG or an approved randomization procedure for human studies.
pub fn condition_order_for_seed(seed: u64) -> [Condition; 4] {
    let mut state = seed;
    let mut order = [
        Condition::Control,
        Condition::Targeted,
        Condition::Sham,
        Condition::Unrelated,
    ];
    for index in (1..order.len()).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        let selected = (value % (index as u64 + 1)) as usize;
        order.swap(index, selected);
    }
    order
}

/// A complete paired intervention set. All conditions share the same frozen
/// state, seed, session, and task; only the declared intervention changes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterventionSet {
    /// Untreated observation.
    pub control: Observation,
    /// Targeted observation.
    pub targeted: Observation,
    /// Sham observation.
    pub sham: Observation,
    /// Unrelated-feature observation.
    pub unrelated: Observation,
    /// Seed used by the intervention assignment procedure.
    pub randomization_seed: u64,
    /// Recorded order; this crate validates the record but does not claim to
    /// have randomized it.
    pub condition_order: [Condition; 4],
}

impl InterventionSet {
    /// Validate exact intervention isolation and event identity.
    pub fn validate(&self, task: &TaskSpec) -> Result<(), HarnessError> {
        task.validate()?;
        let observations = [&self.control, &self.targeted, &self.sham, &self.unrelated];
        let ids: BTreeSet<&str> = observations
            .iter()
            .map(|observation| observation.event_id.as_str())
            .collect();

        if ids.len() != observations.len()
            || observations.iter().any(|observation| {
                observation.event_id.trim().is_empty()
                    || observation.session_id.trim().is_empty()
                    || observation.history_id.trim().is_empty()
                    || observation.task_id != task.task_id
                    || observation.initial_partner_feature.dimension != task.target_dimension
                    || observation.current_partner_feature.dimension != task.target_dimension
                    || observation.unrelated_feature.dimension != task.unrelated_dimension
            })
        {
            return Err(HarnessError::InvalidIntervention);
        }

        let baseline = &self.control;
        for observation in [&self.targeted, &self.sham, &self.unrelated] {
            if observation.session_id != baseline.session_id
                || observation.history_id != baseline.history_id
                || observation.seed != baseline.seed
                || observation.initial_partner_feature != baseline.initial_partner_feature
            {
                return Err(HarnessError::InvalidIntervention);
            }
        }

        if self.targeted.current_partner_feature.value == baseline.current_partner_feature.value
            || self.sham.current_partner_feature != baseline.current_partner_feature
            || self.unrelated.current_partner_feature != baseline.current_partner_feature
            || self.unrelated.unrelated_feature == baseline.unrelated_feature
            || self.targeted.unrelated_feature != baseline.unrelated_feature
            || self.sham.unrelated_feature != baseline.unrelated_feature
        {
            return Err(HarnessError::InvalidIntervention);
        }

        let assigned: BTreeSet<Condition> = self.condition_order.into_iter().collect();
        let expected: BTreeSet<Condition> = [
            Condition::Control,
            Condition::Targeted,
            Condition::Sham,
            Condition::Unrelated,
        ]
        .into_iter()
        .collect();
        if assigned != expected
            || self.condition_order != condition_order_for_seed(self.randomization_seed)
        {
            return Err(HarnessError::InvalidAssignment);
        }
        Ok(())
    }
}

/// Action mechanism label, retained so sensitivity is not conflated with
/// transformation quality.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionMechanism {
    /// Constant action independent of the current input.
    Fixed,
    /// Copies the current partner feature.
    Imitation,
    /// Uses only the frozen initial prompt/state.
    OneWay,
    /// Replays an action from another session.
    Yoked,
    /// Applies the task rule to the current partner feature.
    RuleBased,
    /// Improperly depends on an unrelated feature.
    DistractorDependent,
}

/// An agent's next symbolic musical action.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentAction {
    /// Deterministic action event identity.
    pub event_id: String,
    /// Source partner observation event.
    pub source_event_id: String,
    /// Produced musical feature.
    pub feature: Feature,
    /// Mechanism label.
    pub mechanism: ActionMechanism,
}

/// Policy interface for adapters and deterministic controls.
pub trait AgentPolicy {
    /// Stable policy identifier.
    fn policy_id(&self) -> &str;

    /// Produce one next action from the observed state.
    fn act(&self, observation: &Observation, task: &TaskSpec) -> Result<AgentAction, HarnessError>;
}

/// Policy that always emits the same feature value.
pub struct FixedResponsePolicy {
    /// Stable policy identifier.
    pub id: String,
    /// Constant response value.
    pub value: i32,
}

impl AgentPolicy for FixedResponsePolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn act(&self, observation: &Observation, task: &TaskSpec) -> Result<AgentAction, HarnessError> {
        make_action(
            observation,
            self.policy_id(),
            task,
            self.value,
            ActionMechanism::Fixed,
        )
    }
}

/// Policy that copies the current partner feature without transformation.
pub struct ImitationOnlyPolicy {
    /// Stable policy identifier.
    pub id: String,
}

impl AgentPolicy for ImitationOnlyPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn act(&self, observation: &Observation, task: &TaskSpec) -> Result<AgentAction, HarnessError> {
        make_action(
            observation,
            self.policy_id(),
            task,
            observation.current_partner_feature.value,
            ActionMechanism::Imitation,
        )
    }
}

/// Policy that acts only on the frozen initial partner feature.
pub struct OneWayPromptPolicy {
    /// Stable policy identifier.
    pub id: String,
}

impl AgentPolicy for OneWayPromptPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn act(&self, observation: &Observation, task: &TaskSpec) -> Result<AgentAction, HarnessError> {
        let value = task
            .expected_response
            .apply(observation.initial_partner_feature.value)?;
        make_action(
            observation,
            self.policy_id(),
            task,
            value,
            ActionMechanism::OneWay,
        )
    }
}

/// Policy that replays a value captured from a distinct prior session.
pub struct YokedReplayPolicy {
    /// Stable policy identifier.
    pub id: String,
    /// Source session of the replayed action.
    pub source_session_id: String,
    /// Replayed target feature.
    pub replayed_feature: Feature,
}

impl AgentPolicy for YokedReplayPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn act(&self, observation: &Observation, task: &TaskSpec) -> Result<AgentAction, HarnessError> {
        if self.source_session_id.trim().is_empty()
            || self.source_session_id == observation.session_id
            || self.replayed_feature.dimension != task.target_dimension
        {
            return Err(HarnessError::InvalidYoke);
        }
        make_action(
            observation,
            self.policy_id(),
            task,
            self.replayed_feature.value,
            ActionMechanism::Yoked,
        )
    }
}

/// Positive-control policy: applies the fixture's complementary response
/// rule to the current partner input. This is an oracle, not a candidate
/// Melothaea implementation.
pub struct RuleBasedPolicy {
    /// Stable policy identifier.
    pub id: String,
}

impl AgentPolicy for RuleBasedPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn act(&self, observation: &Observation, task: &TaskSpec) -> Result<AgentAction, HarnessError> {
        let value = task
            .expected_response
            .apply(observation.current_partner_feature.value)?;
        make_action(
            observation,
            self.policy_id(),
            task,
            value,
            ActionMechanism::RuleBased,
        )
    }
}

/// Deliberately incorrect policy that lets the unrelated feature change its
/// target action. It is a negative control for evaluator specificity.
pub struct UnrelatedFeatureShortcutPolicy {
    /// Stable policy identifier.
    pub id: String,
}

impl AgentPolicy for UnrelatedFeatureShortcutPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn act(&self, observation: &Observation, task: &TaskSpec) -> Result<AgentAction, HarnessError> {
        let value = ((i64::from(observation.current_partner_feature.value)
            + i64::from(observation.unrelated_feature.value))
        .rem_euclid(i64::from(task.expected_response.modulus))) as i32;
        make_action(
            observation,
            self.policy_id(),
            task,
            value,
            ActionMechanism::DistractorDependent,
        )
    }
}

fn make_action(
    observation: &Observation,
    policy_id: &str,
    task: &TaskSpec,
    value: i32,
    mechanism: ActionMechanism,
) -> Result<AgentAction, HarnessError> {
    if policy_id.trim().is_empty() || observation.task_id != task.task_id {
        return Err(HarnessError::InvalidPolicy);
    }
    Ok(AgentAction {
        event_id: format!("{}::{policy_id}::action", observation.event_id),
        source_event_id: observation.event_id.clone(),
        feature: Feature {
            dimension: task.target_dimension,
            value,
        },
        mechanism,
    })
}

/// Separate contrasts for sensitivity, sham/repeat floor, specificity and
/// oracle-rule agreement. These are not a composite musical-quality score.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalContrast {
    /// Circular feature distance between treated and untreated agent actions.
    pub targeted_delta: u32,
    /// Difference between sham and untreated agent actions.
    pub sham_delta: u32,
    /// Difference in the target action caused by an unrelated input change.
    pub unrelated_target_delta: u32,
    /// Whether treatment changed the target action at all.
    pub input_sensitive: bool,
    /// Whether sham and unrelated interventions left the target action stable.
    pub target_specific: bool,
    /// Whether both actions match the fixture's declared response relation.
    /// This is a synthetic oracle check, not a human musical-relevance score.
    pub oracle_rule_match: bool,
    /// Strict synthetic-oracle gate: sensitivity, specificity and relation
    /// match must all hold.
    pub qualified_synthetic_contingency: bool,
}

/// Full replayable output for one Track A paired fixture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrialRecordV1 {
    /// Versioned trial manifest.
    pub manifest: TrialManifestV1,
    /// Exact task definition.
    pub task: TaskSpec,
    /// Exact paired intervention inputs.
    pub interventions: InterventionSet,
    /// Actions in control, targeted, sham, unrelated order.
    pub actions: [AgentAction; 4],
    /// Per-dimension synthetic causal contrasts.
    pub contrast: CausalContrast,
}

/// Immutable identity and assignment record for one synthetic trial.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrialManifestV1 {
    /// Schema version.
    pub schema_version: u16,
    /// Metric implementation version.
    pub metric_version: String,
    /// Task identity.
    pub task_id: String,
    /// Session identity.
    pub session_id: String,
    /// Shared task seed.
    pub task_seed: u64,
    /// Intervention-assignment seed.
    pub randomization_seed: u64,
    /// Policy identity and caller-supplied version.
    pub policy_id: String,
    /// Policy version/configuration identity.
    pub policy_version: String,
    /// Exact source revision; a 40-character lowercase Git SHA.
    pub code_revision: String,
    /// Input event IDs in condition order.
    pub input_event_ids: [String; 4],
    /// Output event IDs in condition order.
    pub output_event_ids: [String; 4],
    /// Recorded condition order.
    pub condition_order: [Condition; 4],
}

impl TrialManifestV1 {
    /// Validate required identities, exact event binding and schema version.
    pub fn validate(&self) -> Result<(), HarnessError> {
        let unique_inputs: BTreeSet<&str> =
            self.input_event_ids.iter().map(String::as_str).collect();
        let unique_outputs: BTreeSet<&str> =
            self.output_event_ids.iter().map(String::as_str).collect();
        let valid_sha = self.code_revision.len() == 40
            && self
                .code_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        let assigned: BTreeSet<Condition> = self.condition_order.into_iter().collect();
        let expected: BTreeSet<Condition> = [
            Condition::Control,
            Condition::Targeted,
            Condition::Sham,
            Condition::Unrelated,
        ]
        .into_iter()
        .collect();

        if self.schema_version != RMI_SCHEMA_VERSION
            || self.metric_version != RMI_METRIC_VERSION
            || self.task_id.trim().is_empty()
            || self.session_id.trim().is_empty()
            || self.policy_id.trim().is_empty()
            || self.policy_version.trim().is_empty()
            || !valid_sha
            || unique_inputs.len() != 4
            || unique_outputs.len() != 4
            || self.input_event_ids.iter().any(|id| id.trim().is_empty())
            || self.output_event_ids.iter().any(|id| id.trim().is_empty())
            || self
                .input_event_ids
                .iter()
                .zip(self.output_event_ids.iter())
                .any(|(input, output)| output != &format!("{input}::{}::action", self.policy_id))
            || assigned != expected
            || self.condition_order != condition_order_for_seed(self.randomization_seed)
        {
            return Err(HarnessError::InvalidManifest);
        }
        Ok(())
    }
}

/// Inputs required to create a versioned trial record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrialRunConfig {
    /// Caller-supplied policy version/configuration identity.
    pub policy_version: String,
    /// Exact Git revision of the harness/candidate adapter.
    pub code_revision: String,
}

/// Run one policy on all paired conditions and produce a replayable record.
pub fn run_trial(
    task: TaskSpec,
    interventions: InterventionSet,
    policy: &dyn AgentPolicy,
    config: TrialRunConfig,
) -> Result<TrialRecordV1, HarnessError> {
    interventions.validate(&task)?;
    if config.policy_version.trim().is_empty() {
        return Err(HarnessError::InvalidManifest);
    }

    let observations = [
        &interventions.control,
        &interventions.targeted,
        &interventions.sham,
        &interventions.unrelated,
    ];
    let mut actions = Vec::with_capacity(4);
    for observation in observations {
        actions.push(policy.act(observation, &task)?);
    }
    let actions: [AgentAction; 4] = actions
        .try_into()
        .map_err(|_| HarnessError::InvalidPolicy)?;

    for (action, observation) in actions.iter().zip(observations.iter()) {
        let expected_event_id = format!("{}::{}::action", observation.event_id, policy.policy_id());
        if action.source_event_id != observation.event_id || action.event_id != expected_event_id {
            return Err(HarnessError::InvalidPolicy);
        }
    }

    let contrast = calculate_contrast(&task, &interventions, &actions)?;
    let manifest = TrialManifestV1 {
        schema_version: RMI_SCHEMA_VERSION,
        metric_version: RMI_METRIC_VERSION.to_owned(),
        task_id: task.task_id.clone(),
        session_id: interventions.control.session_id.clone(),
        task_seed: interventions.control.seed,
        randomization_seed: interventions.randomization_seed,
        policy_id: policy.policy_id().to_owned(),
        policy_version: config.policy_version,
        code_revision: config.code_revision,
        input_event_ids: [
            interventions.control.event_id.clone(),
            interventions.targeted.event_id.clone(),
            interventions.sham.event_id.clone(),
            interventions.unrelated.event_id.clone(),
        ],
        output_event_ids: [
            actions[0].event_id.clone(),
            actions[1].event_id.clone(),
            actions[2].event_id.clone(),
            actions[3].event_id.clone(),
        ],
        condition_order: interventions.condition_order,
    };
    manifest.validate()?;
    Ok(TrialRecordV1 {
        manifest,
        task,
        interventions,
        actions,
        contrast,
    })
}

/// Compute separately reported targeted, sham and unrelated contrasts.
pub fn calculate_contrast(
    task: &TaskSpec,
    interventions: &InterventionSet,
    actions: &[AgentAction; 4],
) -> Result<CausalContrast, HarnessError> {
    interventions.validate(task)?;
    let observations = [
        &interventions.control,
        &interventions.targeted,
        &interventions.sham,
        &interventions.unrelated,
    ];
    if actions
        .iter()
        .zip(observations.iter())
        .any(|(action, observation)| {
            action.feature.dimension != task.target_dimension
                || action.source_event_id != observation.event_id
                || !action
                    .event_id
                    .starts_with(&format!("{}::", observation.event_id))
                || action.event_id.trim().is_empty()
        })
    {
        return Err(HarnessError::InvalidPolicy);
    }

    let rule = task.expected_response;
    let targeted_delta = rule.distance(actions[0].feature.value, actions[1].feature.value)?;
    let sham_delta = rule.distance(actions[0].feature.value, actions[2].feature.value)?;
    let unrelated_target_delta =
        rule.distance(actions[0].feature.value, actions[3].feature.value)?;
    let input_sensitive = targeted_delta > 0;
    let target_specific = sham_delta == 0 && unrelated_target_delta == 0;
    let oracle_rule_match = actions[0].feature.value
        == rule.apply(interventions.control.current_partner_feature.value)?
        && actions[1].feature.value
            == rule.apply(interventions.targeted.current_partner_feature.value)?;
    let qualified_synthetic_contingency = input_sensitive && target_specific && oracle_rule_match;

    Ok(CausalContrast {
        targeted_delta,
        sham_delta,
        unrelated_target_delta,
        input_sensitive,
        target_specific,
        oracle_rule_match,
        qualified_synthetic_contingency,
    })
}

/// A controlled partner policy for testing agent-to-partner influence.
pub trait ControlledPartner {
    /// Stable policy identifier.
    fn policy_id(&self) -> &str;

    /// Produce the partner's next feature from the fixed prior state and
    /// current agent action.
    fn respond(
        &self,
        prior_partner_feature: Feature,
        agent_action: &AgentAction,
        target_dimension: Dimension,
        rule: ResponseRule,
    ) -> Result<Feature, HarnessError>;
}

/// Partner that ignores the agent's action.
pub struct FixedPartnerPolicy {
    /// Stable policy identifier.
    pub id: String,
    /// Constant next-partner value.
    pub value: i32,
}

impl ControlledPartner for FixedPartnerPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn respond(
        &self,
        prior_partner_feature: Feature,
        _agent_action: &AgentAction,
        target_dimension: Dimension,
        _rule: ResponseRule,
    ) -> Result<Feature, HarnessError> {
        if prior_partner_feature.dimension != target_dimension || self.id.trim().is_empty() {
            return Err(HarnessError::InvalidPolicy);
        }
        Ok(Feature {
            dimension: target_dimension,
            value: self.value,
        })
    }
}

/// Partner whose next action is causally contingent on the agent's response.
pub struct ResponsivePartnerPolicy {
    /// Stable policy identifier.
    pub id: String,
}

impl ControlledPartner for ResponsivePartnerPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn respond(
        &self,
        prior_partner_feature: Feature,
        agent_action: &AgentAction,
        target_dimension: Dimension,
        rule: ResponseRule,
    ) -> Result<Feature, HarnessError> {
        if prior_partner_feature.dimension != target_dimension
            || agent_action.feature.dimension != target_dimension
            || self.id.trim().is_empty()
        {
            return Err(HarnessError::InvalidPolicy);
        }
        Ok(Feature {
            dimension: target_dimension,
            value: rule.apply(agent_action.feature.value)?,
        })
    }
}

/// Agent-to-controlled-partner causal contrast. This is simulator evidence
/// only; it does not establish influence over human performers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartnerInfluence {
    /// Partner next action under the untreated agent response.
    pub control_next: Feature,
    /// Partner next action under the treated agent response.
    pub treated_next: Feature,
    /// Circular difference in the partner's next action.
    pub targeted_delta: u32,
}

/// Measure agent-to-partner influence from matched agent actions.
pub fn measure_partner_influence(
    partner: &dyn ControlledPartner,
    prior_partner_feature: Feature,
    control_agent_action: &AgentAction,
    treated_agent_action: &AgentAction,
    task: &TaskSpec,
) -> Result<PartnerInfluence, HarnessError> {
    task.validate()?;
    if prior_partner_feature.dimension != task.target_dimension
        || control_agent_action.feature.dimension != task.target_dimension
        || treated_agent_action.feature.dimension != task.target_dimension
    {
        return Err(HarnessError::InvalidPolicy);
    }
    let control_next = partner.respond(
        prior_partner_feature,
        control_agent_action,
        task.target_dimension,
        task.expected_response,
    )?;
    let treated_next = partner.respond(
        prior_partner_feature,
        treated_agent_action,
        task.target_dimension,
        task.expected_response,
    )?;
    if control_next.dimension != task.target_dimension
        || treated_next.dimension != task.target_dimension
    {
        return Err(HarnessError::InvalidPolicy);
    }
    let targeted_delta = task
        .expected_response
        .distance(control_next.value, treated_next.value)?;
    Ok(PartnerInfluence {
        control_next,
        treated_next,
        targeted_delta,
    })
}

/// Reverse-direction causal contrast against the controlled partner policy.
/// This is synthetic simulator evidence, not evidence about human musicians.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartnerCausalContrast {
    /// Partner's next feature under the control agent action.
    pub control_next: Feature,
    /// Partner's next feature under the targeted agent action.
    pub targeted_next: Feature,
    /// Partner's next feature under the sham agent action.
    pub sham_next: Feature,
    /// Partner's next feature under the unrelated-feature agent action.
    pub unrelated_next: Feature,
    /// Circular difference between targeted and control partner actions.
    pub targeted_delta: u32,
    /// Circular difference between sham and control partner actions.
    pub sham_delta: u32,
    /// Circular difference caused by the unrelated-feature intervention.
    pub unrelated_delta: u32,
    /// Whether the targeted agent intervention changes the partner action.
    pub input_sensitive: bool,
    /// Whether sham and unrelated conditions leave the partner target stable.
    pub target_specific: bool,
}

/// Measure the reverse causal direction under all four paired conditions.
/// The shared pre-intervention partner state is frozen across conditions.
pub fn measure_partner_contrast(
    partner: &dyn ControlledPartner,
    prior_partner_feature: Feature,
    actions: &[AgentAction; 4],
    task: &TaskSpec,
    interventions: &InterventionSet,
) -> Result<PartnerCausalContrast, HarnessError> {
    calculate_contrast(task, interventions, actions)?;
    if prior_partner_feature.dimension != task.target_dimension {
        return Err(HarnessError::InvalidPolicy);
    }

    let mut next = Vec::with_capacity(4);
    for action in actions {
        let result = partner.respond(
            prior_partner_feature,
            action,
            task.target_dimension,
            task.expected_response,
        )?;
        if result.dimension != task.target_dimension {
            return Err(HarnessError::InvalidPolicy);
        }
        next.push(result);
    }
    let next: [Feature; 4] = next.try_into().map_err(|_| HarnessError::InvalidPolicy)?;
    let rule = task.expected_response;
    let targeted_delta = rule.distance(next[0].value, next[1].value)?;
    let sham_delta = rule.distance(next[0].value, next[2].value)?;
    let unrelated_delta = rule.distance(next[0].value, next[3].value)?;

    Ok(PartnerCausalContrast {
        control_next: next[0],
        targeted_next: next[1],
        sham_next: next[2],
        unrelated_next: next[3],
        targeted_delta,
        sham_delta,
        unrelated_delta,
        input_sensitive: targeted_delta > 0,
        target_specific: sham_delta == 0 && unrelated_delta == 0,
    })
}

/// Two-direction synthetic trial. Its joint flag is only a finite-oracle
/// result: human musical relevance and collaborator-valued outcomes remain
/// outside this record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReciprocalTrialRecordV1 {
    /// Partner-to-agent synthetic contrast and artifact/event record.
    pub agent: TrialRecordV1,
    /// Agent-to-controlled-partner synthetic contrast.
    pub partner: PartnerCausalContrast,
    /// True only when both directional synthetic gates pass.
    pub qualified_synthetic_reciprocity: bool,
}

/// Run both causal directions over one paired synthetic intervention set.
pub fn run_reciprocal_trial(
    task: TaskSpec,
    interventions: InterventionSet,
    agent_policy: &dyn AgentPolicy,
    partner_policy: &dyn ControlledPartner,
    prior_partner_feature: Feature,
    config: TrialRunConfig,
) -> Result<ReciprocalTrialRecordV1, HarnessError> {
    let agent = run_trial(task.clone(), interventions.clone(), agent_policy, config)?;
    let partner = measure_partner_contrast(
        partner_policy,
        prior_partner_feature,
        &agent.actions,
        &task,
        &interventions,
    )?;
    let qualified_synthetic_reciprocity = agent.contrast.qualified_synthetic_contingency
        && partner.input_sensitive
        && partner.target_specific;
    Ok(ReciprocalTrialRecordV1 {
        agent,
        partner,
        qualified_synthetic_reciprocity,
    })
}

/// A measurement that is either supported by the adapter or explicitly
/// unavailable. Unsupported modalities must not be imputed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetricMeasurement<T> {
    /// A value measured through the declared adapter.
    Measured(T),
    /// The adapter cannot expose this metric for the current modality.
    NotApplicable {
        /// Machine-readable reason code.
        reason: String,
    },
}

impl<T> MetricMeasurement<T> {
    /// Construct an explicit not-applicable value.
    pub fn not_applicable(reason: impl Into<String>) -> Self {
        Self::NotApplicable {
            reason: reason.into(),
        }
    }
}

/// Errors produced by task, intervention, policy, and manifest validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HarnessError {
    /// Invalid task definition or response domain.
    InvalidTask,
    /// Intervention pair changed more than the declared treatment.
    InvalidIntervention,
    /// Condition assignment is not a permutation of the four conditions.
    InvalidAssignment,
    /// Policy identity, action, or feature dimension is invalid.
    InvalidPolicy,
    /// A yoked source is missing, same-session, or dimension-incompatible.
    InvalidYoke,
    /// Manifest is incomplete, inconsistent, or not versioned as expected.
    InvalidManifest,
}

impl fmt::Display for HarnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HarnessError {}

#[cfg(test)]
mod tests {
    use super::*;

    const CODE_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn fixture() -> (TaskSpec, InterventionSet) {
        let task = TaskSpec {
            task_id: "motif-complement-01".to_owned(),
            target_dimension: Dimension::Motif,
            unrelated_dimension: Dimension::Texture,
            expected_response: ResponseRule {
                modulus: 12,
                offset: 7,
            },
        };
        let observation = |event_id: &str, target: i32, texture: i32| Observation {
            event_id: event_id.to_owned(),
            session_id: "session-a".to_owned(),
            history_id: "history-frozen-01".to_owned(),
            task_id: task.task_id.clone(),
            seed: 41,
            initial_partner_feature: Feature {
                dimension: Dimension::Motif,
                value: 2,
            },
            current_partner_feature: Feature {
                dimension: Dimension::Motif,
                value: target,
            },
            unrelated_feature: Feature {
                dimension: Dimension::Texture,
                value: texture,
            },
        };
        let interventions = InterventionSet {
            control: observation("event-control", 2, 3),
            targeted: observation("event-targeted", 5, 3),
            sham: observation("event-sham", 2, 3),
            unrelated: observation("event-unrelated", 2, 8),
            randomization_seed: 991,
            condition_order: condition_order_for_seed(991),
        };
        (task, interventions)
    }

    fn config() -> TrialRunConfig {
        TrialRunConfig {
            policy_version: "fixture-v1".to_owned(),
            code_revision: CODE_REVISION.to_owned(),
        }
    }

    #[test]
    fn rule_based_positive_control_passes_all_synthetic_gates() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task,
            interventions,
            &RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            },
            config(),
        )
        .unwrap();
        assert_eq!(record.contrast.targeted_delta, 3);
        assert_eq!(record.contrast.sham_delta, 0);
        assert_eq!(record.contrast.unrelated_target_delta, 0);
        assert!(record.contrast.qualified_synthetic_contingency);
        assert_eq!(record.manifest.input_event_ids.len(), 4);
    }

    #[test]
    fn fixed_response_cannot_pass_targeted_contingency() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task,
            interventions,
            &FixedResponsePolicy {
                id: "fixed".to_owned(),
                value: 4,
            },
            config(),
        )
        .unwrap();
        assert_eq!(record.contrast.targeted_delta, 0);
        assert!(!record.contrast.qualified_synthetic_contingency);
    }

    #[test]
    fn imitation_is_input_sensitive_but_fails_complementarity_oracle() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task,
            interventions,
            &ImitationOnlyPolicy {
                id: "imitation".to_owned(),
            },
            config(),
        )
        .unwrap();
        assert_eq!(record.contrast.targeted_delta, 3);
        assert!(record.contrast.input_sensitive);
        assert!(!record.contrast.oracle_rule_match);
        assert!(!record.contrast.qualified_synthetic_contingency);
    }

    #[test]
    fn one_way_prompt_does_not_respond_to_current_partner_intervention() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task,
            interventions,
            &OneWayPromptPolicy {
                id: "one-way".to_owned(),
            },
            config(),
        )
        .unwrap();
        assert_eq!(record.contrast.targeted_delta, 0);
        assert!(!record.contrast.qualified_synthetic_contingency);
    }

    #[test]
    fn yoked_replay_does_not_respond_to_the_current_session() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task,
            interventions,
            &YokedReplayPolicy {
                id: "yoke".to_owned(),
                source_session_id: "session-prior".to_owned(),
                replayed_feature: Feature {
                    dimension: Dimension::Motif,
                    value: 9,
                },
            },
            config(),
        )
        .unwrap();
        assert_eq!(record.contrast.targeted_delta, 0);
        assert!(!record.contrast.qualified_synthetic_contingency);
    }

    #[test]
    fn same_session_yoke_is_rejected() {
        let (task, interventions) = fixture();
        let result = run_trial(
            task,
            interventions,
            &YokedReplayPolicy {
                id: "invalid-yoke".to_owned(),
                source_session_id: "session-a".to_owned(),
                replayed_feature: Feature {
                    dimension: Dimension::Motif,
                    value: 9,
                },
            },
            config(),
        );
        assert_eq!(result, Err(HarnessError::InvalidYoke));
    }

    #[test]
    fn unrelated_intervention_must_actually_change_unrelated_feature() {
        let (task, mut interventions) = fixture();
        interventions.unrelated.unrelated_feature.value = 3;
        assert_eq!(
            interventions.validate(&task),
            Err(HarnessError::InvalidIntervention)
        );
    }

    #[test]
    fn no_op_target_intervention_is_rejected_instead_of_counted_as_treatment() {
        let (task, mut interventions) = fixture();
        interventions.targeted.current_partner_feature.value = 2;
        assert_eq!(
            interventions.validate(&task),
            Err(HarnessError::InvalidIntervention)
        );
    }

    #[test]
    fn intervention_pair_rejects_seed_or_history_leakage() {
        let (task, mut interventions) = fixture();
        interventions.targeted.seed += 1;
        assert_eq!(
            interventions.validate(&task),
            Err(HarnessError::InvalidIntervention)
        );

        let (task, mut interventions) = fixture();
        interventions.unrelated.history_id = "different-history".to_owned();
        assert_eq!(
            interventions.validate(&task),
            Err(HarnessError::InvalidIntervention)
        );
    }

    #[test]
    fn duplicate_input_event_identity_is_rejected() {
        let (task, mut interventions) = fixture();
        interventions.targeted.event_id = interventions.control.event_id.clone();
        assert_eq!(
            interventions.validate(&task),
            Err(HarnessError::InvalidIntervention)
        );
    }

    #[test]
    fn controlled_responsive_partner_passes_agent_to_partner_effect() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task.clone(),
            interventions,
            &RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            },
            config(),
        )
        .unwrap();
        let partner = ResponsivePartnerPolicy {
            id: "responsive-partner".to_owned(),
        };
        let prior = Feature {
            dimension: Dimension::Motif,
            value: 1,
        };
        let influence = measure_partner_influence(
            &partner,
            prior,
            &record.actions[0],
            &record.actions[1],
            &task,
        )
        .unwrap();
        assert_eq!(influence.targeted_delta, 3);
    }

    #[test]
    fn fixed_partner_has_zero_agent_to_partner_effect() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task.clone(),
            interventions,
            &RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            },
            config(),
        )
        .unwrap();
        let partner = FixedPartnerPolicy {
            id: "fixed-partner".to_owned(),
            value: 6,
        };
        let prior = Feature {
            dimension: Dimension::Motif,
            value: 1,
        };
        let influence = measure_partner_influence(
            &partner,
            prior,
            &record.actions[0],
            &record.actions[1],
            &task,
        )
        .unwrap();
        assert_eq!(influence.targeted_delta, 0);
    }

    #[test]
    fn trial_manifest_binds_exact_events_and_revision() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task,
            interventions,
            &RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            },
            config(),
        )
        .unwrap();
        assert_eq!(record.manifest.schema_version, RMI_SCHEMA_VERSION);
        assert_eq!(record.manifest.code_revision, CODE_REVISION);
        assert_eq!(
            record.manifest.output_event_ids[1],
            record.actions[1].event_id
        );
        let mut invalid = record.manifest;
        invalid.code_revision = "main".to_owned();
        assert_eq!(invalid.validate(), Err(HarnessError::InvalidManifest));
    }

    #[test]
    fn unsupported_modality_measurement_remains_not_applicable() {
        let timing: MetricMeasurement<u64> =
            MetricMeasurement::not_applicable("audio_only_adapter_has_no_midi_clock");
        assert_eq!(
            timing,
            MetricMeasurement::NotApplicable {
                reason: "audio_only_adapter_has_no_midi_clock".to_owned()
            }
        );
    }

    #[test]
    fn deterministic_replay_produces_identical_trial_record() {
        let (task_a, interventions_a) = fixture();
        let (task_b, interventions_b) = fixture();
        let policy = RuleBasedPolicy {
            id: "rule-positive".to_owned(),
        };
        let first = run_trial(task_a, interventions_a, &policy, config()).unwrap();
        let second = run_trial(task_b, interventions_b, &policy, config()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn condition_order_is_reproducible_and_bound_to_assignment_seed() {
        assert_eq!(condition_order_for_seed(991), condition_order_for_seed(991));
        let (task, mut interventions) = fixture();
        interventions.condition_order.swap(0, 1);
        assert_eq!(
            interventions.validate(&task),
            Err(HarnessError::InvalidAssignment)
        );
    }

    #[test]
    fn unrelated_feature_shortcut_is_detected_as_nontargeted_contingency() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task,
            interventions,
            &UnrelatedFeatureShortcutPolicy {
                id: "distractor-shortcut".to_owned(),
            },
            config(),
        )
        .unwrap();
        assert!(record.contrast.input_sensitive);
        assert!(!record.contrast.target_specific);
        assert!(record.contrast.unrelated_target_delta > 0);
        assert!(!record.contrast.qualified_synthetic_contingency);
    }

    #[test]
    fn sham_noop_effect_is_zero_for_each_deterministic_control() {
        let (task, interventions) = fixture();
        let policies: Vec<Box<dyn AgentPolicy>> = vec![
            Box::new(FixedResponsePolicy {
                id: "fixed".to_owned(),
                value: 4,
            }),
            Box::new(ImitationOnlyPolicy {
                id: "imitation".to_owned(),
            }),
            Box::new(OneWayPromptPolicy {
                id: "one-way".to_owned(),
            }),
            Box::new(YokedReplayPolicy {
                id: "yoke".to_owned(),
                source_session_id: "session-prior".to_owned(),
                replayed_feature: Feature {
                    dimension: Dimension::Motif,
                    value: 9,
                },
            }),
            Box::new(RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            }),
        ];
        for policy in policies {
            let record = run_trial(
                task.clone(),
                interventions.clone(),
                policy.as_ref(),
                config(),
            )
            .unwrap();
            assert_eq!(record.contrast.sham_delta, 0);
        }
    }

    #[test]
    fn role_exchange_oracle_is_not_motif_specific() {
        let (mut task, mut interventions) = fixture();
        task.task_id = "role-complement-01".to_owned();
        task.target_dimension = Dimension::Role;
        task.expected_response = ResponseRule {
            modulus: 4,
            offset: 2,
        };
        for observation in [
            &mut interventions.control,
            &mut interventions.targeted,
            &mut interventions.sham,
            &mut interventions.unrelated,
        ] {
            observation.task_id = task.task_id.clone();
            observation.initial_partner_feature = Feature {
                dimension: Dimension::Role,
                value: 0,
            };
            observation.current_partner_feature.dimension = Dimension::Role;
        }
        interventions.control.current_partner_feature.value = 0;
        interventions.targeted.current_partner_feature.value = 1;
        interventions.sham.current_partner_feature.value = 0;
        interventions.unrelated.current_partner_feature.value = 0;

        let record = run_trial(
            task,
            interventions,
            &RuleBasedPolicy {
                id: "role-rule".to_owned(),
            },
            config(),
        )
        .unwrap();
        assert_eq!(record.contrast.targeted_delta, 1);
        assert!(record.contrast.qualified_synthetic_contingency);
    }

    struct CrossBoundPolicy;

    impl AgentPolicy for CrossBoundPolicy {
        fn policy_id(&self) -> &str {
            "cross-bound"
        }

        fn act(
            &self,
            _observation: &Observation,
            task: &TaskSpec,
        ) -> Result<AgentAction, HarnessError> {
            Ok(AgentAction {
                event_id: "reused-output-event".to_owned(),
                source_event_id: "event-control".to_owned(),
                feature: Feature {
                    dimension: task.target_dimension,
                    value: 1,
                },
                mechanism: ActionMechanism::Fixed,
            })
        }
    }

    #[test]
    fn adapter_cannot_bind_multiple_outputs_to_one_input_event() {
        let (task, interventions) = fixture();
        assert_eq!(
            run_trial(task, interventions, &CrossBoundPolicy, config()),
            Err(HarnessError::InvalidPolicy)
        );
    }

    #[test]
    fn public_contrast_rejects_actions_cross_bound_to_other_observations() {
        let (task, interventions) = fixture();
        let policy = RuleBasedPolicy {
            id: "rule-positive".to_owned(),
        };
        let observations = [
            &interventions.control,
            &interventions.targeted,
            &interventions.sham,
            &interventions.unrelated,
        ];
        let mut actions: Vec<AgentAction> = observations
            .iter()
            .map(|observation| policy.act(observation, &task).unwrap())
            .collect();
        actions.swap(0, 1);
        let actions: [AgentAction; 4] = actions.try_into().unwrap();
        assert_eq!(
            calculate_contrast(&task, &interventions, &actions),
            Err(HarnessError::InvalidPolicy)
        );
    }

    #[test]
    fn manifest_rejects_output_event_rebound_to_another_input() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task,
            interventions,
            &RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            },
            config(),
        )
        .unwrap();
        let mut invalid = record.manifest;
        invalid.output_event_ids.swap(0, 1);
        assert_eq!(invalid.validate(), Err(HarnessError::InvalidManifest));
    }

    struct WrongDimensionPartner;

    impl ControlledPartner for WrongDimensionPartner {
        fn policy_id(&self) -> &str {
            "wrong-dimension"
        }

        fn respond(
            &self,
            _prior_partner_feature: Feature,
            _agent_action: &AgentAction,
            _target_dimension: Dimension,
            _rule: ResponseRule,
        ) -> Result<Feature, HarnessError> {
            Ok(Feature {
                dimension: Dimension::Texture,
                value: 1,
            })
        }
    }

    #[test]
    fn partner_adapter_cannot_return_an_unrelated_feature_dimension() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task.clone(),
            interventions,
            &RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            },
            config(),
        )
        .unwrap();
        let prior = Feature {
            dimension: Dimension::Motif,
            value: 1,
        };
        assert_eq!(
            measure_partner_influence(
                &WrongDimensionPartner,
                prior,
                &record.actions[0],
                &record.actions[1],
                &task
            ),
            Err(HarnessError::InvalidPolicy)
        );
    }

    #[test]
    fn four_condition_partner_contrast_separates_response_from_sham_and_unrelated() {
        let (task, interventions) = fixture();
        let record = run_trial(
            task.clone(),
            interventions.clone(),
            &RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            },
            config(),
        )
        .unwrap();
        let prior = Feature {
            dimension: Dimension::Motif,
            value: 1,
        };

        let responsive = measure_partner_contrast(
            &ResponsivePartnerPolicy {
                id: "responsive".to_owned(),
            },
            prior,
            &record.actions,
            &task,
            &interventions,
        )
        .unwrap();
        assert_eq!(responsive.targeted_delta, 3);
        assert_eq!(responsive.sham_delta, 0);
        assert_eq!(responsive.unrelated_delta, 0);
        assert!(responsive.input_sensitive);
        assert!(responsive.target_specific);

        let fixed = measure_partner_contrast(
            &FixedPartnerPolicy {
                id: "fixed".to_owned(),
                value: 6,
            },
            prior,
            &record.actions,
            &task,
            &interventions,
        )
        .unwrap();
        assert_eq!(fixed.targeted_delta, 0);
        assert!(!fixed.input_sensitive);
        assert!(fixed.target_specific);
    }

    #[test]
    fn reciprocal_trial_requires_agent_and_partner_directional_gates() {
        let (task, interventions) = fixture();
        let prior = Feature {
            dimension: Dimension::Motif,
            value: 1,
        };
        let reciprocal = run_reciprocal_trial(
            task.clone(),
            interventions.clone(),
            &RuleBasedPolicy {
                id: "rule-positive".to_owned(),
            },
            &ResponsivePartnerPolicy {
                id: "responsive".to_owned(),
            },
            prior,
            config(),
        )
        .unwrap();
        assert!(reciprocal.agent.contrast.qualified_synthetic_contingency);
        assert!(reciprocal.partner.input_sensitive);
        assert!(reciprocal.partner.target_specific);
        assert!(reciprocal.qualified_synthetic_reciprocity);

        let imitation = run_reciprocal_trial(
            task,
            interventions,
            &ImitationOnlyPolicy {
                id: "imitation".to_owned(),
            },
            &ResponsivePartnerPolicy {
                id: "responsive".to_owned(),
            },
            prior,
            config(),
        )
        .unwrap();
        assert!(imitation.partner.input_sensitive);
        assert!(!imitation.agent.contrast.qualified_synthetic_contingency);
        assert!(!imitation.qualified_synthetic_reciprocity);
    }
}
