//! Campaign-level selection over upstream, decision-conditioned EVSI estimates.
//!
//! This crate does not infer sensor performance or replace the Subterra
//! single-decision planner. It selects a bounded portfolio of already-scored
//! inspection options, charging shared deployment bundles once and enforcing
//! separate resource and conservative risk bounds.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

const MAX_DECISIONS: usize = 16;
const MAX_OPTIONS: usize = 64;
const MAX_RISK_PPM: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceVector {
    pub energy_millijoules: u64,
    pub duration_milliseconds: u64,
    pub data_bytes: u64,
    pub exposure_milliunits: u64,
}

impl ResourceVector {
    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            energy_millijoules: self
                .energy_millijoules
                .checked_add(other.energy_millijoules)?,
            duration_milliseconds: self
                .duration_milliseconds
                .checked_add(other.duration_milliseconds)?,
            data_bytes: self.data_bytes.checked_add(other.data_bytes)?,
            exposure_milliunits: self
                .exposure_milliunits
                .checked_add(other.exposure_milliunits)?,
        })
    }

    fn within(self, budget: Self) -> bool {
        self.energy_millijoules <= budget.energy_millijoules
            && self.duration_milliseconds <= budget.duration_milliseconds
            && self.data_bytes <= budget.data_bytes
            && self.exposure_milliunits <= budget.exposure_milliunits
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CampaignBudget {
    pub resources: ResourceVector,
    /// Sum of supplied per-action probability upper bounds. This is a union
    /// bound and does not assume event independence.
    pub mission_interruption_upper_ppm: u32,
    pub robot_loss_upper_ppm: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventoryDisposition {
    CompleteWithinDeclaredScope,
    Incomplete,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssuranceBindings {
    pub common_cause_graph_digest: String,
    pub fault_domain_profile_digest: String,
    pub common_cause_assessment_digest: String,
    pub portfolio_value_assessment_digest: String,
    pub inventory_disposition: InventoryDisposition,
    /// True only when an external assessment supports summing the supplied
    /// per-decision EVSI values for this portfolio.
    pub additive_value_factorization_complete: bool,
}

impl AssuranceBindings {
    fn disposition(&self) -> AssuranceDisposition {
        let digests_present = [
            &self.common_cause_graph_digest,
            &self.fault_domain_profile_digest,
            &self.common_cause_assessment_digest,
            &self.portfolio_value_assessment_digest,
        ]
        .iter()
        .all(|digest| !digest.trim().is_empty());

        if digests_present
            && self.inventory_disposition == InventoryDisposition::CompleteWithinDeclaredScope
            && self.additive_value_factorization_complete
        {
            AssuranceDisposition::EligibleUnderDeclaredAssessment
        } else {
            AssuranceDisposition::ResearchOnlyInsufficientEvidence
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssuranceDisposition {
    EligibleUnderDeclaredAssessment,
    ResearchOnlyInsufficientEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRequirement {
    pub decision_id: String,
    pub mandatory: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentBundle {
    pub bundle_id: String,
    /// One-time cost charged if at least one option in this bundle is selected.
    pub fixed_resources: ResourceVector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectionOption {
    pub option_id: String,
    pub decision_id: String,
    pub deployment_bundle_id: String,
    /// Upstream gross EVSI in a common, declared integer loss unit.
    pub gross_evsi_micro_units: u64,
    /// Kept separate from EVSI; this measures reduction in declared
    /// unverified-state debt, not a monetized benefit.
    pub unknown_state_debt_reduction_units: u64,
    pub variable_resources: ResourceVector,
    /// Per-option upper bounds in parts per million. Campaign aggregation uses
    /// the union bound, never an independence assumption.
    pub mission_interruption_upper_ppm: u32,
    pub robot_loss_upper_ppm: u32,
    pub external_gate_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignProblem {
    pub campaign_id: String,
    pub decisions: Vec<DecisionRequirement>,
    pub deployment_bundles: Vec<DeploymentBundle>,
    pub options: Vec<InspectionOption>,
    pub budget: CampaignBudget,
    pub assurance: AssuranceBindings,
    pub maximum_search_nodes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CampaignError {
    EmptyIdentifier,
    DuplicateIdentifier,
    MissingDecisionReference,
    MissingDeploymentBundle,
    InvalidRiskBound,
    InvalidBudget,
    ProblemTooLarge,
    NoFeasibleCampaign,
    SearchLimitWithoutFeasibleIncumbent,
    ArithmeticOverflow,
    ZeroSearchNodeLimit,
}

impl std::fmt::Display for CampaignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CampaignError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDisposition {
    OptimalityProven,
    NodeLimitReached,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignSelection {
    pub selected_option_ids: Vec<String>,
    pub covered_decision_ids: Vec<String>,
    pub additive_evsi_estimate_micro_units: u64,
    pub unknown_state_debt_reduction_units: u64,
    pub resources: ResourceVector,
    pub mission_interruption_upper_ppm: u32,
    pub robot_loss_upper_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignReport {
    pub campaign_id: String,
    pub selection: CampaignSelection,
    pub search_disposition: SearchDisposition,
    pub evaluated_nodes: u64,
    pub assurance_disposition: AssuranceDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreedyMetric {
    GrossEvsi,
    EvsiPerEnergy,
    EvsiPerDuration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreedyReport {
    pub metric: GreedyMetric,
    pub selection: CampaignSelection,
}

#[derive(Debug, Clone)]
struct DecisionGroup {
    decision_id: String,
    mandatory: bool,
    options: Vec<InspectionOption>,
}

#[derive(Debug, Clone)]
struct PreparedProblem {
    groups: Vec<DecisionGroup>,
    bundles: BTreeMap<String, DeploymentBundle>,
}

#[derive(Debug, Clone)]
struct PartialSelection {
    index: usize,
    selected_option_ids: Vec<String>,
    covered_decision_ids: Vec<String>,
    used_bundles: BTreeSet<String>,
    resources: ResourceVector,
    mission_interruption_upper_ppm: u32,
    robot_loss_upper_ppm: u32,
    gross_evsi_micro_units: u64,
    unknown_state_debt_reduction_units: u64,
}

impl PartialSelection {
    fn empty() -> Self {
        Self {
            index: 0,
            selected_option_ids: Vec::new(),
            covered_decision_ids: Vec::new(),
            used_bundles: BTreeSet::new(),
            resources: ResourceVector::default(),
            mission_interruption_upper_ppm: 0,
            robot_loss_upper_ppm: 0,
            gross_evsi_micro_units: 0,
            unknown_state_debt_reduction_units: 0,
        }
    }

    fn into_selection(mut self) -> CampaignSelection {
        self.selected_option_ids.sort();
        self.covered_decision_ids.sort();
        CampaignSelection {
            selected_option_ids: self.selected_option_ids,
            covered_decision_ids: self.covered_decision_ids,
            additive_evsi_estimate_micro_units: self.gross_evsi_micro_units,
            unknown_state_debt_reduction_units: self.unknown_state_debt_reduction_units,
            resources: self.resources,
            mission_interruption_upper_ppm: self.mission_interruption_upper_ppm,
            robot_loss_upper_ppm: self.robot_loss_upper_ppm,
        }
    }
}

fn prepare(problem: &CampaignProblem) -> Result<PreparedProblem, CampaignError> {
    if problem.campaign_id.trim().is_empty() {
        return Err(CampaignError::EmptyIdentifier);
    }
    if problem.maximum_search_nodes == 0 {
        return Err(CampaignError::ZeroSearchNodeLimit);
    }
    if problem.budget.mission_interruption_upper_ppm > MAX_RISK_PPM
        || problem.budget.robot_loss_upper_ppm > MAX_RISK_PPM
    {
        return Err(CampaignError::InvalidBudget);
    }
    if problem.decisions.len() > MAX_DECISIONS || problem.options.len() > MAX_OPTIONS {
        return Err(CampaignError::ProblemTooLarge);
    }

    let mut decisions = BTreeMap::<String, bool>::new();
    for decision in &problem.decisions {
        if decision.decision_id.trim().is_empty() {
            return Err(CampaignError::EmptyIdentifier);
        }
        if decisions
            .insert(decision.decision_id.clone(), decision.mandatory)
            .is_some()
        {
            return Err(CampaignError::DuplicateIdentifier);
        }
    }

    let mut bundles = BTreeMap::<String, DeploymentBundle>::new();
    for bundle in &problem.deployment_bundles {
        if bundle.bundle_id.trim().is_empty() {
            return Err(CampaignError::EmptyIdentifier);
        }
        if bundles
            .insert(bundle.bundle_id.clone(), bundle.clone())
            .is_some()
        {
            return Err(CampaignError::DuplicateIdentifier);
        }
    }

    let mut option_ids = BTreeSet::new();
    let mut grouped = BTreeMap::<String, Vec<InspectionOption>>::new();
    for option in &problem.options {
        if option.option_id.trim().is_empty()
            || option.decision_id.trim().is_empty()
            || option.deployment_bundle_id.trim().is_empty()
        {
            return Err(CampaignError::EmptyIdentifier);
        }
        if !option_ids.insert(option.option_id.as_str()) {
            return Err(CampaignError::DuplicateIdentifier);
        }
        if !decisions.contains_key(&option.decision_id) {
            return Err(CampaignError::MissingDecisionReference);
        }
        if !bundles.contains_key(&option.deployment_bundle_id) {
            return Err(CampaignError::MissingDeploymentBundle);
        }
        if option.mission_interruption_upper_ppm > MAX_RISK_PPM
            || option.robot_loss_upper_ppm > MAX_RISK_PPM
        {
            return Err(CampaignError::InvalidRiskBound);
        }
        if option.external_gate_passed {
            grouped
                .entry(option.decision_id.clone())
                .or_default()
                .push(option.clone());
        }
    }

    let mut groups = Vec::with_capacity(decisions.len());
    for (decision_id, mandatory) in decisions {
        let mut options = grouped.remove(&decision_id).unwrap_or_default();
        options.sort_by(|a, b| {
            b.gross_evsi_micro_units
                .cmp(&a.gross_evsi_micro_units)
                .then_with(|| a.option_id.cmp(&b.option_id))
        });
        groups.push(DecisionGroup {
            decision_id,
            mandatory,
            options,
        });
    }
    // Mandatory decisions are explored first; stable ID ordering resolves ties.
    groups.sort_by(|a, b| {
        b.mandatory
            .cmp(&a.mandatory)
            .then_with(|| a.decision_id.cmp(&b.decision_id))
    });

    Ok(PreparedProblem { groups, bundles })
}

fn apply_option(
    problem: &CampaignProblem,
    bundles: &BTreeMap<String, DeploymentBundle>,
    state: &PartialSelection,
    option: &InspectionOption,
) -> Result<Option<PartialSelection>, CampaignError> {
    let mut next = state.clone();
    let bundle = bundles
        .get(&option.deployment_bundle_id)
        .ok_or(CampaignError::MissingDeploymentBundle)?;

    let mut added_resources = option.variable_resources;
    if !next.used_bundles.contains(&option.deployment_bundle_id) {
        added_resources = added_resources
            .checked_add(bundle.fixed_resources)
            .ok_or(CampaignError::ArithmeticOverflow)?;
        next.used_bundles
            .insert(option.deployment_bundle_id.clone());
    }
    next.resources = next
        .resources
        .checked_add(added_resources)
        .ok_or(CampaignError::ArithmeticOverflow)?;

    next.mission_interruption_upper_ppm = next
        .mission_interruption_upper_ppm
        .checked_add(option.mission_interruption_upper_ppm)
        .ok_or(CampaignError::ArithmeticOverflow)?;
    next.robot_loss_upper_ppm = next
        .robot_loss_upper_ppm
        .checked_add(option.robot_loss_upper_ppm)
        .ok_or(CampaignError::ArithmeticOverflow)?;
    next.gross_evsi_micro_units = next
        .gross_evsi_micro_units
        .checked_add(option.gross_evsi_micro_units)
        .ok_or(CampaignError::ArithmeticOverflow)?;
    next.unknown_state_debt_reduction_units = next
        .unknown_state_debt_reduction_units
        .checked_add(option.unknown_state_debt_reduction_units)
        .ok_or(CampaignError::ArithmeticOverflow)?;

    if !next.resources.within(problem.budget.resources)
        || next.mission_interruption_upper_ppm > problem.budget.mission_interruption_upper_ppm
        || next.robot_loss_upper_ppm > problem.budget.robot_loss_upper_ppm
    {
        return Ok(None);
    }

    next.selected_option_ids.push(option.option_id.clone());
    next.covered_decision_ids.push(option.decision_id.clone());
    Ok(Some(next))
}

fn better(candidate: &PartialSelection, current: &PartialSelection) -> bool {
    match candidate
        .gross_evsi_micro_units
        .cmp(&current.gross_evsi_micro_units)
    {
        Ordering::Greater => return true,
        Ordering::Less => return false,
        Ordering::Equal => {}
    }
    match candidate
        .unknown_state_debt_reduction_units
        .cmp(&current.unknown_state_debt_reduction_units)
    {
        Ordering::Greater => return true,
        Ordering::Less => return false,
        Ordering::Equal => {}
    }
    let candidate_cost = (
        candidate.resources.energy_millijoules,
        candidate.resources.duration_milliseconds,
        candidate.resources.data_bytes,
        candidate.resources.exposure_milliunits,
    );
    let current_cost = (
        current.resources.energy_millijoules,
        current.resources.duration_milliseconds,
        current.resources.data_bytes,
        current.resources.exposure_milliunits,
    );
    match candidate_cost.cmp(&current_cost) {
        Ordering::Less => return true,
        Ordering::Greater => return false,
        Ordering::Equal => {}
    }
    let mut candidate_ids = candidate.selected_option_ids.clone();
    let mut current_ids = current.selected_option_ids.clone();
    candidate_ids.sort();
    current_ids.sort();
    candidate_ids < current_ids
}

struct PortfolioSearch<'a> {
    problem: &'a CampaignProblem,
    groups: &'a [DecisionGroup],
    bundles: &'a BTreeMap<String, DeploymentBundle>,
    best: Option<PartialSelection>,
    evaluated_nodes: u64,
    hit_node_limit: bool,
}

impl PortfolioSearch<'_> {
    fn visit(&mut self, mut state: PartialSelection) -> Result<(), CampaignError> {
        if self.evaluated_nodes >= self.problem.maximum_search_nodes {
            self.hit_node_limit = true;
            return Ok(());
        }
        self.evaluated_nodes += 1;

        if state.index == self.groups.len() {
            if self.best.as_ref().is_none_or(|best| better(&state, best))
            {
                self.best = Some(state);
            }
            return Ok(());
        }

        if let Some(best) = &self.best {
            let optimistic_value = self.groups[state.index..]
                .iter()
                .map(|group| {
                    group
                        .options
                        .iter()
                        .map(|option| option.gross_evsi_micro_units)
                        .max()
                        .unwrap_or(0)
                })
                .fold(u128::from(state.gross_evsi_micro_units), |total, value| {
                    total + u128::from(value)
                });
            if optimistic_value < u128::from(best.gross_evsi_micro_units) {
                return Ok(());
            }
        }

        let group = self.groups[state.index].clone();
        for option in &group.options {
            if let Some(mut next) = apply_option(self.problem, self.bundles, &state, option)? {
                next.index += 1;
                self.visit(next)?;
                if self.hit_node_limit {
                    return Ok(());
                }
            }
        }

        if !group.mandatory {
            state.index += 1;
            self.visit(state)?;
        }
        Ok(())
    }
}

pub fn optimize_campaign(problem: &CampaignProblem) -> Result<CampaignReport, CampaignError> {
    let prepared = prepare(problem)?;
    let mut search = PortfolioSearch {
        problem,
        groups: &prepared.groups,
        bundles: &prepared.bundles,
        best: None,
        evaluated_nodes: 0,
        hit_node_limit: false,
    };
    search.visit(PartialSelection::empty())?;

    let Some(selection) = search.best else {
        return Err(if search.hit_node_limit {
            CampaignError::SearchLimitWithoutFeasibleIncumbent
        } else {
            CampaignError::NoFeasibleCampaign
        });
    };
    let search_disposition = if search.hit_node_limit {
        SearchDisposition::NodeLimitReached
    } else {
        SearchDisposition::OptimalityProven
    };

    Ok(CampaignReport {
        campaign_id: problem.campaign_id.clone(),
        selection: selection.into_selection(),
        search_disposition,
        evaluated_nodes: search.evaluated_nodes,
        assurance_disposition: problem.assurance.disposition(),
    })
}

fn compare_greedy_options(
    a: &InspectionOption,
    b: &InspectionOption,
    metric: GreedyMetric,
) -> Ordering {
    let score = match metric {
        GreedyMetric::GrossEvsi => b.gross_evsi_micro_units.cmp(&a.gross_evsi_micro_units),
        GreedyMetric::EvsiPerEnergy => {
            let a_cost = u128::from(a.variable_resources.energy_millijoules.max(1));
            let b_cost = u128::from(b.variable_resources.energy_millijoules.max(1));
            (u128::from(b.gross_evsi_micro_units) * a_cost)
                .cmp(&(u128::from(a.gross_evsi_micro_units) * b_cost))
        }
        GreedyMetric::EvsiPerDuration => {
            let a_cost = u128::from(a.variable_resources.duration_milliseconds.max(1));
            let b_cost = u128::from(b.variable_resources.duration_milliseconds.max(1));
            (u128::from(b.gross_evsi_micro_units) * a_cost)
                .cmp(&(u128::from(a.gross_evsi_micro_units) * b_cost))
        }
    };
    score.then_with(|| a.option_id.cmp(&b.option_id))
}

pub fn greedy_baseline(
    problem: &CampaignProblem,
    metric: GreedyMetric,
) -> Result<GreedyReport, CampaignError> {
    let prepared = prepare(problem)?;
    let mut options = prepared
        .groups
        .iter()
        .flat_map(|group| group.options.iter().cloned())
        .collect::<Vec<_>>();
    options.sort_by(|a, b| compare_greedy_options(a, b, metric));

    let mut state = PartialSelection::empty();
    let mut covered = BTreeSet::new();
    for option in &options {
        if covered.contains(&option.decision_id) {
            continue;
        }
        if let Some(next) = apply_option(problem, &prepared.bundles, &state, option)? {
            state = next;
            covered.insert(option.decision_id.clone());
        }
    }

    if prepared
        .groups
        .iter()
        .any(|group| group.mandatory && !covered.contains(&group.decision_id))
    {
        return Err(CampaignError::NoFeasibleCampaign);
    }

    Ok(GreedyReport {
        metric,
        selection: state.into_selection(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assurance() -> AssuranceBindings {
        AssuranceBindings {
            common_cause_graph_digest: "graph:synthetic".into(),
            fault_domain_profile_digest: "profile:synthetic".into(),
            common_cause_assessment_digest: "assessment:synthetic".into(),
            portfolio_value_assessment_digest: "factorization:synthetic".into(),
            inventory_disposition: InventoryDisposition::CompleteWithinDeclaredScope,
            additive_value_factorization_complete: true,
        }
    }

    fn option(id: &str, decision: &str, bundle: &str, value: u64, energy: u64) -> InspectionOption {
        InspectionOption {
            option_id: id.into(),
            decision_id: decision.into(),
            deployment_bundle_id: bundle.into(),
            gross_evsi_micro_units: value,
            unknown_state_debt_reduction_units: value / 2,
            variable_resources: ResourceVector {
                energy_millijoules: energy,
                duration_milliseconds: 1,
                data_bytes: 1,
                exposure_milliunits: 1,
            },
            mission_interruption_upper_ppm: 0,
            robot_loss_upper_ppm: 0,
            external_gate_passed: true,
        }
    }

    fn problem(
        decisions: Vec<DecisionRequirement>,
        bundles: Vec<DeploymentBundle>,
        options: Vec<InspectionOption>,
        energy_budget: u64,
    ) -> CampaignProblem {
        CampaignProblem {
            campaign_id: "synthetic-campaign".into(),
            decisions,
            deployment_bundles: bundles,
            options,
            budget: CampaignBudget {
                resources: ResourceVector {
                    energy_millijoules: energy_budget,
                    duration_milliseconds: 100,
                    data_bytes: 100,
                    exposure_milliunits: 100,
                },
                mission_interruption_upper_ppm: 100_000,
                robot_loss_upper_ppm: 100_000,
            },
            assurance: assurance(),
            maximum_search_nodes: 100_000,
        }
    }

    fn empty_bundle(id: &str) -> DeploymentBundle {
        DeploymentBundle {
            bundle_id: id.into(),
            fixed_resources: ResourceVector::default(),
        }
    }

    #[test]
    fn exact_campaign_beats_gross_evsi_greedy_under_the_same_budget() {
        let p = problem(
            vec![
                DecisionRequirement {
                    decision_id: "asset_a".into(),
                    mandatory: false,
                },
                DecisionRequirement {
                    decision_id: "asset_b".into(),
                    mandatory: false,
                },
            ],
            vec![empty_bundle("deploy")],
            vec![
                option("a_expensive", "asset_a", "deploy", 9, 9),
                option("a_efficient", "asset_a", "deploy", 6, 4),
                option("b_value", "asset_b", "deploy", 8, 6),
            ],
            10,
        );
        let exact = optimize_campaign(&p).unwrap();
        let greedy = greedy_baseline(&p, GreedyMetric::GrossEvsi).unwrap();
        assert_eq!(
            exact.selection.selected_option_ids,
            vec!["a_efficient", "b_value"]
        );
        assert_eq!(exact.selection.additive_evsi_estimate_micro_units, 14);
        assert_eq!(greedy.selection.additive_evsi_estimate_micro_units, 9);
        assert_eq!(
            exact.search_disposition,
            SearchDisposition::OptimalityProven
        );
    }

    #[test]
    fn shared_deployment_bundle_is_charged_exactly_once() {
        let bundle = DeploymentBundle {
            bundle_id: "corridor_west".into(),
            fixed_resources: ResourceVector {
                energy_millijoules: 10,
                duration_milliseconds: 10,
                data_bytes: 10,
                exposure_milliunits: 10,
            },
        };
        let p = problem(
            vec![
                DecisionRequirement {
                    decision_id: "pipe_1".into(),
                    mandatory: true,
                },
                DecisionRequirement {
                    decision_id: "cable_1".into(),
                    mandatory: true,
                },
            ],
            vec![bundle],
            vec![
                option("pipe_scan", "pipe_1", "corridor_west", 5, 10),
                option("cable_scan", "cable_1", "corridor_west", 5, 10),
            ],
            30,
        );
        let report = optimize_campaign(&p).unwrap();
        assert_eq!(report.selection.resources.energy_millijoules, 30);
        assert_eq!(report.selection.selected_option_ids.len(), 2);
    }

    #[test]
    fn risk_bounds_use_a_conservative_union_bound() {
        let mut a = option("a", "asset_a", "deploy", 5, 1);
        let mut b = option("b", "asset_b", "deploy", 5, 1);
        a.robot_loss_upper_ppm = 30_000;
        b.robot_loss_upper_ppm = 30_000;
        let mut p = problem(
            vec![
                DecisionRequirement {
                    decision_id: "asset_a".into(),
                    mandatory: true,
                },
                DecisionRequirement {
                    decision_id: "asset_b".into(),
                    mandatory: true,
                },
            ],
            vec![empty_bundle("deploy")],
            vec![a, b],
            10,
        );
        p.budget.robot_loss_upper_ppm = 50_000;
        assert_eq!(
            optimize_campaign(&p),
            Err(CampaignError::NoFeasibleCampaign)
        );
        p.budget.robot_loss_upper_ppm = 60_000;
        let report = optimize_campaign(&p).unwrap();
        assert_eq!(report.selection.robot_loss_upper_ppm, 60_000);
    }

    #[test]
    fn unknown_inventory_cannot_be_assurance_eligible() {
        let mut p = problem(
            vec![DecisionRequirement {
                decision_id: "asset".into(),
                mandatory: true,
            }],
            vec![empty_bundle("deploy")],
            vec![option("scan", "asset", "deploy", 3, 1)],
            5,
        );
        p.assurance.inventory_disposition = InventoryDisposition::Unknown;
        let report = optimize_campaign(&p).unwrap();
        assert_eq!(
            report.assurance_disposition,
            AssuranceDisposition::ResearchOnlyInsufficientEvidence
        );
    }

    #[test]
    fn additive_evsi_requires_a_pinned_factorization_assessment() {
        let mut p = problem(
            vec![DecisionRequirement {
                decision_id: "asset".into(),
                mandatory: true,
            }],
            vec![empty_bundle("deploy")],
            vec![option("scan", "asset", "deploy", 3, 1)],
            5,
        );
        p.assurance.additive_value_factorization_complete = false;
        p.assurance.portfolio_value_assessment_digest.clear();
        let report = optimize_campaign(&p).unwrap();
        assert_eq!(
            report.assurance_disposition,
            AssuranceDisposition::ResearchOnlyInsufficientEvidence
        );
    }

    #[test]
    fn mandatory_decisions_cannot_be_silently_omitted() {
        let p = problem(
            vec![DecisionRequirement {
                decision_id: "critical_support".into(),
                mandatory: true,
            }],
            vec![empty_bundle("deploy")],
            vec![option("verify", "critical_support", "deploy", 1, 10)],
            9,
        );
        assert_eq!(
            optimize_campaign(&p),
            Err(CampaignError::NoFeasibleCampaign)
        );
        assert_eq!(
            greedy_baseline(&p, GreedyMetric::GrossEvsi),
            Err(CampaignError::NoFeasibleCampaign)
        );
    }

    #[test]
    fn node_limit_reports_an_incumbent_without_claiming_optimality() {
        let mut p = problem(
            vec![DecisionRequirement {
                decision_id: "optional".into(),
                mandatory: false,
            }],
            vec![empty_bundle("deploy")],
            vec![option("scan", "optional", "deploy", 1, 1)],
            5,
        );
        p.maximum_search_nodes = 2;
        let report = optimize_campaign(&p).unwrap();
        assert_eq!(
            report.search_disposition,
            SearchDisposition::NodeLimitReached
        );
        assert!(!report.selection.selected_option_ids.is_empty());
    }

    #[test]
    fn duplicate_options_are_rejected() {
        let mut p = problem(
            vec![DecisionRequirement {
                decision_id: "asset".into(),
                mandatory: false,
            }],
            vec![empty_bundle("deploy")],
            vec![
                option("duplicate", "asset", "deploy", 1, 1),
                option("duplicate", "asset", "deploy", 2, 1),
            ],
            5,
        );
        p.maximum_search_nodes = 100;
        assert_eq!(
            optimize_campaign(&p),
            Err(CampaignError::DuplicateIdentifier)
        );
    }

    #[test]
    fn gate_rejected_option_is_not_selected() {
        let mut rejected = option("unsafe", "asset", "deploy", 100, 1);
        rejected.external_gate_passed = false;
        let p = problem(
            vec![DecisionRequirement {
                decision_id: "asset".into(),
                mandatory: false,
            }],
            vec![empty_bundle("deploy")],
            vec![rejected, option("eligible", "asset", "deploy", 2, 1)],
            5,
        );
        let report = optimize_campaign(&p).unwrap();
        assert_eq!(report.selection.selected_option_ids, vec!["eligible"]);
    }

    #[test]
    fn missing_reference_and_invalid_risk_fail_closed() {
        let mut p = problem(
            vec![DecisionRequirement {
                decision_id: "asset".into(),
                mandatory: true,
            }],
            vec![empty_bundle("deploy")],
            vec![option("scan", "asset", "deploy", 1, 1)],
            5,
        );
        p.options[0].deployment_bundle_id = "missing".into();
        assert_eq!(
            optimize_campaign(&p),
            Err(CampaignError::MissingDeploymentBundle)
        );
        p.options[0].deployment_bundle_id = "deploy".into();
        p.options[0].robot_loss_upper_ppm = MAX_RISK_PPM + 1;
        assert_eq!(optimize_campaign(&p), Err(CampaignError::InvalidRiskBound));
    }

    #[test]
    fn missing_assurance_digests_only_downgrade_the_claim() {
        let mut p = problem(
            vec![DecisionRequirement {
                decision_id: "asset".into(),
                mandatory: true,
            }],
            vec![empty_bundle("deploy")],
            vec![option("scan", "asset", "deploy", 1, 1)],
            5,
        );
        p.assurance.common_cause_assessment_digest.clear();
        let report = optimize_campaign(&p).unwrap();
        assert_eq!(
            report.assurance_disposition,
            AssuranceDisposition::ResearchOnlyInsufficientEvidence
        );
        assert_eq!(
            report.search_disposition,
            SearchDisposition::OptimalityProven
        );
    }
}
