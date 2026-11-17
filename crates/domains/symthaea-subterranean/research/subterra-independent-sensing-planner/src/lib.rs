//! Bounded exact planning for fault-domain-diverse inspection acquisition portfolios.
//!
//! This module consumes a transitive dependency closure from Mycelix. It does
//! not infer independence from modality/vendor labels or build a duplicate graph.
//! The objective is a Pareto frontier over declared physical resource and risk
//! dimensions; no arbitrary scalarization or additive per-sensor EVSI is used.

use std::collections::{BTreeMap, BTreeSet};

use subterra_corroboration_assurance::{DependencyIdentity, InventoryDisposition};
use subterra_inspection_campaign::ResourceVector;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedAcquisitionOption {
    pub option_id: String,
    pub source_id: String,
    pub claim_id: String,
    pub asset_id: String,
    pub asset_revision: String,
    pub coordinate_frame_digest: String,
    pub region_scope_digest: String,
    pub deployment_bundle_id: String,
    pub deployment_bundle_digest: String,
    pub deployment_bundle_resource_model_digest: String,
    pub deployment_bundle_assessment_digest: String,
    pub dependency_graph_digest: String,
    pub fault_domain_profile_digest: String,
    pub dependency_assessment_digest: String,
    pub inventory: InventoryDisposition,
    /// Transitive, claim-critical dependencies from Mycelix LEX-NET-030.
    pub claim_critical_dependencies: Vec<DependencyIdentity>,
    pub resource_model_digest: String,
    pub resource_assessment_digest: String,
    pub risk_model_digest: String,
    pub risk_assessment_digest: String,
    pub resource_assessed_at_unix_ms: u64,
    pub resource_assessment_valid_until_unix_ms: u64,
    pub risk_assessed_at_unix_ms: u64,
    pub risk_assessment_valid_until_unix_ms: u64,
    pub resources: ResourceVector,
    pub mission_interruption_upper_ppm: u32,
    pub robot_loss_upper_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentBundleCost {
    pub bundle_id: String,
    pub bundle_digest: String,
    pub resource_model_digest: String,
    pub assessment_digest: String,
    pub assessed_at_unix_ms: u64,
    pub valid_until_unix_ms: u64,
    /// Fixed mobilisation/setup cost, charged once per distinct bundle in a portfolio.
    pub fixed_resources: ResourceVector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedDeploymentBundle {
    pub bundle_id: String,
    pub bundle_digest: String,
    pub assessment_digest: String,
    pub fixed_resources: ResourceVector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierDependencyClosure {
    pub verifier_id: String,
    pub inventory: InventoryDisposition,
    pub claim_critical_dependencies: Vec<DependencyIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedAcquisitionContext {
    pub claim_id: String,
    pub asset_id: String,
    pub asset_revision: String,
    pub coordinate_frame_digest: String,
    pub region_scope_digest: String,
    pub dependency_graph_digest: String,
    pub fault_domain_profile_digest: String,
    pub dependency_assessment_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcquisitionBudget {
    pub resources: ResourceVector,
    pub mission_interruption_upper_ppm: u32,
    pub robot_loss_upper_ppm: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcquisitionPolicy {
    /// Exactly this many mutually disjoint acquisition sources are selected.
    pub required_independent_sources: usize,
    pub evaluation_time_unix_ms: u64,
    pub maximum_resource_assessment_age_ms: u64,
    pub maximum_risk_assessment_age_ms: u64,
    pub maximum_future_clock_skew_ms: u64,
    pub maximum_search_nodes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RejectedOptionReason {
    EmptyOptionId,
    EmptySourceId,
    EmptyDigest,
    ClaimMismatch,
    AssetMismatch,
    AssetRevisionMismatch,
    CoordinateFrameMismatch,
    RegionScopeMismatch,
    DependencyGraphMismatch,
    FaultDomainProfileMismatch,
    DependencyAssessmentMismatch,
    InventoryIncomplete,
    InventoryUnknown,
    MissingDeploymentBundleCost,
    DeploymentBundleDigestMismatch,
    DeploymentBundleAssessmentFromFuture,
    DeploymentBundleAssessmentStale,
    DeploymentBundleAssessmentExpired,
    EmptyDependencyClosure,
    EmptyDependencyIdentity,
    VerifierOverlap,
    ResourceAssessmentFromFuture,
    ResourceAssessmentExpired,
    ResourceAssessmentStale,
    RiskAssessmentFromFuture,
    RiskAssessmentExpired,
    RiskAssessmentStale,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedAcquisitionOption {
    pub option_id: String,
    pub reasons: Vec<RejectedOptionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceConflict {
    pub left_option_id: String,
    pub right_option_id: String,
    pub shared_claim_critical_dependencies: Vec<DependencyIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcquisitionPortfolio {
    pub selected_option_ids: Vec<String>,
    pub selected_source_ids: Vec<String>,
    pub selected_deployment_bundles: Vec<SelectedDeploymentBundle>,
    /// Sum of per-source variable acquisition costs.
    pub variable_resources: ResourceVector,
    /// Sum of each unique selected deployment bundle's fixed resources, once.
    pub shared_fixed_resources: ResourceVector,
    /// Total = variable_resources + shared_fixed_resources.
    pub resources: ResourceVector,
    /// Conservative union bound; no independence assumption is used.
    pub mission_interruption_upper_ppm: u32,
    /// Conservative union bound; no independence assumption is used.
    pub robot_loss_upper_ppm: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDisposition {
    ExactParetoFrontierProven,
    NodeLimitReached,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndependentSensingPlanReport {
    pub pareto_frontier: Vec<AcquisitionPortfolio>,
    pub rejected_options: Vec<RejectedAcquisitionOption>,
    pub source_conflicts: Vec<SourceConflict>,
    pub explored_nodes: u64,
    pub disposition: SearchDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanningError {
    EmptyRequiredSourceCount,
    DuplicateOptionId(String),
    DuplicateSourceId(String),
    DuplicateDeploymentBundleId(String),
    ConflictingDeploymentBundleDigest(String),
    EmptyVerifierIdentity,
    IncompleteVerifierInventory,
    UnknownVerifierInventory,
    EmptyVerifierDependencyClosure,
    EmptyExpectedContext,
    InvalidSearchNodeLimit,
}

fn missing(value: &str) -> bool {
    value.trim().is_empty()
}

fn dependency_set(
    dependencies: &[DependencyIdentity],
) -> Result<BTreeSet<DependencyIdentity>, RejectedOptionReason> {
    if dependencies.is_empty() {
        return Err(RejectedOptionReason::EmptyDependencyClosure);
    }
    let mut result = BTreeSet::new();
    for dependency in dependencies {
        if missing(&dependency.dimension) || missing(&dependency.identity) {
            return Err(RejectedOptionReason::EmptyDependencyIdentity);
        }
        result.insert(dependency.clone());
    }
    Ok(result)
}

fn checked_resources_add(left: &ResourceVector, right: &ResourceVector) -> Option<ResourceVector> {
    Some(ResourceVector {
        energy_millijoules: left.energy_millijoules.checked_add(right.energy_millijoules)?,
        duration_milliseconds: left
            .duration_milliseconds
            .checked_add(right.duration_milliseconds)?,
        data_bytes: left.data_bytes.checked_add(right.data_bytes)?,
        exposure_milliunits: left.exposure_milliunits.checked_add(right.exposure_milliunits)?,
    })
}

fn resources_within(value: &ResourceVector, budget: &ResourceVector) -> bool {
    value.energy_millijoules <= budget.energy_millijoules
        && value.duration_milliseconds <= budget.duration_milliseconds
        && value.data_bytes <= budget.data_bytes
        && value.exposure_milliunits <= budget.exposure_milliunits
}

fn resources_dominate(left: &ResourceVector, right: &ResourceVector) -> bool {
    left.energy_millijoules <= right.energy_millijoules
        && left.duration_milliseconds <= right.duration_milliseconds
        && left.data_bytes <= right.data_bytes
        && left.exposure_milliunits <= right.exposure_milliunits
}

fn portfolio_dominates(left: &AcquisitionPortfolio, right: &AcquisitionPortfolio) -> bool {
    let no_worse = resources_dominate(&left.resources, &right.resources)
        && left.mission_interruption_upper_ppm <= right.mission_interruption_upper_ppm
        && left.robot_loss_upper_ppm <= right.robot_loss_upper_ppm;
    let strictly_better = left.resources != right.resources
        || left.mission_interruption_upper_ppm != right.mission_interruption_upper_ppm
        || left.robot_loss_upper_ppm != right.robot_loss_upper_ppm;
    no_worse && strictly_better
}

fn option_rejection_reasons(
    option: &PlannedAcquisitionOption,
    bundle_cost: Option<&DeploymentBundleCost>,
    expected: &ExpectedAcquisitionContext,
    policy: AcquisitionPolicy,
    verifier_dependencies: &BTreeSet<DependencyIdentity>,
) -> Vec<RejectedOptionReason> {
    let mut reasons = Vec::new();
    if missing(&option.option_id) {
        reasons.push(RejectedOptionReason::EmptyOptionId);
    }
    if missing(&option.source_id) {
        reasons.push(RejectedOptionReason::EmptySourceId);
    }
    match bundle_cost {
        None => reasons.push(RejectedOptionReason::MissingDeploymentBundleCost),
        Some(bundle) => {
            if bundle.bundle_id != option.deployment_bundle_id
                || bundle.bundle_digest != option.deployment_bundle_digest
                || bundle.resource_model_digest != option.deployment_bundle_resource_model_digest
                || bundle.assessment_digest != option.deployment_bundle_assessment_digest
            {
                reasons.push(RejectedOptionReason::DeploymentBundleDigestMismatch);
            }
            if [
                bundle.bundle_id.as_str(),
                bundle.bundle_digest.as_str(),
                bundle.resource_model_digest.as_str(),
                bundle.assessment_digest.as_str(),
            ].iter().any(|value| missing(value)) {
                reasons.push(RejectedOptionReason::EmptyDigest);
            }
            let latest_allowed = policy
                .evaluation_time_unix_ms
                .saturating_add(policy.maximum_future_clock_skew_ms);
            if bundle.assessed_at_unix_ms > latest_allowed {
                reasons.push(RejectedOptionReason::DeploymentBundleAssessmentFromFuture);
            } else if policy
                .evaluation_time_unix_ms
                .saturating_sub(bundle.assessed_at_unix_ms)
                > policy.maximum_resource_assessment_age_ms
            {
                reasons.push(RejectedOptionReason::DeploymentBundleAssessmentStale);
            }
            if bundle.valid_until_unix_ms <= policy.evaluation_time_unix_ms {
                reasons.push(RejectedOptionReason::DeploymentBundleAssessmentExpired);
            }
        }
    }
    if [
        option.deployment_bundle_id.as_str(),
        option.deployment_bundle_digest.as_str(),
        option.dependency_graph_digest.as_str(),
        option.fault_domain_profile_digest.as_str(),
        option.dependency_assessment_digest.as_str(),
        option.resource_model_digest.as_str(),
        option.resource_assessment_digest.as_str(),
        option.risk_model_digest.as_str(),
        option.risk_assessment_digest.as_str(),
    ]
    .iter()
    .any(|value| missing(value))
    {
        reasons.push(RejectedOptionReason::EmptyDigest);
    }
    if option.claim_id != expected.claim_id {
        reasons.push(RejectedOptionReason::ClaimMismatch);
    }
    if option.asset_id != expected.asset_id {
        reasons.push(RejectedOptionReason::AssetMismatch);
    }
    if option.asset_revision != expected.asset_revision {
        reasons.push(RejectedOptionReason::AssetRevisionMismatch);
    }
    if option.coordinate_frame_digest != expected.coordinate_frame_digest {
        reasons.push(RejectedOptionReason::CoordinateFrameMismatch);
    }
    if option.region_scope_digest != expected.region_scope_digest {
        reasons.push(RejectedOptionReason::RegionScopeMismatch);
    }
    if option.dependency_graph_digest != expected.dependency_graph_digest {
        reasons.push(RejectedOptionReason::DependencyGraphMismatch);
    }
    if option.fault_domain_profile_digest != expected.fault_domain_profile_digest {
        reasons.push(RejectedOptionReason::FaultDomainProfileMismatch);
    }
    if option.dependency_assessment_digest != expected.dependency_assessment_digest {
        reasons.push(RejectedOptionReason::DependencyAssessmentMismatch);
    }

    match option.inventory {
        InventoryDisposition::CompleteWithinDeclaredScope => {}
        InventoryDisposition::Incomplete => reasons.push(RejectedOptionReason::InventoryIncomplete),
        InventoryDisposition::Unknown => reasons.push(RejectedOptionReason::InventoryUnknown),
    }
    match dependency_set(&option.claim_critical_dependencies) {
        Ok(dependencies) => {
            if !dependencies.is_disjoint(verifier_dependencies) {
                reasons.push(RejectedOptionReason::VerifierOverlap);
            }
        }
        Err(reason) => reasons.push(reason),
    }

    if option.resource_assessment_valid_until_unix_ms <= policy.evaluation_time_unix_ms {
        reasons.push(RejectedOptionReason::ResourceAssessmentExpired);
    }
    let latest_allowed = policy
        .evaluation_time_unix_ms
        .saturating_add(policy.maximum_future_clock_skew_ms);
    if option.resource_assessed_at_unix_ms > latest_allowed {
        reasons.push(RejectedOptionReason::ResourceAssessmentFromFuture);
    } else if policy
        .evaluation_time_unix_ms
        .saturating_sub(option.resource_assessed_at_unix_ms)
        > policy.maximum_resource_assessment_age_ms
    {
        reasons.push(RejectedOptionReason::ResourceAssessmentStale);
    }
    if option.risk_assessed_at_unix_ms > latest_allowed {
        reasons.push(RejectedOptionReason::RiskAssessmentFromFuture);
    } else if policy
        .evaluation_time_unix_ms
        .saturating_sub(option.risk_assessed_at_unix_ms)
        > policy.maximum_risk_assessment_age_ms
    {
        reasons.push(RejectedOptionReason::RiskAssessmentStale);
    }
    if option.risk_assessment_valid_until_unix_ms <= policy.evaluation_time_unix_ms {
        reasons.push(RejectedOptionReason::RiskAssessmentExpired);
    }

    reasons.sort();
    reasons.dedup();
    reasons
}

struct SearchState<'a> {
    options: &'a [PlannedAcquisitionOption],
    dependency_sets: &'a [BTreeSet<DependencyIdentity>],
    option_bundle_indices: &'a [usize],
    bundle_costs: &'a [DeploymentBundleCost],
    budget: &'a AcquisitionBudget,
    required_count: usize,
    maximum_nodes: u64,
    explored_nodes: u64,
    limited: bool,
    portfolios: Vec<AcquisitionPortfolio>,
}

impl SearchState<'_> {
    fn visit(
        &mut self,
        start: usize,
        selected: &mut Vec<usize>,
        variable_resources: ResourceVector,
        shared_fixed_resources: ResourceVector,
        resources: ResourceVector,
        interruption: u32,
        robot_loss: u32,
        selected_bundles: &mut BTreeSet<usize>,
    ) {
        if self.limited {
            return;
        }
        if self.explored_nodes >= self.maximum_nodes {
            self.limited = true;
            return;
        }
        self.explored_nodes += 1;

        if selected.len() == self.required_count {
            self.portfolios.push(AcquisitionPortfolio {
                selected_option_ids: selected
                    .iter()
                    .map(|index| self.options[*index].option_id.clone())
                    .collect(),
                selected_source_ids: selected
                    .iter()
                    .map(|index| self.options[*index].source_id.clone())
                    .collect(),
                selected_deployment_bundles: selected_bundles
                    .iter()
                    .map(|index| {
                        let bundle = &self.bundle_costs[*index];
                        SelectedDeploymentBundle {
                            bundle_id: bundle.bundle_id.clone(),
                            bundle_digest: bundle.bundle_digest.clone(),
                            assessment_digest: bundle.assessment_digest.clone(),
                            fixed_resources: bundle.fixed_resources.clone(),
                        }
                    })
                    .collect(),
                variable_resources,
                shared_fixed_resources,
                resources,
                mission_interruption_upper_ppm: interruption,
                robot_loss_upper_ppm: robot_loss,
            });
            return;
        }

        let remaining_needed = self.required_count - selected.len();
        if self.options.len().saturating_sub(start) < remaining_needed {
            return;
        }

        let last_start = self.options.len() - remaining_needed;
        for index in start..=last_start {
            if self.limited {
                break;
            }
            if selected
                .iter()
                .any(|selected_index| {
                    !self.dependency_sets[index].is_disjoint(&self.dependency_sets[*selected_index])
                })
            {
                continue;
            }
            let bundle_index = self.option_bundle_indices[index];
            let is_new_bundle = !selected_bundles.contains(&bundle_index);
            let Some(next_variable_resources) =
                checked_resources_add(&variable_resources, &self.options[index].resources)
            else {
                continue;
            };
            let next_shared_fixed_resources = if is_new_bundle {
                let Some(value) = checked_resources_add(
                    &shared_fixed_resources,
                    &self.bundle_costs[bundle_index].fixed_resources,
                ) else {
                    continue;
                };
                value
            } else {
                shared_fixed_resources.clone()
            };
            let Some(next_resources) = checked_resources_add(
                &next_variable_resources,
                &next_shared_fixed_resources,
            ) else {
                continue;
            };
            let Some(next_interruption) =
                interruption.checked_add(self.options[index].mission_interruption_upper_ppm)
            else {
                continue;
            };
            let Some(next_robot_loss) =
                robot_loss.checked_add(self.options[index].robot_loss_upper_ppm)
            else {
                continue;
            };
            if !resources_within(&next_resources, &self.budget.resources)
                || next_interruption > self.budget.mission_interruption_upper_ppm
                || next_robot_loss > self.budget.robot_loss_upper_ppm
            {
                continue;
            }
            selected.push(index);
            if is_new_bundle {
                selected_bundles.insert(bundle_index);
            }
            self.visit(
                index + 1,
                selected,
                next_variable_resources,
                next_shared_fixed_resources,
                next_resources,
                next_interruption,
                next_robot_loss,
                selected_bundles,
            );
            if is_new_bundle {
                selected_bundles.remove(&bundle_index);
            }
            selected.pop();
        }
    }
}

pub fn plan_independent_acquisition_portfolio(
    mut options: Vec<PlannedAcquisitionOption>,
    deployment_bundles: &[DeploymentBundleCost],
    expected: &ExpectedAcquisitionContext,
    verifier: &VerifierDependencyClosure,
    policy: AcquisitionPolicy,
    budget: &AcquisitionBudget,
) -> Result<IndependentSensingPlanReport, PlanningError> {
    if policy.required_independent_sources == 0 {
        return Err(PlanningError::EmptyRequiredSourceCount);
    }
    if policy.maximum_search_nodes == 0 {
        return Err(PlanningError::InvalidSearchNodeLimit);
    }
    if [
        expected.claim_id.as_str(),
        expected.asset_id.as_str(),
        expected.asset_revision.as_str(),
        expected.coordinate_frame_digest.as_str(),
        expected.region_scope_digest.as_str(),
        expected.dependency_graph_digest.as_str(),
        expected.fault_domain_profile_digest.as_str(),
        expected.dependency_assessment_digest.as_str(),
    ]
    .iter()
    .any(|value| missing(value))
    {
        return Err(PlanningError::EmptyExpectedContext);
    }
    if missing(&verifier.verifier_id) {
        return Err(PlanningError::EmptyVerifierIdentity);
    }
    match verifier.inventory {
        InventoryDisposition::CompleteWithinDeclaredScope => {}
        InventoryDisposition::Incomplete => return Err(PlanningError::IncompleteVerifierInventory),
        InventoryDisposition::Unknown => return Err(PlanningError::UnknownVerifierInventory),
    }
    let verifier_dependencies = dependency_set(&verifier.claim_critical_dependencies)
        .map_err(|_| PlanningError::EmptyVerifierDependencyClosure)?;

    options.sort_by(|a, b| a.option_id.cmp(&b.option_id));
    let mut bundle_indices = BTreeMap::new();
    for (index, bundle) in deployment_bundles.iter().enumerate() {
        if bundle_indices.insert(bundle.bundle_id.as_str(), index).is_some() {
            return Err(PlanningError::DuplicateDeploymentBundleId(bundle.bundle_id.clone()));
        }
    }
    let mut bundle_digests = BTreeMap::new();
    for option in &options {
        if let Some(existing) = bundle_digests.insert(
            option.deployment_bundle_id.as_str(),
            option.deployment_bundle_digest.as_str(),
        ) {
            if existing != option.deployment_bundle_digest {
                return Err(PlanningError::ConflictingDeploymentBundleDigest(
                    option.deployment_bundle_id.clone(),
                ));
            }
        }
    }
    let mut option_ids = BTreeSet::new();
    let mut source_ids = BTreeSet::new();
    for option in &options {
        if !option_ids.insert(option.option_id.as_str()) {
            return Err(PlanningError::DuplicateOptionId(option.option_id.clone()));
        }
        if !source_ids.insert(option.source_id.as_str()) {
            return Err(PlanningError::DuplicateSourceId(option.source_id.clone()));
        }
    }

    let mut rejected_options = Vec::new();
    let mut valid_options = Vec::new();
    let mut valid_dependency_sets = Vec::new();
    let mut valid_bundle_indices = Vec::new();
    for option in options {
        let bundle_cost = bundle_indices
            .get(option.deployment_bundle_id.as_str())
            .map(|index| &deployment_bundles[*index]);
        let reasons =
            option_rejection_reasons(&option, bundle_cost, expected, policy, &verifier_dependencies);
        if !reasons.is_empty() {
            rejected_options.push(RejectedAcquisitionOption {
                option_id: option.option_id,
                reasons,
            });
            continue;
        }
        let dependencies = dependency_set(&option.claim_critical_dependencies)
            .map_err(|_| PlanningError::EmptyVerifierDependencyClosure)?;
        valid_dependency_sets.push(dependencies);
        let bundle_index = *bundle_indices
            .get(option.deployment_bundle_id.as_str())
            .expect("validated bundle cost is present");
        valid_bundle_indices.push(bundle_index);
        valid_options.push(option);
    }

    let mut source_conflicts = Vec::new();
    for left in 0..valid_options.len() {
        for right in (left + 1)..valid_options.len() {
            let shared = valid_dependency_sets[left]
                .intersection(&valid_dependency_sets[right])
                .cloned()
                .collect::<Vec<_>>();
            if !shared.is_empty() {
                source_conflicts.push(SourceConflict {
                    left_option_id: valid_options[left].option_id.clone(),
                    right_option_id: valid_options[right].option_id.clone(),
                    shared_claim_critical_dependencies: shared,
                });
            }
        }
    }

    let mut search = SearchState {
        options: &valid_options,
        dependency_sets: &valid_dependency_sets,
        option_bundle_indices: &valid_bundle_indices,
        bundle_costs: deployment_bundles,
        budget,
        required_count: policy.required_independent_sources,
        maximum_nodes: policy.maximum_search_nodes,
        explored_nodes: 0,
        limited: false,
        portfolios: Vec::new(),
    };
    search.visit(
        0,
        &mut Vec::new(),
        ResourceVector::default(),
        ResourceVector::default(),
        ResourceVector::default(),
        0,
        0,
        &mut BTreeSet::new(),
    );

    let mut frontier = Vec::new();
    search.portfolios.sort_by(|a, b| {
        (
            a.resources.energy_millijoules,
            a.resources.duration_milliseconds,
            a.resources.data_bytes,
            a.resources.exposure_milliunits,
            a.mission_interruption_upper_ppm,
            a.robot_loss_upper_ppm,
            &a.selected_option_ids,
        )
            .cmp(&(
                b.resources.energy_millijoules,
                b.resources.duration_milliseconds,
                b.resources.data_bytes,
                b.resources.exposure_milliunits,
                b.mission_interruption_upper_ppm,
                b.robot_loss_upper_ppm,
                &b.selected_option_ids,
            ))
    });
    for portfolio in search.portfolios {
        if frontier
            .iter()
            .any(|existing| portfolio_dominates(existing, &portfolio))
        {
            continue;
        }
        frontier.retain(|existing| !portfolio_dominates(&portfolio, existing));
        frontier.push(portfolio);
    }
    frontier.sort_by(|a, b| a.selected_option_ids.cmp(&b.selected_option_ids));

    Ok(IndependentSensingPlanReport {
        pareto_frontier: frontier,
        rejected_options,
        source_conflicts,
        explored_nodes: search.explored_nodes,
        disposition: if search.limited {
            SearchDisposition::NodeLimitReached
        } else {
            SearchDisposition::ExactParetoFrontierProven
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    fn dep(dimension: &str, identity: &str) -> DependencyIdentity {
        DependencyIdentity {
            dimension: dimension.into(),
            identity: identity.into(),
        }
    }

    fn option(id: &str, source: &str, dependencies: Vec<DependencyIdentity>) -> PlannedAcquisitionOption {
        PlannedAcquisitionOption {
            option_id: id.into(),
            source_id: source.into(),
            claim_id: "claim-1".into(),
            asset_id: "asset-1".into(),
            asset_revision: "rev-1".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            deployment_bundle_id: "bundle-1".into(),
            deployment_bundle_digest: "sha256:bundle".into(),
            deployment_bundle_resource_model_digest: "sha256:bundle-resource-model".into(),
            deployment_bundle_assessment_digest: "sha256:bundle-assessment".into(),
            dependency_graph_digest: "sha256:graph".into(),
            fault_domain_profile_digest: "sha256:profile".into(),
            dependency_assessment_digest: "sha256:assessment".into(),
            inventory: InventoryDisposition::CompleteWithinDeclaredScope,
            claim_critical_dependencies: dependencies,
            resource_model_digest: "sha256:resource-model".into(),
            resource_assessment_digest: format!("sha256:resource-{id}"),
            risk_model_digest: "sha256:risk-model".into(),
            risk_assessment_digest: format!("sha256:risk-{id}"),
            resource_assessed_at_unix_ms: NOW - 100,
            resource_assessment_valid_until_unix_ms: NOW + 10_000,
            risk_assessed_at_unix_ms: NOW - 100,
            risk_assessment_valid_until_unix_ms: NOW + 10_000,
            resources: ResourceVector {
                energy_millijoules: 10,
                duration_milliseconds: 10,
                data_bytes: 10,
                exposure_milliunits: 1,
            },
            mission_interruption_upper_ppm: 10,
            robot_loss_upper_ppm: 10,
        }
    }

    fn bundle_costs() -> Vec<DeploymentBundleCost> {
        vec![DeploymentBundleCost {
            bundle_id: "bundle-1".into(),
            bundle_digest: "sha256:bundle".into(),
            resource_model_digest: "sha256:bundle-resource-model".into(),
            assessment_digest: "sha256:bundle-assessment".into(),
            assessed_at_unix_ms: NOW - 100,
            valid_until_unix_ms: NOW + 10_000,
            fixed_resources: ResourceVector {
                energy_millijoules: 5,
                duration_milliseconds: 2,
                data_bytes: 3,
                exposure_milliunits: 1,
            },
        }]
    }

    fn expected() -> ExpectedAcquisitionContext {
        ExpectedAcquisitionContext {
            claim_id: "claim-1".into(),
            asset_id: "asset-1".into(),
            asset_revision: "rev-1".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            dependency_graph_digest: "sha256:graph".into(),
            fault_domain_profile_digest: "sha256:profile".into(),
            dependency_assessment_digest: "sha256:assessment".into(),
        }
    }

    fn verifier() -> VerifierDependencyClosure {
        VerifierDependencyClosure {
            verifier_id: "independent-verifier".into(),
            inventory: InventoryDisposition::CompleteWithinDeclaredScope,
            claim_critical_dependencies: vec![
                dep("compute", "verifier-host"),
                dep("clock", "verifier-clock"),
                dep("ground-truth", "sealed-reference"),
            ],
        }
    }

    fn policy() -> AcquisitionPolicy {
        AcquisitionPolicy {
            required_independent_sources: 2,
            evaluation_time_unix_ms: NOW,
            maximum_resource_assessment_age_ms: 3_600_000,
            maximum_risk_assessment_age_ms: 3_600_000,
            maximum_future_clock_skew_ms: 5_000,
            maximum_search_nodes: 100_000,
        }
    }

    fn budget() -> AcquisitionBudget {
        AcquisitionBudget {
            resources: ResourceVector {
                energy_millijoules: 100,
                duration_milliseconds: 100,
                data_bytes: 100,
                exposure_milliunits: 100,
            },
            mission_interruption_upper_ppm: 100,
            robot_loss_upper_ppm: 100,
        }
    }

    #[test]
    fn planner_selects_only_disjoint_source_paths() {
        let options = vec![
            option("rgb-a", "robot-a-rgb", vec![dep("robot", "a"), dep("power", "shared-a")]),
            option("depth-a", "robot-a-depth", vec![dep("robot", "a"), dep("power", "shared-a")]),
            option("thermal-b", "robot-b-thermal", vec![dep("robot", "b"), dep("power", "b")]),
        ];
        let report = plan_independent_acquisition_portfolio(
            options,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        )
        .unwrap();
        assert_eq!(report.disposition, SearchDisposition::ExactParetoFrontierProven);
        assert!(!report.pareto_frontier.is_empty());
        for portfolio in &report.pareto_frontier {
            assert_eq!(portfolio.selected_option_ids.len(), 2);
            assert!(!portfolio.selected_option_ids.contains(&"rgb-a".into())
                || !portfolio.selected_option_ids.contains(&"depth-a".into()));
        }
        assert!(report.source_conflicts.iter().any(|conflict| {
            conflict.left_option_id == "depth-a" && conflict.right_option_id == "rgb-a"
        }));
    }

    #[test]
    fn a_third_fault_domain_diverse_path_restores_feasibility() {
        let options = vec![
            option("rgb-a", "robot-a-rgb", vec![dep("robot", "a"), dep("power", "a")]),
            option("depth-a", "robot-a-depth", vec![dep("robot", "a"), dep("power", "a")]),
            option("lidar-b", "robot-b-lidar", vec![dep("robot", "b"), dep("power", "b")]),
            option("acoustic-c", "robot-c-acoustic", vec![dep("robot", "c"), dep("power", "c")]),
        ];
        let mut policy = policy();
        policy.required_independent_sources = 3;
        let report = plan_independent_acquisition_portfolio(
            options,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy,
            &budget(),
        )
        .unwrap();
        assert_eq!(report.disposition, SearchDisposition::ExactParetoFrontierProven);
        assert!(!report.pareto_frontier.is_empty());
        assert!(report.pareto_frontier.iter().all(|p| p.selected_option_ids.len() == 3));
    }

    #[test]
    fn verifier_overlap_rejects_individual_source() {
        let mut options = vec![
            option("a", "source-a", vec![dep("robot", "a"), dep("power", "a")]),
            option("b", "source-b", vec![dep("robot", "b"), dep("power", "b")]),
        ];
        options[0].claim_critical_dependencies.push(dep("compute", "verifier-host"));
        let report = plan_independent_acquisition_portfolio(
            options,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        ).unwrap();
        assert!(report.rejected_options.iter().any(|r| {
            r.option_id == "a" && r.reasons.contains(&RejectedOptionReason::VerifierOverlap)
        }));
        assert!(report.pareto_frontier.is_empty());
    }

    #[test]
    fn incomplete_or_unknown_source_inventories_are_excluded() {
        let mut unknown = option("unknown", "source-unknown", vec![dep("robot", "x")]);
        unknown.inventory = InventoryDisposition::Unknown;
        let mut incomplete = option("incomplete", "source-incomplete", vec![dep("robot", "y")]);
        incomplete.inventory = InventoryDisposition::Incomplete;
        let options = vec![
            unknown,
            incomplete,
            option("good-a", "source-a", vec![dep("robot", "a")]),
            option("good-b", "source-b", vec![dep("robot", "b")]),
        ];
        let report = plan_independent_acquisition_portfolio(
            options,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        ).unwrap();
        assert_eq!(report.rejected_options.len(), 2);
        assert!(!report.pareto_frontier.is_empty());
    }

    #[test]
    fn each_resource_dimension_and_union_bound_risk_is_hard() {
        let options = vec![
            option("energy-high", "source-a", vec![dep("robot", "a")]),
            option("other", "source-b", vec![dep("robot", "b")]),
        ];
        let mut budget = budget();
        budget.resources.energy_millijoules = 15;
        let report = plan_independent_acquisition_portfolio(
            options.clone(),
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget,
        ).unwrap();
        assert!(report.pareto_frontier.is_empty());

        let mut budget = budget();
        budget.mission_interruption_upper_ppm = 15;
        let report = plan_independent_acquisition_portfolio(
            options,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget,
        ).unwrap();
        assert!(report.pareto_frontier.is_empty());
    }

    #[test]
    fn stale_or_future_resource_and_risk_assessments_are_rejected() {
        let mut stale = option("stale", "source-stale", vec![dep("robot", "stale")]);
        stale.resource_assessed_at_unix_ms = NOW - 3_600_001;
        stale.risk_assessed_at_unix_ms = NOW - 3_600_001;
        let report = plan_independent_acquisition_portfolio(
            vec![stale],
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        ).unwrap();
        assert!(report.rejected_options[0]
            .reasons
            .contains(&RejectedOptionReason::ResourceAssessmentStale));
        assert!(report.rejected_options[0]
            .reasons
            .contains(&RejectedOptionReason::RiskAssessmentStale));

        let mut future = option("future", "source-future", vec![dep("robot", "future")]);
        future.resource_assessed_at_unix_ms = NOW + 5_001;
        future.risk_assessed_at_unix_ms = NOW + 5_001;
        let report = plan_independent_acquisition_portfolio(
            vec![future],
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        ).unwrap();
        assert!(report.rejected_options[0]
            .reasons
            .contains(&RejectedOptionReason::ResourceAssessmentFromFuture));
        assert!(report.rejected_options[0]
            .reasons
            .contains(&RejectedOptionReason::RiskAssessmentFromFuture));
    }

    #[test]
    fn node_limit_never_claims_exact_optimality() {
        let options = (0..8)
            .map(|index| {
                option(
                    &format!("option-{index}"),
                    &format!("source-{index}"),
                    vec![dep("robot", &format!("robot-{index}"))],
                )
            })
            .collect();
        let mut policy = policy();
        policy.required_independent_sources = 4;
        policy.maximum_search_nodes = 2;
        let report = plan_independent_acquisition_portfolio(
            options,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy,
            &budget(),
        ).unwrap();
        assert_eq!(report.disposition, SearchDisposition::NodeLimitReached);
    }

    #[test]
    fn input_order_does_not_change_pareto_frontier() {
        let options = vec![
            option("c", "source-c", vec![dep("robot", "c")]),
            option("a", "source-a", vec![dep("robot", "a")]),
            option("b", "source-b", vec![dep("robot", "b")]),
        ];
        let first = plan_independent_acquisition_portfolio(
            options.clone(),
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        ).unwrap();
        let mut reversed = options;
        reversed.reverse();
        let second = plan_independent_acquisition_portfolio(
            reversed,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        ).unwrap();
        assert_eq!(first.pareto_frontier, second.pareto_frontier);
        assert_eq!(first.source_conflicts, second.source_conflicts);
    }

    #[test]
    fn duplicate_candidate_or_source_ids_are_fatal() {
        let duplicate_option = vec![
            option("same", "source-a", vec![dep("robot", "a")]),
            option("same", "source-b", vec![dep("robot", "b")]),
        ];
        assert_eq!(
            plan_independent_acquisition_portfolio(
                duplicate_option,
                &bundle_costs(),
                &expected(),
                &verifier(),
                policy(),
                &budget()
            ),
            Err(PlanningError::DuplicateOptionId("same".into()))
        );

        let duplicate_source = vec![
            option("a", "same-source", vec![dep("robot", "a")]),
            option("b", "same-source", vec![dep("robot", "b")]),
        ];
        assert_eq!(
            plan_independent_acquisition_portfolio(
                duplicate_source,
                &bundle_costs(),
                &expected(),
                &verifier(),
                policy(),
                &budget()
            ),
            Err(PlanningError::DuplicateSourceId("same-source".into()))
        );
    }

    #[test]
    fn shared_deployment_bundle_fixed_cost_is_charged_once() {
        let options = vec![
            option("a", "source-a", vec![dep("robot", "a")]),
            option("b", "source-b", vec![dep("robot", "b")]),
        ];
        let mut budget = budget();
        budget.resources.energy_millijoules = 25;
        let report = plan_independent_acquisition_portfolio(
            options,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget,
        )
        .unwrap();
        assert_eq!(report.pareto_frontier.len(), 1);
        let portfolio = &report.pareto_frontier[0];
        assert_eq!(portfolio.variable_resources.energy_millijoules, 20);
        assert_eq!(portfolio.shared_fixed_resources.energy_millijoules, 5);
        assert_eq!(portfolio.resources.energy_millijoules, 25);
        assert_eq!(portfolio.selected_deployment_bundles.len(), 1);
    }

    #[test]
    fn same_bundle_cost_is_not_omitted_or_double_charged_for_multiple_sources() {
        let mut second = option("b", "source-b", vec![dep("robot", "b")]);
        second.deployment_bundle_id = "bundle-1".into();
        let options = vec![
            option("a", "source-a", vec![dep("robot", "a")]),
            second,
        ];
        let report = plan_independent_acquisition_portfolio(
            options,
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        )
        .unwrap();
        assert_eq!(report.pareto_frontier[0].selected_deployment_bundles.len(), 1);
        assert_eq!(report.pareto_frontier[0].resources.energy_millijoules, 25);
    }

    #[test]
    fn same_deployment_bundle_id_must_pin_one_digest() {
        let mut second = option("b", "source-b", vec![dep("robot", "b")]);
        second.deployment_bundle_id = "bundle-1".into();
        second.deployment_bundle_digest = "sha256:conflicting-bundle".into();
        let result = plan_independent_acquisition_portfolio(
            vec![
                option("a", "source-a", vec![dep("robot", "a")]),
                second,
            ],
            &bundle_costs(),
            &expected(),
            &verifier(),
            policy(),
            &budget(),
        );
        assert_eq!(
            result,
            Err(PlanningError::ConflictingDeploymentBundleDigest(
                "bundle-1".into()
            ))
        );
    }

    #[test]
    fn verifier_inventory_must_be_complete() {
        let mut verifier = verifier();
        verifier.inventory = InventoryDisposition::Unknown;
        assert_eq!(
            plan_independent_acquisition_portfolio(
                vec![],
                &expected(),
                &verifier,
                policy(),
                &budget()
            ),
            Err(PlanningError::UnknownVerifierInventory)
        );
    }
}
