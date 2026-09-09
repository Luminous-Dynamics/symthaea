// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded counterfactual capability-frontier analysis.
//!
//! This module asks a deliberately narrow question:
//!
//! > Under one exact capability graph and one exact activation-assumption set,
//! > which small sets of additional prerequisite capabilities, if treated as
//! > available counterfactually, cause a currently blocked activatable target to
//! > enter the deterministic activation closure?
//!
//! Counterfactual support is not evidence, feasibility, cost, safety, priority,
//! recommendation, or authority. The target capability itself is never offered
//! as a trivial support intervention. Search is exact only within the complete
//! transitive prerequisite cone and configured support-width bound.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CapabilityActivationAssumptionsId, CapabilityActivationAssumptionsV1,
    CapabilityActivationError, CapabilityGraphSnapshotId, CapabilityId,
    ValidatedCapabilityActivationAssumptionsV1, ValidatedCapabilityGraphV1,
    derive_capability_activation_closure,
};

/// Stable schema for deterministic counterfactual search bounds.
pub const CAPABILITY_COUNTERFACTUAL_CONFIG_SCHEMA_V1: &str =
    "symthaea-continuity-capability-counterfactual-config-v1";

const CONFIG_DOMAIN: &[u8] = b"symthaea.continuity.capability-counterfactual-config.v1\0";
const HARD_MAX_UNIVERSE_SIZE: u16 = 32;
const HARD_MAX_SUPPORT_WIDTH: u16 = 6;
const HARD_MAX_OPTIONS_PER_TARGET: u16 = 2_048;
const HARD_MAX_TOTAL_SIMULATIONS: u32 = 100_000;

/// Exact identity of one canonical bounded-search configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityCounterfactualConfigId([u8; 32]);

impl CapabilityCounterfactualConfigId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable deterministic search bounds.
///
/// These bounds are part of the analysis provenance. A result computed through
/// width 2 is not silently interchangeable with one computed through width 4.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityCounterfactualConfigV1 {
    schema_version: String,
    max_universe_size: u16,
    max_support_width: u16,
    max_options_per_target: u16,
    max_total_simulations: u32,
    config_id: CapabilityCounterfactualConfigId,
}

impl CapabilityCounterfactualConfigV1 {
    pub fn new(
        max_universe_size: u16,
        max_support_width: u16,
        max_options_per_target: u16,
        max_total_simulations: u32,
    ) -> Result<Self, CapabilityCounterfactualError> {
        validate_limits(
            max_universe_size,
            max_support_width,
            max_options_per_target,
            max_total_simulations,
        )?;
        let config_id = CapabilityCounterfactualConfigId(hash_config(
            max_universe_size,
            max_support_width,
            max_options_per_target,
            max_total_simulations,
        ));
        Ok(Self {
            schema_version: CAPABILITY_COUNTERFACTUAL_CONFIG_SCHEMA_V1.to_owned(),
            max_universe_size,
            max_support_width,
            max_options_per_target,
            max_total_simulations,
            config_id,
        })
    }

    pub fn id(&self) -> CapabilityCounterfactualConfigId {
        self.config_id
    }

    pub fn max_universe_size(&self) -> usize {
        self.max_universe_size as usize
    }

    pub fn max_support_width(&self) -> usize {
        self.max_support_width as usize
    }

    pub fn max_options_per_target(&self) -> usize {
        self.max_options_per_target as usize
    }

    pub fn max_total_simulations(&self) -> u64 {
        self.max_total_simulations as u64
    }

    pub fn validate(
        &self,
    ) -> Result<ValidatedCapabilityCounterfactualConfigV1, CapabilityCounterfactualError> {
        if self.schema_version != CAPABILITY_COUNTERFACTUAL_CONFIG_SCHEMA_V1 {
            return Err(CapabilityCounterfactualError::UnsupportedConfigSchema(
                self.schema_version.clone(),
            ));
        }
        validate_limits(
            self.max_universe_size,
            self.max_support_width,
            self.max_options_per_target,
            self.max_total_simulations,
        )?;
        let expected = CapabilityCounterfactualConfigId(hash_config(
            self.max_universe_size,
            self.max_support_width,
            self.max_options_per_target,
            self.max_total_simulations,
        ));
        if expected != self.config_id {
            return Err(CapabilityCounterfactualError::ConfigIdentityMismatch);
        }
        Ok(ValidatedCapabilityCounterfactualConfigV1 {
            inner: self.clone(),
        })
    }
}

/// Non-Serde validated search configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCapabilityCounterfactualConfigV1 {
    inner: CapabilityCounterfactualConfigV1,
}

impl ValidatedCapabilityCounterfactualConfigV1 {
    pub fn id(&self) -> CapabilityCounterfactualConfigId {
        self.inner.id()
    }

    pub fn max_universe_size(&self) -> usize {
        self.inner.max_universe_size()
    }

    pub fn max_support_width(&self) -> usize {
        self.inner.max_support_width()
    }

    pub fn max_options_per_target(&self) -> usize {
        self.inner.max_options_per_target()
    }

    pub fn max_total_simulations(&self) -> u64 {
        self.inner.max_total_simulations()
    }

    pub fn as_raw(&self) -> &CapabilityCounterfactualConfigV1 {
        &self.inner
    }
}

/// One inclusion-minimal counterfactual support option found within the search
/// width.
///
/// `assumed_support` is the exogenous intervention. `marginally_activated`
/// contains only capabilities that enter closure endogenously after that
/// intervention; the support set itself is excluded from this count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityCounterfactualOptionV1 {
    target_capability_id: CapabilityId,
    assumed_support: Vec<CapabilityId>,
    counterfactual_assumptions_id: CapabilityActivationAssumptionsId,
    marginally_activated: Vec<CapabilityId>,
}

impl CapabilityCounterfactualOptionV1 {
    pub fn target_capability_id(&self) -> CapabilityId {
        self.target_capability_id
    }

    pub fn assumed_support(&self) -> &[CapabilityId] {
        &self.assumed_support
    }

    pub fn counterfactual_assumptions_id(&self) -> CapabilityActivationAssumptionsId {
        self.counterfactual_assumptions_id
    }

    pub fn marginally_activated(&self) -> &[CapabilityId] {
        &self.marginally_activated
    }

    /// Descriptive count only. This is not a priority score.
    pub fn marginal_activation_count(&self) -> usize {
        self.marginally_activated.len()
    }
}

/// Counterfactual search result for one blocked activatable target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityCounterfactualTargetV1 {
    target_capability_id: CapabilityId,
    dependency_universe: Vec<CapabilityId>,
    complete_through_support_width: usize,
    options: Vec<CapabilityCounterfactualOptionV1>,
}

impl CapabilityCounterfactualTargetV1 {
    pub fn target_capability_id(&self) -> CapabilityId {
        self.target_capability_id
    }

    /// Complete transitive prerequisite cone after removing capabilities already
    /// available in the base closure and excluding the target itself.
    pub fn dependency_universe(&self) -> &[CapabilityId] {
        &self.dependency_universe
    }

    /// Search completeness statement: every subset of the complete dependency
    /// universe up to this width was either simulated or skipped because an
    /// already-found successful strict subset proved it non-minimal.
    pub fn complete_through_support_width(&self) -> usize {
        self.complete_through_support_width
    }

    pub fn options(&self) -> &[CapabilityCounterfactualOptionV1] {
        &self.options
    }
}

/// Deterministic bounded counterfactual frontier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityCounterfactualFrontierV1 {
    source_snapshot_id: CapabilityGraphSnapshotId,
    base_assumptions_id: CapabilityActivationAssumptionsId,
    config_id: CapabilityCounterfactualConfigId,
    base_available_after_closure: Vec<CapabilityId>,
    targets: Vec<CapabilityCounterfactualTargetV1>,
    simulations_evaluated: u64,
}

impl CapabilityCounterfactualFrontierV1 {
    pub fn source_snapshot_id(&self) -> CapabilityGraphSnapshotId {
        self.source_snapshot_id
    }

    pub fn base_assumptions_id(&self) -> CapabilityActivationAssumptionsId {
        self.base_assumptions_id
    }

    pub fn config_id(&self) -> CapabilityCounterfactualConfigId {
        self.config_id
    }

    pub fn base_available_after_closure(&self) -> &[CapabilityId] {
        &self.base_available_after_closure
    }

    pub fn targets(&self) -> &[CapabilityCounterfactualTargetV1] {
        &self.targets
    }

    pub fn simulations_evaluated(&self) -> u64 {
        self.simulations_evaluated
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityCounterfactualError {
    #[error("unsupported capability counterfactual config schema: {0}")]
    UnsupportedConfigSchema(String),
    #[error("counterfactual config identity does not match canonical fields")]
    ConfigIdentityMismatch,
    #[error("counterfactual max universe size must be in 1..={HARD_MAX_UNIVERSE_SIZE}")]
    InvalidUniverseLimit,
    #[error("counterfactual max support width must be in 1..={HARD_MAX_SUPPORT_WIDTH} and not exceed max universe size")]
    InvalidSupportWidth,
    #[error("counterfactual max options per target must be in 1..={HARD_MAX_OPTIONS_PER_TARGET}")]
    InvalidOptionLimit,
    #[error("counterfactual max simulations must be in 1..={HARD_MAX_TOTAL_SIMULATIONS}")]
    InvalidSimulationLimit,
    #[error(transparent)]
    Activation(#[from] CapabilityActivationError),
    #[error("counterfactual analysis references capability unexpectedly absent from graph: {0:?}")]
    MissingDefinition(CapabilityId),
    #[error("target {target:?} has dependency universe size {actual}, above configured maximum {max}")]
    UniverseLimitExceeded {
        target: CapabilityId,
        actual: usize,
        max: usize,
    },
    #[error("target {target:?} needs upper-bound {required_upper_bound} subset simulations but only {remaining} remain")]
    SimulationLimitExceeded {
        target: CapabilityId,
        required_upper_bound: u64,
        remaining: u64,
    },
    #[error("target {target:?} produced more than {max} inclusion-minimal support options")]
    OptionLimitExceeded { target: CapabilityId, max: usize },
    #[error("counterfactual support set did not produce a consistent activation result for target {0:?}")]
    InconsistentCounterfactual(CapabilityId),
}

/// Derive bounded inclusion-minimal counterfactual support sets.
///
/// The search universe for each blocked activatable target is its complete
/// transitive structural prerequisite cone, excluding the target and anything
/// already available after the base activation closure. Every subset through the
/// configured width is considered. A subset is successful only if feeding it
/// back as an explicit additional-availability assumption into CC-03B causes the
/// target to enter closure.
///
/// This is counterfactual model analysis only. It does not claim that any
/// support capability can be realized, restored, purchased, operated safely, or
/// authorized in the world.
pub fn derive_capability_counterfactual_frontier(
    graph: &ValidatedCapabilityGraphV1,
    assumptions: &ValidatedCapabilityActivationAssumptionsV1,
    config: &ValidatedCapabilityCounterfactualConfigV1,
) -> Result<CapabilityCounterfactualFrontierV1, CapabilityCounterfactualError> {
    let base_closure = derive_capability_activation_closure(graph, assumptions)?;
    let base_available_after_closure = base_closure.available_after_closure().to_vec();
    let base_available: BTreeSet<_> = base_available_after_closure.iter().copied().collect();
    let mut simulations_evaluated = 0_u64;
    let mut targets = Vec::with_capacity(base_closure.blocked_activatable().len());

    for blocked in base_closure.blocked_activatable() {
        let target = blocked.capability_id();
        let mut dependency_universe = collect_dependency_cone(graph, target)?;
        dependency_universe.retain(|capability_id| !base_available.contains(capability_id));

        if dependency_universe.len() > config.max_universe_size() {
            return Err(CapabilityCounterfactualError::UniverseLimitExceeded {
                target,
                actual: dependency_universe.len(),
                max: config.max_universe_size(),
            });
        }

        let search_width = config.max_support_width().min(dependency_universe.len());
        let required_upper_bound = subset_count_upper_bound(dependency_universe.len(), search_width);
        let remaining = config
            .max_total_simulations()
            .saturating_sub(simulations_evaluated);
        if required_upper_bound > remaining {
            return Err(CapabilityCounterfactualError::SimulationLimitExceeded {
                target,
                required_upper_bound,
                remaining,
            });
        }

        let mut options: Vec<CapabilityCounterfactualOptionV1> = Vec::new();
        for width in 1..=search_width {
            let combinations = combinations_of_width(&dependency_universe, width);
            for support in combinations {
                // Widths are visited from small to large. If a successful
                // support set is already a subset, this candidate cannot be
                // inclusion-minimal and need not be simulated.
                if options
                    .iter()
                    .any(|option| sorted_subset(option.assumed_support(), &support))
                {
                    continue;
                }

                simulations_evaluated += 1;

                let mut counterfactual_available = assumptions.available().to_vec();
                counterfactual_available.extend_from_slice(&support);
                let counterfactual_raw = CapabilityActivationAssumptionsV1::new(
                    graph,
                    counterfactual_available,
                    assumptions.activatable().to_vec(),
                )?;
                let counterfactual_assumptions_id = counterfactual_raw.id();
                let counterfactual = counterfactual_raw.validate(graph)?;
                let closure = derive_capability_activation_closure(graph, &counterfactual)?;

                if !closure.is_available_after_closure(target) {
                    continue;
                }

                let support_set: BTreeSet<_> = support.iter().copied().collect();
                let marginally_activated: Vec<_> = closure
                    .available_after_closure()
                    .iter()
                    .copied()
                    .filter(|capability_id| {
                        !base_available.contains(capability_id)
                            && !support_set.contains(capability_id)
                    })
                    .collect();

                if marginally_activated.binary_search(&target).is_err() {
                    return Err(CapabilityCounterfactualError::InconsistentCounterfactual(
                        target,
                    ));
                }

                options.push(CapabilityCounterfactualOptionV1 {
                    target_capability_id: target,
                    assumed_support: support,
                    counterfactual_assumptions_id,
                    marginally_activated,
                });
                if options.len() > config.max_options_per_target() {
                    return Err(CapabilityCounterfactualError::OptionLimitExceeded {
                        target,
                        max: config.max_options_per_target(),
                    });
                }
            }
        }

        options.sort_by(|left, right| {
            left.assumed_support
                .len()
                .cmp(&right.assumed_support.len())
                .then_with(|| left.assumed_support.cmp(&right.assumed_support))
        });
        targets.push(CapabilityCounterfactualTargetV1 {
            target_capability_id: target,
            dependency_universe,
            complete_through_support_width: search_width,
            options,
        });
    }

    Ok(CapabilityCounterfactualFrontierV1 {
        source_snapshot_id: graph.id(),
        base_assumptions_id: assumptions.id(),
        config_id: config.id(),
        base_available_after_closure,
        targets,
        simulations_evaluated,
    })
}

fn collect_dependency_cone(
    graph: &ValidatedCapabilityGraphV1,
    target: CapabilityId,
) -> Result<Vec<CapabilityId>, CapabilityCounterfactualError> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![target];

    while let Some(current) = stack.pop() {
        let definition = graph
            .definition(current)
            .ok_or(CapabilityCounterfactualError::MissingDefinition(current))?;
        for referenced in definition.referenced_capabilities() {
            // Directly assuming the target available is intentionally excluded
            // as a trivial intervention, including when a cycle points back to
            // the target.
            if referenced == target {
                continue;
            }
            if seen.insert(referenced) {
                stack.push(referenced);
            }
        }
    }

    Ok(seen.into_iter().collect())
}

fn combinations_of_width(items: &[CapabilityId], width: usize) -> Vec<Vec<CapabilityId>> {
    fn visit(
        items: &[CapabilityId],
        start: usize,
        remaining: usize,
        current: &mut Vec<CapabilityId>,
        out: &mut Vec<Vec<CapabilityId>>,
    ) {
        if remaining == 0 {
            out.push(current.clone());
            return;
        }
        if items.len().saturating_sub(start) < remaining {
            return;
        }
        let last_start = items.len() - remaining;
        for index in start..=last_start {
            current.push(items[index]);
            visit(items, index + 1, remaining - 1, current, out);
            current.pop();
        }
    }

    if width == 0 || width > items.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    visit(items, 0, width, &mut Vec::with_capacity(width), &mut out);
    out
}

fn sorted_subset(left: &[CapabilityId], right: &[CapabilityId]) -> bool {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {
                left_index += 1;
                right_index += 1;
            }
            std::cmp::Ordering::Greater => right_index += 1,
        }
    }
    left_index == left.len()
}

fn subset_count_upper_bound(universe_size: usize, max_width: usize) -> u64 {
    let mut total = 0_u128;
    for width in 1..=max_width.min(universe_size) {
        total = total.saturating_add(binomial(universe_size, width));
    }
    total.min(u64::MAX as u128) as u64
}

fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut value = 1_u128;
    for index in 0..k {
        value = value
            .saturating_mul((n - index) as u128)
            / (index + 1) as u128;
    }
    value
}

fn validate_limits(
    max_universe_size: u16,
    max_support_width: u16,
    max_options_per_target: u16,
    max_total_simulations: u32,
) -> Result<(), CapabilityCounterfactualError> {
    if max_universe_size == 0 || max_universe_size > HARD_MAX_UNIVERSE_SIZE {
        return Err(CapabilityCounterfactualError::InvalidUniverseLimit);
    }
    if max_support_width == 0
        || max_support_width > HARD_MAX_SUPPORT_WIDTH
        || max_support_width > max_universe_size
    {
        return Err(CapabilityCounterfactualError::InvalidSupportWidth);
    }
    if max_options_per_target == 0 || max_options_per_target > HARD_MAX_OPTIONS_PER_TARGET {
        return Err(CapabilityCounterfactualError::InvalidOptionLimit);
    }
    if max_total_simulations == 0 || max_total_simulations > HARD_MAX_TOTAL_SIMULATIONS {
        return Err(CapabilityCounterfactualError::InvalidSimulationLimit);
    }
    Ok(())
}

fn hash_config(
    max_universe_size: u16,
    max_support_width: u16,
    max_options_per_target: u16,
    max_total_simulations: u32,
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(96);
    put_str(&mut bytes, CAPABILITY_COUNTERFACTUAL_CONFIG_SCHEMA_V1);
    bytes.extend_from_slice(&max_universe_size.to_le_bytes());
    bytes.extend_from_slice(&max_support_width.to_le_bytes());
    bytes.extend_from_slice(&max_options_per_target.to_le_bytes());
    bytes.extend_from_slice(&max_total_simulations.to_le_bytes());

    let mut hasher = blake3::Hasher::new();
    hasher.update(CONFIG_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapabilityDefinitionV1, CapabilityGraphSnapshotV1, CapabilityRequirementV1,
    };

    fn leaf(name: &str) -> CapabilityDefinitionV1 {
        CapabilityDefinitionV1::new("org.example", name, None).unwrap()
    }

    fn graph(definitions: Vec<CapabilityDefinitionV1>) -> ValidatedCapabilityGraphV1 {
        CapabilityGraphSnapshotV1::new(definitions)
            .unwrap()
            .validate()
            .unwrap()
    }

    fn assumptions(
        graph: &ValidatedCapabilityGraphV1,
        available: Vec<CapabilityId>,
        activatable: Vec<CapabilityId>,
    ) -> ValidatedCapabilityActivationAssumptionsV1 {
        CapabilityActivationAssumptionsV1::new(graph, available, activatable)
            .unwrap()
            .validate(graph)
            .unwrap()
    }

    fn config(
        universe: u16,
        width: u16,
        options: u16,
        simulations: u32,
    ) -> ValidatedCapabilityCounterfactualConfigV1 {
        CapabilityCounterfactualConfigV1::new(universe, width, options, simulations)
            .unwrap()
            .validate()
            .unwrap()
    }

    #[test]
    fn all_of_requires_the_minimal_joint_support_set() {
        let a = leaf("a");
        let b = leaf("b");
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(
                CapabilityRequirementV1::all_of(vec![
                    CapabilityRequirementV1::leaf(a.id()),
                    CapabilityRequirementV1::leaf(b.id()),
                ])
                .unwrap(),
            ),
        )
        .unwrap();
        let target_id = target.id();
        let mut expected_support = vec![a.id(), b.id()];
        expected_support.sort_unstable();
        let graph = graph(vec![target, a, b]);
        let assumptions = assumptions(&graph, vec![], vec![target_id]);

        let frontier = derive_capability_counterfactual_frontier(
            &graph,
            &assumptions,
            &config(8, 3, 16, 100),
        )
        .unwrap();
        let target = &frontier.targets()[0];
        assert_eq!(target.options().len(), 1);
        assert_eq!(target.options()[0].assumed_support(), expected_support);
        assert_eq!(target.options()[0].marginally_activated(), &[target_id]);
    }

    #[test]
    fn any_of_preserves_distinct_minimal_alternatives() {
        let a = leaf("a");
        let b = leaf("b");
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(
                CapabilityRequirementV1::any_of(vec![
                    CapabilityRequirementV1::leaf(a.id()),
                    CapabilityRequirementV1::leaf(b.id()),
                ])
                .unwrap(),
            ),
        )
        .unwrap();
        let target_id = target.id();
        let a_id = a.id();
        let b_id = b.id();
        let graph = graph(vec![target, a, b]);
        let assumptions = assumptions(&graph, vec![], vec![target_id]);

        let frontier = derive_capability_counterfactual_frontier(
            &graph,
            &assumptions,
            &config(8, 2, 16, 100),
        )
        .unwrap();
        let supports: BTreeSet<Vec<CapabilityId>> = frontier.targets()[0]
            .options()
            .iter()
            .map(|option| option.assumed_support().to_vec())
            .collect();
        assert_eq!(supports.len(), 2);
        assert!(supports.contains(&vec![a_id]));
        assert!(supports.contains(&vec![b_id]));
    }

    #[test]
    fn transitive_search_finds_indirect_bootstrap_support() {
        let b = leaf("b");
        let a = CapabilityDefinitionV1::new(
            "org.example",
            "a",
            Some(CapabilityRequirementV1::leaf(b.id())),
        )
        .unwrap();
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(CapabilityRequirementV1::leaf(a.id())),
        )
        .unwrap();
        let b_id = b.id();
        let a_id = a.id();
        let target_id = target.id();
        let graph = graph(vec![target, a, b]);
        let assumptions = assumptions(&graph, vec![], vec![a_id, target_id]);

        let frontier = derive_capability_counterfactual_frontier(
            &graph,
            &assumptions,
            &config(8, 2, 16, 100),
        )
        .unwrap();
        let options = frontier.targets()[0].options();
        let by_b = options
            .iter()
            .find(|option| option.assumed_support() == [b_id])
            .expect("indirect b support");
        assert_eq!(by_b.marginal_activation_count(), 2);
        assert!(by_b.marginally_activated().binary_search(&a_id).is_ok());
        assert!(by_b.marginally_activated().binary_search(&target_id).is_ok());
        assert!(options.iter().any(|option| option.assumed_support() == [a_id]));
    }

    #[test]
    fn cycle_can_be_broken_only_by_an_explicit_non_target_seed() {
        let a_id = CapabilityId::new("org.example", "a").unwrap();
        let b_id = CapabilityId::new("org.example", "b").unwrap();
        let a = CapabilityDefinitionV1::new(
            "org.example",
            "a",
            Some(CapabilityRequirementV1::leaf(b_id)),
        )
        .unwrap();
        let b = CapabilityDefinitionV1::new(
            "org.example",
            "b",
            Some(CapabilityRequirementV1::leaf(a_id)),
        )
        .unwrap();
        let graph = graph(vec![a, b]);
        let assumptions = assumptions(&graph, vec![], vec![a_id, b_id]);

        let frontier = derive_capability_counterfactual_frontier(
            &graph,
            &assumptions,
            &config(8, 2, 16, 100),
        )
        .unwrap();
        let a_target = frontier
            .targets()
            .iter()
            .find(|target| target.target_capability_id() == a_id)
            .unwrap();
        assert_eq!(a_target.options().len(), 1);
        assert_eq!(a_target.options()[0].assumed_support(), &[b_id]);
    }

    #[test]
    fn target_itself_is_never_a_trivial_support_option() {
        let target_id = CapabilityId::new("org.example", "target").unwrap();
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(CapabilityRequirementV1::leaf(target_id)),
        )
        .unwrap();
        let graph = graph(vec![target]);
        let assumptions = assumptions(&graph, vec![], vec![target_id]);

        let frontier = derive_capability_counterfactual_frontier(
            &graph,
            &assumptions,
            &config(4, 2, 8, 20),
        )
        .unwrap();
        assert!(frontier.targets()[0].dependency_universe().is_empty());
        assert!(frontier.targets()[0].options().is_empty());
    }

    #[test]
    fn complete_dependency_cone_must_fit_the_configured_universe_bound() {
        let a = leaf("a");
        let b = leaf("b");
        let c = leaf("c");
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(
                CapabilityRequirementV1::all_of(vec![
                    CapabilityRequirementV1::leaf(a.id()),
                    CapabilityRequirementV1::leaf(b.id()),
                    CapabilityRequirementV1::leaf(c.id()),
                ])
                .unwrap(),
            ),
        )
        .unwrap();
        let target_id = target.id();
        let graph = graph(vec![target, a, b, c]);
        let assumptions = assumptions(&graph, vec![], vec![target_id]);

        assert!(matches!(
            derive_capability_counterfactual_frontier(
                &graph,
                &assumptions,
                &config(2, 2, 16, 100),
            ),
            Err(CapabilityCounterfactualError::UniverseLimitExceeded { .. })
        ));
    }

    #[test]
    fn simulation_upper_bound_fails_before_partial_analysis() {
        let leaves: Vec<_> = ["a", "b", "c", "d"].into_iter().map(leaf).collect();
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(
                CapabilityRequirementV1::any_of(
                    leaves
                        .iter()
                        .map(|definition| CapabilityRequirementV1::leaf(definition.id()))
                        .collect(),
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let target_id = target.id();
        let mut definitions = vec![target];
        definitions.extend(leaves);
        let graph = graph(definitions);
        let assumptions = assumptions(&graph, vec![], vec![target_id]);

        assert!(matches!(
            derive_capability_counterfactual_frontier(
                &graph,
                &assumptions,
                &config(8, 2, 16, 5),
            ),
            Err(CapabilityCounterfactualError::SimulationLimitExceeded { .. })
        ));
    }

    #[test]
    fn too_many_minimal_options_fails_instead_of_truncating() {
        let a = leaf("a");
        let b = leaf("b");
        let c = leaf("c");
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(
                CapabilityRequirementV1::any_of(vec![
                    CapabilityRequirementV1::leaf(a.id()),
                    CapabilityRequirementV1::leaf(b.id()),
                    CapabilityRequirementV1::leaf(c.id()),
                ])
                .unwrap(),
            ),
        )
        .unwrap();
        let target_id = target.id();
        let graph = graph(vec![target, a, b, c]);
        let assumptions = assumptions(&graph, vec![], vec![target_id]);

        assert!(matches!(
            derive_capability_counterfactual_frontier(
                &graph,
                &assumptions,
                &config(8, 1, 2, 100),
            ),
            Err(CapabilityCounterfactualError::OptionLimitExceeded { .. })
        ));
    }

    #[test]
    fn config_identity_commits_search_bounds() {
        let left = CapabilityCounterfactualConfigV1::new(8, 2, 16, 100).unwrap();
        let right = CapabilityCounterfactualConfigV1::new(8, 3, 16, 100).unwrap();
        assert_ne!(left.id(), right.id());
        assert!(left.validate().is_ok());
        assert!(right.validate().is_ok());
    }
}
