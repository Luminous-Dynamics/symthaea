#![forbid(unsafe_code)]

//! Exact, bounded-horizon kernel-family restart portfolio.
//!
//! At time t the wealth is
//!
//!   sum_(s <= t) nu_s * E_(s,t) + sum_(s > t) nu_s,
//!
//! where E_(s,t) is a CMAQ-054 family e-process restarted at s. Not-yet-started
//! components remain at wealth one and are represented by exact residual
//! prior mass. One parent replay ledger is shared by every child.

use std::collections::HashSet;

use cmaq_053_compositional_risk_budget::{ComposedRiskPlan, Rational};
use cmaq_054_evidence_adaptive_kernel_family::{
    KernelExpertConfig, KernelFamilyError, KernelFamilyPortfolio, KernelFamilyUpdate,
};

use crate::{future_mass_after, weight_for_start, StartPriorError};

const MAX_ORACLE_HORIZON: u64 = 1_024;
const MAX_PAIR_ID_BYTES: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestartPortfolioError {
    InvalidHorizon,
    HorizonExceeded,
    InvalidPairId,
    PairIdTooLong,
    DuplicatePairId,
    PairIdInventoryFull,
    SequenceMismatch,
    AlreadyQuarantined,
    ArithmeticOverflow,
    PosteriorNormalizationFailure,
    StartPrior(StartPriorError),
    KernelFamily(KernelFamilyError),
}

impl From<StartPriorError> for RestartPortfolioError {
    fn from(value: StartPriorError) -> Self {
        Self::StartPrior(value)
    }
}

impl From<KernelFamilyError> for RestartPortfolioError {
    fn from(value: KernelFamilyError) -> Self {
        Self::KernelFamily(value)
    }
}

impl std::fmt::Display for RestartPortfolioError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RestartPortfolioError {}

#[derive(Clone, Debug)]
struct RestartComponent {
    start_time: u64,
    start_prior: Rational,
    family: KernelFamilyPortfolio,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelStartPosterior {
    pub start_time: u64,
    pub expert_id: String,
    pub start_prior: Rational,
    pub kernel_prior: Rational,
    pub joint_posterior: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RestartComponentUpdate {
    pub start_time: u64,
    pub start_prior: Rational,
    pub component_posterior: f64,
    pub log_family_e_value: f64,
    pub family_update: KernelFamilyUpdate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RestartPortfolioUpdate {
    pub sequence: u64,
    pub pair_id: String,
    pub log_mixture_e_value: f64,
    pub log_threshold: f64,
    pub future_start_mass: Rational,
    pub future_start_posterior: f64,
    pub components: Vec<RestartComponentUpdate>,
    pub kernel_start_posteriors: Vec<KernelStartPosterior>,
    pub quarantined: bool,
}

/// Direct all-start correctness oracle.
///
/// Numeric child-monitor state is O(K*T), with one shared O(T) parent replay
/// ledger. Each update replays the new observation through all active start
/// components, yielding O(K*T^2) total expert-update work over horizon T.
/// This is intentionally bounded to small horizons, not a production fast path.
#[derive(Clone, Debug)]
pub struct RestartableKernelFamilyPortfolio {
    risk_plan: ComposedRiskPlan,
    portfolio_contract_sha256: String,
    expert_configurations: Vec<KernelExpertConfig>,
    horizon_cap: u64,
    observations_seen: u64,
    seen_pair_ids: HashSet<String>,
    components: Vec<RestartComponent>,
    log_mixture_e_value: f64,
    log_threshold: f64,
    quarantined: bool,
}

impl RestartableKernelFamilyPortfolio {
    pub fn new(
        risk_plan: ComposedRiskPlan,
        portfolio_contract_sha256: String,
        expert_configurations: Vec<KernelExpertConfig>,
        horizon_cap: u64,
    ) -> Result<Self, RestartPortfolioError> {
        if horizon_cap == 0 || horizon_cap > MAX_ORACLE_HORIZON {
            return Err(RestartPortfolioError::InvalidHorizon);
        }
        if expert_configurations.is_empty() {
            return Err(RestartPortfolioError::KernelFamily(
                KernelFamilyError::EmptyExpertInventory,
            ));
        }

        // Validate the plan, expert inventory and first component threshold
        // before consuming any data. Start one has prior mass exactly 1/2.
        let first_prior = weight_for_start(1)?;
        let _validation_family = KernelFamilyPortfolio::new_with_threshold_scale(
            &risk_plan,
            first_prior,
            portfolio_contract_sha256.clone(),
            expert_configurations.clone(),
        )?;

        let log_threshold = rational_log_inverse(risk_plan.alpha_test())?;
        Ok(Self {
            risk_plan,
            portfolio_contract_sha256,
            expert_configurations,
            horizon_cap,
            observations_seen: 0,
            seen_pair_ids: HashSet::with_capacity(horizon_cap as usize),
            components: Vec::with_capacity(horizon_cap as usize),
            log_mixture_e_value: 0.0,
            log_threshold,
            quarantined: false,
        })
    }

    /// Updates every active child transactionally. Pair identity is checked
    /// once here; child CMAQ-054 monitors retain local sequence checks but use
    /// their externally guarded path so IDs are not duplicated per restart.
    pub fn observe_pair(
        &mut self,
        sequence: u64,
        pair_id: &str,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<RestartPortfolioUpdate, RestartPortfolioError> {
        if self.quarantined {
            return Err(RestartPortfolioError::AlreadyQuarantined);
        }
        if sequence > self.horizon_cap {
            return Err(RestartPortfolioError::HorizonExceeded);
        }
        if pair_id.trim().is_empty() {
            return Err(RestartPortfolioError::InvalidPairId);
        }
        if pair_id.len() > MAX_PAIR_ID_BYTES {
            return Err(RestartPortfolioError::PairIdTooLong);
        }
        if self.seen_pair_ids.contains(pair_id) {
            return Err(RestartPortfolioError::DuplicatePairId);
        }
        if self.seen_pair_ids.len() >= self.horizon_cap as usize {
            return Err(RestartPortfolioError::PairIdInventoryFull);
        }

        let expected_sequence = self
            .observations_seen
            .checked_add(1)
            .ok_or(RestartPortfolioError::ArithmeticOverflow)?;
        if sequence != expected_sequence {
            return Err(RestartPortfolioError::SequenceMismatch);
        }

        // Stage all old children and exactly one new child at the current
        // start. A component sees its start observation as local sequence 1.
        let mut staged_components = self.components.clone();
        let start_prior = weight_for_start(sequence)?;
        let new_family = KernelFamilyPortfolio::new_with_threshold_scale(
            &self.risk_plan,
            start_prior,
            self.portfolio_contract_sha256.clone(),
            self.expert_configurations.clone(),
        )?;
        staged_components.push(RestartComponent {
            start_time: sequence,
            start_prior,
            family: new_family,
        });

        let mut staged_updates = Vec::with_capacity(staged_components.len());
        for component in &mut staged_components {
            if component.family.is_quarantined() {
                // A child can only stop at alpha_test*nu_s, so its weighted
                // contribution must already have stopped the parent.
                return Err(RestartPortfolioError::ArithmeticOverflow);
            }
            let local_sequence = sequence
                .checked_sub(component.start_time)
                .and_then(|difference| difference.checked_add(1))
                .ok_or(RestartPortfolioError::ArithmeticOverflow)?;
            let update = component.family.observe_pair_with_external_replay_guard(
                local_sequence,
                pair_id,
                covariate,
                x,
                y,
            )?;
            staged_updates.push(update);
        }

        let future_start_mass = future_mass_after(sequence)?;
        let next_log_mixture =
            mixture_log_e_value(&staged_components, future_start_mass)?;
        let next_quarantined = next_log_mixture >= self.log_threshold;

        if staged_components
            .iter()
            .any(|component| component.family.is_quarantined())
            && !next_quarantined
        {
            return Err(RestartPortfolioError::PosteriorNormalizationFailure);
        }

        let future_start_posterior =
            (rational_log(future_start_mass)? - next_log_mixture).exp();
        if !future_start_posterior.is_finite()
            || !(0.0..=1.0 + 1e-10).contains(&future_start_posterior)
        {
            return Err(RestartPortfolioError::PosteriorNormalizationFailure);
        }

        let mut posterior_sum = future_start_posterior;
        let mut component_updates = Vec::with_capacity(staged_components.len());
        let mut kernel_start_posteriors = Vec::new();

        for (component, family_update) in
            staged_components.iter().zip(staged_updates.into_iter())
        {
            let log_component = rational_log(component.start_prior)?
                + component.family.log_mixture_e_value();
            let component_posterior = (log_component - next_log_mixture).exp();
            if !component_posterior.is_finite()
                || !(0.0..=1.0 + 1e-10).contains(&component_posterior)
            {
                return Err(RestartPortfolioError::PosteriorNormalizationFailure);
            }
            posterior_sum += component_posterior;

            for expert in &family_update.experts {
                let joint_posterior =
                    component_posterior * expert.posterior_weight;
                if !joint_posterior.is_finite()
                    || !(0.0..=1.0 + 1e-10).contains(&joint_posterior)
                {
                    return Err(RestartPortfolioError::PosteriorNormalizationFailure);
                }
                kernel_start_posteriors.push(KernelStartPosterior {
                    start_time: component.start_time,
                    expert_id: expert.expert_id.clone(),
                    start_prior: component.start_prior,
                    kernel_prior: expert.prior_weight,
                    joint_posterior,
                });
            }

            component_updates.push(RestartComponentUpdate {
                start_time: component.start_time,
                start_prior: component.start_prior,
                component_posterior,
                log_family_e_value: component.family.log_mixture_e_value(),
                family_update,
            });
        }

        if !posterior_sum.is_finite() || (posterior_sum - 1.0).abs() > 1e-8 {
            return Err(RestartPortfolioError::PosteriorNormalizationFailure);
        }

        // Commit only after every child, mixture and posterior calculation
        // succeeded. The parent consumes each pair identity exactly once.
        self.components = staged_components;
        self.seen_pair_ids.insert(pair_id.to_string());
        self.observations_seen = sequence;
        self.log_mixture_e_value = next_log_mixture;
        self.quarantined = next_quarantined;

        Ok(RestartPortfolioUpdate {
            sequence,
            pair_id: pair_id.to_string(),
            log_mixture_e_value: next_log_mixture,
            log_threshold: self.log_threshold,
            future_start_mass,
            future_start_posterior,
            components: component_updates,
            kernel_start_posteriors,
            quarantined: next_quarantined,
        })
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn horizon_cap(&self) -> u64 {
        self.horizon_cap
    }

    pub fn active_start_count(&self) -> usize {
        self.components.len()
    }

    pub fn parent_replay_inventory_size(&self) -> usize {
        self.seen_pair_ids.len()
    }

    pub fn log_mixture_e_value(&self) -> f64 {
        self.log_mixture_e_value
    }

    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    /// (start time, exact start prior, exact scaled family threshold).
    pub fn component_thresholds(&self) -> Vec<(u64, Rational, Rational)> {
        self.components
            .iter()
            .map(|component| {
                (
                    component.start_time,
                    component.start_prior,
                    component.family.alpha_portfolio(),
                )
            })
            .collect()
    }

    pub fn risk_plan(&self) -> &ComposedRiskPlan {
        &self.risk_plan
    }

    pub fn portfolio_contract_sha256(&self) -> &str {
        &self.portfolio_contract_sha256
    }
}

fn mixture_log_e_value(
    components: &[RestartComponent],
    future_start_mass: Rational,
) -> Result<f64, RestartPortfolioError> {
    let mut terms = Vec::with_capacity(components.len() + 1);
    for component in components {
        let term = rational_log(component.start_prior)?
            + component.family.log_mixture_e_value();
        if !term.is_finite() {
            return Err(RestartPortfolioError::ArithmeticOverflow);
        }
        terms.push(term);
    }
    terms.push(rational_log(future_start_mass)?);
    log_sum_exp(&terms)
}

fn log_sum_exp(values: &[f64]) -> Result<f64, RestartPortfolioError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(RestartPortfolioError::ArithmeticOverflow);
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled_sum = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f64>();
    let result = maximum + scaled_sum.ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(RestartPortfolioError::ArithmeticOverflow)
    }
}

fn rational_log(value: Rational) -> Result<f64, RestartPortfolioError> {
    let result =
        (value.numerator() as f64).ln() - (value.denominator() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(RestartPortfolioError::ArithmeticOverflow)
    }
}

fn rational_log_inverse(value: Rational) -> Result<f64, RestartPortfolioError> {
    let result =
        (value.denominator() as f64).ln() - (value.numerator() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(RestartPortfolioError::ArithmeticOverflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_046_convex_target_approachability::BoxTarget;
    use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
    use cmaq_053_compositional_risk_budget::{
        FleetBudgetPlan, FleetScope, RepresentationWeight, RiskPlanBindings,
    };

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const SHA_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const SHA_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    fn risk_plan() -> ComposedRiskPlan {
        let fleet = FleetBudgetPlan::new(
            Rational::probability(1, 20).unwrap(),
            SHA_A.into(),
            SHA_B.into(),
        )
        .unwrap();
        let scope = FleetScope {
            stream_id: "stream-055".into(),
            asset_id: "asset-1".into(),
            channel_id: "channel-1".into(),
            deployment_id: "deployment-1".into(),
            asset_index: 1,
            channel_index: 1,
            deployment_index: 1,
        };
        let local = fleet.local_alpha_for(&scope).unwrap();
        let alpha_test = local.divided_by_integer(2).unwrap();
        let delta = local.divided_by_integer(2).unwrap();
        ComposedRiskPlan::new(
            &fleet,
            scope,
            alpha_test,
            delta,
            vec![
                RepresentationWeight {
                    member_id: "low-frequency".into(),
                    weight: Rational::new(1, 2).unwrap(),
                },
                RepresentationWeight {
                    member_id: "high-frequency".into(),
                    weight: Rational::new(1, 2).unwrap(),
                },
            ],
            RiskPlanBindings {
                deployment_plan_sha256: SHA_C.into(),
                reference_manifest_sha256: SHA_D.into(),
                target_envelope_sha256: SHA_E.into(),
                monitor_build_sha256: SHA_F.into(),
                representation_inventory_sha256: SHA_1.into(),
                calibration_method_sha256: SHA_2.into(),
            },
        )
        .unwrap()
    }

    fn expert(
        id: &str,
        frequency: f64,
        prior_weight: Rational,
        response_sha: &str,
        covariate_sha: &str,
        target_sha: &str,
    ) -> KernelExpertConfig {
        KernelExpertConfig {
            expert_id: id.into(),
            response_map: FourierFeatureMap::new(
                1,
                vec![vec![frequency]],
                response_sha.into(),
            )
            .unwrap(),
            covariate_map: FourierFeatureMap::new(
                1,
                vec![vec![0.5]],
                covariate_sha.into(),
            )
            .unwrap(),
            target: BoxTarget::new(vec![-0.005; 4], vec![0.005; 4]).unwrap(),
            prior_weight,
            target_sha256: target_sha.into(),
            response_reference_sha256: SHA_E.into(),
            covariate_reference_sha256: SHA_F.into(),
        }
    }

    fn expert_configs() -> Vec<KernelExpertConfig> {
        vec![
            expert(
                "low-frequency",
                1.0,
                Rational::new(1, 2).unwrap(),
                SHA_A,
                SHA_B,
                SHA_C,
            ),
            expert(
                "high-frequency",
                0.5,
                Rational::new(1, 2).unwrap(),
                SHA_D,
                SHA_E,
                SHA_F,
            ),
        ]
    }

    fn portfolio(horizon: u64) -> RestartableKernelFamilyPortfolio {
        RestartableKernelFamilyPortfolio::new(
            risk_plan(),
            SHA_1.into(),
            expert_configs(),
            horizon,
        )
        .unwrap()
    }

    #[test]
    fn initial_restart_mixture_has_unit_wealth_and_future_mass_one() {
        let family = portfolio(16);
        assert_eq!(family.observations_seen(), 0);
        assert_eq!(family.active_start_count(), 0);
        assert_eq!(family.parent_replay_inventory_size(), 0);
        assert!(family.log_mixture_e_value().abs() < 1e-12);
        assert_eq!(future_mass_after(0).unwrap(), Rational::one());
    }

    #[test]
    fn equal_pairs_keep_all_start_components_at_unit_wealth() {
        let mut family = portfolio(16);
        for sequence in 1..=16u64 {
            let update = family
                .observe_pair(
                    sequence,
                    &format!("equal-pair-{sequence}"),
                    &[(sequence as f64 / 10.0).sin()],
                    &[0.25],
                    &[0.25],
                )
                .unwrap();
            assert!(!update.quarantined);
            assert!(update.log_mixture_e_value.abs() < 1e-10);
            assert_eq!(update.future_start_mass, future_mass_after(sequence).unwrap());
            assert_eq!(update.components.len(), sequence as usize);
            assert!(
                (update.future_start_posterior - update.future_start_mass.numerator() as f64
                    / update.future_start_mass.denominator() as f64)
                    .abs()
                    < 1e-10
            );
        }
        assert_eq!(family.active_start_count(), 16);
        assert_eq!(family.parent_replay_inventory_size(), 16);
    }

    #[test]
    fn restart_components_receive_exact_parent_scaled_thresholds() {
        let mut family = portfolio(16);
        family
            .observe_pair(1, "first", &[0.0], &[0.25], &[0.25])
            .unwrap();
        family
            .observe_pair(2, "second", &[0.0], &[0.25], &[0.25])
            .unwrap();

        let thresholds = family.component_thresholds();
        assert_eq!(
            thresholds[0],
            (
                1,
                Rational::new(1, 2).unwrap(),
                Rational::new(1, 640).unwrap()
            )
        );
        assert_eq!(
            thresholds[1],
            (
                2,
                Rational::new(1, 12).unwrap(),
                Rational::new(1, 3_840).unwrap()
            )
        );

        let first_member_thresholds =
            family.components[0].family.posterior_weights().unwrap();
        let second_member_thresholds =
            family.components[1].family.posterior_weights().unwrap();
        assert_eq!(first_member_thresholds[0].member_alpha, (1, 1_280));
        assert_eq!(first_member_thresholds[1].member_alpha, (1, 1_280));
        assert_eq!(second_member_thresholds[0].member_alpha, (1, 7_680));
        assert_eq!(second_member_thresholds[1].member_alpha, (1, 7_680));
    }

    #[test]
    fn parent_replay_ledger_rejects_duplicates_without_partial_mutation() {
        let mut family = portfolio(16);
        family
            .observe_pair(1, "unique-pair", &[0.0], &[0.25], &[0.25])
            .unwrap();
        let before = family.clone();
        assert_eq!(
            family.observe_pair(2, "unique-pair", &[0.0], &[0.25], &[0.25]),
            Err(RestartPortfolioError::DuplicatePairId)
        );
        assert_eq!(family.observations_seen(), before.observations_seen());
        assert_eq!(family.active_start_count(), before.active_start_count());
        assert_eq!(family.log_mixture_e_value(), before.log_mixture_e_value());
        assert_eq!(
            family.parent_replay_inventory_size(),
            before.parent_replay_inventory_size()
        );
    }

    #[test]
    fn a_late_child_failure_does_not_commit_or_consume_pair_id() {
        let mut family = portfolio(16);
        family
            .observe_pair(1, "first", &[0.0], &[0.25], &[0.25])
            .unwrap();
        let before = family.clone();

        assert!(matches!(
            family.observe_pair(2, "retryable", &[0.0, 1.0], &[0.25], &[0.25]),
            Err(RestartPortfolioError::KernelFamily(
                KernelFamilyError::FeatureMap(_)
            ))
        ));
        assert_eq!(family.observations_seen(), before.observations_seen());
        assert_eq!(family.active_start_count(), before.active_start_count());
        assert_eq!(
            family.parent_replay_inventory_size(),
            before.parent_replay_inventory_size()
        );

        family
            .observe_pair(2, "retryable", &[0.0], &[0.25], &[0.25])
            .unwrap();
        assert_eq!(family.observations_seen(), 2);
        assert_eq!(family.parent_replay_inventory_size(), 2);
    }

    #[test]
    fn start_and_kernel_posteriors_normalize_including_future_mass() {
        let mut family = portfolio(16);
        let update = family
            .observe_pair(1, "posterior", &[0.0], &[0.0], &[0.0])
            .unwrap();
        let joint_sum = update
            .kernel_start_posteriors
            .iter()
            .map(|item| item.joint_posterior)
            .sum::<f64>();
        assert!((joint_sum + update.future_start_posterior - 1.0).abs() < 1e-10);
        assert_eq!(update.kernel_start_posteriors.len(), 2);
        assert!((update.components[0].component_posterior - 0.5).abs() < 1e-10);
        assert!((update.future_start_posterior - 0.5).abs() < 1e-10);
    }

    #[test]
    fn horizon_limit_is_fail_closed() {
        assert_eq!(
            RestartableKernelFamilyPortfolio::new(
                risk_plan(),
                SHA_1.into(),
                expert_configs(),
                0,
            )
            .err(),
            Some(RestartPortfolioError::InvalidHorizon)
        );
        assert_eq!(
            RestartableKernelFamilyPortfolio::new(
                risk_plan(),
                SHA_1.into(),
                expert_configs(),
                MAX_ORACLE_HORIZON + 1,
            )
            .err(),
            Some(RestartPortfolioError::InvalidHorizon)
        );

        let mut family = portfolio(2);
        family
            .observe_pair(1, "one", &[0.0], &[0.25], &[0.25])
            .unwrap();
        family
            .observe_pair(2, "two", &[0.0], &[0.25], &[0.25])
            .unwrap();
        assert_eq!(
            family.observe_pair(3, "three", &[0.0], &[0.25], &[0.25]),
            Err(RestartPortfolioError::HorizonExceeded)
        );
    }
}
