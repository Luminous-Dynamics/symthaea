#![forbid(unsafe_code)]

//! Exact composition of fleet, calibration and representation risk budgets.
//!
//! A deployment's local alpha is allocated by the countable three-level
//! telescoping fleet hierarchy. The reference-envelope failure budget and
//! sequential-test budget must sum to no more than that local allocation.
//! A fixed mixture of representation e-processes shares alpha_test; it does
//! not split alpha_test again across its members.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const ALLOCATION_RULE_VERSION: u32 = 1;
const DIGEST_HEX_LEN: usize = 64;
const MAX_ID_BYTES: usize = 256;
const MAX_REPRESENTATION_MEMBERS: usize = 1_024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RiskBudgetError {
    InvalidFraction,
    InvalidProbability,
    ArithmeticOverflow,
    InvalidDigest(&'static str),
    InvalidIdentifier(&'static str),
    InvalidIndex,
    RiskBudgetExceeded,
    EmptyRepresentationInventory,
    TooManyRepresentationMembers,
    DuplicateRepresentationMember,
    RepresentationWeightSumMismatch,
    FleetPlanMismatch,
    DuplicateDeploymentPlan,
    DuplicateStreamId,
    DuplicateScope,
    ScopeIdentityMismatch,
}

impl std::fmt::Display for RiskBudgetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RiskBudgetError {}

/// A reduced positive rational in (0,1]. All authoritative risk arithmetic
/// uses this type; no floating-point approximation participates in a policy
/// decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    numerator: u128,
    denominator: u128,
}

impl Rational {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, RiskBudgetError> {
        if numerator == 0 || denominator == 0 || numerator > denominator {
            return Err(RiskBudgetError::InvalidFraction);
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    /// Creates a strict probability in (0,1).
    pub fn probability(numerator: u128, denominator: u128) -> Result<Self, RiskBudgetError> {
        if numerator == 0 || denominator == 0 || numerator >= denominator {
            return Err(RiskBudgetError::InvalidProbability);
        }
        Self::new(numerator, denominator)
    }

    pub fn one() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    pub fn numerator(self) -> u128 {
        self.numerator
    }

    pub fn denominator(self) -> u128 {
        self.denominator
    }

    pub fn is_one(self) -> bool {
        self.numerator == self.denominator
    }

    /// Exact reduced addition. The sum must remain in (0,1].
    pub fn checked_add(self, other: Self) -> Result<Self, RiskBudgetError> {
        let common = gcd(self.denominator, other.denominator);
        let left_factor = other.denominator / common;
        let right_factor = self.denominator / common;
        let numerator = self
            .numerator
            .checked_mul(left_factor)
            .and_then(|value| {
                other
                    .numerator
                    .checked_mul(right_factor)
                    .and_then(|other_value| value.checked_add(other_value))
            })
            .ok_or(RiskBudgetError::ArithmeticOverflow)?;
        let denominator = self
            .denominator
            .checked_mul(left_factor)
            .ok_or(RiskBudgetError::ArithmeticOverflow)?;
        Self::new(numerator, denominator)
    }

    /// Divides this fraction by a positive integer, cancelling against the
    /// numerator before multiplying the denominator.
    pub fn divided_by_integer(self, divisor: u128) -> Result<Self, RiskBudgetError> {
        if divisor == 0 {
            return Err(RiskBudgetError::InvalidFraction);
        }
        let cancellation = gcd(self.numerator, divisor);
        let numerator = self.numerator / cancellation;
        let reduced_divisor = divisor / cancellation;
        let denominator = self
            .denominator
            .checked_mul(reduced_divisor)
            .ok_or(RiskBudgetError::ArithmeticOverflow)?;
        Self::new(numerator, denominator)
    }

    /// Compares two rational values without cross-multiplication overflow.
    pub fn compare(self, other: Self) -> Ordering {
        compare_fractions(
            self.numerator,
            self.denominator,
            other.numerator,
            other.denominator,
        )
    }

    /// Tests a + b <= limit using exact checked integer arithmetic. If the
    /// required intermediate representation overflows, the caller fails
    /// closed instead of accepting an unverified budget.
    pub fn checked_sum_le(
        left: Self,
        right: Self,
        limit: Self,
    ) -> Result<bool, RiskBudgetError> {
        let common = gcd(left.denominator, right.denominator);
        let left_factor = right.denominator / common;
        let right_factor = left.denominator / common;
        let sum_numerator = left
            .numerator
            .checked_mul(left_factor)
            .and_then(|value| {
                right
                    .numerator
                    .checked_mul(right_factor)
                    .and_then(|other_value| value.checked_add(other_value))
            })
            .ok_or(RiskBudgetError::ArithmeticOverflow)?;
        let sum_denominator = left
            .denominator
            .checked_mul(left_factor)
            .ok_or(RiskBudgetError::ArithmeticOverflow)?;
        Ok(compare_fractions(
            sum_numerator,
            sum_denominator,
            limit.numerator,
            limit.denominator,
        ) != Ordering::Greater)
    }
}

/// The immutable global fleet budget and its authenticated-by-an-external
/// authority plan/inventory identifiers. This crate checks digest shape; it
/// does not authenticate the digests or the authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FleetBudgetPlan {
    global_alpha: Rational,
    fleet_plan_sha256: String,
    fleet_inventory_sha256: String,
}

impl FleetBudgetPlan {
    pub fn new(
        global_alpha: Rational,
        fleet_plan_sha256: String,
        fleet_inventory_sha256: String,
    ) -> Result<Self, RiskBudgetError> {
        if global_alpha.numerator >= global_alpha.denominator {
            return Err(RiskBudgetError::InvalidProbability);
        }
        validate_digest(&fleet_plan_sha256, "fleet_plan_sha256")?;
        validate_digest(&fleet_inventory_sha256, "fleet_inventory_sha256")?;
        Ok(Self {
            global_alpha,
            fleet_plan_sha256,
            fleet_inventory_sha256,
        })
    }

    pub fn global_alpha(&self) -> Rational {
        self.global_alpha
    }

    pub fn fleet_plan_sha256(&self) -> &str {
        &self.fleet_plan_sha256
    }

    pub fn fleet_inventory_sha256(&self) -> &str {
        &self.fleet_inventory_sha256
    }

    /// Computes alpha_global / [i(i+1) j(j+1) k(k+1)] for one-based
    /// asset/channel/deployment indices.
    pub fn local_alpha_for(
        &self,
        scope: &FleetScope,
    ) -> Result<Rational, RiskBudgetError> {
        validate_scope(scope)?;
        let factors = [
            telescoping_factor(scope.asset_index)?,
            telescoping_factor(scope.channel_index)?,
            telescoping_factor(scope.deployment_index)?,
        ];
        let mut allocation = self.global_alpha;
        for factor in factors {
            allocation = allocation.divided_by_integer(factor)?;
        }
        Ok(allocation)
    }
}

/// One deployment's identity and one-based precommitted hierarchy indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FleetScope {
    pub stream_id: String,
    pub asset_id: String,
    pub channel_id: String,
    pub deployment_id: String,
    pub asset_index: u64,
    pub channel_index: u64,
    pub deployment_index: u64,
}

/// One fixed representation member and its prior mixture weight. Member
/// weights are not separate alpha allocations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationWeight {
    pub member_id: String,
    pub weight: Rational,
}

/// Digests binding this deployment's calibration, target, executable and
/// representation inventory. A syntactically valid digest is not proof of
/// source authenticity or content correctness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiskPlanBindings {
    pub deployment_plan_sha256: String,
    pub reference_manifest_sha256: String,
    pub target_envelope_sha256: String,
    pub monitor_build_sha256: String,
    pub representation_inventory_sha256: String,
    pub calibration_method_sha256: String,
}

/// Serialization-friendly receipt for the complete risk composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompositionReceipt {
    pub allocation_rule_version: u32,
    pub fleet_plan_sha256: String,
    pub fleet_inventory_sha256: String,
    pub global_alpha: Rational,
    pub scope: FleetScope,
    pub local_alpha: Rational,
    pub alpha_test: Rational,
    pub delta_reference_failure: Rational,
    pub representation_weights: Vec<RepresentationWeight>,
    pub bindings: RiskPlanBindings,
}

/// Validated immutable plan for one fleet deployment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposedRiskPlan {
    receipt: CompositionReceipt,
}

impl ComposedRiskPlan {
    pub fn new(
        fleet_plan: &FleetBudgetPlan,
        scope: FleetScope,
        alpha_test: Rational,
        delta_reference_failure: Rational,
        representation_weights: Vec<RepresentationWeight>,
        bindings: RiskPlanBindings,
    ) -> Result<Self, RiskBudgetError> {
        validate_scope(&scope)?;
        validate_probability(alpha_test)?;
        validate_probability(delta_reference_failure)?;
        validate_bindings(&bindings)?;
        validate_representation_weights(&representation_weights)?;

        let local_alpha = fleet_plan.local_alpha_for(&scope)?;
        if !Rational::checked_sum_le(alpha_test, delta_reference_failure, local_alpha)? {
            return Err(RiskBudgetError::RiskBudgetExceeded);
        }

        Ok(Self {
            receipt: CompositionReceipt {
                allocation_rule_version: ALLOCATION_RULE_VERSION,
                fleet_plan_sha256: fleet_plan.fleet_plan_sha256.clone(),
                fleet_inventory_sha256: fleet_plan.fleet_inventory_sha256.clone(),
                global_alpha: fleet_plan.global_alpha,
                scope,
                local_alpha,
                alpha_test,
                delta_reference_failure,
                representation_weights,
                bindings,
            },
        })
    }

    pub fn receipt(&self) -> &CompositionReceipt {
        &self.receipt
    }

    pub fn local_alpha(&self) -> Rational {
        self.receipt.local_alpha
    }

    pub fn alpha_test(&self) -> Rational {
        self.receipt.alpha_test
    }

    pub fn delta_reference_failure(&self) -> Rational {
        self.receipt.delta_reference_failure
    }
}

/// Consume-once registration for one fleet-plan/inventory pair.
///
/// This in-memory ledger is only a single-process structural guard. Production
/// deployment requires a durable authenticated cross-host ledger with atomic
/// compare-and-consume semantics.
#[derive(Debug)]
pub struct CompositionLedger {
    fleet_plan_sha256: String,
    fleet_inventory_sha256: String,
    global_alpha: Rational,
    consumed_deployment_plan_ids: HashSet<String>,
    consumed_stream_ids: HashSet<String>,
    consumed_scopes: HashSet<(u64, u64, u64)>,
    asset_indices: HashMap<u64, String>,
    asset_ids: HashMap<String, u64>,
    channel_indices: HashMap<(u64, u64), String>,
    channel_ids: HashMap<(String, String), u64>,
    deployment_indices: HashMap<(u64, u64, u64), String>,
    deployment_ids: HashMap<(String, String, String), u64>,
    receipts: Vec<CompositionReceipt>,
}

impl CompositionLedger {
    pub fn new(fleet_plan: &FleetBudgetPlan) -> Self {
        Self {
            fleet_plan_sha256: fleet_plan.fleet_plan_sha256.clone(),
            fleet_inventory_sha256: fleet_plan.fleet_inventory_sha256.clone(),
            global_alpha: fleet_plan.global_alpha,
            consumed_deployment_plan_ids: HashSet::new(),
            consumed_stream_ids: HashSet::new(),
            consumed_scopes: HashSet::new(),
            asset_indices: HashMap::new(),
            asset_ids: HashMap::new(),
            channel_indices: HashMap::new(),
            channel_ids: HashMap::new(),
            deployment_indices: HashMap::new(),
            deployment_ids: HashMap::new(),
            receipts: Vec::new(),
        }
    }

    /// Registers one immutable deployment plan. Every validation and receipt
    /// calculation is completed before ledger state is mutated.
    pub fn register_once(
        &mut self,
        fleet_plan: &FleetBudgetPlan,
        plan: &ComposedRiskPlan,
    ) -> Result<CompositionReceipt, RiskBudgetError> {
        let receipt = plan.receipt();
        if self.fleet_plan_sha256 != fleet_plan.fleet_plan_sha256
            || self.fleet_inventory_sha256 != fleet_plan.fleet_inventory_sha256
            || self.global_alpha != fleet_plan.global_alpha
            || receipt.fleet_plan_sha256 != fleet_plan.fleet_plan_sha256
            || receipt.fleet_inventory_sha256 != fleet_plan.fleet_inventory_sha256
            || receipt.global_alpha != fleet_plan.global_alpha
        {
            return Err(RiskBudgetError::FleetPlanMismatch);
        }

        let scope = &receipt.scope;
        validate_scope(scope)?;
        let plan_id = &receipt.bindings.deployment_plan_sha256;
        if self.consumed_deployment_plan_ids.contains(plan_id) {
            return Err(RiskBudgetError::DuplicateDeploymentPlan);
        }
        if self.consumed_stream_ids.contains(&scope.stream_id) {
            return Err(RiskBudgetError::DuplicateStreamId);
        }
        let scope_key = (
            scope.asset_index,
            scope.channel_index,
            scope.deployment_index,
        );
        if self.consumed_scopes.contains(&scope_key) {
            return Err(RiskBudgetError::DuplicateScope);
        }

        // Enforce one-to-one identity/index bindings at each hierarchy level.
        if self
            .asset_indices
            .get(&scope.asset_index)
            .is_some_and(|asset_id| asset_id != &scope.asset_id)
            || self
                .asset_ids
                .get(&scope.asset_id)
                .is_some_and(|asset_index| *asset_index != scope.asset_index)
        {
            return Err(RiskBudgetError::ScopeIdentityMismatch);
        }

        let channel_index_key = (scope.asset_index, scope.channel_index);
        let channel_identity_key = (scope.asset_id.clone(), scope.channel_id.clone());
        if self
            .channel_indices
            .get(&channel_index_key)
            .is_some_and(|channel_id| channel_id != &scope.channel_id)
            || self
                .channel_ids
                .get(&channel_identity_key)
                .is_some_and(|channel_index| *channel_index != scope.channel_index)
        {
            return Err(RiskBudgetError::ScopeIdentityMismatch);
        }

        if self
            .deployment_indices
            .get(&scope_key)
            .is_some_and(|deployment_id| deployment_id != &scope.deployment_id)
            || self
                .deployment_ids
                .get(&(
                    scope.asset_id.clone(),
                    scope.channel_id.clone(),
                    scope.deployment_id.clone(),
                ))
                .is_some_and(|deployment_index| *deployment_index != scope.deployment_index)
        {
            return Err(RiskBudgetError::ScopeIdentityMismatch);
        }

        // Receipt cloning happens before mutation; all later inserts are
        // infallible allocations apart from process-level memory exhaustion.
        let committed_receipt = receipt.clone();
        self.consumed_deployment_plan_ids.insert(plan_id.clone());
        self.consumed_stream_ids.insert(scope.stream_id.clone());
        self.consumed_scopes.insert(scope_key);
        self.asset_indices
            .insert(scope.asset_index, scope.asset_id.clone());
        self.asset_ids
            .insert(scope.asset_id.clone(), scope.asset_index);
        self.channel_indices
            .insert(channel_index_key, scope.channel_id.clone());
        self.channel_ids
            .insert(channel_identity_key, scope.channel_index);
        self.deployment_indices
            .insert(scope_key, scope.deployment_id.clone());
        self.deployment_ids.insert(
            (
                scope.asset_id.clone(),
                scope.channel_id.clone(),
                scope.deployment_id.clone(),
            ),
            scope.deployment_index,
        );
        self.receipts.push(committed_receipt.clone());
        Ok(committed_receipt)
    }

    pub fn registration_count(&self) -> usize {
        self.receipts.len()
    }

    pub fn receipts(&self) -> &[CompositionReceipt] {
        &self.receipts
    }

    pub fn has_deployment_plan(&self, plan_sha256: &str) -> bool {
        self.consumed_deployment_plan_ids.contains(plan_sha256)
    }

    pub fn has_stream_id(&self, stream_id: &str) -> bool {
        self.consumed_stream_ids.contains(stream_id)
    }

    pub fn has_scope(&self, scope: &FleetScope) -> bool {
        self.consumed_scopes.contains(&(
            scope.asset_index,
            scope.channel_index,
            scope.deployment_index,
        ))
    }
}

fn validate_probability(value: Rational) -> Result<(), RiskBudgetError> {
    if value.numerator == 0 || value.denominator == 0 || value.numerator >= value.denominator {
        Err(RiskBudgetError::InvalidProbability)
    } else {
        Ok(())
    }
}

fn validate_scope(scope: &FleetScope) -> Result<(), RiskBudgetError> {
    validate_identifier(&scope.stream_id, "stream_id")?;
    validate_identifier(&scope.asset_id, "asset_id")?;
    validate_identifier(&scope.channel_id, "channel_id")?;
    validate_identifier(&scope.deployment_id, "deployment_id")?;
    if scope.asset_index == 0 || scope.channel_index == 0 || scope.deployment_index == 0 {
        return Err(RiskBudgetError::InvalidIndex);
    }
    Ok(())
}

fn validate_bindings(bindings: &RiskPlanBindings) -> Result<(), RiskBudgetError> {
    validate_digest(
        &bindings.deployment_plan_sha256,
        "deployment_plan_sha256",
    )?;
    validate_digest(
        &bindings.reference_manifest_sha256,
        "reference_manifest_sha256",
    )?;
    validate_digest(
        &bindings.target_envelope_sha256,
        "target_envelope_sha256",
    )?;
    validate_digest(&bindings.monitor_build_sha256, "monitor_build_sha256")?;
    validate_digest(
        &bindings.representation_inventory_sha256,
        "representation_inventory_sha256",
    )?;
    validate_digest(
        &bindings.calibration_method_sha256,
        "calibration_method_sha256",
    )?;
    Ok(())
}

fn validate_representation_weights(
    members: &[RepresentationWeight],
) -> Result<(), RiskBudgetError> {
    if members.is_empty() {
        return Err(RiskBudgetError::EmptyRepresentationInventory);
    }
    if members.len() > MAX_REPRESENTATION_MEMBERS {
        return Err(RiskBudgetError::TooManyRepresentationMembers);
    }

    let mut ids = HashSet::with_capacity(members.len());
    let mut total = None;
    for member in members {
        validate_identifier(&member.member_id, "representation_member_id")?;
        validate_probability_or_one(member.weight)?;
        if !ids.insert(member.member_id.as_str()) {
            return Err(RiskBudgetError::DuplicateRepresentationMember);
        }
        total = Some(match total {
            None => member.weight,
            Some(value) => value
                .checked_add(member.weight)
                .map_err(|_| RiskBudgetError::RepresentationWeightSumMismatch)?,
        });
    }
    if total != Some(Rational::one()) {
        return Err(RiskBudgetError::RepresentationWeightSumMismatch);
    }
    Ok(())
}

fn validate_probability_or_one(value: Rational) -> Result<(), RiskBudgetError> {
    if value.numerator == 0 || value.denominator == 0 || value.numerator > value.denominator {
        Err(RiskBudgetError::InvalidFraction)
    } else {
        Ok(())
    }
}

fn validate_identifier(value: &str, label: &'static str) -> Result<(), RiskBudgetError> {
    if value.trim().is_empty() || value.len() > MAX_ID_BYTES {
        Err(RiskBudgetError::InvalidIdentifier(label))
    } else {
        Ok(())
    }
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), RiskBudgetError> {
    let valid = value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(RiskBudgetError::InvalidDigest(label))
    }
}

fn telescoping_factor(index_one_based: u64) -> Result<u128, RiskBudgetError> {
    if index_one_based == 0 {
        return Err(RiskBudgetError::InvalidIndex);
    }
    let index = u128::from(index_one_based);
    let next = index
        .checked_add(1)
        .ok_or(RiskBudgetError::ArithmeticOverflow)?;
    index
        .checked_mul(next)
        .ok_or(RiskBudgetError::ArithmeticOverflow)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Continued-fraction comparison avoids cross-products and therefore avoids
/// overflow even for normalized u128 fractions.
fn compare_fractions(
    mut left_numerator: u128,
    mut left_denominator: u128,
    mut right_numerator: u128,
    mut right_denominator: u128,
) -> Ordering {
    let mut reversed = false;
    loop {
        let left_whole = left_numerator / left_denominator;
        let right_whole = right_numerator / right_denominator;
        if left_whole != right_whole {
            let ordering = left_whole.cmp(&right_whole);
            return if reversed { ordering.reverse() } else { ordering };
        }

        let left_remainder = left_numerator % left_denominator;
        let right_remainder = right_numerator % right_denominator;
        if left_remainder == 0 || right_remainder == 0 {
            let ordering = match (left_remainder == 0, right_remainder == 0) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                (false, false) => unreachable!(),
            };
            return if reversed { ordering.reverse() } else { ordering };
        }

        left_numerator = left_denominator;
        left_denominator = left_remainder;
        right_numerator = right_denominator;
        right_denominator = right_remainder;
        reversed = !reversed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLEET_PLAN_SHA: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const FLEET_INVENTORY_SHA: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn fleet_plan() -> FleetBudgetPlan {
        FleetBudgetPlan::new(
            Rational::probability(1, 20).unwrap(),
            FLEET_PLAN_SHA.into(),
            FLEET_INVENTORY_SHA.into(),
        )
        .unwrap()
    }

    fn scope(asset: u64, channel: u64, deployment: u64, suffix: &str) -> FleetScope {
        FleetScope {
            stream_id: format!("stream-{suffix}"),
            asset_id: format!("asset-{asset}"),
            channel_id: format!("channel-{channel}"),
            deployment_id: format!("deployment-{deployment}"),
            asset_index: asset,
            channel_index: channel,
            deployment_index: deployment,
        }
    }

    fn digest(value: u64) -> String {
        format!("{value:064x}")
    }

    fn bindings(value: u64) -> RiskPlanBindings {
        RiskPlanBindings {
            deployment_plan_sha256: digest(value),
            reference_manifest_sha256: digest(value + 100),
            target_envelope_sha256: digest(value + 200),
            monitor_build_sha256: digest(value + 300),
            representation_inventory_sha256: digest(value + 400),
            calibration_method_sha256: digest(value + 500),
        }
    }

    fn equal_weights() -> Vec<RepresentationWeight> {
        vec![
            RepresentationWeight {
                member_id: "context".into(),
                weight: Rational::new(1, 2).unwrap(),
            },
            RepresentationWeight {
                member_id: "product".into(),
                weight: Rational::new(1, 2).unwrap(),
            },
        ]
    }

    fn plan_for(
        fleet: &FleetBudgetPlan,
        deployment: u64,
        scope: FleetScope,
        alpha_test: Rational,
        delta: Rational,
        weights: Vec<RepresentationWeight>,
    ) -> Result<ComposedRiskPlan, RiskBudgetError> {
        ComposedRiskPlan::new(
            fleet,
            scope,
            alpha_test,
            delta,
            weights,
            bindings(deployment),
        )
    }

    #[test]
    fn rational_values_are_reduced_and_comparison_is_exact() {
        assert_eq!(Rational::new(2, 40).unwrap(), Rational::new(1, 20).unwrap());
        assert_eq!(
            Rational::new(1, 3).unwrap().compare(Rational::new(2, 5).unwrap()),
            Ordering::Less
        );
        assert_eq!(
            Rational::new(u128::MAX - 1, u128::MAX)
                .unwrap()
                .compare(Rational::new(u128::MAX - 2, u128::MAX - 1).unwrap()),
            Ordering::Greater
        );
    }

    #[test]
    fn local_allocation_uses_all_three_telescoping_levels() {
        let fleet = fleet_plan();
        let local = fleet
            .local_alpha_for(&scope(1, 2, 3, "one"))
            .unwrap();
        assert_eq!(local, Rational::new(1, 2_880).unwrap());
    }

    #[test]
    fn one_dimensional_telescoping_partial_sums_are_exact() {
        for count in 1..=1_000u128 {
            let mut total = Rational::new(1, 2).unwrap();
            for index in 2..=count {
                let index_u64 = u64::try_from(index).unwrap();
                let term = Rational::new(1, telescoping_factor(index_u64).unwrap()).unwrap();
                total = total.checked_add(term).unwrap();
            }
            assert_eq!(
                total,
                Rational::new(count, count + 1).unwrap()
            );
        }
    }

    #[test]
    fn finite_three_level_grid_never_exceeds_global_alpha() {
        let fleet = fleet_plan();
        let mut total = None;
        for asset in 1..=4 {
            for channel in 1..=4 {
                for deployment in 1..=4 {
                    let local = fleet
                        .local_alpha_for(&scope(asset, channel, deployment, "grid"))
                        .unwrap();
                    total = Some(match total {
                        None => local,
                        Some(value) => value.checked_add(local).unwrap(),
                    });
                }
            }
        }
        assert_ne!(
            total.unwrap().compare(fleet.global_alpha()),
            Ordering::Greater
        );
    }

    #[test]
    fn exact_alpha_split_boundary_is_accepted() {
        let fleet = fleet_plan();
        let scope_value = scope(1, 1, 1, "boundary");
        let local = fleet.local_alpha_for(&scope_value).unwrap();
        let half = local.divided_by_integer(2).unwrap();
        let plan = plan_for(
            &fleet,
            1,
            scope_value,
            half,
            half,
            equal_weights(),
        )
        .unwrap();
        assert_eq!(plan.local_alpha(), local);
        assert_eq!(plan.alpha_test(), half);
        assert_eq!(plan.delta_reference_failure(), half);
    }

    #[test]
    fn alpha_split_over_budget_is_rejected() {
        let fleet = fleet_plan();
        let result = plan_for(
            &fleet,
            2,
            scope(1, 1, 1, "over"),
            Rational::probability(1, 100).unwrap(),
            Rational::probability(1, 100).unwrap(),
            equal_weights(),
        );
        assert_eq!(result, Err(RiskBudgetError::RiskBudgetExceeded));
    }

    #[test]
    fn representation_members_share_one_test_budget() {
        let fleet = fleet_plan();
        let scope_value = scope(1, 1, 1, "portfolio");
        let local = fleet.local_alpha_for(&scope_value).unwrap();
        let alpha_test = local.divided_by_integer(2).unwrap();
        let delta = local.divided_by_integer(2).unwrap();
        let weights = vec![
            RepresentationWeight {
                member_id: "context".into(),
                weight: Rational::new(1, 4).unwrap(),
            },
            RepresentationWeight {
                member_id: "continuous".into(),
                weight: Rational::new(1, 4).unwrap(),
            },
            RepresentationWeight {
                member_id: "multi-scale".into(),
                weight: Rational::new(1, 2).unwrap(),
            },
        ];
        let plan = plan_for(&fleet, 3, scope_value, alpha_test, delta, weights).unwrap();
        assert_eq!(plan.receipt().representation_weights.len(), 3);
        assert_eq!(plan.receipt().alpha_test, alpha_test);
        assert_eq!(plan.receipt().delta_reference_failure, delta);
        let weight_sum = plan
            .receipt()
            .representation_weights
            .iter()
            .map(|member| member.weight)
            .reduce(|left, right| left.checked_add(right).unwrap())
            .unwrap();
        assert_eq!(weight_sum, Rational::one());
    }

    #[test]
    fn representation_inventory_must_be_nonempty_unique_and_sum_to_one() {
        let fleet = fleet_plan();
        let local = fleet
            .local_alpha_for(&scope(1, 1, 1, "bad-weights"))
            .unwrap();
        let half = local.divided_by_integer(2).unwrap();
        assert_eq!(
            plan_for(
                &fleet,
                4,
                scope(1, 1, 1, "empty-weights"),
                half,
                half,
                vec![],
            ),
            Err(RiskBudgetError::EmptyRepresentationInventory)
        );

        let invalid = vec![
            RepresentationWeight {
                member_id: "one".into(),
                weight: Rational::new(1, 2).unwrap(),
            },
            RepresentationWeight {
                member_id: "two".into(),
                weight: Rational::new(1, 4).unwrap(),
            },
        ];
        assert_eq!(
            plan_for(
                &fleet,
                5,
                scope(1, 1, 1, "partial-weights"),
                half,
                half,
                invalid,
            ),
            Err(RiskBudgetError::RepresentationWeightSumMismatch)
        );

        let duplicate = vec![
            RepresentationWeight {
                member_id: "same".into(),
                weight: Rational::new(1, 2).unwrap(),
            },
            RepresentationWeight {
                member_id: "same".into(),
                weight: Rational::new(1, 2).unwrap(),
            },
        ];
        assert_eq!(
            plan_for(
                &fleet,
                6,
                scope(1, 1, 1, "duplicate-weights"),
                half,
                half,
                duplicate,
            ),
            Err(RiskBudgetError::DuplicateRepresentationMember)
        );
    }

    #[test]
    fn ledger_registration_is_consume_once_and_replay_safe_in_process() {
        let fleet = fleet_plan();
        let scope_value = scope(1, 1, 1, "registered");
        let local = fleet.local_alpha_for(&scope_value).unwrap();
        let half = local.divided_by_integer(2).unwrap();
        let plan = plan_for(
            &fleet,
            7,
            scope_value.clone(),
            half,
            half,
            equal_weights(),
        )
        .unwrap();
        let mut ledger = CompositionLedger::new(&fleet);
        ledger.register_once(&fleet, &plan).unwrap();
        assert_eq!(ledger.registration_count(), 1);
        assert!(ledger.has_stream_id("stream-registered"));
        assert!(ledger.has_scope(&scope_value));
        assert!(ledger.has_deployment_plan(&digest(7)));
        assert_eq!(
            ledger.register_once(&fleet, &plan),
            Err(RiskBudgetError::DuplicateDeploymentPlan)
        );
        assert_eq!(ledger.registration_count(), 1);
    }

    #[test]
    fn identity_index_remapping_is_rejected_without_mutation() {
        let fleet = fleet_plan();
        let first_scope = scope(1, 1, 1, "first");
        let local = fleet.local_alpha_for(&first_scope).unwrap();
        let half = local.divided_by_integer(2).unwrap();
        let first = plan_for(
            &fleet,
            8,
            first_scope,
            half,
            half,
            equal_weights(),
        )
        .unwrap();

        let mut remapped_scope = scope(1, 2, 1, "remapped");
        remapped_scope.asset_id = "different-asset".into();
        let remapped_local = fleet.local_alpha_for(&remapped_scope).unwrap();
        let remapped_half = remapped_local.divided_by_integer(2).unwrap();
        let remapped = plan_for(
            &fleet,
            9,
            remapped_scope,
            remapped_half,
            remapped_half,
            equal_weights(),
        )
        .unwrap();

        let mut ledger = CompositionLedger::new(&fleet);
        ledger.register_once(&fleet, &first).unwrap();
        assert_eq!(
            ledger.register_once(&fleet, &remapped),
            Err(RiskBudgetError::ScopeIdentityMismatch)
        );
        assert_eq!(ledger.registration_count(), 1);
    }

    #[test]
    fn wrong_fleet_plan_is_rejected() {
        let fleet = fleet_plan();
        let other = FleetBudgetPlan::new(
            Rational::probability(1, 10).unwrap(),
            digest(10),
            digest(11),
        )
        .unwrap();
        let scope_value = scope(1, 1, 1, "wrong-plan");
        let local = fleet.local_alpha_for(&scope_value).unwrap();
        let half = local.divided_by_integer(2).unwrap();
        let plan = plan_for(
            &fleet,
            12,
            scope_value,
            half,
            half,
            equal_weights(),
        )
        .unwrap();
        let mut ledger = CompositionLedger::new(&fleet);
        assert_eq!(
            ledger.register_once(&other, &plan),
            Err(RiskBudgetError::FleetPlanMismatch)
        );
        assert_eq!(ledger.registration_count(), 0);
    }

    #[test]
    fn large_denominator_overflow_fails_closed() {
        let fleet = FleetBudgetPlan::new(
            Rational::probability(1, u128::MAX).unwrap(),
            FLEET_PLAN_SHA.into(),
            FLEET_INVENTORY_SHA.into(),
        )
        .unwrap();
        assert_eq!(
            fleet.local_alpha_for(&scope(1, 1, 1, "overflow")),
            Err(RiskBudgetError::ArithmeticOverflow)
        );
    }

    #[test]
    fn malformed_digest_and_zero_index_fail_closed() {
        let fleet = FleetBudgetPlan::new(
            Rational::probability(1, 20).unwrap(),
            "bad".into(),
            FLEET_INVENTORY_SHA.into(),
        );
        assert_eq!(fleet, Err(RiskBudgetError::InvalidDigest("fleet_plan_sha256")));

        let valid = fleet_plan();
        let mut invalid_scope = scope(1, 1, 1, "zero-index");
        invalid_scope.asset_index = 0;
        assert_eq!(
            valid.local_alpha_for(&invalid_scope),
            Err(RiskBudgetError::InvalidIndex)
        );
    }
}
