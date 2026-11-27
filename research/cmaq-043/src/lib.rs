#![forbid(unsafe_code)]

//! Fleet-wide anytime risk-budget allocation.
//!
//! The hierarchy uses the telescoping allocation w_i = 1/(i(i+1)) at each
//! scope level. For asset, evidence channel, and deployment/restart indices,
//! the local alpha is alpha_global * w_asset * w_channel * w_deployment.
//! The weights sum to one over each countable index set, so all local alpha
//! allocations sum to at most the global alpha. No independence assumption
//! between monitor streams is needed for the union-bound claim.

use std::collections::HashSet;

const ALLOCATION_RULE_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BudgetError {
    InvalidGlobalAlpha,
    InvalidDigest(&'static str),
    EmptyStreamId,
    EmptyAssetId,
    EmptyChannelId,
    EmptyDeploymentId,
    ZeroAssetIndex,
    ZeroChannelIndex,
    ZeroDeploymentIndex,
    ArithmeticOverflow,
    DuplicateStreamId,
    DuplicateScope,
    PlanMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BudgetScope {
    pub stream_id: String,
    pub asset_id: String,
    pub channel_id: String,
    pub deployment_id: String,
    /// One-based, precommitted priority index.
    pub asset_index: u64,
    /// One-based, precommitted priority index.
    pub channel_index: u64,
    /// One-based, precommitted deployment/restart ordinal.
    pub deployment_index: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocationReceipt {
    pub rule_version: u32,
    pub plan_sha256: String,
    pub inventory_sha256: String,
    pub stream_id: String,
    pub asset_id: String,
    pub channel_id: String,
    pub deployment_id: String,
    pub asset_index: u64,
    pub channel_index: u64,
    pub deployment_index: u64,
    pub global_alpha_numerator: u128,
    pub global_alpha_denominator: u128,
    /// Exact local alpha = alpha_numerator / alpha_denominator.
    pub alpha_numerator: u128,
    pub alpha_denominator: u128,
}

impl AllocationReceipt {
    /// Log of the Ville threshold 1/alpha. This is a display/interop value;
    /// exact policy comparisons should use the rational alpha fields.
    pub fn log_ville_threshold(&self) -> f64 {
        (self.alpha_denominator as f64).ln() - (self.alpha_numerator as f64).ln()
    }

    /// Exact reciprocal threshold as a rational numerator/denominator.
    pub fn ville_threshold_ratio(&self) -> (u128, u128) {
        (self.alpha_denominator, self.alpha_numerator)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetPlan {
    global_alpha_numerator: u128,
    global_alpha_denominator: u128,
    plan_sha256: String,
    inventory_sha256: String,
}

impl BudgetPlan {
    pub fn new(
        global_alpha_numerator: u128,
        global_alpha_denominator: u128,
        plan_sha256: String,
        inventory_sha256: String,
    ) -> Result<Self, BudgetError> {
        if global_alpha_numerator == 0
            || global_alpha_denominator == 0
            || global_alpha_numerator >= global_alpha_denominator
        {
            return Err(BudgetError::InvalidGlobalAlpha);
        }
        validate_digest(&plan_sha256, "plan_sha256")?;
        validate_digest(&inventory_sha256, "inventory_sha256")?;

        let divisor = gcd(global_alpha_numerator, global_alpha_denominator);
        Ok(Self {
            global_alpha_numerator: global_alpha_numerator / divisor,
            global_alpha_denominator: global_alpha_denominator / divisor,
            plan_sha256,
            inventory_sha256,
        })
    }

    /// Computes a local allocation without consuming it. Operational use
    /// should go through AllocationLedger::allocate_once.
    pub fn allocation_for(
        &self,
        scope: &BudgetScope,
    ) -> Result<AllocationReceipt, BudgetError> {
        validate_scope(scope)?;

        let factors = [
            telescoping_denominator(scope.asset_index)?,
            telescoping_denominator(scope.channel_index)?,
            telescoping_denominator(scope.deployment_index)?,
        ];

        let (alpha_numerator, alpha_denominator) = divide_fraction_by_factors(
            self.global_alpha_numerator,
            self.global_alpha_denominator,
            &factors,
        )?;

        Ok(AllocationReceipt {
            rule_version: ALLOCATION_RULE_VERSION,
            plan_sha256: self.plan_sha256.clone(),
            inventory_sha256: self.inventory_sha256.clone(),
            stream_id: scope.stream_id.clone(),
            asset_id: scope.asset_id.clone(),
            channel_id: scope.channel_id.clone(),
            deployment_id: scope.deployment_id.clone(),
            asset_index: scope.asset_index,
            channel_index: scope.channel_index,
            deployment_index: scope.deployment_index,
            global_alpha_numerator: self.global_alpha_numerator,
            global_alpha_denominator: self.global_alpha_denominator,
            alpha_numerator,
            alpha_denominator,
        })
    }

    pub fn global_alpha(&self) -> (u128, u128) {
        (self.global_alpha_numerator, self.global_alpha_denominator)
    }

    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    pub fn inventory_sha256(&self) -> &str {
        &self.inventory_sha256
    }
}

/// Consume-once allocator for one process/authority instance.
///
/// This in-memory ledger is not a substitute for a durable, authenticated
/// ledger across machines, process restarts, or deployment controllers.
#[derive(Debug)]
pub struct AllocationLedger {
    plan_sha256: String,
    inventory_sha256: String,
    global_alpha_numerator: u128,
    global_alpha_denominator: u128,
    consumed_stream_ids: HashSet<String>,
    consumed_scopes: HashSet<(u64, u64, u64)>,
    receipts: Vec<AllocationReceipt>,
}

impl AllocationLedger {
    /// Binds this in-memory ledger to exactly one immutable budget plan.
    pub fn new(plan: &BudgetPlan) -> Self {
        Self {
            plan_sha256: plan.plan_sha256.clone(),
            inventory_sha256: plan.inventory_sha256.clone(),
            global_alpha_numerator: plan.global_alpha_numerator,
            global_alpha_denominator: plan.global_alpha_denominator,
            consumed_stream_ids: HashSet::new(),
            consumed_scopes: HashSet::new(),
            receipts: Vec::new(),
        }
    }

    pub fn allocate_once(
        &mut self,
        plan: &BudgetPlan,
        scope: &BudgetScope,
    ) -> Result<AllocationReceipt, BudgetError> {
        validate_scope(scope)?;
        if self.plan_sha256 != plan.plan_sha256
            || self.inventory_sha256 != plan.inventory_sha256
            || self.global_alpha_numerator != plan.global_alpha_numerator
            || self.global_alpha_denominator != plan.global_alpha_denominator
        {
            return Err(BudgetError::PlanMismatch);
        }
        if self.consumed_stream_ids.contains(&scope.stream_id) {
            return Err(BudgetError::DuplicateStreamId);
        }
        let scope_key = (
            scope.asset_index,
            scope.channel_index,
            scope.deployment_index,
        );
        if self.consumed_scopes.contains(&scope_key) {
            return Err(BudgetError::DuplicateScope);
        }

        // Compute the receipt before mutating the ledger.
        let receipt = plan.allocation_for(scope)?;
        self.consumed_stream_ids.insert(scope.stream_id.clone());
        self.consumed_scopes.insert(scope_key);
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    pub fn allocation_count(&self) -> usize {
        self.receipts.len()
    }

    pub fn receipts(&self) -> &[AllocationReceipt] {
        &self.receipts
    }

    pub fn has_stream_id(&self, stream_id: &str) -> bool {
        self.consumed_stream_ids.contains(stream_id)
    }

    pub fn has_scope(&self, scope: &BudgetScope) -> bool {
        self.consumed_scopes.contains(&(
            scope.asset_index,
            scope.channel_index,
            scope.deployment_index,
        ))
    }
}

fn validate_scope(scope: &BudgetScope) -> Result<(), BudgetError> {
    if scope.stream_id.trim().is_empty() {
        return Err(BudgetError::EmptyStreamId);
    }
    if scope.asset_id.trim().is_empty() {
        return Err(BudgetError::EmptyAssetId);
    }
    if scope.channel_id.trim().is_empty() {
        return Err(BudgetError::EmptyChannelId);
    }
    if scope.deployment_id.trim().is_empty() {
        return Err(BudgetError::EmptyDeploymentId);
    }
    if scope.asset_index == 0 {
        return Err(BudgetError::ZeroAssetIndex);
    }
    if scope.channel_index == 0 {
        return Err(BudgetError::ZeroChannelIndex);
    }
    if scope.deployment_index == 0 {
        return Err(BudgetError::ZeroDeploymentIndex);
    }
    Ok(())
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), BudgetError> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(BudgetError::InvalidDigest(label))
    }
}

fn telescoping_denominator(index_one_based: u64) -> Result<u128, BudgetError> {
    if index_one_based == 0 {
        return Err(BudgetError::ZeroAssetIndex);
    }
    let index = u128::from(index_one_based);
    index
        .checked_mul(
            index
                .checked_add(1)
                .ok_or(BudgetError::ArithmeticOverflow)?,
        )
        .ok_or(BudgetError::ArithmeticOverflow)
}

/// Divides a rational by a set of integer factors, cancelling against the
/// numerator before every multiplication to reduce overflow exposure.
fn divide_fraction_by_factors(
    mut numerator: u128,
    mut denominator: u128,
    factors: &[u128],
) -> Result<(u128, u128), BudgetError> {
    for factor in factors {
        if *factor == 0 {
            return Err(BudgetError::ArithmeticOverflow);
        }
        let divisor = gcd(numerator, *factor);
        numerator /= divisor;
        let reduced_factor = *factor / divisor;
        denominator = denominator
            .checked_mul(reduced_factor)
            .ok_or(BudgetError::ArithmeticOverflow)?;
    }
    let divisor = gcd(numerator, denominator);
    Ok((numerator / divisor, denominator / divisor))
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAN_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const INVENTORY_SHA: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn plan() -> BudgetPlan {
        BudgetPlan::new(1, 20, PLAN_SHA.into(), INVENTORY_SHA.into()).unwrap()
    }

    fn scope(asset: u64, channel: u64, deployment: u64, stream: &str) -> BudgetScope {
        BudgetScope {
            stream_id: stream.into(),
            asset_id: format!("asset-{asset}"),
            channel_id: format!("channel-{channel}"),
            deployment_id: format!("deployment-{deployment}"),
            asset_index: asset,
            channel_index: channel,
            deployment_index: deployment,
        }
    }

    fn add_fractions(
        left: (u128, u128),
        right: (u128, u128),
    ) -> Result<(u128, u128), BudgetError> {
        let common = gcd(left.1, right.1);
        let left_factor = right.1 / common;
        let right_factor = left.1 / common;
        let numerator = left
            .0
            .checked_mul(left_factor)
            .and_then(|value| value.checked_add(right.0.checked_mul(right_factor)?))
            .ok_or(BudgetError::ArithmeticOverflow)?;
        let denominator = left
            .1
            .checked_mul(left_factor)
            .ok_or(BudgetError::ArithmeticOverflow)?;
        let divisor = gcd(numerator, denominator);
        Ok((numerator / divisor, denominator / divisor))
    }

    fn bounded_weight_sum(max_index: u64) -> Result<(u128, u128), BudgetError> {
        let mut total = (0u128, 1u128);
        for index in 1..=max_index {
            total = add_fractions(
                total,
                (1, telescoping_denominator(index)?),
            )?;
        }
        Ok(total)
    }

    #[test]
    fn global_alpha_is_reduced_exactly() {
        assert_eq!(
            BudgetPlan::new(2, 40, PLAN_SHA.into(), INVENTORY_SHA.into())
                .unwrap()
                .global_alpha(),
            (1, 20)
        );
    }

    #[test]
    fn telescoping_partial_sums_are_exact() {
        for count in 1..=1_000u128 {
            let index = u64::try_from(count).unwrap();
            let observed = bounded_weight_sum(index).unwrap();
            assert_eq!(observed, (count, count + 1));
        }
    }

    #[test]
    fn bounded_three_level_grid_never_exceeds_global_alpha() {
        let budget_plan = plan();
        let mut total_alpha = (0u128, 1u128);
        for asset in 1..=4u64 {
            for channel in 1..=4u64 {
                for deployment in 1..=4u64 {
                    let scope = scope(
                        asset,
                        channel,
                        deployment,
                        &format!("{asset}-{channel}-{deployment}"),
                    );
                    let receipt = budget_plan.allocation_for(&scope).unwrap();
                    total_alpha = add_fractions(
                        total_alpha,
                        (receipt.alpha_numerator, receipt.alpha_denominator),
                    )
                    .unwrap();
                }
            }
        }

        // The finite 4×4×4 grid covers (4/5)^3 of the countable allocation.
        assert_eq!(total_alpha, (64, 2_500));
        assert!(total_alpha.0 * 20 <= total_alpha.1);
    }

    #[test]
    fn local_allocation_is_exact_and_threshold_is_its_reciprocal() {
        let receipt = plan().allocation_for(&scope(1, 1, 1, "stream-1")).unwrap();
        assert_eq!((receipt.alpha_numerator, receipt.alpha_denominator), (1, 160));
        assert_eq!(receipt.ville_threshold_ratio(), (160, 1));
        assert!((receipt.log_ville_threshold() - 160.0f64.ln()).abs() < 1e-12);
        assert_eq!(receipt.rule_version, 1);
    }

    #[test]
    fn later_indices_receive_no_more_risk_budget() {
        let plan = plan();
        let first = plan.allocation_for(&scope(1, 1, 1, "stream-1")).unwrap();
        let later_asset = plan.allocation_for(&scope(2, 1, 1, "stream-2")).unwrap();
        let later_channel = plan.allocation_for(&scope(1, 2, 1, "stream-3")).unwrap();
        let later_deployment = plan.allocation_for(&scope(1, 1, 2, "stream-4")).unwrap();
        assert_eq!(
            (first.alpha_numerator, first.alpha_denominator),
            (1, 160)
        );
        assert_eq!(
            (later_asset.alpha_numerator, later_asset.alpha_denominator),
            (1, 480)
        );
        assert_eq!(
            (later_channel.alpha_numerator, later_channel.alpha_denominator),
            (1, 480)
        );
        assert_eq!(
            (later_deployment.alpha_numerator, later_deployment.alpha_denominator),
            (1, 480)
        );
    }

    #[test]
    fn ledger_consumes_each_scope_and_stream_once() {
        let plan = plan();
        let mut ledger = AllocationLedger::new(&plan);
        let first_scope = scope(1, 1, 1, "stream-1");
        let receipt = ledger.allocate_once(&plan, &first_scope).unwrap();
        assert_eq!(ledger.allocation_count(), 1);
        assert_eq!(ledger.receipts(), &[receipt]);
        assert!(ledger.has_stream_id("stream-1"));
        assert!(ledger.has_scope(&first_scope));
        assert_eq!(
            ledger.allocate_once(&plan, &first_scope),
            Err(BudgetError::DuplicateStreamId)
        );

        let other_stream_same_scope = scope(1, 1, 1, "stream-2");
        assert_eq!(
            ledger.allocate_once(&plan, &other_stream_same_scope),
            Err(BudgetError::DuplicateScope)
        );
        assert_eq!(ledger.allocation_count(), 1);
    }

    #[test]
    fn same_stream_id_cannot_claim_another_scope() {
        let plan = plan();
        let mut ledger = AllocationLedger::new(&plan);
        ledger.allocate_once(&plan, &scope(1, 1, 1, "same-stream")).unwrap();
        assert_eq!(
            ledger.allocate_once(&plan, &scope(2, 1, 1, "same-stream")),
            Err(BudgetError::DuplicateStreamId)
        );
    }

    #[test]
    fn ledger_is_bound_to_exactly_one_budget_plan() {
        let first_plan = plan();
        let other_plan =
            BudgetPlan::new(1, 20, INVENTORY_SHA.into(), PLAN_SHA.into()).unwrap();
        let mut ledger = AllocationLedger::new(&first_plan);
        assert_eq!(
            ledger.allocate_once(&other_plan, &scope(1, 1, 1, "stream")),
            Err(BudgetError::PlanMismatch)
        );
        assert_eq!(ledger.allocation_count(), 0);
    }

    #[test]
    fn invalid_indices_and_empty_ids_are_rejected() {
        let plan = plan();
        assert_eq!(
            plan.allocation_for(&scope(0, 1, 1, "stream")),
            Err(BudgetError::ZeroAssetIndex)
        );
        assert_eq!(
            plan.allocation_for(&scope(1, 0, 1, "stream")),
            Err(BudgetError::ZeroChannelIndex)
        );
        assert_eq!(
            plan.allocation_for(&scope(1, 1, 0, "stream")),
            Err(BudgetError::ZeroDeploymentIndex)
        );
        let mut invalid = scope(1, 1, 1, "");
        assert_eq!(
            plan.allocation_for(&invalid),
            Err(BudgetError::EmptyStreamId)
        );
        invalid = scope(1, 1, 1, "stream");
        invalid.asset_id = " ".into();
        assert_eq!(
            plan.allocation_for(&invalid),
            Err(BudgetError::EmptyAssetId)
        );
    }

    #[test]
    fn malformed_or_invalid_global_bindings_are_rejected() {
        assert_eq!(
            BudgetPlan::new(0, 20, PLAN_SHA.into(), INVENTORY_SHA.into()).err(),
            Some(BudgetError::InvalidGlobalAlpha)
        );
        assert_eq!(
            BudgetPlan::new(1, 1, PLAN_SHA.into(), INVENTORY_SHA.into()).err(),
            Some(BudgetError::InvalidGlobalAlpha)
        );
        assert_eq!(
            BudgetPlan::new(1, 20, "bad".into(), INVENTORY_SHA.into()).err(),
            Some(BudgetError::InvalidDigest("plan_sha256"))
        );
    }

    #[test]
    fn arithmetic_overflow_fails_closed() {
        let extreme = BudgetPlan::new(
            1,
            u128::MAX,
            PLAN_SHA.into(),
            INVENTORY_SHA.into(),
        )
        .unwrap();
        assert_eq!(
            extreme.allocation_for(&scope(u64::MAX, u64::MAX, u64::MAX, "overflow")),
            Err(BudgetError::ArithmeticOverflow)
        );
    }

    #[test]
    fn ledger_does_not_mutate_on_allocation_overflow() {
        let extreme = BudgetPlan::new(
            1,
            u128::MAX,
            PLAN_SHA.into(),
            INVENTORY_SHA.into(),
        )
        .unwrap();
        let mut ledger = AllocationLedger::new(&extreme);
        let scope = scope(u64::MAX, u64::MAX, u64::MAX, "overflow");
        assert_eq!(
            ledger.allocate_once(&extreme, &scope),
            Err(BudgetError::ArithmeticOverflow)
        );
        assert_eq!(ledger.allocation_count(), 0);
        assert!(!ledger.has_stream_id("overflow"));
    }
}
