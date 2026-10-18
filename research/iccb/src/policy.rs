//! Finite-horizon capital policy comparison over synthetic demand outcomes.
//!
//! This is an exhaustively enumerated *policy catalog*, not a general optimizer
//! and not a calibrated real-options model. It compares one-shot investment,
//! wait-and-invest, staged-and-adapt, and abandon while reporting the economic
//! consequences of delay rather than assuming that staging always wins.

const BPS: i128 = 10_000;
const BPS_SQUARED: i128 = 100_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemandCase {
    pub id: String,
    /// Probability in basis points; all cases must sum to 10_000.
    pub probability_bps: u32,
    pub full_capacity_cashflow_period1_minor: i64,
    pub full_capacity_cashflow_period2_minor: i64,
    pub staged_cashflow_period1_minor: i64,
    pub expanded_cashflow_period2_minor: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapitalPolicyScenario {
    pub id: String,
    /// Maximum authorized capital commitment, exclusive of financing/opportunity costs.
    pub capital_budget_minor: i64,
    pub full_commitment_minor: i64,
    pub staged_commitment_minor: i64,
    pub expansion_commitment_minor: i64,
    /// Explicit cost incurred for waiting one period (lost revenue, financing, option expiry, etc.).
    pub wait_cost_minor: i64,
    /// Period discount factor in basis points; 9_000 means 0.90 per period.
    pub discount_factor_bps: u32,
    pub demand_cases: Vec<DemandCase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CapitalPolicy {
    /// Conservative tie-break default; no capital is committed.
    Abandon,
    /// Pay full capex now and receive both period cashflows.
    CommitFullNow,
    /// Wait for the demand signal; invest at period 1 only when continuation value is positive.
    WaitThenInvest,
    /// Pay a first tranche now; after observing the signal, expand only when continuation value is positive.
    StageThenAdapt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyInfeasibility {
    CommitmentExceedsBudget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyInputError {
    EmptyScenarioId,
    EmptyDemandCases,
    EmptyDemandCaseId,
    ProbabilitiesDoNotSumTo10000,
    DiscountFactorOutOfRange,
    NegativeBudgetOrCommitment,
    CommitmentArithmeticOverflow,
    ExpectedValueOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyAssessment {
    pub policy: CapitalPolicy,
    pub feasible: bool,
    pub infeasibility: Option<PolicyInfeasibility>,
    pub maximum_authorized_commitment_minor: i64,
    /// None means the policy is infeasible under the declared capital budget.
    pub expected_npv_minor: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyComparison {
    pub scenario_id: String,
    pub assessments: Vec<PolicyAssessment>,
    pub best_policy_within_catalog: CapitalPolicy,
    /// Difference between the selected policy and the one-shot baseline.
    pub improvement_vs_one_shot_minor: Option<i64>,
}

impl CapitalPolicyScenario {
    pub fn evaluate(&self) -> Result<PolicyComparison, PolicyInputError> {
        self.validate()?;

        let mut assessments = vec![PolicyAssessment {
            policy: CapitalPolicy::Abandon,
            feasible: true,
            infeasibility: None,
            maximum_authorized_commitment_minor: 0,
            expected_npv_minor: Some(0),
        }];

        assessments.push(self.assess_policy(
            CapitalPolicy::CommitFullNow,
            self.full_commitment_minor,
        )?);

        let wait_result = self.assess_policy(
            CapitalPolicy::WaitThenInvest,
            self.full_commitment_minor,
        )?;
        assessments.push(wait_result);

        let staged_capital = self
            .staged_commitment_minor
            .checked_add(self.expansion_commitment_minor)
            .ok_or(PolicyInputError::CommitmentArithmeticOverflow)?;
        assessments.push(
            self.assess_policy(CapitalPolicy::StageThenAdapt, staged_capital)?,
        );

        // Abandon is first and wins exact ties. This is a deterministic, conservative tie break.
        let mut best = CapitalPolicy::Abandon;
        let mut best_value = 0_i64;
        for assessment in &assessments {
            if let Some(value) = assessment.expected_npv_minor {
                if value > best_value {
                    best = assessment.policy;
                    best_value = value;
                }
            }
        }

        let one_shot_value = assessments
            .iter()
            .find(|assessment| assessment.policy == CapitalPolicy::CommitFullNow)
            .and_then(|assessment| assessment.expected_npv_minor);

        let improvement_vs_one_shot_minor = match one_shot_value {
            Some(value) => Some(
                best_value
                    .checked_sub(value)
                    .ok_or(PolicyInputError::ExpectedValueOverflow)?,
            ),
            None => None,
        };

        Ok(PolicyComparison {
            scenario_id: self.id.clone(),
            assessments,
            best_policy_within_catalog: best,
            improvement_vs_one_shot_minor,
        })
    }

    fn validate(&self) -> Result<(), PolicyInputError> {
        if self.id.trim().is_empty() {
            return Err(PolicyInputError::EmptyScenarioId);
        }
        if self.demand_cases.is_empty() {
            return Err(PolicyInputError::EmptyDemandCases);
        }
        if self.discount_factor_bps > 10_000 {
            return Err(PolicyInputError::DiscountFactorOutOfRange);
        }
        if self.capital_budget_minor < 0
            || self.full_commitment_minor < 0
            || self.staged_commitment_minor < 0
            || self.expansion_commitment_minor < 0
            || self.wait_cost_minor < 0
        {
            return Err(PolicyInputError::NegativeBudgetOrCommitment);
        }
        self.staged_commitment_minor
            .checked_add(self.expansion_commitment_minor)
            .ok_or(PolicyInputError::CommitmentArithmeticOverflow)?;

        let mut probability_sum = 0_u64;
        for case in &self.demand_cases {
            if case.id.trim().is_empty() {
                return Err(PolicyInputError::EmptyDemandCaseId);
            }
            probability_sum += u64::from(case.probability_bps);
        }
        if probability_sum != 10_000 {
            return Err(PolicyInputError::ProbabilitiesDoNotSumTo10000);
        }
        Ok(())
    }

    fn assess_policy(
        &self,
        policy: CapitalPolicy,
        maximum_authorized_commitment_minor: i64,
    ) -> Result<PolicyAssessment, PolicyInputError> {
        let infeasibility = if maximum_authorized_commitment_minor > self.capital_budget_minor {
            Some(PolicyInfeasibility::CommitmentExceedsBudget)
        } else {
            None
        };

        if let Some(reason) = infeasibility {
            return Ok(PolicyAssessment {
                policy,
                feasible: false,
                infeasibility: Some(reason),
                maximum_authorized_commitment_minor,
                expected_npv_minor: None,
            });
        }

        let mut weighted_value = 0_i128;
        for case in &self.demand_cases {
            let case_value = match policy {
                CapitalPolicy::Abandon => 0,
                CapitalPolicy::CommitFullNow => {
                    -(i128::from(self.full_commitment_minor))
                        + discount_period1(
                            case.full_capacity_cashflow_period1_minor,
                            self.discount_factor_bps,
                        )
                        + discount_period2(
                            case.full_capacity_cashflow_period2_minor,
                            self.discount_factor_bps,
                        )
                }
                CapitalPolicy::WaitThenInvest => {
                    let continuation_pv = -discount_period1(
                        self.full_commitment_minor,
                        self.discount_factor_bps,
                    ) + discount_period2(
                        case.full_capacity_cashflow_period2_minor,
                        self.discount_factor_bps,
                    );
                    -i128::from(self.wait_cost_minor) + continuation_pv.max(0)
                }
                CapitalPolicy::StageThenAdapt => {
                    let continuation_pv = -discount_period1(
                        self.expansion_commitment_minor,
                        self.discount_factor_bps,
                    ) + discount_period2(
                        case.expanded_cashflow_period2_minor,
                        self.discount_factor_bps,
                    );
                    -i128::from(self.staged_commitment_minor)
                        + discount_period1(
                            case.staged_cashflow_period1_minor,
                            self.discount_factor_bps,
                        )
                        + continuation_pv.max(0)
                }
            };
            weighted_value += i128::from(case.probability_bps) * case_value;
        }

        // Conservative fixed-point arithmetic: Euclidean division rounds negative
        // expected values downward instead of making losses look artificially smaller.
        let expected = weighted_value.div_euclid(BPS);
        let expected_npv_minor =
            i64::try_from(expected).map_err(|_| PolicyInputError::ExpectedValueOverflow)?;

        Ok(PolicyAssessment {
            policy,
            feasible: true,
            infeasibility: None,
            maximum_authorized_commitment_minor,
            expected_npv_minor: Some(expected_npv_minor),
        })
    }
}

fn discount_period1(amount_minor: i64, discount_factor_bps: u32) -> i128 {
    (i128::from(amount_minor) * i128::from(discount_factor_bps)).div_euclid(BPS)
}

fn discount_period2(amount_minor: i64, discount_factor_bps: u32) -> i128 {
    let factor = i128::from(discount_factor_bps);
    (i128::from(amount_minor) * factor * factor).div_euclid(BPS_SQUARED)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> CapitalPolicyScenario {
        CapitalPolicyScenario {
            id: "ICCB-POLICY-UNCERTAINTY-001".into(),
            capital_budget_minor: 120,
            full_commitment_minor: 100,
            staged_commitment_minor: 15,
            expansion_commitment_minor: 90,
            wait_cost_minor: 70,
            discount_factor_bps: 9_000,
            demand_cases: vec![
                DemandCase {
                    id: "high-demand".into(),
                    probability_bps: 5_000,
                    full_capacity_cashflow_period1_minor: 40,
                    full_capacity_cashflow_period2_minor: 200,
                    staged_cashflow_period1_minor: 30,
                    expanded_cashflow_period2_minor: 180,
                },
                DemandCase {
                    id: "low-demand".into(),
                    probability_bps: 5_000,
                    full_capacity_cashflow_period1_minor: 10,
                    full_capacity_cashflow_period2_minor: 20,
                    staged_cashflow_period1_minor: 10,
                    expanded_cashflow_period2_minor: 20,
                },
            ],
        }
    }

    fn value(comparison: &PolicyComparison, policy: CapitalPolicy) -> i64 {
        comparison
            .assessments
            .iter()
            .find(|assessment| assessment.policy == policy)
            .and_then(|assessment| assessment.expected_npv_minor)
            .expect("policy should be feasible in this fixture")
    }

    #[test]
    fn staged_adaptive_policy_can_win_when_the_wait_cost_is_high() {
        let comparison = scenario().evaluate().unwrap();
        assert_eq!(
            comparison.best_policy_within_catalog,
            CapitalPolicy::StageThenAdapt
        );
        assert!(value(&comparison, CapitalPolicy::StageThenAdapt)
            > value(&comparison, CapitalPolicy::CommitFullNow));
        assert!(value(&comparison, CapitalPolicy::StageThenAdapt)
            > value(&comparison, CapitalPolicy::WaitThenInvest));
    }

    #[test]
    fn waiting_can_win_when_the_information_value_exceeds_delay_cost() {
        let mut input = scenario();
        input.wait_cost_minor = 0;
        let comparison = input.evaluate().unwrap();
        assert_eq!(
            comparison.best_policy_within_catalog,
            CapitalPolicy::WaitThenInvest
        );
    }

    #[test]
    fn one_shot_can_win_when_early_cashflows_are_strong_and_likely() {
        let mut input = scenario();
        input.wait_cost_minor = 100;
        input.demand_cases = vec![
            DemandCase {
                id: "high".into(),
                probability_bps: 9_500,
                full_capacity_cashflow_period1_minor: 200,
                full_capacity_cashflow_period2_minor: 200,
                staged_cashflow_period1_minor: 10,
                expanded_cashflow_period2_minor: 160,
            },
            DemandCase {
                id: "low".into(),
                probability_bps: 500,
                full_capacity_cashflow_period1_minor: 100,
                full_capacity_cashflow_period2_minor: 100,
                staged_cashflow_period1_minor: 10,
                expanded_cashflow_period2_minor: 80,
            },
        ];
        let comparison = input.evaluate().unwrap();
        assert_eq!(
            comparison.best_policy_within_catalog,
            CapitalPolicy::CommitFullNow
        );
    }

    #[test]
    fn budget_constraint_can_make_abandon_the_only_feasible_policy() {
        let mut input = scenario();
        input.capital_budget_minor = 50;
        let comparison = input.evaluate().unwrap();
        assert_eq!(
            comparison.best_policy_within_catalog,
            CapitalPolicy::Abandon
        );
        assert!(comparison
            .assessments
            .iter()
            .filter(|assessment| assessment.policy != CapitalPolicy::Abandon)
            .all(|assessment| !assessment.feasible));
    }

    #[test]
    fn bad_probability_mass_fails_closed() {
        let mut input = scenario();
        input.demand_cases[0].probability_bps = 4_999;
        assert_eq!(
            input.evaluate(),
            Err(PolicyInputError::ProbabilitiesDoNotSumTo10000)
        );
    }

    #[test]
    fn same_input_produces_identical_policy_comparison() {
        let input = scenario();
        assert_eq!(input.evaluate(), input.evaluate());
    }
}
