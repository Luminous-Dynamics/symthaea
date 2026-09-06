// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ESE-A4: deterministic synthetic economic mechanism-recovery fixture.
//!
//! This module is intentionally a worked synthetic scenario rather than a
//! general economic simulator. Two genuinely different model paradigms consume
//! the same ETIR claim and preserve the same exact accounting invariant. Their
//! outputs are synthetic model behavior only and carry no empirical, policy, or
//! governance authority.

use symthaea_economics::{
    AccountId, ClaimId, DoubleEntryLedger, EconomicClaim, EconomicVariable, EconomicsError,
    EmpiricalClaim, EmpiricalClaimMode, FalsificationCriterion, JournalEntry, MechanismId,
    MechanismSpec, ModelAdapterDeclaration, ModelId, ModelParadigm, Posting, Prediction,
    PredictionId, ResponseDirection, Result, StateDomain, TheoryId, TheoryIr, UnitId, VariableId,
};

/// Fixed demand contraction used by the public A4 synthetic fixture: 20%.
pub const A4_DEMAND_SHOCK_BPS: u16 = 2_000;
const BASIS_POINTS: u16 = 10_000;
const BASELINE_DEMAND_ATOMS: u64 = 1_000;
const BASELINE_EMPLOYMENT: u64 = 10;
const DEMAND_ATOMS_PER_JOB: u64 = 100;
const WAGE_ATOMS_PER_JOB: u64 = 50;

const DEMAND_VARIABLE: &str = "macro:nominal_demand";
const EMPLOYMENT_VARIABLE: &str = "labor:employment";
const RIGIDITY_MECHANISM: &str = "nominal_rigidity";
const DEMAND_EMPLOYMENT_CLAIM: &str = "demand_employment";
const EMPLOYMENT_PREDICTION: &str = "employment_response";
const THEORY_ID: &str = "synthetic:nominal_rigidity_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SyntheticHousehold {
    baseline_budget_atoms: u64,
    reservation_demand_atoms: u64,
}

const HOUSEHOLDS: [SyntheticHousehold; 4] = [
    SyntheticHousehold {
        baseline_budget_atoms: 400,
        reservation_demand_atoms: 250,
    },
    SyntheticHousehold {
        baseline_budget_atoms: 300,
        reservation_demand_atoms: 250,
    },
    SyntheticHousehold {
        baseline_budget_atoms: 200,
        reservation_demand_atoms: 150,
    },
    SyntheticHousehold {
        baseline_budget_atoms: 100,
        reservation_demand_atoms: 80,
    },
];

/// Observable synthetic output shared by both A4 model paradigms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyntheticEconomicOutcome {
    pub baseline_demand_atoms: u64,
    pub shocked_demand_atoms: u64,
    pub baseline_employment: u64,
    pub shocked_employment: u64,
}

impl SyntheticEconomicOutcome {
    pub fn employment_decreased(self) -> bool {
        self.shocked_employment < self.baseline_employment
    }
}

/// One model's declaration, deterministic synthetic output, and exact ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticEconomicRun {
    pub adapter: ModelAdapterDeclaration,
    pub outcome: SyntheticEconomicOutcome,
    pub ledger: DoubleEntryLedger,
}

/// Complete A4 comparison. Both models are bound to `theory`, but neither model
/// owns the theory or upgrades its synthetic output into external evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticEconomicComparison {
    pub theory: TheoryIr,
    pub agent_based: SyntheticEconomicRun,
    pub system_dynamics: SyntheticEconomicRun,
}

fn invalid(context: &'static str) -> EconomicsError {
    EconomicsError::InvalidParameter { context }
}

fn numerical(context: &'static str) -> EconomicsError {
    EconomicsError::NumericalFailure { context }
}

fn reduce_by_basis_points(amount: u64, reduction_bps: u16) -> Result<u64> {
    if reduction_bps > BASIS_POINTS {
        return Err(invalid("synthetic demand shock exceeds 100 percent"));
    }
    let retained = u128::from(BASIS_POINTS - reduction_bps);
    let scaled = u128::from(amount)
        .checked_mul(retained)
        .ok_or(numerical("synthetic demand shock multiplication overflow"))?
        / u128::from(BASIS_POINTS);
    u64::try_from(scaled).map_err(|_| numerical("synthetic demand shock conversion overflow"))
}

fn ceil_div(numerator: u64, denominator: u64) -> Result<u64> {
    if denominator == 0 {
        return Err(invalid("synthetic employment denominator"));
    }
    Ok(numerator.div_ceil(denominator))
}

fn variable(
    id: &str,
    domain: StateDomain,
    unit: &str,
    description: &str,
) -> Result<EconomicVariable> {
    EconomicVariable::new(
        VariableId::new(id)?,
        domain,
        UnitId::new(unit)?,
        description,
    )
}

/// Build the exact shared ETIR theory consumed by both A4 paradigms.
pub fn a4_theory() -> Result<TheoryIr> {
    let demand = variable(
        DEMAND_VARIABLE,
        StateDomain::Financial,
        "currency_atoms_per_period",
        "Nominal final demand in the synthetic fixture",
    )?;
    let employment = variable(
        EMPLOYMENT_VARIABLE,
        StateDomain::Institutional,
        "persons",
        "Employed synthetic workers",
    )?;
    let mechanism_id = MechanismId::new(RIGIDITY_MECHANISM)?;
    let claim = EconomicClaim::Empirical(EmpiricalClaim::new(
        ClaimId::new(DEMAND_EMPLOYMENT_CLAIM)?,
        "A negative nominal-demand shock reduces employment when nominal adjustment is rigid.",
        EmpiricalClaimMode::Mechanistic,
        "deterministic_A4_synthetic_fixture_only",
        vec![mechanism_id.clone()],
        vec![Prediction::new(
            PredictionId::new(EMPLOYMENT_PREDICTION)?,
            VariableId::new(EMPLOYMENT_VARIABLE)?,
            ResponseDirection::Decrease,
            "1_period",
            Some("negative_nominal_demand_shock".into()),
        )?],
        vec![FalsificationCriterion::new(
            PredictionId::new(EMPLOYMENT_PREDICTION)?,
            "The shocked synthetic fixture does not reduce employment relative to baseline.",
        )?],
    )?);

    TheoryIr::new(
        TheoryId::new(THEORY_ID)?,
        vec![demand, employment],
        vec![MechanismSpec::new(
            mechanism_id,
            "Nominal rigidity transmits a demand contraction into lower employment.",
            vec![VariableId::new(DEMAND_VARIABLE)?],
            vec![VariableId::new(EMPLOYMENT_VARIABLE)?],
        )?],
        vec![claim],
    )
}

fn adapter(
    theory: &TheoryIr,
    id: &str,
    paradigm: ModelParadigm,
) -> Result<ModelAdapterDeclaration> {
    let adapter = ModelAdapterDeclaration::new(
        ModelId::new(id)?,
        theory.id().clone(),
        vec![paradigm],
        vec![ClaimId::new(DEMAND_EMPLOYMENT_CLAIM)?],
        vec![VariableId::new(EMPLOYMENT_VARIABLE)?],
    )?;
    theory.validate_adapter(&adapter)?;
    Ok(adapter)
}

/// Deterministic heterogeneous-agent prediction used by A4 and later synthetic
/// identification fixtures. Kept crate-private so A4 does not become a general
/// public simulation API.
pub(crate) fn heterogeneous_agent_prediction(
    shock_bps: u16,
) -> Result<SyntheticEconomicOutcome> {
    let baseline_demand_atoms = HOUSEHOLDS.iter().try_fold(0_u64, |total, household| {
        total
            .checked_add(household.baseline_budget_atoms)
            .ok_or(numerical("synthetic household baseline demand overflow"))
    })?;
    if baseline_demand_atoms != BASELINE_DEMAND_ATOMS {
        return Err(invalid("synthetic household fixture baseline drift"));
    }

    let shocked_demand_atoms = HOUSEHOLDS.iter().try_fold(0_u64, |total, household| {
        let shocked_budget = reduce_by_basis_points(household.baseline_budget_atoms, shock_bps)?;
        let active_demand = if shocked_budget >= household.reservation_demand_atoms {
            shocked_budget
        } else {
            0
        };
        total
            .checked_add(active_demand)
            .ok_or(numerical("synthetic household shocked demand overflow"))
    })?;

    Ok(SyntheticEconomicOutcome {
        baseline_demand_atoms,
        shocked_demand_atoms,
        baseline_employment: ceil_div(baseline_demand_atoms, DEMAND_ATOMS_PER_JOB)?,
        shocked_employment: ceil_div(shocked_demand_atoms, DEMAND_ATOMS_PER_JOB)?,
    })
}

/// Deterministic aggregate system-dynamics prediction used by A4 and later
/// synthetic identification fixtures. Kept crate-private for the same reason as
/// [`heterogeneous_agent_prediction`].
pub(crate) fn aggregate_system_dynamics_prediction(
    shock_bps: u16,
) -> Result<SyntheticEconomicOutcome> {
    let shocked_demand_atoms = reduce_by_basis_points(BASELINE_DEMAND_ATOMS, shock_bps)?;
    let shocked_employment_u128 = u128::from(BASELINE_EMPLOYMENT)
        .checked_mul(u128::from(shocked_demand_atoms))
        .ok_or(numerical(
            "synthetic aggregate employment multiplication overflow",
        ))?
        / u128::from(BASELINE_DEMAND_ATOMS);
    let shocked_employment = u64::try_from(shocked_employment_u128)
        .map_err(|_| numerical("synthetic aggregate employment conversion overflow"))?;

    Ok(SyntheticEconomicOutcome {
        baseline_demand_atoms: BASELINE_DEMAND_ATOMS,
        shocked_demand_atoms,
        baseline_employment: BASELINE_EMPLOYMENT,
        shocked_employment,
    })
}

fn account(id: &str) -> Result<AccountId> {
    AccountId::new(id)
}

fn accounting_episode(outcome: SyntheticEconomicOutcome) -> Result<DoubleEntryLedger> {
    let unit = UnitId::new("synthetic_currency:atom")?;
    let household = account("household:deposit")?;
    let firm = account("firm:deposit")?;
    let government = account("government:deposit")?;
    let bank_deposits = account("bank:deposit_liability")?;
    let bank_loan_asset = account("bank:loan_asset")?;
    let firm_loan_liability = account("firm:loan_liability")?;

    let mut ledger = DoubleEntryLedger::new(unit.clone());
    for id in [
        household.clone(),
        firm.clone(),
        government.clone(),
        bank_deposits.clone(),
        bank_loan_asset.clone(),
        firm_loan_liability.clone(),
    ] {
        ledger.register_account(id)?;
    }

    ledger.apply(&JournalEntry::new(
        unit.clone(),
        vec![
            Posting::new(household.clone(), 20_000)?,
            Posting::new(firm.clone(), 20_000)?,
            Posting::new(government.clone(), 5_000)?,
            Posting::new(bank_deposits.clone(), -45_000)?,
        ],
        Some("synthetic opening deposits".into()),
    )?)?;

    ledger.apply(&JournalEntry::new(
        unit.clone(),
        vec![
            Posting::new(firm.clone(), 1_000)?,
            Posting::new(bank_deposits, -1_000)?,
            Posting::new(bank_loan_asset, 1_000)?,
            Posting::new(firm_loan_liability, -1_000)?,
        ],
        Some("synthetic bank credit creation".into()),
    )?)?;

    let wage_atoms_u64 = outcome
        .shocked_employment
        .checked_mul(WAGE_ATOMS_PER_JOB)
        .ok_or(numerical("synthetic wage multiplication overflow"))?;
    let wage_atoms = i128::from(wage_atoms_u64);
    ledger.apply(&JournalEntry::new(
        unit.clone(),
        vec![
            Posting::new(firm.clone(), -wage_atoms)?,
            Posting::new(household.clone(), wage_atoms)?,
        ],
        Some("synthetic wage payment".into()),
    )?)?;

    let tax_atoms = wage_atoms / 10;
    ledger.apply(&JournalEntry::new(
        unit.clone(),
        vec![
            Posting::new(household.clone(), -tax_atoms)?,
            Posting::new(government, tax_atoms)?,
        ],
        Some("synthetic household tax".into()),
    )?)?;

    let purchase_atoms = i128::from(outcome.shocked_demand_atoms);
    ledger.apply(&JournalEntry::new(
        unit,
        vec![
            Posting::new(household, -purchase_atoms)?,
            Posting::new(firm, purchase_atoms)?,
        ],
        Some("synthetic final-goods purchase".into()),
    )?)?;

    if !ledger.is_balanced() {
        return Err(invalid(
            "synthetic accounting episode lost double-entry balance",
        ));
    }
    Ok(ledger)
}

/// Execute the complete fixed, deterministic A4 comparison.
pub fn run_a4_synthetic_comparison() -> Result<SyntheticEconomicComparison> {
    let theory = a4_theory()?;
    let agent_outcome = heterogeneous_agent_prediction(A4_DEMAND_SHOCK_BPS)?;
    let system_outcome = aggregate_system_dynamics_prediction(A4_DEMAND_SHOCK_BPS)?;

    let agent_based = SyntheticEconomicRun {
        adapter: adapter(
            &theory,
            "synthetic:heterogeneous_households_v1",
            ModelParadigm::AgentBased,
        )?,
        outcome: agent_outcome,
        ledger: accounting_episode(agent_outcome)?,
    };
    let system_dynamics = SyntheticEconomicRun {
        adapter: adapter(
            &theory,
            "synthetic:aggregate_adjustment_v1",
            ModelParadigm::SystemDynamics,
        )?,
        outcome: system_outcome,
        ledger: accounting_episode(system_outcome)?,
    };

    Ok(SyntheticEconomicComparison {
        theory,
        agent_based,
        system_dynamics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn employment_prediction_direction(
        comparison: &SyntheticEconomicComparison,
    ) -> ResponseDirection {
        let EconomicClaim::Empirical(claim) = &comparison.theory.claims()[0] else {
            panic!("A4 claim must remain empirical");
        };
        claim.predictions()[0].direction()
    }

    #[test]
    fn a4_fixture_is_exactly_deterministic() {
        let first = run_a4_synthetic_comparison().unwrap();
        let second = run_a4_synthetic_comparison().unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn internal_predictors_fail_closed_on_impossible_shock() {
        assert!(heterogeneous_agent_prediction(10_001).is_err());
        assert!(aggregate_system_dynamics_prediction(10_001).is_err());
    }

    #[test]
    fn two_distinct_paradigms_bind_the_same_etir_claim() {
        let comparison = run_a4_synthetic_comparison().unwrap();
        comparison
            .theory
            .validate_adapter(&comparison.agent_based.adapter)
            .unwrap();
        comparison
            .theory
            .validate_adapter(&comparison.system_dynamics.adapter)
            .unwrap();
        assert_eq!(
            comparison.agent_based.adapter.implemented_claims(),
            comparison.system_dynamics.adapter.implemented_claims()
        );
        assert_ne!(
            comparison.agent_based.adapter.paradigms(),
            comparison.system_dynamics.adapter.paradigms()
        );
    }

    #[test]
    fn both_models_recover_the_etir_direction_but_not_identical_magnitude() {
        let comparison = run_a4_synthetic_comparison().unwrap();
        assert_eq!(
            employment_prediction_direction(&comparison),
            ResponseDirection::Decrease
        );
        assert!(comparison.agent_based.outcome.employment_decreased());
        assert!(comparison.system_dynamics.outcome.employment_decreased());
        assert_eq!(comparison.agent_based.outcome.shocked_employment, 6);
        assert_eq!(comparison.system_dynamics.outcome.shocked_employment, 8);
        assert_ne!(
            comparison.agent_based.outcome.shocked_employment,
            comparison.system_dynamics.outcome.shocked_employment
        );
    }

    #[test]
    fn both_model_episodes_preserve_exact_accounting() {
        let comparison = run_a4_synthetic_comparison().unwrap();
        assert!(comparison.agent_based.ledger.is_balanced());
        assert!(comparison.system_dynamics.ledger.is_balanced());
    }

    #[test]
    fn bank_credit_creation_remains_explicit_and_balanced() {
        let comparison = run_a4_synthetic_comparison().unwrap();
        let bank_asset = AccountId::new("bank:loan_asset").unwrap();
        let firm_liability = AccountId::new("firm:loan_liability").unwrap();
        for ledger in [
            &comparison.agent_based.ledger,
            &comparison.system_dynamics.ledger,
        ] {
            assert_eq!(ledger.balance(&bank_asset).unwrap(), 1_000);
            assert_eq!(ledger.balance(&firm_liability).unwrap(), -1_000);
            assert!(ledger.is_balanced());
        }
    }

    #[test]
    fn heterogeneous_agent_response_is_not_aggregate_rescaling() {
        let comparison = run_a4_synthetic_comparison().unwrap();
        assert_eq!(comparison.agent_based.outcome.baseline_demand_atoms, 1_000);
        assert_eq!(comparison.agent_based.outcome.shocked_demand_atoms, 560);
        assert_eq!(
            comparison.system_dynamics.outcome.baseline_demand_atoms,
            1_000
        );
        assert_eq!(
            comparison.system_dynamics.outcome.shocked_demand_atoms,
            800
        );
    }
}
