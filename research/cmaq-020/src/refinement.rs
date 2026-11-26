//! Budgeted counterexample-guided refinement with candidate-visible probe receipts.
//!
//! The counterexample selector reveals a permitted query identity, not its
//! evaluator response value. The candidate must execute that query and submit
//! the observation returned by the environment. The evaluator validates the
//! receipt against its deterministic reference record, charges the frozen
//! query cost, and scores the candidate's revised partition separately.
//!
//! Receipt digests are syntax-checked here. Independent byte recomputation,
//! source authentication and signed custody remain runner responsibilities.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    AbstractionScore, CounterexampleDimension, CounterexampleProbe, ResponseKey,
    StateResponseContract, StateResponseRecord, assess_pair, score_abstraction,
    select_counterexample_probe, validate_record_set,
};

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Candidate-visible output of actually executing one selected probe on both
/// states. These values are observations, not the evaluator's hidden response
/// table; the runner must bind them to independently authenticated execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProbeObservationReceiptV1 {
    pub contract_id: String,
    pub left_state_id: String,
    pub right_state_id: String,
    pub response_key: ResponseKey,
    pub left_observed_value: i64,
    pub right_observed_value: i64,
    pub source_manifest_sha256: String,
    /// Digest of the exact observation receipt bytes, recomputed by the runner.
    pub receipt_sha256: String,
}

/// Auditable charge for one successfully admitted probe observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeCharge {
    pub cost_units: u64,
    pub cumulative_cost_units: u64,
    pub remaining_budget_units: u64,
}

/// Per-candidate, per-contract probe budget with duplicate-receipt protection.
#[derive(Clone, Debug)]
pub struct ProbeBudgetLedger {
    contract_id: String,
    contract_sha256: String,
    source_manifest_sha256: String,
    budget_limit_units: u64,
    spent_units: u64,
    accepted_receipt_sha256: BTreeSet<String>,
}

impl ProbeBudgetLedger {
    pub fn new(
        contract: &StateResponseContract,
        budget_limit_units: u64,
    ) -> Result<Self, String> {
        contract.validate()?;
        Ok(Self {
            contract_id: contract.contract_id.clone(),
            contract_sha256: contract.contract_sha256.clone(),
            source_manifest_sha256: contract.source_manifest_sha256.clone(),
            budget_limit_units,
            spent_units: 0,
            accepted_receipt_sha256: BTreeSet::new(),
        })
    }

    pub fn budget_limit_units(&self) -> u64 {
        self.budget_limit_units
    }

    pub fn spent_units(&self) -> u64 {
        self.spent_units
    }

    pub fn remaining_budget_units(&self) -> u64 {
        self.budget_limit_units.saturating_sub(self.spent_units)
    }

    pub fn accepted_receipt_count(&self) -> usize {
        self.accepted_receipt_sha256.len()
    }

    /// Validate and charge a probe that the evaluator selected from the
    /// candidate's current merged pairs. Charging is transactional: malformed,
    /// mismatched, replayed or over-budget receipts do not mutate the ledger.
    pub fn charge_probe(
        &mut self,
        current_records: &[StateResponseRecord],
        contract: &StateResponseContract,
        probe: &CounterexampleProbe,
        receipt: &ProbeObservationReceiptV1,
    ) -> Result<ProbeCharge, String> {
        validate_record_set(current_records, contract)?;
        if self.contract_id != contract.contract_id
            || self.contract_sha256 != contract.contract_sha256
            || self.source_manifest_sha256 != contract.source_manifest_sha256
        {
            return Err(
                "budget ledger belongs to a different frozen contract or source manifest"
                    .to_owned(),
            );
        }

        let selected = select_counterexample_probe(current_records, contract)?
            .ok_or_else(|| "no remaining counterexample probe exists".to_owned())?;
        if &selected != probe {
            return Err(
                "receipt must correspond to the evaluator's currently selected counterexample"
                    .to_owned(),
            );
        }

        let (key, selected_cost) = match (&probe.dimension, probe.probe_cost_units) {
            (CounterexampleDimension::Response { key }, Some(cost)) => (key, cost),
            (CounterexampleDimension::Invariant { .. }, None) => {
                return Err(
                    "structural invariant witnesses require a separate verification-cost ledger"
                        .to_owned(),
                );
            }
            _ => return Err("counterexample dimension and probe cost are inconsistent".to_owned()),
        };
        let contract_cost = contract
            .probe_cost_units
            .get(key)
            .copied()
            .ok_or_else(|| "selected response key has no registered probe cost".to_owned())?;
        if selected_cost != contract_cost {
            return Err("selected probe cost differs from the frozen contract".to_owned());
        }

        if receipt.contract_id != contract.contract_id
            || receipt.left_state_id != probe.left_state_id
            || receipt.right_state_id != probe.right_state_id
            || receipt.response_key != *key
        {
            return Err("probe receipt identity does not match the selected counterexample".to_owned());
        }
        if receipt.source_manifest_sha256 != contract.source_manifest_sha256 {
            return Err("probe receipt source manifest differs from the frozen contract".to_owned());
        }
        if !is_sha256(&receipt.receipt_sha256) {
            return Err("probe receipt digest must be a 64-character SHA-256 hex value".to_owned());
        }
        if self
            .accepted_receipt_sha256
            .contains(&receipt.receipt_sha256)
        {
            return Err("probe receipt replay detected".to_owned());
        }

        let by_id: BTreeMap<_, _> = current_records
            .iter()
            .map(|record| (record.state_id.as_str(), record))
            .collect();
        let left = by_id
            .get(receipt.left_state_id.as_str())
            .ok_or_else(|| "probe receipt references an unknown left state".to_owned())?;
        let right = by_id
            .get(receipt.right_state_id.as_str())
            .ok_or_else(|| "probe receipt references an unknown right state".to_owned())?;
        let expected_left = left.responses[key];
        let expected_right = right.responses[key];
        if receipt.left_observed_value != expected_left
            || receipt.right_observed_value != expected_right
        {
            return Err(
                "probe receipt observations do not match the deterministic environment result"
                    .to_owned(),
            );
        }
        if receipt.left_observed_value == receipt.right_observed_value {
            return Err("selected counterexample receipt did not observe a separating result".to_owned());
        }

        let next_spent = self
            .spent_units
            .checked_add(contract_cost)
            .ok_or_else(|| "cumulative probe cost overflow".to_owned())?;
        if next_spent > self.budget_limit_units {
            return Err("probe budget would be exceeded".to_owned());
        }

        self.spent_units = next_spent;
        self.accepted_receipt_sha256
            .insert(receipt.receipt_sha256.clone());
        Ok(ProbeCharge {
            cost_units: contract_cost,
            cumulative_cost_units: next_spent,
            remaining_budget_units: self.remaining_budget_units(),
        })
    }
}

/// Evaluator-side effect of a candidate's before/after state partition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefinementEffectiveness {
    pub before: AbstractionScore,
    pub after: AbstractionScore,
    pub false_merges_resolved: usize,
    pub false_merges_added: usize,
    pub false_splits_resolved: usize,
    pub false_splits_added: usize,
    pub selected_pair_separated: bool,
}

/// Cost and quality reported for one budgeted refinement attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BudgetedRefinementScore {
    pub charge: ProbeCharge,
    pub effectiveness: RefinementEffectiveness,
    /// Probe-cost units per false merge removed; None when no merge was removed.
    pub cost_per_resolved_false_merge_units: Option<f64>,
}

/// Compare partitions without allowing the candidate to rewrite evaluator
/// records, source identity, ground-truth invariants or response signatures.
pub fn score_refinement_transition(
    before_records: &[StateResponseRecord],
    after_records: &[StateResponseRecord],
    contract: &StateResponseContract,
    probe: &CounterexampleProbe,
) -> Result<RefinementEffectiveness, String> {
    validate_record_set(before_records, contract)?;
    validate_record_set(after_records, contract)?;

    let before_by_id: BTreeMap<_, _> = before_records
        .iter()
        .map(|record| (record.state_id.as_str(), record))
        .collect();
    let after_by_id: BTreeMap<_, _> = after_records
        .iter()
        .map(|record| (record.state_id.as_str(), record))
        .collect();
    if before_by_id.keys().ne(after_by_id.keys()) {
        return Err("refinement must preserve the exact evaluated state-ID set".to_owned());
    }

    for (state_id, before) in &before_by_id {
        let after = after_by_id[state_id];
        if before.domain_id != after.domain_id
            || before.environment_family_id != after.environment_family_id
            || before.environment_cluster_id != after.environment_cluster_id
            || before.source_state_sha256 != after.source_state_sha256
            || before.invariant_signature != after.invariant_signature
            || before.responses != after.responses
        {
            return Err(format!(
                "candidate changed evaluator-owned source or reference fields for state {state_id}"
            ));
        }
    }

    let before_checkpoint = &before_records[0].candidate_checkpoint_sha256;
    let after_checkpoint = &after_records[0].candidate_checkpoint_sha256;
    if before_checkpoint == after_checkpoint {
        return Err("a refinement step must bind a new candidate checkpoint digest".to_owned());
    }

    let selected_left = after_by_id
        .get(probe.left_state_id.as_str())
        .ok_or_else(|| "selected probe left state is absent after refinement".to_owned())?;
    let selected_right = after_by_id
        .get(probe.right_state_id.as_str())
        .ok_or_else(|| "selected probe right state is absent after refinement".to_owned())?;

    let before_score = score_abstraction(before_records, contract)?;
    let after_score = score_abstraction(after_records, contract)?;

    // Count pair-level transitions rather than net aggregate score deltas.
    // A lower global error count can otherwise conceal new errors on different
    // pairs while incorrectly labelling those as "resolved".
    let before_states: Vec<_> = before_by_id.values().copied().collect();
    let after_states: Vec<_> = after_by_id.values().copied().collect();
    let mut false_merges_resolved = 0;
    let mut false_merges_added = 0;
    let mut false_splits_resolved = 0;
    let mut false_splits_added = 0;

    for left_index in 0..before_states.len() {
        for right_index in (left_index + 1)..before_states.len() {
            let before_left = before_states[left_index];
            let before_right = before_states[right_index];
            let after_left = after_by_id[before_left.state_id.as_str()];
            let after_right = after_by_id[before_right.state_id.as_str()];
            let equivalent = assess_pair(before_left, before_right, contract)?.exact_equivalent;
            let before_merged =
                before_left.abstract_state_id == before_right.abstract_state_id;
            let after_merged = after_left.abstract_state_id == after_right.abstract_state_id;

            let before_false_merge = before_merged && !equivalent;
            let after_false_merge = after_merged && !equivalent;
            let before_false_split = !before_merged && equivalent;
            let after_false_split = !after_merged && equivalent;

            false_merges_resolved += usize::from(before_false_merge && !after_false_merge);
            false_merges_added += usize::from(!before_false_merge && after_false_merge);
            false_splits_resolved += usize::from(before_false_split && !after_false_split);
            false_splits_added += usize::from(!before_false_split && after_false_split);
        }
    }

    Ok(RefinementEffectiveness {
        before: before_score,
        after: after_score,
        false_merges_resolved,
        false_merges_added,
        false_splits_resolved,
        false_splits_added,
        selected_pair_separated: selected_left.abstract_state_id
            != selected_right.abstract_state_id,
    })
}

/// Charge first, then score the candidate's refinement. If after-record
/// validation fails, the already-executed probe remains charged: execution
/// cost is not refunded because the candidate submitted an invalid update.
pub fn score_budgeted_refinement_step(
    before_records: &[StateResponseRecord],
    after_records: &[StateResponseRecord],
    contract: &StateResponseContract,
    probe: &CounterexampleProbe,
    receipt: &ProbeObservationReceiptV1,
    ledger: &mut ProbeBudgetLedger,
) -> Result<BudgetedRefinementScore, String> {
    let charge = ledger.charge_probe(before_records, contract, probe, receipt)?;
    let effectiveness =
        score_refinement_transition(before_records, after_records, contract, probe)?;
    let cost_per_resolved_false_merge_units =
        (effectiveness.false_merges_resolved > 0).then_some(
            charge.cost_units as f64 / effectiveness.false_merges_resolved as f64,
        );
    Ok(BudgetedRefinementScore {
        charge,
        effectiveness,
        cost_per_resolved_false_merge_units,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> StateResponseContract {
        let mut contract = StateResponseContract {
            contract_id: "budgeted-cegar-v1".to_owned(),
            contract_sha256: "a".repeat(64),
            source_manifest_sha256: "b".repeat(64),
            domain_ids: ["toy-domain".to_owned()].into_iter().collect(),
            intervention_ids: ["probe_fast", "probe_slow"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            query_ids: ["displacement".to_owned()].into_iter().collect(),
            horizons: [1].into_iter().collect(),
            invariant_ids: ["mass_class".to_owned()].into_iter().collect(),
            tolerance_units: BTreeMap::new(),
            probe_cost_units: BTreeMap::new(),
        };
        for key in contract.expected_keys() {
            contract.tolerance_units.insert(key.clone(), 1);
            let cost = if key.intervention_id == "probe_fast" { 1 } else { 4 };
            contract.probe_cost_units.insert(key, cost);
        }
        contract
    }

    fn record(
        state_id: &str,
        abstract_state_id: &str,
        fast: i64,
        slow: i64,
        checkpoint: &str,
        source_seed: char,
    ) -> StateResponseRecord {
        let contract = contract();
        let responses = contract
            .expected_keys()
            .into_iter()
            .map(|key| {
                let value = if key.intervention_id == "probe_fast" {
                    fast
                } else {
                    slow
                };
                (key, value)
            })
            .collect();
        StateResponseRecord {
            state_id: state_id.to_owned(),
            domain_id: "toy-domain".to_owned(),
            environment_family_id: "toy-mechanism".to_owned(),
            environment_cluster_id: "toy-cluster".to_owned(),
            source_state_sha256: source_seed.to_string().repeat(64),
            candidate_checkpoint_sha256: checkpoint.to_owned(),
            abstract_state_id: abstract_state_id.to_owned(),
            invariant_signature: [("mass_class".to_owned(), 1)].into_iter().collect(),
            responses,
        }
    }

    fn receipt(
        left: &str,
        right: &str,
        key: ResponseKey,
        left_value: i64,
        right_value: i64,
    ) -> ProbeObservationReceiptV1 {
        ProbeObservationReceiptV1 {
            contract_id: "budgeted-cegar-v1".to_owned(),
            left_state_id: left.to_owned(),
            right_state_id: right.to_owned(),
            response_key: key,
            left_observed_value: left_value,
            right_observed_value: right_value,
            source_manifest_sha256: "b".repeat(64),
            receipt_sha256: "f".repeat(64),
        }
    }

    #[test]
    fn executed_probe_receipt_charges_budget_and_scores_refinement() {
        let contract = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 10, 10, &"c".repeat(64), '2'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        let key = ResponseKey::new("probe_fast", "displacement", 1);
        assert_eq!(
            probe.dimension,
            CounterexampleDimension::Response { key: key.clone() }
        );

        let after = vec![
            record("a", "class-a", 0, 0, &"d".repeat(64), '1'),
            record("b", "class-b", 10, 10, &"d".repeat(64), '2'),
        ];
        let observation = receipt("a", "b", key, 0, 10);
        let mut ledger = ProbeBudgetLedger::new(&contract, 1).expect("valid budget");
        let score = score_budgeted_refinement_step(
            &before,
            &after,
            &contract,
            &probe,
            &observation,
            &mut ledger,
        )
        .expect("valid refinement step");

        assert_eq!(score.charge.cost_units, 1);
        assert_eq!(score.charge.cumulative_cost_units, 1);
        assert_eq!(score.charge.remaining_budget_units, 0);
        assert_eq!(score.effectiveness.before.false_merges, 1);
        assert_eq!(score.effectiveness.after.false_merges, 0);
        assert_eq!(score.effectiveness.false_merges_resolved, 1);
        assert_eq!(score.effectiveness.false_splits_added, 0);
        assert!(score.effectiveness.selected_pair_separated);
        assert_eq!(score.cost_per_resolved_false_merge_units, Some(1.0));
        assert_eq!(ledger.accepted_receipt_count(), 1);
    }

    #[test]
    fn one_probe_can_resolve_multiple_merges_but_over_refinement_creates_false_splits() {
        let contract = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 0, 0, &"c".repeat(64), '2'),
            record("c", "merged", 10, 10, &"c".repeat(64), '3'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("one distinct mechanism remains merged");
        let key = ResponseKey::new("probe_fast", "displacement", 1);
        let observation = receipt("a", "c", key, 0, 10);
        let after = vec![
            record("a", "singleton-a", 0, 0, &"d".repeat(64), '1'),
            record("b", "singleton-b", 0, 0, &"d".repeat(64), '2'),
            record("c", "singleton-c", 10, 10, &"d".repeat(64), '3'),
        ];
        let mut ledger = ProbeBudgetLedger::new(&contract, 10).expect("valid budget");
        let score = score_budgeted_refinement_step(
            &before,
            &after,
            &contract,
            &probe,
            &observation,
            &mut ledger,
        )
        .expect("score over-refinement explicitly");

        assert_eq!(score.effectiveness.false_merges_resolved, 2);
        assert_eq!(score.effectiveness.false_splits_added, 1);
        assert_eq!(score.effectiveness.after.false_merges, 0);
        assert_eq!(score.effectiveness.after.false_splits, 1);
        assert_eq!(score.cost_per_resolved_false_merge_units, Some(0.5));
    }

    #[test]
    fn budget_exhaustion_does_not_charge_or_accept_a_receipt() {
        let contract = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 10, 10, &"c".repeat(64), '2'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        let observation = receipt(
            "a",
            "b",
            ResponseKey::new("probe_fast", "displacement", 1),
            0,
            10,
        );
        let mut ledger = ProbeBudgetLedger::new(&contract, 0).expect("valid zero budget");
        assert!(
            ledger
                .charge_probe(&before, &contract, &probe, &observation)
                .is_err()
        );
        assert_eq!(ledger.spent_units(), 0);
        assert_eq!(ledger.accepted_receipt_count(), 0);
    }

    #[test]
    fn malformed_or_forged_observation_is_rejected_without_charging() {
        let contract = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 10, 10, &"c".repeat(64), '2'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        let mut observation = receipt(
            "a",
            "b",
            ResponseKey::new("probe_fast", "displacement", 1),
            0,
            9,
        );
        let mut ledger = ProbeBudgetLedger::new(&contract, 5).expect("valid budget");
        assert!(
            ledger
                .charge_probe(&before, &contract, &probe, &observation)
                .is_err()
        );
        assert_eq!(ledger.spent_units(), 0);

        observation.right_observed_value = 10;
        observation.receipt_sha256 = "not-a-sha".to_owned();
        assert!(
            ledger
                .charge_probe(&before, &contract, &probe, &observation)
                .is_err()
        );
        assert_eq!(ledger.spent_units(), 0);
    }

    #[test]
    fn duplicate_receipt_replay_is_rejected_without_double_charging() {
        let contract = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 10, 10, &"c".repeat(64), '2'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        let observation = receipt(
            "a",
            "b",
            ResponseKey::new("probe_fast", "displacement", 1),
            0,
            10,
        );
        let mut ledger = ProbeBudgetLedger::new(&contract, 5).expect("valid budget");
        assert!(
            ledger
                .charge_probe(&before, &contract, &probe, &observation)
                .is_ok()
        );
        assert!(
            ledger
                .charge_probe(&before, &contract, &probe, &observation)
                .is_err()
        );
        assert_eq!(ledger.spent_units(), 1);
        assert_eq!(ledger.accepted_receipt_count(), 1);
    }

    #[test]
    fn post_probe_reference_mutation_is_rejected_but_probe_cost_is_not_refunded() {
        let contract = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 10, 10, &"c".repeat(64), '2'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        let observation = receipt(
            "a",
            "b",
            ResponseKey::new("probe_fast", "displacement", 1),
            0,
            10,
        );
        let mut after = vec![
            record("a", "class-a", 0, 0, &"d".repeat(64), '1'),
            record("b", "class-b", 10, 10, &"d".repeat(64), '2'),
        ];
        after[1]
            .responses
            .insert(ResponseKey::new("probe_slow", "displacement", 1), 99);
        let mut ledger = ProbeBudgetLedger::new(&contract, 5).expect("valid budget");
        assert!(
            score_budgeted_refinement_step(
                &before,
                &after,
                &contract,
                &probe,
                &observation,
                &mut ledger,
            )
            .is_err()
        );
        assert_eq!(ledger.spent_units(), 1);
        assert_eq!(ledger.accepted_receipt_count(), 1);
    }

    #[test]
    fn budget_ledger_rejects_same_id_contract_or_manifest_substitution() {
        let original = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 10, 10, &"c".repeat(64), '2'),
        ];
        let probe = select_counterexample_probe(&before, &original)
            .expect("valid contract")
            .expect("counterexample exists");
        let observation = receipt(
            "a",
            "b",
            ResponseKey::new("probe_fast", "displacement", 1),
            0,
            10,
        );

        let mut ledger = ProbeBudgetLedger::new(&original, 5).expect("valid budget");
        let mut changed_contract_digest = original.clone();
        changed_contract_digest.contract_sha256 = "c".repeat(64);
        assert!(
            ledger
                .charge_probe(
                    &before,
                    &changed_contract_digest,
                    &probe,
                    &observation,
                )
                .is_err()
        );

        let mut changed_source_manifest = original.clone();
        changed_source_manifest.source_manifest_sha256 = "d".repeat(64);
        assert!(
            ledger
                .charge_probe(
                    &before,
                    &changed_source_manifest,
                    &probe,
                    &observation,
                )
                .is_err()
        );
        assert_eq!(ledger.spent_units(), 0);
        assert_eq!(ledger.accepted_receipt_count(), 0);
    }

    #[test]
    fn pair_level_accounting_detects_error_migration_with_unchanged_totals() {
        let contract = contract();
        let before = vec![
            record("a", "before-left", 0, 0, &"c".repeat(64), '1'),
            record("b", "before-right", 0, 0, &"c".repeat(64), '2'),
            record("c", "before-left", 10, 10, &"c".repeat(64), '3'),
            record("d", "before-right", 10, 10, &"c".repeat(64), '4'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        let after = vec![
            record("a", "after-left", 0, 0, &"d".repeat(64), '1'),
            record("b", "after-right", 0, 0, &"d".repeat(64), '2'),
            record("c", "after-right", 10, 10, &"d".repeat(64), '3'),
            record("d", "after-left", 10, 10, &"d".repeat(64), '4'),
        ];

        let effect = score_refinement_transition(&before, &after, &contract, &probe)
            .expect("valid pair-level comparison");
        assert_eq!(effect.before.false_merges, 2);
        assert_eq!(effect.after.false_merges, 2);
        assert_eq!(effect.false_merges_resolved, 2);
        assert_eq!(effect.false_merges_added, 2);
        assert_eq!(effect.before.false_splits, 2);
        assert_eq!(effect.after.false_splits, 2);
        assert_eq!(effect.false_splits_resolved, 0);
        assert_eq!(effect.false_splits_added, 0);
    }

    #[test]
    fn candidate_must_bind_refinement_to_a_new_checkpoint() {
        let contract = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 10, 10, &"c".repeat(64), '2'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        let after = vec![
            record("a", "class-a", 0, 0, &"c".repeat(64), '1'),
            record("b", "class-b", 10, 10, &"c".repeat(64), '2'),
        ];
        assert!(
            score_refinement_transition(&before, &after, &contract, &probe).is_err()
        );
    }

    #[test]
    fn candidate_that_does_not_use_the_probe_is_scored_as_no_progress() {
        let contract = contract();
        let before = vec![
            record("a", "merged", 0, 0, &"c".repeat(64), '1'),
            record("b", "merged", 10, 10, &"c".repeat(64), '2'),
        ];
        let probe = select_counterexample_probe(&before, &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        let after = vec![
            record("a", "still-merged", 0, 0, &"d".repeat(64), '1'),
            record("b", "still-merged", 10, 10, &"d".repeat(64), '2'),
        ];
        let result =
            score_refinement_transition(&before, &after, &contract, &probe)
                .expect("valid but ineffective refinement");
        assert_eq!(result.false_merges_resolved, 0);
        assert!(!result.selected_pair_separated);
    }
}
