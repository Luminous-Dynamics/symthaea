use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ResponseKey {
    pub intervention_id: String,
    pub query_id: String,
    pub horizon: u32,
}

impl ResponseKey {
    pub fn new(intervention_id: &str, query_id: &str, horizon: u32) -> Self {
        Self {
            intervention_id: intervention_id.to_owned(),
            query_id: query_id.to_owned(),
            horizon,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StateResponseContract {
    pub contract_id: String,
    pub intervention_ids: BTreeSet<String>,
    pub query_ids: BTreeSet<String>,
    pub horizons: BTreeSet<u32>,
    pub invariant_ids: BTreeSet<String>,
    /// Positive integer response tolerance for each registered response key.
    pub tolerance_units: BTreeMap<ResponseKey, u64>,
}

impl StateResponseContract {
    pub fn expected_keys(&self) -> BTreeSet<ResponseKey> {
        self.intervention_ids
            .iter()
            .flat_map(|intervention_id| {
                self.query_ids.iter().flat_map(move |query_id| {
                    self.horizons.iter().map(move |horizon| ResponseKey {
                        intervention_id: intervention_id.clone(),
                        query_id: query_id.clone(),
                        horizon: *horizon,
                    })
                })
            })
            .collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.contract_id.trim().is_empty() {
            return Err("contract_id must be non-empty".to_owned());
        }
        if self.intervention_ids.is_empty() || self.intervention_ids.iter().any(|id| id.trim().is_empty()) {
            return Err("intervention set must be non-empty with non-empty IDs".to_owned());
        }
        if self.query_ids.is_empty() || self.query_ids.iter().any(|id| id.trim().is_empty()) {
            return Err("query set must be non-empty with non-empty IDs".to_owned());
        }
        if self.horizons.is_empty() || self.horizons.contains(&0) {
            return Err("horizons must be non-empty and strictly positive".to_owned());
        }
        if self.invariant_ids.iter().any(|id| id.trim().is_empty()) {
            return Err("invariant IDs must be non-empty".to_owned());
        }

        let expected = self.expected_keys();
        let supplied: BTreeSet<_> = self.tolerance_units.keys().cloned().collect();
        if supplied != expected {
            return Err("tolerance keys must exactly cover the intervention × query × horizon contract".to_owned());
        }
        if self.tolerance_units.values().any(|tolerance| *tolerance == 0) {
            return Err("response tolerances must be positive".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StateResponseRecord {
    pub state_id: String,
    /// Candidate-assigned abstraction label. It is not trusted as evaluator truth.
    pub abstract_state_id: String,
    /// Evaluator-owned invariant values/class identity.
    pub invariant_signature: BTreeMap<String, i64>,
    /// Evaluator-owned outcome values in the contract's canonical integer units.
    pub responses: BTreeMap<ResponseKey, i64>,
}

impl StateResponseRecord {
    pub fn validate_against(&self, contract: &StateResponseContract) -> Result<(), String> {
        if self.state_id.trim().is_empty() || self.abstract_state_id.trim().is_empty() {
            return Err("state and abstraction IDs must be non-empty".to_owned());
        }
        let supplied_invariants: BTreeSet<_> = self.invariant_signature.keys().cloned().collect();
        if supplied_invariants != contract.invariant_ids {
            return Err(format!("state {} has incomplete or unexpected invariant fields", self.state_id));
        }
        let supplied_responses: BTreeSet<_> = self.responses.keys().cloned().collect();
        if supplied_responses != contract.expected_keys() {
            return Err(format!("state {} has incomplete or out-of-contract response coverage", self.state_id));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AbstractionScore {
    pub compared_pairs: usize,
    pub candidate_merged_pairs: usize,
    pub candidate_separated_pairs: usize,
    pub false_merges: usize,
    pub false_splits: usize,
    pub max_normalized_pair_distortion: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairAssessment {
    pub exact_equivalent: bool,
    pub normalized_worst_case_distortion: f64,
}

pub fn assess_pair(
    left: &StateResponseRecord,
    right: &StateResponseRecord,
    contract: &StateResponseContract,
) -> Result<PairAssessment, String> {
    contract.validate()?;
    left.validate_against(contract)?;
    right.validate_against(contract)?;
    let mut maximum = 0.0_f64;
    for (key, left_value) in &left.responses {
        let right_value = right.responses.get(key).copied().unwrap_or_default();
        let tolerance = contract.tolerance_units.get(key).copied().unwrap_or(1);
        let distance = left_value.abs_diff(right_value) as f64 / tolerance as f64;
        maximum = maximum.max(distance);
    }
    Ok(PairAssessment {
        exact_equivalent: left.invariant_signature == right.invariant_signature
            && left.responses == right.responses,
        normalized_worst_case_distortion: maximum,
    })
}

pub fn approximate_response_equivalent(
    left: &StateResponseRecord,
    right: &StateResponseRecord,
    contract: &StateResponseContract,
    epsilon: f64,
) -> Result<bool, String> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("epsilon must be finite and non-negative".to_owned());
    }
    if left.invariant_signature != right.invariant_signature {
        return Ok(false);
    }
    Ok(assess_pair(left, right, contract)?.normalized_worst_case_distortion <= epsilon)
}

/// Score the candidate's state partition against evaluator-owned response
/// equivalence. Every unordered state pair is one comparison; task rows or
/// intervention cells are not counted as independent statistical units.
pub fn score_abstraction(
    records: &[StateResponseRecord],
    contract: &StateResponseContract,
) -> Result<AbstractionScore, String> {
    contract.validate()?;
    if records.is_empty() {
        return Err("at least one state record is required".to_owned());
    }

    let mut state_ids = BTreeSet::new();
    for record in records {
        record.validate_against(contract)?;
        if !state_ids.insert(record.state_id.clone()) {
            return Err(format!("duplicate state ID: {}", record.state_id));
        }
    }

    let mut score = AbstractionScore::default();
    for left_index in 0..records.len() {
        for right_index in (left_index + 1)..records.len() {
            let left = &records[left_index];
            let right = &records[right_index];
            let assessment = assess_pair(left, right, contract)?;
            let equivalent = assessment.exact_equivalent;
            score.max_normalized_pair_distortion = score
                .max_normalized_pair_distortion
                .max(assessment.normalized_worst_case_distortion);
            let merged = left.abstract_state_id == right.abstract_state_id;
            score.compared_pairs += 1;
            if merged {
                score.candidate_merged_pairs += 1;
                if !equivalent {
                    score.false_merges += 1;
                }
            } else {
                score.candidate_separated_pairs += 1;
                if equivalent {
                    score.false_splits += 1;
                }
            }
        }
    }
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> StateResponseContract {
        let intervention_ids = ["push_left", "push_right"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        let query_ids = ["displacement", "rotation"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        let horizons = [1, 5].into_iter().collect();
        let invariant_ids = ["mass_class", "contact_topology"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        let mut result = StateResponseContract {
            contract_id: "fixture-contract-v1".to_owned(),
            intervention_ids,
            query_ids,
            horizons,
            invariant_ids,
            tolerance_units: BTreeMap::new(),
        };
        for key in result.expected_keys() {
            result.tolerance_units.insert(key, 10);
        }
        result
    }

    fn record(
        state_id: &str,
        abstract_state_id: &str,
        outcome_offset: i64,
        mass_class: i64,
    ) -> StateResponseRecord {
        let contract = contract();
        let invariant_signature = [
            ("mass_class".to_owned(), mass_class),
            ("contact_topology".to_owned(), 1),
        ]
        .into_iter()
        .collect();
        let responses = contract
            .expected_keys()
            .into_iter()
            .enumerate()
            .map(|(index, key)| (key, index as i64 + outcome_offset))
            .collect();
        StateResponseRecord {
            state_id: state_id.to_owned(),
            abstract_state_id: abstract_state_id.to_owned(),
            invariant_signature,
            responses,
        }
    }

    #[test]
    fn different_raw_states_with_identical_registered_responses_are_equivalent() {
        let contract = contract();
        let left = record("raw-a", "compressed-1", 0, 2);
        let right = record("raw-b", "compressed-1", 0, 2);
        assert!(assess_pair(&left, &right, &contract).expect("valid pair").exact_equivalent);
        assert_eq!(
            score_abstraction(&[left, right], &contract).expect("valid fixture").false_merges,
            0
        );
    }

    #[test]
    fn hidden_intervention_response_prevents_false_merge() {
        let contract = contract();
        let left = record("raw-a", "same-compressed-state", 0, 2);
        let right = record("raw-b", "same-compressed-state", 1, 2);
        let score = score_abstraction(&[left, right], &contract).expect("valid fixture");
        assert_eq!(score.candidate_merged_pairs, 1);
        assert_eq!(score.false_merges, 1);
    }

    #[test]
    fn candidate_partition_reports_false_splits() {
        let contract = contract();
        let left = record("raw-a", "compressed-a", 0, 2);
        let right = record("raw-b", "compressed-b", 0, 2);
        let score = score_abstraction(&[left, right], &contract).expect("valid fixture");
        assert_eq!(score.false_splits, 1);
    }

    #[test]
    fn invariant_mismatch_prevents_equivalence_even_when_responses_match() {
        let contract = contract();
        let left = record("raw-a", "same-compressed-state", 0, 2);
        let right = record("raw-b", "same-compressed-state", 0, 3);
        assert!(!assess_pair(&left, &right, &contract).expect("valid pair").exact_equivalent);
        assert_eq!(
            score_abstraction(&[left, right], &contract).expect("valid fixture").false_merges,
            1
        );
    }

    #[test]
    fn approximate_tolerance_is_non_transitive_and_must_not_be_clustered() {
        let contract = contract();
        let a = record("a", "a", 0, 2);
        let b = record("b", "b", 8, 2);
        let c = record("c", "c", 16, 2);
        assert!(approximate_response_equivalent(&a, &b, &contract, 1.0).expect("valid comparison"));
        assert!(approximate_response_equivalent(&b, &c, &contract, 1.0).expect("valid comparison"));
        assert!(!approximate_response_equivalent(&a, &c, &contract, 1.0).expect("valid comparison"));
    }

    #[test]
    fn missing_or_extra_response_cells_are_rejected() {
        let contract = contract();
        let mut missing = record("raw-a", "state-a", 0, 2);
        missing.responses.pop_first();
        assert!(missing.validate_against(&contract).is_err());

        let mut extra = record("raw-b", "state-b", 0, 2);
        extra.responses.insert(ResponseKey::new("unknown", "displacement", 1), 0);
        assert!(extra.validate_against(&contract).is_err());
    }

    #[test]
    fn contract_rejects_zero_tolerance_and_incomplete_grid() {
        let mut invalid = contract();
        let key = invalid.expected_keys().into_iter().next().expect("nonempty");
        invalid.tolerance_units.insert(key, 0);
        assert!(invalid.validate().is_err());

        let mut incomplete = contract();
        incomplete.tolerance_units.pop_first();
        assert!(incomplete.validate().is_err());
    }

    #[test]
    fn duplicate_state_ids_are_rejected() {
        let contract = contract();
        let left = record("same-id", "compressed-a", 0, 2);
        let right = record("same-id", "compressed-b", 0, 2);
        assert!(score_abstraction(&[left, right], &contract).is_err());
    }

    #[test]
    fn non_finite_or_negative_epsilon_is_rejected() {
        let contract = contract();
        let left = record("a", "a", 0, 2);
        let right = record("b", "b", 0, 2);
        assert!(approximate_response_equivalent(&left, &right, &contract, f64::NAN).is_err());
        assert!(approximate_response_equivalent(&left, &right, &contract, -0.1).is_err());
    }
}
