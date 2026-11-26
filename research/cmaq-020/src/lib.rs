use std::collections::{BTreeMap, BTreeSet};

pub mod failure_fidelity;
pub mod refinement;

pub use failure_fidelity::{
    BoundaryRateClusterSummary, BoundaryRateFidelityScore, BoundaryRatePair,
    BoundaryRatePairMeasurement, BoundarySide, FailureClusterCounts, FailureFidelityContract,
    FailureFidelityScore, FailureSignature, FailureTraceRecord, OutcomeLabel,
    score_boundary_rate_pairs, score_failure_fidelity,
};

pub use refinement::{
    BudgetedRefinementScore, ProbeBudgetLedger, ProbeCharge, ProbeObservationReceiptV1,
    RefinementEffectiveness, score_budgeted_refinement_step, score_refinement_transition,
};

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
    /// SHA-256 of the canonical frozen contract payload (excluding this field).
    pub contract_sha256: String,
    /// SHA-256 of the source dataset/generator manifest used by the evaluator.
    pub source_manifest_sha256: String,
    /// Registered domains admitted by this exact contract.
    pub domain_ids: BTreeSet<String>,
    pub intervention_ids: BTreeSet<String>,
    pub query_ids: BTreeSet<String>,
    pub horizons: BTreeSet<u32>,
    pub invariant_ids: BTreeSet<String>,
    /// Positive integer response tolerance for each registered response key.
    pub tolerance_units: BTreeMap<ResponseKey, u64>,
    /// Positive relative cost to acquire each registered intervention response.
    pub probe_cost_units: BTreeMap<ResponseKey, u64>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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
        if !is_sha256(&self.contract_sha256) {
            return Err("contract_sha256 must be a 64-character hexadecimal digest".to_owned());
        }
        if !is_sha256(&self.source_manifest_sha256) {
            return Err(
                "source_manifest_sha256 must be a 64-character hexadecimal digest".to_owned(),
            );
        }
        if self.domain_ids.is_empty()
            || self
                .domain_ids
                .iter()
                .any(|domain| domain.trim().is_empty())
        {
            return Err("domain_ids must be non-empty and contain no empty IDs".to_owned());
        }
        if self.intervention_ids.is_empty()
            || self.intervention_ids.iter().any(|id| id.trim().is_empty())
        {
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
            return Err(
                "tolerance keys must exactly cover the intervention × query × horizon contract"
                    .to_owned(),
            );
        }
        if self
            .tolerance_units
            .values()
            .any(|tolerance| *tolerance == 0)
        {
            return Err("response tolerances must be positive".to_owned());
        }
        let supplied_probe_costs: BTreeSet<_> = self.probe_cost_units.keys().cloned().collect();
        if supplied_probe_costs != expected {
            return Err(
                "probe-cost keys must exactly cover the intervention × query × horizon contract"
                    .to_owned(),
            );
        }
        if self.probe_cost_units.values().any(|cost| *cost == 0) {
            return Err("response probe costs must be positive".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StateResponseRecord {
    pub state_id: String,
    /// Domain identity; must be admitted by the frozen contract.
    pub domain_id: String,
    /// Environment/mechanism family and independent cluster identities.
    pub environment_family_id: String,
    pub environment_cluster_id: String,
    /// Digest of the exact state snapshot from which this record was derived.
    pub source_state_sha256: String,
    /// Frozen candidate checkpoint digest used to produce this state's abstraction.
    pub candidate_checkpoint_sha256: String,
    /// Candidate-assigned abstraction label. It is not trusted as evaluator truth.
    pub abstract_state_id: String,
    /// Evaluator-owned invariant values/class identity.
    pub invariant_signature: BTreeMap<String, i64>,
    /// Evaluator-owned outcome values in the contract's canonical integer units.
    pub responses: BTreeMap<ResponseKey, i64>,
}

impl StateResponseRecord {
    pub fn validate_against(&self, contract: &StateResponseContract) -> Result<(), String> {
        if self.state_id.trim().is_empty()
            || self.abstract_state_id.trim().is_empty()
            || self.domain_id.trim().is_empty()
            || self.environment_family_id.trim().is_empty()
            || self.environment_cluster_id.trim().is_empty()
        {
            return Err(
                "state, abstraction, domain, family and cluster IDs must be non-empty".to_owned(),
            );
        }
        if !contract.domain_ids.contains(&self.domain_id) {
            return Err(format!(
                "state {} uses a domain outside the frozen contract",
                self.state_id
            ));
        }
        if !is_sha256(&self.source_state_sha256) {
            return Err(format!(
                "state {} has an invalid source-state digest",
                self.state_id
            ));
        }
        if !is_sha256(&self.candidate_checkpoint_sha256) {
            return Err(format!(
                "state {} has an invalid candidate-checkpoint digest",
                self.state_id
            ));
        }
        let supplied_invariants: BTreeSet<_> = self.invariant_signature.keys().cloned().collect();
        if supplied_invariants != contract.invariant_ids {
            return Err(format!(
                "state {} has incomplete or unexpected invariant fields",
                self.state_id
            ));
        }
        let supplied_responses: BTreeSet<_> = self.responses.keys().cloned().collect();
        if supplied_responses != contract.expected_keys() {
            return Err(format!(
                "state {} has incomplete or out-of-contract response coverage",
                self.state_id
            ));
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
    /// Number of candidate abstraction classes (a coarse compression proxy).
    pub candidate_class_count: usize,
    /// Number of singleton classes (one state retained per class).
    pub singleton_class_count: usize,
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
    if left.candidate_checkpoint_sha256 != right.candidate_checkpoint_sha256 {
        return Err("pair assessment cannot mix candidate checkpoints".to_owned());
    }
    if left.environment_cluster_id == right.environment_cluster_id
        && (left.domain_id != right.domain_id
            || left.environment_family_id != right.environment_family_id)
    {
        return Err(
            "one environment cluster cannot alias multiple domain/family identities".to_owned(),
        );
    }
    Ok(assess_pair_validated(left, right, contract))
}

fn assess_pair_validated(
    left: &StateResponseRecord,
    right: &StateResponseRecord,
    contract: &StateResponseContract,
) -> PairAssessment {
    let mut maximum = 0.0_f64;
    for (key, left_value) in &left.responses {
        let right_value = right.responses[key];
        let tolerance = contract.tolerance_units[key];
        let distance = left_value.abs_diff(right_value) as f64 / tolerance as f64;
        maximum = maximum.max(distance);
    }
    PairAssessment {
        exact_equivalent: left.invariant_signature == right.invariant_signature
            && left.responses == right.responses,
        normalized_worst_case_distortion: maximum,
    }
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
    let assessment = assess_pair(left, right, contract)?;
    Ok(left.invariant_signature == right.invariant_signature
        && assessment.normalized_worst_case_distortion <= epsilon)
}

fn validate_record_set(
    records: &[StateResponseRecord],
    contract: &StateResponseContract,
) -> Result<(), String> {
    contract.validate()?;
    if records.is_empty() {
        return Err("at least one state record is required".to_owned());
    }

    let mut state_ids = BTreeSet::new();
    let mut frozen_checkpoint: Option<&str> = None;
    let mut cluster_metadata: BTreeMap<&str, (&str, &str)> = BTreeMap::new();

    for record in records {
        record.validate_against(contract)?;
        if !state_ids.insert(record.state_id.as_str()) {
            return Err(format!("duplicate state ID: {}", record.state_id));
        }
        match frozen_checkpoint {
            Some(checkpoint) if checkpoint != record.candidate_checkpoint_sha256.as_str() => {
                return Err(
                    "one evaluation must use exactly one frozen candidate checkpoint".to_owned(),
                );
            }
            None => frozen_checkpoint = Some(&record.candidate_checkpoint_sha256),
            _ => {}
        }
        let metadata = (
            record.domain_id.as_str(),
            record.environment_family_id.as_str(),
        );
        match cluster_metadata.get(record.environment_cluster_id.as_str()) {
            Some(previous) if *previous != metadata => {
                return Err(format!(
                    "environment cluster {} is reused across domains or environment families",
                    record.environment_cluster_id
                ));
            }
            None => {
                cluster_metadata.insert(record.environment_cluster_id.as_str(), metadata);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Score the candidate's state partition against evaluator-owned response
/// equivalence. Every unordered state pair is one comparison; task rows or
/// intervention cells are not counted as independent statistical units.
pub fn score_abstraction(
    records: &[StateResponseRecord],
    contract: &StateResponseContract,
) -> Result<AbstractionScore, String> {
    validate_record_set(records, contract)?;

    let mut class_sizes: BTreeMap<&str, usize> = BTreeMap::new();
    for record in records {
        *class_sizes
            .entry(record.abstract_state_id.as_str())
            .or_default() += 1;
    }

    let mut score = AbstractionScore {
        candidate_class_count: class_sizes.len(),
        singleton_class_count: class_sizes.values().filter(|size| **size == 1).count(),
        ..AbstractionScore::default()
    };
    for left_index in 0..records.len() {
        for right_index in (left_index + 1)..records.len() {
            let left = &records[left_index];
            let right = &records[right_index];
            let assessment = assess_pair_validated(left, right, contract);
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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CounterexampleDimension {
    Invariant { invariant_id: String },
    Response { key: ResponseKey },
}

/// An evaluator-selected separating query. It deliberately contains no
/// evaluator response value, target label or hidden state field value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CounterexampleProbe {
    pub contract_id: String,
    pub left_state_id: String,
    pub right_state_id: String,
    pub dimension: CounterexampleDimension,
    /// None means the witness is a structural invariant mismatch; no action-query cost is assigned here.
    pub probe_cost_units: Option<u64>,
}

/// Find the cheapest deterministic counterexample among states that the
/// candidate currently assigns to the same abstraction class.
///
/// The evaluator returns only the separating invariant/query identity. It
/// never exposes the response or invariant values used to find the counterexample.
/// Verification/reconstruction cost for invariant witnesses is charged separately.
pub fn select_counterexample_probe(
    records: &[StateResponseRecord],
    contract: &StateResponseContract,
) -> Result<Option<CounterexampleProbe>, String> {
    validate_record_set(records, contract)?;

    let mut groups: BTreeMap<String, Vec<&StateResponseRecord>> = BTreeMap::new();
    for record in records {
        groups
            .entry(record.abstract_state_id.clone())
            .or_default()
            .push(record);
    }

    let mut candidates = Vec::new();
    for group in groups.values_mut() {
        group.sort_by(|left, right| left.state_id.cmp(&right.state_id));
        for left_index in 0..group.len() {
            for right_index in (left_index + 1)..group.len() {
                let left = group[left_index];
                let right = group[right_index];

                if let Some((invariant_id, _)) = left
                    .invariant_signature
                    .iter()
                    .find(|(key, value)| right.invariant_signature.get(*key) != Some(*value))
                {
                    candidates.push(CounterexampleProbe {
                        contract_id: contract.contract_id.clone(),
                        left_state_id: left.state_id.clone(),
                        right_state_id: right.state_id.clone(),
                        dimension: CounterexampleDimension::Invariant {
                            invariant_id: invariant_id.clone(),
                        },
                        probe_cost_units: None,
                    });
                    continue;
                }

                let cheapest_witness = left
                    .responses
                    .iter()
                    .filter_map(|(key, left_value)| {
                        let right_value = right.responses.get(key)?;
                        if left_value == right_value {
                            return None;
                        }
                        Some((contract.probe_cost_units[key], key))
                    })
                    .min_by(|(left_cost, left_key), (right_cost, right_key)| {
                        left_cost
                            .cmp(right_cost)
                            .then_with(|| left_key.cmp(right_key))
                    });

                if let Some((cost, key)) = cheapest_witness {
                    candidates.push(CounterexampleProbe {
                        contract_id: contract.contract_id.clone(),
                        left_state_id: left.state_id.clone(),
                        right_state_id: right.state_id.clone(),
                        dimension: CounterexampleDimension::Response { key: key.clone() },
                        probe_cost_units: Some(cost),
                    });
                }
            }
        }
    }

    candidates.sort_by(|left, right| {
        left.probe_cost_units
            .cmp(&right.probe_cost_units)
            .then_with(|| left.left_state_id.cmp(&right.left_state_id))
            .then_with(|| left.right_state_id.cmp(&right.right_state_id))
            .then_with(|| left.dimension.cmp(&right.dimension))
    });
    Ok(candidates.into_iter().next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> StateResponseContract {
        let intervention_ids = ["no_op", "push_left", "push_right"]
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
            contract_sha256: "a".repeat(64),
            source_manifest_sha256: "b".repeat(64),
            domain_ids: ["rigid-object-v1".to_owned()].into_iter().collect(),
            intervention_ids,
            query_ids,
            horizons,
            invariant_ids,
            tolerance_units: BTreeMap::new(),
            probe_cost_units: BTreeMap::new(),
        };
        for key in result.expected_keys() {
            result.tolerance_units.insert(key.clone(), 10);
            result.probe_cost_units.insert(key, 5);
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
            domain_id: "rigid-object-v1".to_owned(),
            environment_family_id: "family-a".to_owned(),
            environment_cluster_id: format!("cluster-{state_id}"),
            source_state_sha256: "c".repeat(64),
            candidate_checkpoint_sha256: "d".repeat(64),
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
        assert!(
            assess_pair(&left, &right, &contract)
                .expect("valid pair")
                .exact_equivalent
        );
        assert_eq!(
            score_abstraction(&[left, right], &contract)
                .expect("valid fixture")
                .false_merges,
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
        assert!(
            !assess_pair(&left, &right, &contract)
                .expect("valid pair")
                .exact_equivalent
        );
        assert_eq!(
            score_abstraction(&[left, right], &contract)
                .expect("valid fixture")
                .false_merges,
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
        assert!(
            !approximate_response_equivalent(&a, &c, &contract, 1.0).expect("valid comparison")
        );
    }

    #[test]
    fn missing_or_extra_response_cells_are_rejected() {
        let contract = contract();
        let mut missing = record("raw-a", "state-a", 0, 2);
        missing.responses.pop_first();
        assert!(missing.validate_against(&contract).is_err());

        let mut extra = record("raw-b", "state-b", 0, 2);
        extra
            .responses
            .insert(ResponseKey::new("unknown", "displacement", 1), 0);
        assert!(extra.validate_against(&contract).is_err());
    }

    #[test]
    fn contract_rejects_zero_tolerance_and_incomplete_grid() {
        let mut invalid = contract();
        let key = invalid
            .expected_keys()
            .into_iter()
            .next()
            .expect("nonempty");
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
    fn toy_world_exposes_copy_appearance_and_response_quotient_tradeoffs() {
        let contract = contract();
        let mut states = vec![
            record("sticky-red", "red", 0, 2),
            record("sticky-blue", "blue", 0, 2),
            record("slippery-red", "red", 8, 2),
            record("slippery-blue", "blue", 8, 2),
        ];
        let no_op_keys: Vec<_> = contract
            .expected_keys()
            .into_iter()
            .filter(|key| key.intervention_id == "no_op")
            .collect();
        for state in &mut states {
            for key in &no_op_keys {
                state.responses.insert(key.clone(), 0);
            }
        }

        // Copy/no-op grouping: identical no-op outcomes do not imply
        // equivalent responses to registered interventions.
        let mut copy_only = states.clone();
        for state in &mut copy_only {
            state.abstract_state_id = "copied-current-state".to_owned();
        }
        let copy_score = score_abstraction(&copy_only, &contract).expect("copy-only partition");
        assert_eq!(copy_score.candidate_class_count, 1);
        assert_eq!(copy_score.false_merges, 4);

        // Appearance grouping both merges different mechanisms and splits
        // same-mechanism states under an irrelevant nuisance change.
        let appearance_score = score_abstraction(&states, &contract).expect("appearance partition");
        assert_eq!(appearance_score.candidate_class_count, 2);
        assert_eq!(appearance_score.false_merges, 2);
        assert_eq!(appearance_score.false_splits, 2);

        // Evaluator-only quotient by identical response signatures is a
        // diagnostic upper bound, not a candidate-visible baseline.
        let mut response_quotient = states;
        response_quotient[0].abstract_state_id = "sticky".to_owned();
        response_quotient[1].abstract_state_id = "sticky".to_owned();
        response_quotient[2].abstract_state_id = "slippery".to_owned();
        response_quotient[3].abstract_state_id = "slippery".to_owned();
        let quotient_score =
            score_abstraction(&response_quotient, &contract).expect("response quotient");
        assert_eq!(quotient_score.candidate_class_count, 2);
        assert_eq!(quotient_score.false_merges, 0);
        assert_eq!(quotient_score.false_splits, 0);
    }

    #[test]
    fn abstraction_score_reports_partition_cardinality_and_singletons() {
        let contract = contract();
        let first = record("state-a", "class-shared", 0, 2);
        let second = record("state-b", "class-shared", 0, 2);
        let third = record("state-c", "class-singleton", 0, 2);
        let score = score_abstraction(&[first, second, third], &contract).expect("valid partition");
        assert_eq!(score.candidate_class_count, 2);
        assert_eq!(score.singleton_class_count, 1);
        assert_eq!(score.false_splits, 2);
    }

    #[test]
    fn copy_no_op_response_does_not_hide_action_distinguishable_states() {
        let mut contract = contract();
        let separating_key = ResponseKey::new("push_right", "displacement", 1);
        contract.probe_cost_units.insert(separating_key.clone(), 1);

        let left = record("state-a", "copied-current-state", 0, 2);
        let mut right = record("state-b", "copied-current-state", 1, 2);
        let no_op_key = ResponseKey::new("no_op", "displacement", 1);
        right
            .responses
            .insert(no_op_key.clone(), left.responses[&no_op_key]);

        assert_eq!(left.responses[&no_op_key], right.responses[&no_op_key]);
        assert_ne!(
            left.responses[&separating_key],
            right.responses[&separating_key]
        );

        let score = score_abstraction(&[left.clone(), right.clone()], &contract)
            .expect("valid copy-control fixture");
        assert_eq!(score.false_merges, 1);

        let probe = select_counterexample_probe(&[left, right], &contract)
            .expect("valid contract")
            .expect("action consequence distinguishes the states");
        assert_eq!(
            probe.dimension,
            CounterexampleDimension::Response {
                key: separating_key
            }
        );
    }

    #[test]
    fn domain_scope_and_source_digests_are_mandatory() {
        let contract = contract();
        let mut out_of_domain = record("state-a", "class-a", 0, 2);
        out_of_domain.domain_id = "unknown-domain".to_owned();
        assert!(out_of_domain.validate_against(&contract).is_err());

        let mut invalid_source = record("state-b", "class-b", 0, 2);
        invalid_source.source_state_sha256 = "not-a-digest".to_owned();
        assert!(invalid_source.validate_against(&contract).is_err());

        let mut invalid_contract = contract.clone();
        invalid_contract.source_manifest_sha256 = "bad".to_owned();
        assert!(invalid_contract.validate().is_err());

        invalid_contract.source_manifest_sha256 = "b".repeat(64);
        invalid_contract.contract_sha256 = "bad".to_owned();
        assert!(invalid_contract.validate().is_err());
    }

    #[test]
    fn one_evaluation_cannot_mix_candidate_checkpoints() {
        let contract = contract();
        let left = record("state-a", "class-a", 0, 2);
        let mut right = record("state-b", "class-b", 0, 2);
        right.candidate_checkpoint_sha256 = "e".repeat(64);
        assert!(assess_pair(&left, &right, &contract).is_err());
        assert!(score_abstraction(&[left, right], &contract).is_err());
        let left = record("state-a", "class-a", 0, 2);
        let mut right = record("state-b", "class-b", 0, 2);
        right.candidate_checkpoint_sha256 = "e".repeat(64);
        assert!(select_counterexample_probe(&[left, right], &contract).is_err());
    }

    #[test]
    fn independent_cluster_cannot_alias_domains_or_environment_families() {
        let contract = contract();
        let left = record("state-a", "class-a", 0, 2);
        let mut right = record("state-b", "class-b", 1, 2);
        right.environment_cluster_id = left.environment_cluster_id.clone();
        right.environment_family_id = "family-b".to_owned();
        assert!(score_abstraction(&[left, right], &contract).is_err());
    }

    #[test]
    fn same_domain_family_and_cluster_identity_passes_validation() {
        let contract = contract();
        let left = record("state-a", "class-a", 0, 2);
        let mut right = record("state-b", "class-b", 0, 2);
        right.environment_cluster_id = left.environment_cluster_id.clone();
        // States in one independent cluster may share a domain and family.
        right.environment_family_id = left.environment_family_id.clone();
        let score = score_abstraction(&[left, right], &contract)
            .expect("same checkpoint and cluster identity");
        assert_eq!(score.compared_pairs, 1);
    }

    #[test]
    fn counterexample_probe_returns_a_query_without_leaking_response_values() {
        let mut contract = contract();
        let expected_keys = contract.expected_keys();
        for key in expected_keys {
            contract.probe_cost_units.insert(key, 5);
        }
        let cheapest = ResponseKey::new("push_right", "rotation", 5);
        contract.probe_cost_units.insert(cheapest.clone(), 1);
        let left = record("state-a", "same-class", 0, 2);
        let right = record("state-b", "same-class", 1, 2);

        let probe = select_counterexample_probe(&[left, right], &contract)
            .expect("valid contract")
            .expect("counterexample exists");
        assert_eq!(probe.contract_id, contract.contract_id);
        assert_eq!(probe.left_state_id, "state-a");
        assert_eq!(probe.right_state_id, "state-b");
        assert_eq!(
            probe.dimension,
            CounterexampleDimension::Response { key: cheapest }
        );
        assert_eq!(probe.probe_cost_units, Some(1));
    }

    #[test]
    fn counterexample_probe_prefers_invariant_witness_without_value_disclosure() {
        let contract = contract();
        let left = record("state-a", "same-class", 0, 2);
        let right = record("state-b", "same-class", 0, 3);
        let probe = select_counterexample_probe(&[left, right], &contract)
            .expect("valid contract")
            .expect("invariant witness exists");
        assert_eq!(
            probe.dimension,
            CounterexampleDimension::Invariant {
                invariant_id: "mass_class".to_owned()
            }
        );
        assert_eq!(probe.probe_cost_units, None);
    }

    #[test]
    fn counterexample_probe_ignores_pairs_already_in_separate_classes() {
        let contract = contract();
        let left = record("state-a", "class-a", 0, 2);
        let right = record("state-b", "class-b", 1, 2);
        assert!(
            select_counterexample_probe(&[left, right], &contract)
                .expect("valid contract")
                .is_none()
        );
    }

    #[test]
    fn counterexample_probe_is_deterministic_across_input_order() {
        let contract = contract();
        let left = record("state-a", "same-class", 0, 2);
        let middle = record("state-b", "same-class", 1, 2);
        let right = record("state-c", "same-class", 2, 2);
        let forward =
            select_counterexample_probe(&[left.clone(), middle.clone(), right.clone()], &contract)
                .expect("valid contract");
        let reverse =
            select_counterexample_probe(&[right, middle, left], &contract).expect("valid contract");
        assert_eq!(forward, reverse);
    }

    #[test]
    fn counterexample_refinement_can_remove_a_false_merge_one_pair_at_a_time() {
        let contract = contract();
        let left = record("state-a", "class-0", 0, 2);
        let middle = record("state-b", "class-0", 1, 2);
        let right = record("state-c", "class-0", 2, 2);
        let first =
            select_counterexample_probe(&[left.clone(), middle.clone(), right.clone()], &contract)
                .expect("valid contract")
                .expect("first counterexample");
        assert_eq!(
            (first.left_state_id.as_str(), first.right_state_id.as_str()),
            ("state-a", "state-b")
        );

        let mut refined_middle = middle;
        refined_middle.abstract_state_id = "class-1".to_owned();
        let second = select_counterexample_probe(
            &[left.clone(), refined_middle.clone(), right.clone()],
            &contract,
        )
        .expect("valid contract")
        .expect("remaining counterexample");
        assert_eq!(
            (
                second.left_state_id.as_str(),
                second.right_state_id.as_str()
            ),
            ("state-a", "state-c")
        );

        let mut refined_right = right;
        refined_right.abstract_state_id = "class-2".to_owned();
        assert!(
            select_counterexample_probe(&[left, refined_middle, refined_right], &contract)
                .expect("valid contract")
                .is_none()
        );
    }

    #[test]
    fn contract_rejects_missing_or_zero_probe_costs() {
        let mut missing = contract();
        missing.probe_cost_units.pop_first();
        assert!(missing.validate().is_err());

        let mut zero = contract();
        let key = zero.expected_keys().into_iter().next().expect("nonempty");
        zero.probe_cost_units.insert(key, 0);
        assert!(zero.validate().is_err());
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
