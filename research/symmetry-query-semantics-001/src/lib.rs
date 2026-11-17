// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact query aggregation over an independently verified finite quotient.
//!
//! A valid orbit partition is not enough: the requested query determines how
//! orbit representatives must be aggregated. This module consumes the finite
//! quotient certificate and independently checks the query-relevant fields
//! over every orbit before evaluating the quotient query.

use std::fmt;

use symmetry_quotient_reference::{
    FiniteActionV1, QualificationBudgetV1, QuotientCertificateV1, QuotientErrorV1, SearchProblemV1,
    verify_certificate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKindV1 {
    Exists,
    Minimum,
    Maximum,
    Enumerate,
    Count,
    WeightedSum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationLawV1 {
    BooleanAny,
    ExtremeValueWithCompleteTieLift,
    CompleteMemberLift,
    OrbitCardinalityCount,
    OrbitCardinalityWeightedSum,
}

impl QueryKindV1 {
    pub const fn aggregation_law(self) -> AggregationLawV1 {
        match self {
            Self::Exists => AggregationLawV1::BooleanAny,
            Self::Minimum | Self::Maximum => AggregationLawV1::ExtremeValueWithCompleteTieLift,
            Self::Enumerate => AggregationLawV1::CompleteMemberLift,
            Self::Count => AggregationLawV1::OrbitCardinalityCount,
            Self::WeightedSum => AggregationLawV1::OrbitCardinalityWeightedSum,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::Exists => 0,
            Self::Minimum => 1,
            Self::Maximum => 2,
            Self::Enumerate => 3,
            Self::Count => 4,
            Self::WeightedSum => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryRowV1 {
    /// The query's candidate predicate. It must be constant on every orbit.
    pub included: bool,
    /// Exact objective value, used only by Minimum and Maximum.
    pub objective: i128,
    /// Exact signed weight, used only by WeightedSum.
    pub weight: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteQueryV1 {
    pub query_id: String,
    pub kind: QueryKindV1,
    /// Rows are aligned with SearchProblemV1::candidate_ids by exact index.
    pub rows: Vec<QueryRowV1>,
}

impl FiniteQueryV1 {
    /// Canonical, versioned query-contract bytes.
    ///
    /// The bytes are carried in the receipt for exact replay. This v0.1
    /// implementation does not claim a cryptographic digest or signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symmetry_query.v1\0".to_vec();
        append_bytes(&mut bytes, self.query_id.as_bytes());
        bytes.push(self.kind.tag());
        bytes.extend_from_slice(&(self.rows.len() as u64).to_le_bytes());
        for row in &self.rows {
            bytes.push(u8::from(row.included));
            bytes.extend_from_slice(&row.objective.to_le_bytes());
            bytes.extend_from_slice(&row.weight.to_le_bytes());
        }
        bytes
    }
}

fn append_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryFieldV1 {
    Predicate,
    Objective,
    Weight,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryOutcomeV1 {
    Exists(bool),
    Minimum {
        value: Option<i128>,
        candidate_ids: Vec<String>,
    },
    Maximum {
        value: Option<i128>,
        candidate_ids: Vec<String>,
    },
    Enumerate(Vec<String>),
    Count(u64),
    WeightedSum(i128),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrbitContributionV1 {
    pub representative_id: String,
    pub orbit_size: usize,
    pub included: bool,
    pub representative_objective: Option<i128>,
    pub representative_weight: Option<i128>,
    /// Meaning is fixed by QueryKindV1::aggregation_law.
    /// Enumeration and min/max have no additive contribution.
    pub additive_contribution: Option<i128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryReceiptV1 {
    pub schema_version: u32,
    pub problem_id: String,
    pub action_id: String,
    pub query_id: String,
    pub query_kind: QueryKindV1,
    pub aggregation_law: AggregationLawV1,
    pub query_contract_bytes: Vec<u8>,
    pub result: QueryOutcomeV1,
    pub orbit_contributions: Vec<OrbitContributionV1>,
    pub orbit_count: usize,
    pub original_candidate_count: usize,
    /// Number of representatives whose query-relevant row was inspected.
    pub representative_evaluations: usize,
    /// Number of original candidates the exhaustive query would inspect.
    pub exhaustive_candidate_evaluations: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryErrorV1 {
    InvalidQueryId,
    CandidateRowCountMismatch,
    QueryInvariantViolation {
        field: QueryFieldV1,
        group_element: usize,
        representative_index: usize,
        member_index: usize,
    },
    ArithmeticOverflow,
    InvalidOrbit,
    ReceiptMismatch,
    BaseQuotient(QuotientErrorV1),
}

impl fmt::Display for QueryErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for QueryErrorV1 {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_query_shape(
    problem: &SearchProblemV1,
    query: &FiniteQueryV1,
) -> Result<(), QueryErrorV1> {
    if !nonempty(&query.query_id) {
        return Err(QueryErrorV1::InvalidQueryId);
    }
    if query.rows.len() != problem.candidate_ids.len() {
        return Err(QueryErrorV1::CandidateRowCountMismatch);
    }
    Ok(())
}

fn check_orbit_query_invariance(
    query: &FiniteQueryV1,
    representative_index: usize,
    member_index: usize,
    group_element: usize,
) -> Result<(), QueryErrorV1> {
    let representative = query
        .rows
        .get(representative_index)
        .ok_or(QueryErrorV1::InvalidOrbit)?;
    let member = query
        .rows
        .get(member_index)
        .ok_or(QueryErrorV1::InvalidOrbit)?;

    if representative.included != member.included {
        return Err(QueryErrorV1::QueryInvariantViolation {
            field: QueryFieldV1::Predicate,
            group_element,
            representative_index,
            member_index,
        });
    }

    match query.kind {
        QueryKindV1::Minimum | QueryKindV1::Maximum => {
            if representative.objective != member.objective {
                return Err(QueryErrorV1::QueryInvariantViolation {
                    field: QueryFieldV1::Objective,
                    group_element,
                    representative_index,
                    member_index,
                });
            }
        }
        QueryKindV1::WeightedSum => {
            if representative.weight != member.weight {
                return Err(QueryErrorV1::QueryInvariantViolation {
                    field: QueryFieldV1::Weight,
                    group_element,
                    representative_index,
                    member_index,
                });
            }
        }
        QueryKindV1::Exists | QueryKindV1::Enumerate | QueryKindV1::Count => {}
    }
    Ok(())
}

fn sorted_ids(mut candidate_ids: Vec<String>) -> Vec<String> {
    candidate_ids.sort();
    candidate_ids
}

pub fn evaluate_quotient_query(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    quotient: &QuotientCertificateV1,
    query: &FiniteQueryV1,
    budget: QualificationBudgetV1,
) -> Result<QueryReceiptV1, QueryErrorV1> {
    validate_query_shape(problem, query)?;
    verify_certificate(problem, action, quotient, budget).map_err(QueryErrorV1::BaseQuotient)?;

    let candidate_count = problem.candidate_ids.len();
    let orbit_count = quotient.orbits.len();
    let mut contributions = Vec::with_capacity(orbit_count);

    for orbit in &quotient.orbits {
        let representative_index = orbit.representative_index;
        let representative = query
            .rows
            .get(representative_index)
            .ok_or(QueryErrorV1::InvalidOrbit)?;

        for member in &orbit.members {
            check_orbit_query_invariance(
                query,
                representative_index,
                member.candidate_index,
                member.group_element,
            )?;
        }

        let included = representative.included;
        let (representative_objective, representative_weight) = match query.kind {
            QueryKindV1::Minimum | QueryKindV1::Maximum => (Some(representative.objective), None),
            QueryKindV1::WeightedSum => (None, Some(representative.weight)),
            QueryKindV1::Exists | QueryKindV1::Enumerate | QueryKindV1::Count => (None, None),
        };

        let additive_contribution = match query.kind {
            QueryKindV1::Exists => Some(included as i128),
            QueryKindV1::Count if included => Some(
                i128::try_from(orbit.members.len())
                    .map_err(|_| QueryErrorV1::ArithmeticOverflow)?,
            ),
            QueryKindV1::Count => Some(0),
            QueryKindV1::WeightedSum if included => {
                let orbit_size = i128::try_from(orbit.members.len())
                    .map_err(|_| QueryErrorV1::ArithmeticOverflow)?;
                Some(
                    representative
                        .weight
                        .checked_mul(orbit_size)
                        .ok_or(QueryErrorV1::ArithmeticOverflow)?,
                )
            }
            QueryKindV1::WeightedSum => Some(0),
            QueryKindV1::Minimum | QueryKindV1::Maximum | QueryKindV1::Enumerate => None,
        };

        contributions.push(OrbitContributionV1 {
            representative_id: orbit.representative_id.clone(),
            orbit_size: orbit.members.len(),
            included,
            representative_objective,
            representative_weight,
            additive_contribution,
        });
    }

    let result = match query.kind {
        QueryKindV1::Exists => {
            QueryOutcomeV1::Exists(contributions.iter().any(|item| item.included))
        }
        QueryKindV1::Minimum | QueryKindV1::Maximum => {
            let mut optimum: Option<i128> = None;
            for orbit in &quotient.orbits {
                let row = &query.rows[orbit.representative_index];
                if !row.included {
                    continue;
                }
                optimum = Some(match (query.kind, optimum) {
                    (QueryKindV1::Minimum, Some(current)) => current.min(row.objective),
                    (QueryKindV1::Maximum, Some(current)) => current.max(row.objective),
                    (_, None) => row.objective,
                    _ => unreachable!("only minimum and maximum reach this branch"),
                });
            }

            let candidate_ids = if let Some(value) = optimum {
                let mut ids = Vec::new();
                for orbit in &quotient.orbits {
                    let row = &query.rows[orbit.representative_index];
                    if row.included && row.objective == value {
                        ids.extend(
                            orbit
                                .members
                                .iter()
                                .map(|member| member.candidate_id.clone()),
                        );
                    }
                }
                sorted_ids(ids)
            } else {
                Vec::new()
            };

            match query.kind {
                QueryKindV1::Minimum => QueryOutcomeV1::Minimum {
                    value: optimum,
                    candidate_ids,
                },
                QueryKindV1::Maximum => QueryOutcomeV1::Maximum {
                    value: optimum,
                    candidate_ids,
                },
                _ => unreachable!("the enclosing match restricts this branch"),
            }
        }
        QueryKindV1::Enumerate => {
            let ids = quotient
                .orbits
                .iter()
                .filter(|orbit| query.rows[orbit.representative_index].included)
                .flat_map(|orbit| {
                    orbit
                        .members
                        .iter()
                        .map(|member| member.candidate_id.clone())
                })
                .collect();
            QueryOutcomeV1::Enumerate(sorted_ids(ids))
        }
        QueryKindV1::Count => {
            let count = contributions.iter().try_fold(0u64, |total, item| {
                let contribution = u64::try_from(item.additive_contribution.unwrap_or(0))
                    .map_err(|_| QueryErrorV1::ArithmeticOverflow)?;
                total
                    .checked_add(contribution)
                    .ok_or(QueryErrorV1::ArithmeticOverflow)
            })?;
            QueryOutcomeV1::Count(count)
        }
        QueryKindV1::WeightedSum => {
            let sum = contributions.iter().try_fold(0i128, |total, item| {
                total
                    .checked_add(item.additive_contribution.unwrap_or(0))
                    .ok_or(QueryErrorV1::ArithmeticOverflow)
            })?;
            QueryOutcomeV1::WeightedSum(sum)
        }
    };

    Ok(QueryReceiptV1 {
        schema_version: 1,
        problem_id: problem.problem_id.clone(),
        action_id: action.action_id.clone(),
        query_id: query.query_id.clone(),
        query_kind: query.kind,
        aggregation_law: query.kind.aggregation_law(),
        query_contract_bytes: query.canonical_bytes(),
        result,
        orbit_contributions: contributions,
        orbit_count,
        original_candidate_count: candidate_count,
        representative_evaluations: orbit_count,
        exhaustive_candidate_evaluations: candidate_count,
    })
}

/// Independent direct-query baseline. It deliberately evaluates each original
/// candidate and does not inspect or use the quotient certificate.
pub fn evaluate_exhaustive_query(
    problem: &SearchProblemV1,
    query: &FiniteQueryV1,
) -> Result<QueryOutcomeV1, QueryErrorV1> {
    validate_query_shape(problem, query)?;
    let included: Vec<usize> = query
        .rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row.included.then_some(index))
        .collect();

    let result = match query.kind {
        QueryKindV1::Exists => QueryOutcomeV1::Exists(!included.is_empty()),
        QueryKindV1::Minimum => {
            let optimum = included
                .iter()
                .map(|index| query.rows[*index].objective)
                .min();
            let ids = optimum.map_or_else(Vec::new, |value| {
                sorted_ids(
                    included
                        .iter()
                        .filter(|index| query.rows[**index].objective == value)
                        .map(|index| problem.candidate_ids[*index].clone())
                        .collect(),
                )
            });
            QueryOutcomeV1::Minimum {
                value: optimum,
                candidate_ids: ids,
            }
        }
        QueryKindV1::Maximum => {
            let optimum = included
                .iter()
                .map(|index| query.rows[*index].objective)
                .max();
            let ids = optimum.map_or_else(Vec::new, |value| {
                sorted_ids(
                    included
                        .iter()
                        .filter(|index| query.rows[**index].objective == value)
                        .map(|index| problem.candidate_ids[*index].clone())
                        .collect(),
                )
            });
            QueryOutcomeV1::Maximum {
                value: optimum,
                candidate_ids: ids,
            }
        }
        QueryKindV1::Enumerate => QueryOutcomeV1::Enumerate(sorted_ids(
            included
                .iter()
                .map(|index| problem.candidate_ids[*index].clone())
                .collect(),
        )),
        QueryKindV1::Count => QueryOutcomeV1::Count(
            u64::try_from(included.len()).map_err(|_| QueryErrorV1::ArithmeticOverflow)?,
        ),
        QueryKindV1::WeightedSum => {
            let sum = included.iter().try_fold(0i128, |total, index| {
                total
                    .checked_add(query.rows[*index].weight)
                    .ok_or(QueryErrorV1::ArithmeticOverflow)
            })?;
            QueryOutcomeV1::WeightedSum(sum)
        }
    };
    Ok(result)
}

/// Independently replay the base quotient certificate and the complete query
/// calculation. The verifier compares all receipt fields, including the exact
/// canonical query-contract bytes and per-orbit contributions.
pub fn verify_query_receipt(
    problem: &SearchProblemV1,
    action: &FiniteActionV1,
    quotient: &QuotientCertificateV1,
    query: &FiniteQueryV1,
    receipt: &QueryReceiptV1,
    budget: QualificationBudgetV1,
) -> Result<(), QueryErrorV1> {
    if receipt.schema_version != 1
        || receipt.query_contract_bytes != query.canonical_bytes()
        || receipt.problem_id != problem.problem_id
        || receipt.action_id != action.action_id
        || receipt.query_id != query.query_id
        || receipt.query_kind != query.kind
        || receipt.aggregation_law != query.kind.aggregation_law()
    {
        return Err(QueryErrorV1::ReceiptMismatch);
    }

    let replayed = evaluate_quotient_query(problem, action, quotient, query, budget)?;
    if *receipt != replayed {
        return Err(QueryErrorV1::ReceiptMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symmetry_quotient_reference::{
        CandidateSemanticsV1, CompileOutcomeV1, FiniteGroupV1, compile_quotient,
    };

    fn d4_action() -> FiniteActionV1 {
        let mut multiplication = vec![vec![0; 8]; 8];
        for left in 0..8 {
            let left_rotation = left % 4;
            let left_reflection = left / 4;
            for right in 0..8 {
                let right_rotation = right % 4;
                let right_reflection = right / 4;
                let signed_rotation = if left_reflection == 0 {
                    right_rotation
                } else {
                    (4 - right_rotation) % 4
                };
                let product_rotation = (left_rotation + signed_rotation) % 4;
                let product_reflection = left_reflection ^ right_reflection;
                multiplication[left][right] = product_reflection * 4 + product_rotation;
            }
        }

        let mut permutations = Vec::with_capacity(8);
        for element in 0..8 {
            let rotation = element % 4;
            let reflection = element / 4;
            let mut permutation = vec![0; 4];
            for vertex in 0..4 {
                let signed_vertex = if reflection == 0 {
                    vertex
                } else {
                    (4 - vertex) % 4
                };
                permutation[vertex] = (rotation + signed_vertex) % 4;
            }
            permutations.push(permutation);
        }

        let pair_permutations = permutations
            .iter()
            .map(|permutation| {
                let mut action = vec![0; 16];
                for source in 0..4 {
                    for target in 0..4 {
                        action[source * 4 + target] = permutation[source] * 4 + permutation[target];
                    }
                }
                action
            })
            .collect();

        FiniteActionV1 {
            action_id: "D4-ordered-vertex-pair-action-v1".to_owned(),
            group: FiniteGroupV1 {
                group_id: "D4-order-8-v1".to_owned(),
                identity: 0,
                multiplication,
            },
            permutations: pair_permutations,
        }
    }

    fn fixture() -> (SearchProblemV1, FiniteActionV1, QuotientCertificateV1) {
        let candidate_ids: Vec<String> = (0..16).map(|index| format!("pair-{index:02}")).collect();
        let semantics = CandidateSemanticsV1 {
            in_search_domain: true,
            admissible: true,
            feasible: true,
            objective_cost: 0,
            constraint_signature: vec![0],
            boundary_condition_id: "d4-frozen-boundary-v1".to_owned(),
            output_equivalence_id: "ordered-pair-output-v1".to_owned(),
        };
        let problem = SearchProblemV1 {
            problem_id: "d4-ordered-pair-query-fixture-v1".to_owned(),
            semantics: vec![semantics; 16],
            candidate_ids,
        };
        let action = d4_action();
        let result = compile_quotient(&problem, &action, QualificationBudgetV1::default()).unwrap();
        let CompileOutcomeV1::Authorized(certificate) = result else {
            panic!("the D4 query fixture must admit an exact quotient");
        };
        (problem, action, certificate)
    }

    fn query_rows(invariant_predicate: bool) -> Vec<QueryRowV1> {
        (0..16)
            .map(|index| {
                let source: usize = index / 4;
                let target: usize = index % 4;
                let raw_distance = source.abs_diff(target);
                let distance = raw_distance.min(4 - raw_distance);
                QueryRowV1 {
                    included: if invariant_predicate {
                        distance != 0
                    } else {
                        true
                    },
                    objective: distance as i128,
                    weight: (distance + 1) as i128,
                }
            })
            .collect()
    }

    fn query(kind: QueryKindV1, rows: Vec<QueryRowV1>) -> FiniteQueryV1 {
        FiniteQueryV1 {
            query_id: format!("d4-{kind:?}-query-v1"),
            kind,
            rows,
        }
    }

    fn evaluate(
        problem: &SearchProblemV1,
        action: &FiniteActionV1,
        certificate: &QuotientCertificateV1,
        query: &FiniteQueryV1,
    ) -> QueryReceiptV1 {
        evaluate_quotient_query(
            problem,
            action,
            certificate,
            query,
            QualificationBudgetV1::default(),
        )
        .unwrap()
    }

    #[test]
    fn all_six_query_kinds_match_the_unquotiented_d4_oracle() {
        let (problem, action, certificate) = fixture();
        let rows = query_rows(true);
        let kinds = [
            QueryKindV1::Exists,
            QueryKindV1::Minimum,
            QueryKindV1::Maximum,
            QueryKindV1::Enumerate,
            QueryKindV1::Count,
            QueryKindV1::WeightedSum,
        ];

        for kind in kinds {
            let query = query(kind, rows.clone());
            let quotient_receipt = evaluate(&problem, &action, &certificate, &query);
            let exhaustive = evaluate_exhaustive_query(&problem, &query).unwrap();
            assert_eq!(quotient_receipt.result, exhaustive, "{kind:?}");
            verify_query_receipt(
                &problem,
                &action,
                &certificate,
                &query,
                &quotient_receipt,
                QualificationBudgetV1::default(),
            )
            .unwrap();
        }
    }

    #[test]
    fn d4_count_uses_orbit_sizes_not_one_per_orbit_or_group_order() {
        let (problem, action, certificate) = fixture();
        let query = query(QueryKindV1::Count, query_rows(true));
        let receipt = evaluate(&problem, &action, &certificate, &query);

        assert_eq!(receipt.result, QueryOutcomeV1::Count(12));
        assert_eq!(receipt.orbit_count, 3);
        assert_eq!(receipt.original_candidate_count, 16);
        assert_eq!(
            receipt
                .orbit_contributions
                .iter()
                .map(|item| item.orbit_size)
                .collect::<Vec<_>>(),
            vec![4, 8, 4]
        );
        assert_ne!(12, receipt.orbit_count as u64);
        assert_ne!(
            12,
            (action.group.multiplication.len() * receipt.orbit_count) as u64
        );
    }

    #[test]
    fn min_and_max_preserve_every_original_tied_candidate() {
        let (problem, action, certificate) = fixture();
        let rows = query_rows(false);

        let minimum = evaluate(
            &problem,
            &action,
            &certificate,
            &query(QueryKindV1::Minimum, rows.clone()),
        );
        let maximum = evaluate(
            &problem,
            &action,
            &certificate,
            &query(QueryKindV1::Maximum, rows),
        );

        let QueryOutcomeV1::Minimum {
            value,
            candidate_ids,
        } = minimum.result
        else {
            panic!("expected minimum result");
        };
        assert_eq!(value, Some(0));
        assert_eq!(candidate_ids.len(), 4);
        let QueryOutcomeV1::Maximum {
            value,
            candidate_ids,
        } = maximum.result
        else {
            panic!("expected maximum result");
        };
        assert_eq!(value, Some(2));
        assert_eq!(candidate_ids.len(), 4);
    }

    #[test]
    fn non_invariant_predicate_blocks_quotient_query() {
        let (problem, action, certificate) = fixture();
        let mut rows = query_rows(true);
        rows[0].included = !rows[0].included;
        let query = query(QueryKindV1::Count, rows);

        assert!(matches!(
            evaluate_quotient_query(
                &problem,
                &action,
                &certificate,
                &query,
                QualificationBudgetV1::default()
            ),
            Err(QueryErrorV1::QueryInvariantViolation {
                field: QueryFieldV1::Predicate,
                ..
            })
        ));
    }

    #[test]
    fn non_invariant_objective_blocks_minimum_but_is_irrelevant_to_count() {
        let (problem, action, certificate) = fixture();
        let mut rows = query_rows(false);
        rows[0].objective += 7;

        assert!(matches!(
            evaluate_quotient_query(
                &problem,
                &action,
                &certificate,
                &query(QueryKindV1::Minimum, rows.clone()),
                QualificationBudgetV1::default()
            ),
            Err(QueryErrorV1::QueryInvariantViolation {
                field: QueryFieldV1::Objective,
                ..
            })
        ));

        let count = evaluate(
            &problem,
            &action,
            &certificate,
            &query(QueryKindV1::Count, rows),
        );
        assert_eq!(count.result, QueryOutcomeV1::Count(16));
    }

    #[test]
    fn non_invariant_weight_blocks_weighted_sum() {
        let (problem, action, certificate) = fixture();
        let mut rows = query_rows(true);
        rows[0].weight += 1;
        assert!(matches!(
            evaluate_quotient_query(
                &problem,
                &action,
                &certificate,
                &query(QueryKindV1::WeightedSum, rows),
                QualificationBudgetV1::default()
            ),
            Err(QueryErrorV1::QueryInvariantViolation {
                field: QueryFieldV1::Weight,
                ..
            })
        ));
    }

    #[test]
    fn weighted_sum_uses_signed_orbit_multiplicity_and_checked_arithmetic() {
        let (problem, action, certificate) = fixture();
        let mut rows = query_rows(false);
        for row in &mut rows {
            row.weight = -3;
        }
        let receipt = evaluate(
            &problem,
            &action,
            &certificate,
            &query(QueryKindV1::WeightedSum, rows),
        );
        assert_eq!(receipt.result, QueryOutcomeV1::WeightedSum(-48));

        let mut overflowing = query_rows(false);
        for row in &mut overflowing {
            row.weight = i128::MAX;
        }
        assert_eq!(
            evaluate_quotient_query(
                &problem,
                &action,
                &certificate,
                &query(QueryKindV1::WeightedSum, overflowing),
                QualificationBudgetV1::default()
            ),
            Err(QueryErrorV1::ArithmeticOverflow)
        );
    }

    #[test]
    fn forged_receipt_or_mutated_query_contract_is_rejected() {
        let (problem, action, certificate) = fixture();
        let query = query(QueryKindV1::Count, query_rows(true));
        let mut receipt = evaluate(&problem, &action, &certificate, &query);
        receipt.result = QueryOutcomeV1::Count(3);

        assert_eq!(
            verify_query_receipt(
                &problem,
                &action,
                &certificate,
                &query,
                &receipt,
                QualificationBudgetV1::default()
            ),
            Err(QueryErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn evaluation_receipt_exposes_candidate_work_reduction_without_claiming_wall_time() {
        let (problem, action, certificate) = fixture();
        let query = query(QueryKindV1::Minimum, query_rows(false));
        let receipt = evaluate(&problem, &action, &certificate, &query);

        assert_eq!(receipt.representative_evaluations, 3);
        assert_eq!(receipt.exhaustive_candidate_evaluations, 16);
    }
}
