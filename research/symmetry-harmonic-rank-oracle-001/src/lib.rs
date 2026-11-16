// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact invariant-subspace rank templates for a bounded family of proper
//! rotational subgroups of SO(3).
//!
//! This is a representation-theoretic proposal oracle, not a SymFP
//! reproduction. It does not construct an equivariant GNN block, compute an
//! observed singular spectrum, or qualify a physical symmetry.
//!
//! For the degree-l irreducible representation V_l of SO(3):
//! - C_n fixes weights m divisible by n, giving 1 + 2 floor(l / n).
//! - Proper rotational D_n (order 2n) has fixed dimension
//!   (1 + 2 floor(l / n) + (-1)^l) / 2.
//!
//! The dihedral expression is independently checked against weight-pair
//! counting: floor(l / n), plus the m=0 mode when l is even.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GroupHypothesisV1 {
    /// Proper rotations around one fixed axis, of order n.
    Cyclic { order: usize },
    /// Proper rotational dihedral group, of order 2n (n >= 2).
    RotationalDihedral { order: usize },
}

impl GroupHypothesisV1 {
    pub fn order(self) -> Result<usize, TemplateErrorV1> {
        match self {
            Self::Cyclic { order: 0 } => Err(TemplateErrorV1::ZeroGroupOrder),
            Self::Cyclic { order } => Ok(order),
            Self::RotationalDihedral { order: 0 | 1 } => Err(TemplateErrorV1::InvalidDihedralOrder),
            Self::RotationalDihedral { order } => order
                .checked_mul(2)
                .ok_or(TemplateErrorV1::ArithmeticOverflow),
        }
    }

    fn rotational_parameter(self) -> usize {
        match self {
            Self::Cyclic { order } | Self::RotationalDihedral { order } => order,
        }
    }

    fn validate(self) -> Result<(), TemplateErrorV1> {
        let _ = self.order()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemplateBudgetV1 {
    pub maximum_group_parameter: usize,
    pub maximum_degree: usize,
    pub maximum_work_units: usize,
}

impl Default for TemplateBudgetV1 {
    fn default() -> Self {
        Self {
            maximum_group_parameter: 64,
            maximum_degree: 32,
            maximum_work_units: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStatusV1 {
    NoMatch,
    Unique,
    Ambiguous,
}

/// Matching group hypotheses only. This type carries no action, physical,
/// equivalence, or pruning authority.
/// A rank-template match is permanently proposal-only in this API surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateAuthorityV1 {
    CandidateProposalOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateProposalV1 {
    pub maximum_degree: usize,
    pub maximum_group_parameter: usize,
    pub observed_signature: Vec<usize>,
    pub matching_hypotheses: Vec<GroupHypothesisV1>,
    pub status: MatchStatusV1,
    /// First degree at which any two hypotheses in the declared candidate
    /// family have different multiplicities. None means the family remains
    /// indistinguishable through the selected cutoff.
    pub first_family_divergence_degree: Option<usize>,
    pub authority: TemplateAuthorityV1,
}

impl GroupHypothesisV1 {
    pub fn canonical_label(self) -> String {
        match self {
            Self::Cyclic { order } => format!("C{order}"),
            Self::RotationalDihedral { order } => format!("D{order}"),
        }
    }
}

impl TemplateProposalV1 {
    /// Stable, versioned byte representation for replay and external hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let observed = self
            .observed_signature
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut encoded = format!(
            "luminous.symm_harmonic_rank_proposal.v1\nL={}\nmax_order={}\nobserved={}\nstatus={:?}\nfirst_divergence={}\nauthority={:?}\n",
            self.maximum_degree,
            self.maximum_group_parameter,
            observed,
            self.status,
            self.first_family_divergence_degree
                .map_or_else(|| "none".to_owned(), |degree| degree.to_string()),
            self.authority
        );
        for hypothesis in &self.matching_hypotheses {
            encoded.push_str(&hypothesis.canonical_label());
            encoded.push('\n');
        }
        encoded.into_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateErrorV1 {
    ZeroGroupOrder,
    InvalidDihedralOrder,
    EmptySignature,
    EmptyCandidateFamily,
    GroupParameterExceeded { required: usize, maximum: usize },
    DegreeExceeded { required: usize, maximum: usize },
    WorkBudgetExceeded { required: usize, maximum: usize },
    ArithmeticOverflow,
}

impl fmt::Display for TemplateErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TemplateErrorV1 {}

fn preflight(
    hypothesis: GroupHypothesisV1,
    maximum_degree: usize,
    budget: TemplateBudgetV1,
) -> Result<(), TemplateErrorV1> {
    hypothesis.validate()?;
    let parameter = hypothesis.rotational_parameter();
    if parameter > budget.maximum_group_parameter {
        return Err(TemplateErrorV1::GroupParameterExceeded {
            required: parameter,
            maximum: budget.maximum_group_parameter,
        });
    }
    if maximum_degree > budget.maximum_degree {
        return Err(TemplateErrorV1::DegreeExceeded {
            required: maximum_degree,
            maximum: budget.maximum_degree,
        });
    }
    let work = maximum_degree
        .checked_add(1)
        .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
    if work > budget.maximum_work_units {
        return Err(TemplateErrorV1::WorkBudgetExceeded {
            required: work,
            maximum: budget.maximum_work_units,
        });
    }
    Ok(())
}

/// Exact multiplicity of the trivial representation in V_l restricted to C_n.
///
/// The fixed weights are m = 0, ±n, ±2n, ..., ±floor(l/n)n.
pub fn cyclic_fixed_dimension(degree: usize, order: usize) -> Result<usize, TemplateErrorV1> {
    if order == 0 {
        return Err(TemplateErrorV1::ZeroGroupOrder);
    }
    (degree / order)
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(TemplateErrorV1::ArithmeticOverflow)
}

/// Independent explicit weight count for C_n. It is intentionally simple and
/// used as an oracle for the closed-form cyclic formula.
#[cfg(test)]
fn cyclic_weight_count(degree: usize, order: usize) -> Result<usize, TemplateErrorV1> {
    if order == 0 {
        return Err(TemplateErrorV1::ZeroGroupOrder);
    }
    let mut count = 1usize;
    let mut weight = order;
    while weight <= degree {
        count = count
            .checked_add(2)
            .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
        let Some(next) = weight.checked_add(order) else {
            break;
        };
        weight = next;
    }
    Ok(count)
}

/// Character-average formula for the trivial multiplicity in the proper
/// rotational D_n group of order 2n:
///
/// (1 + 2 floor(l/n) + (-1)^l) / 2.
///
/// The expression is evaluated in integer arithmetic; no floating-point
/// trigonometry or rounding is involved.
pub fn dihedral_character_fixed_dimension(
    degree: usize,
    order: usize,
) -> Result<usize, TemplateErrorV1> {
    if order < 2 {
        return Err(TemplateErrorV1::InvalidDihedralOrder);
    }
    let cyclic_character_sum = cyclic_fixed_dimension(degree, order)?;
    let half_turn_character = if degree.is_multiple_of(2) { 1 } else { 0 };
    let numerator = if half_turn_character == 1 {
        cyclic_character_sum
            .checked_add(1)
            .ok_or(TemplateErrorV1::ArithmeticOverflow)?
    } else {
        cyclic_character_sum
            .checked_sub(1)
            .ok_or(TemplateErrorV1::ArithmeticOverflow)?
    };
    if !numerator.is_multiple_of(2) {
        return Err(TemplateErrorV1::ArithmeticOverflow);
    }
    Ok(numerator / 2)
}

/// Independent weight-pair derivation for D_n:
/// each positive multiple of n contributes one invariant ±m combination;
/// m=0 contributes one extra mode exactly when l is even.
#[cfg(test)]
fn dihedral_weight_pair_count(degree: usize, order: usize) -> Result<usize, TemplateErrorV1> {
    if order < 2 {
        return Err(TemplateErrorV1::InvalidDihedralOrder);
    }
    let mut count = 0usize;
    let mut weight = order;
    while weight <= degree {
        count = count
            .checked_add(1)
            .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
        let Some(next) = weight.checked_add(order) else {
            break;
        };
        weight = next;
    }
    if degree % 2 == 0 {
        count = count
            .checked_add(1)
            .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
    }
    Ok(count)
}

pub fn fixed_dimension(
    hypothesis: GroupHypothesisV1,
    degree: usize,
) -> Result<usize, TemplateErrorV1> {
    hypothesis.validate()?;
    match hypothesis {
        GroupHypothesisV1::Cyclic { order } => cyclic_fixed_dimension(degree, order),
        GroupHypothesisV1::RotationalDihedral { order } => {
            dihedral_character_fixed_dimension(degree, order)
        }
    }
}

/// Generate the exact degree-resolved rank template [m_0, ..., m_L].
pub fn rank_template(
    hypothesis: GroupHypothesisV1,
    maximum_degree: usize,
    budget: TemplateBudgetV1,
) -> Result<Vec<usize>, TemplateErrorV1> {
    preflight(hypothesis, maximum_degree, budget)?;
    let capacity = maximum_degree
        .checked_add(1)
        .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
    let mut signature = Vec::with_capacity(capacity);
    for degree in 0..=maximum_degree {
        let multiplicity = fixed_dimension(hypothesis, degree)?;
        let representation_dimension = degree
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
        if multiplicity > representation_dimension {
            return Err(TemplateErrorV1::ArithmeticOverflow);
        }
        signature.push(multiplicity);
    }
    Ok(signature)
}

/// Exact subgroup relation for the declared co-axial proper-rotation family.
///
/// C_d <= C_n and D_d <= D_n when d divides n; C_d <= D_n when d divides n;
/// and C_n <= D_n. Other orientation embeddings are outside this v0.1 model.
pub fn is_subgroup_of(
    subgroup: GroupHypothesisV1,
    supergroup: GroupHypothesisV1,
) -> Result<bool, TemplateErrorV1> {
    subgroup.validate()?;
    supergroup.validate()?;
    let d = subgroup.rotational_parameter();
    let n = supergroup.rotational_parameter();
    let divides = n.is_multiple_of(d);

    Ok(match (subgroup, supergroup) {
        (GroupHypothesisV1::Cyclic { .. }, GroupHypothesisV1::Cyclic { .. }) => divides,
        (GroupHypothesisV1::Cyclic { .. }, GroupHypothesisV1::RotationalDihedral { .. }) => divides,
        (
            GroupHypothesisV1::RotationalDihedral { .. },
            GroupHypothesisV1::RotationalDihedral { .. },
        ) => divides,
        (GroupHypothesisV1::RotationalDihedral { .. }, GroupHypothesisV1::Cyclic { .. }) => false,
    })
}

/// Check the exact fixed-space dimension inequality for every degree 0..=L.
/// If H <= G, then V_l^G is a subspace of V_l^H and dim(V_l^G) <= dim(V_l^H).
pub fn verify_subgroup_monotonicity(
    subgroup: GroupHypothesisV1,
    supergroup: GroupHypothesisV1,
    maximum_degree: usize,
    budget: TemplateBudgetV1,
) -> Result<bool, TemplateErrorV1> {
    preflight(subgroup, maximum_degree, budget)?;
    preflight(supergroup, maximum_degree, budget)?;
    if !is_subgroup_of(subgroup, supergroup)? {
        return Ok(false);
    }
    for degree in 0..=maximum_degree {
        let subgroup_dimension = fixed_dimension(subgroup, degree)?;
        let supergroup_dimension = fixed_dimension(supergroup, degree)?;
        if supergroup_dimension > subgroup_dimension {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Return the first degree at which two exact rank templates differ within L.
/// None means the selected cutoff does not distinguish them.
pub fn first_divergence_degree(
    left: GroupHypothesisV1,
    right: GroupHypothesisV1,
    maximum_degree: usize,
    budget: TemplateBudgetV1,
) -> Result<Option<usize>, TemplateErrorV1> {
    preflight(left, maximum_degree, budget)?;
    preflight(right, maximum_degree, budget)?;
    for degree in 0..=maximum_degree {
        if fixed_dimension(left, degree)? != fixed_dimension(right, degree)? {
            return Ok(Some(degree));
        }
    }
    Ok(None)
}

/// Return the first degree where any two hypotheses in a declared family differ.
/// The family is assumed to use the same canonical co-axial embedding conventions.
pub fn family_first_divergence_degree(
    hypotheses: &[GroupHypothesisV1],
    maximum_degree: usize,
    budget: TemplateBudgetV1,
) -> Result<Option<usize>, TemplateErrorV1> {
    if hypotheses.is_empty() {
        return Err(TemplateErrorV1::EmptyCandidateFamily);
    }
    for hypothesis in hypotheses {
        preflight(*hypothesis, maximum_degree, budget)?;
    }
    let required_work = hypotheses
        .len()
        .checked_mul(
            maximum_degree
                .checked_add(1)
                .ok_or(TemplateErrorV1::ArithmeticOverflow)?,
        )
        .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
    if required_work > budget.maximum_work_units {
        return Err(TemplateErrorV1::WorkBudgetExceeded {
            required: required_work,
            maximum: budget.maximum_work_units,
        });
    }
    if hypotheses.len() < 2 {
        return Ok(None);
    }

    for degree in 0..=maximum_degree {
        let first = fixed_dimension(hypotheses[0], degree)?;
        for hypothesis in hypotheses.iter().skip(1) {
            if fixed_dimension(*hypothesis, degree)? != first {
                return Ok(Some(degree));
            }
        }
    }
    Ok(None)
}

/// Match a truncated exact rank signature against all C_n and proper D_n
/// hypotheses up to the maximum group parameter. All matches are returned.
/// A non-unique result is AMBIGUOUS; it is never collapsed to a guessed label.
pub fn match_signature(
    observed: &[usize],
    maximum_group_parameter: usize,
    budget: TemplateBudgetV1,
) -> Result<TemplateProposalV1, TemplateErrorV1> {
    if observed.is_empty() {
        return Err(TemplateErrorV1::EmptySignature);
    }
    if maximum_group_parameter == 0 {
        return Err(TemplateErrorV1::ZeroGroupOrder);
    }
    if maximum_group_parameter > budget.maximum_group_parameter {
        return Err(TemplateErrorV1::GroupParameterExceeded {
            required: maximum_group_parameter,
            maximum: budget.maximum_group_parameter,
        });
    }
    let maximum_degree = observed
        .len()
        .checked_sub(1)
        .ok_or(TemplateErrorV1::EmptySignature)?;
    if maximum_degree > budget.maximum_degree {
        return Err(TemplateErrorV1::DegreeExceeded {
            required: maximum_degree,
            maximum: budget.maximum_degree,
        });
    }
    let candidate_count = maximum_group_parameter
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
    // One complete family sweep finds the first divergence; a second
    // complete sweep compares every candidate with the observed signature.
    let required_work = candidate_count
        .checked_mul(observed.len())
        .and_then(|value| value.checked_mul(2))
        .ok_or(TemplateErrorV1::ArithmeticOverflow)?;
    if required_work > budget.maximum_work_units {
        return Err(TemplateErrorV1::WorkBudgetExceeded {
            required: required_work,
            maximum: budget.maximum_work_units,
        });
    }

    let mut candidates = Vec::with_capacity(candidate_count);
    for order in 1..=maximum_group_parameter {
        candidates.push(GroupHypothesisV1::Cyclic { order });
        if order >= 2 {
            candidates.push(GroupHypothesisV1::RotationalDihedral { order });
        }
    }

    let first_family_divergence_degree =
        family_first_divergence_degree(&candidates, maximum_degree, budget)?;
    let mut matches = Vec::new();
    for hypothesis in &candidates {
        if rank_template(*hypothesis, maximum_degree, budget)? == observed {
            matches.push(*hypothesis);
        }
    }

    matches.sort();

    let status = match matches.len() {
        0 => MatchStatusV1::NoMatch,
        1 => MatchStatusV1::Unique,
        _ => MatchStatusV1::Ambiguous,
    };
    Ok(TemplateProposalV1 {
        maximum_degree,
        maximum_group_parameter,
        observed_signature: observed.to_vec(),
        matching_hypotheses: matches,
        status,
        first_family_divergence_degree,
        authority: TemplateAuthorityV1::CandidateProposalOnly,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> TemplateBudgetV1 {
        TemplateBudgetV1 {
            maximum_group_parameter: 64,
            maximum_degree: 32,
            maximum_work_units: 100_000,
        }
    }

    #[test]
    fn cyclic_closed_form_matches_explicit_weight_count_through_full_fixture_grid() {
        for order in 1..=64 {
            for degree in 0..=32 {
                assert_eq!(
                    cyclic_fixed_dimension(degree, order).unwrap(),
                    cyclic_weight_count(degree, order).unwrap(),
                    "C_{order}, l={degree}"
                );
            }
        }
    }

    #[test]
    fn dihedral_character_average_matches_independent_weight_pair_count() {
        for order in 2..=64 {
            for degree in 0..=32 {
                assert_eq!(
                    dihedral_character_fixed_dimension(degree, order).unwrap(),
                    dihedral_weight_pair_count(degree, order).unwrap(),
                    "D_{order}, l={degree}"
                );
            }
        }
    }

    #[test]
    fn dihedral_character_average_uses_the_cyclic_character_sum_and_half_turn_parity() {
        for order in 2..=64 {
            for degree in 0..=32 {
                let cyclic = cyclic_fixed_dimension(degree, order).unwrap();
                let expected = if degree % 2 == 0 {
                    (cyclic + 1) / 2
                } else {
                    (cyclic - 1) / 2
                };
                assert_eq!(
                    dihedral_character_fixed_dimension(degree, order).unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn every_template_multiplicity_is_bounded_by_irrep_dimension() {
        for order in 1..=64 {
            for degree in 0..=32 {
                let cyclic = cyclic_fixed_dimension(degree, order).unwrap();
                assert!(cyclic <= 2 * degree + 1);
                if order >= 2 {
                    let dihedral = dihedral_character_fixed_dimension(degree, order).unwrap();
                    assert!(dihedral <= 2 * degree + 1);
                }
            }
        }
    }

    #[test]
    fn cyclic_group_is_subgroup_of_same_order_rotational_dihedral_group() {
        for order in 2_usize..=64 {
            assert!(
                is_subgroup_of(
                    GroupHypothesisV1::Cyclic { order },
                    GroupHypothesisV1::RotationalDihedral { order }
                )
                .unwrap()
            );
            assert!(
                verify_subgroup_monotonicity(
                    GroupHypothesisV1::Cyclic { order },
                    GroupHypothesisV1::RotationalDihedral { order },
                    32,
                    budget()
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn divisor_subgroups_obey_degreewise_monotonicity() {
        for parent_order in 2_usize..=64 {
            for divisor in 1..=parent_order {
                if !parent_order.is_multiple_of(divisor) {
                    continue;
                }
                assert!(
                    verify_subgroup_monotonicity(
                        GroupHypothesisV1::Cyclic { order: divisor },
                        GroupHypothesisV1::Cyclic {
                            order: parent_order
                        },
                        32,
                        budget()
                    )
                    .unwrap()
                );
                assert!(
                    verify_subgroup_monotonicity(
                        GroupHypothesisV1::Cyclic { order: divisor },
                        GroupHypothesisV1::RotationalDihedral {
                            order: parent_order
                        },
                        32,
                        budget()
                    )
                    .unwrap()
                );
                if divisor >= 2 {
                    assert!(
                        verify_subgroup_monotonicity(
                            GroupHypothesisV1::RotationalDihedral { order: divisor },
                            GroupHypothesisV1::RotationalDihedral {
                                order: parent_order
                            },
                            32,
                            budget()
                        )
                        .unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn truncated_signatures_return_all_ambiguous_group_hypotheses() {
        let signature = rank_template(GroupHypothesisV1::Cyclic { order: 3 }, 2, budget()).unwrap();
        let proposal = match_signature(&signature, 6, budget()).unwrap();

        assert_eq!(signature, vec![1, 1, 1]);
        assert_eq!(proposal.status, MatchStatusV1::Ambiguous);
        assert_eq!(proposal.first_family_divergence_degree, Some(1));
        assert_eq!(
            proposal.authority,
            TemplateAuthorityV1::CandidateProposalOnly
        );
        assert_eq!(
            proposal.matching_hypotheses,
            vec![
                GroupHypothesisV1::Cyclic { order: 3 },
                GroupHypothesisV1::Cyclic { order: 4 },
                GroupHypothesisV1::Cyclic { order: 5 },
                GroupHypothesisV1::Cyclic { order: 6 },
            ]
        );
    }

    #[test]
    fn truncated_dihedral_signatures_return_all_ambiguous_dihedral_hypotheses() {
        let signature = rank_template(
            GroupHypothesisV1::RotationalDihedral { order: 3 },
            2,
            budget(),
        )
        .unwrap();
        let proposal = match_signature(&signature, 6, budget()).unwrap();

        assert_eq!(signature, vec![1, 0, 1]);
        assert_eq!(proposal.status, MatchStatusV1::Ambiguous);
        assert_eq!(
            proposal.matching_hypotheses,
            vec![
                GroupHypothesisV1::RotationalDihedral { order: 3 },
                GroupHypothesisV1::RotationalDihedral { order: 4 },
                GroupHypothesisV1::RotationalDihedral { order: 5 },
                GroupHypothesisV1::RotationalDihedral { order: 6 },
            ]
        );
    }

    #[test]
    fn extending_degree_cutoff_resolves_cyclic_and_dihedral_orders() {
        let c3 = GroupHypothesisV1::Cyclic { order: 3 };
        let c4 = GroupHypothesisV1::Cyclic { order: 4 };
        let d3 = GroupHypothesisV1::RotationalDihedral { order: 3 };
        let d4 = GroupHypothesisV1::RotationalDihedral { order: 4 };

        assert_eq!(first_divergence_degree(c3, c4, 2, budget()).unwrap(), None);
        assert_eq!(
            first_divergence_degree(c3, c4, 3, budget()).unwrap(),
            Some(3)
        );
        assert_eq!(first_divergence_degree(d3, d4, 2, budget()).unwrap(), None);
        assert_eq!(
            first_divergence_degree(d3, d4, 3, budget()).unwrap(),
            Some(3)
        );
    }

    #[test]
    fn canonical_proposal_bytes_bind_the_observed_signature_and_search_family() {
        let c3 = rank_template(GroupHypothesisV1::Cyclic { order: 3 }, 3, budget()).unwrap();
        let c4 = rank_template(GroupHypothesisV1::Cyclic { order: 4 }, 3, budget()).unwrap();
        let proposal_c3 = match_signature(&c3, 6, budget()).unwrap();
        let proposal_c4 = match_signature(&c4, 6, budget()).unwrap();

        assert_ne!(
            proposal_c3.observed_signature,
            proposal_c4.observed_signature
        );
        assert_ne!(proposal_c3.canonical_bytes(), proposal_c4.canonical_bytes());
        assert_eq!(proposal_c3.maximum_group_parameter, 6);
        assert_eq!(proposal_c3.maximum_degree, 3);
    }

    #[test]
    fn family_divergence_is_reported_without_forcing_an_early_label() {
        let family = [
            GroupHypothesisV1::Cyclic { order: 3 },
            GroupHypothesisV1::Cyclic { order: 4 },
            GroupHypothesisV1::RotationalDihedral { order: 3 },
            GroupHypothesisV1::RotationalDihedral { order: 4 },
        ];
        assert_eq!(
            family_first_divergence_degree(&family, 0, budget()).unwrap(),
            None
        );
        assert_eq!(
            family_first_divergence_degree(&family, 1, budget()).unwrap(),
            Some(1)
        );
    }

    #[test]
    fn family_divergence_rejects_an_empty_candidate_family() {
        assert_eq!(
            family_first_divergence_degree(&[], 4, budget()),
            Err(TemplateErrorV1::EmptyCandidateFamily)
        );
    }

    #[test]
    fn dihedral_template_has_expected_low_degree_values() {
        let d3 = GroupHypothesisV1::RotationalDihedral { order: 3 };
        assert_eq!(
            rank_template(d3, 6, budget()).unwrap(),
            vec![1, 0, 1, 1, 2, 1, 3]
        );
    }

    #[test]
    fn invalid_orders_empty_signatures_and_budget_exhaustion_fail_closed() {
        assert_eq!(
            cyclic_fixed_dimension(2, 0),
            Err(TemplateErrorV1::ZeroGroupOrder)
        );
        assert_eq!(
            dihedral_character_fixed_dimension(2, 1),
            Err(TemplateErrorV1::InvalidDihedralOrder)
        );
        assert_eq!(
            match_signature(&[], 4, budget()),
            Err(TemplateErrorV1::EmptySignature)
        );
        let tiny_budget = TemplateBudgetV1 {
            maximum_work_units: 1,
            ..budget()
        };
        assert!(matches!(
            match_signature(&[1, 1, 1], 6, tiny_budget),
            Err(TemplateErrorV1::WorkBudgetExceeded { .. })
        ));
    }

    #[test]
    fn ambiguous_template_proposal_contains_no_pruning_authority() {
        let signature = rank_template(
            GroupHypothesisV1::RotationalDihedral { order: 4 },
            1,
            budget(),
        )
        .unwrap();
        let proposal = match_signature(&signature, 8, budget()).unwrap();
        assert_eq!(proposal.status, MatchStatusV1::Ambiguous);
        assert!(!proposal.matching_hypotheses.is_empty());
        assert_eq!(
            proposal.authority,
            TemplateAuthorityV1::CandidateProposalOnly
        );
        let first = proposal.canonical_bytes();
        let second = match_signature(&signature, 8, budget())
            .unwrap()
            .canonical_bytes();
        assert_eq!(first, second);
    }
}
