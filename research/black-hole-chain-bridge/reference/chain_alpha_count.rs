//! Independent combinatorial oracle for the finite Chain Theory alpha candidate.
//!
//! This file reproduces only the finite slot/record count and readout arithmetic.
//! It is not a port of the author's full Lean proof, does not construct the
//! qutrit/qubit operator matrices, does not verify the tensor-factor semantics,
//! and does not establish the physical identification with alpha(0).
//!
//! Model slice:
//! - Two reads, each with a read-end and a far-end slot: [qutrit, qubit,
//!   qutrit, qubit] with basis dimensions [8, 3, 8, 3].
//! - Exchange of reads maps slots 0<->2 and 1<->3.
//! - Two-slot products are decomposed into exchange-even/odd combinations.
//! - Same-end products contribute X sectors; mixed-end products contribute
//!   the two Y sectors; qubit far-end commutators add C(3,2) records because
//!   the d=2 Weyl phase is a sign.
//!
//! Compare against the author's ChainAlpha.lean (Lean 4.35.0):
//! https://alexwg.org/publications/ChainAlpha.lean
//! This compact implementation independently reconstructs the combinatorial
//! sector arithmetic; it does not certify the Lean theorem's full derivation.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Assignment {
    slots: [Option<usize>; 4],
}

impl Assignment {
    fn exchanged(self) -> Self {
        Self {
            slots: [self.slots[2], self.slots[3], self.slots[0], self.slots[1]],
        }
    }

    fn is_same_end(self) -> bool {
        let occupied: Vec<usize> = self.slots.iter().enumerate()
            .filter_map(|(slot, value)| value.map(|_| slot))
            .collect();
        occupied.len() == 2 && occupied[0] % 2 == occupied[1] % 2
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SectorCounts {
    /// (+1,+1): odd same-end products plus one member of each mixed-end orbit.
    pub plus_plus: usize,
    /// (+1,-1): the other eigenvector from each mixed-end exchange orbit.
    pub plus_minus: usize,
    /// (-1,+1): even/fixed same-end products.
    pub minus_plus: usize,
    /// (-1,-1): qubit commutator records.
    pub minus_minus: usize,
}

impl SectorCounts {
    fn add(&mut self, x: i8, y: i8, count: usize) {
        match (x, y) {
            (1, 1) => self.plus_plus += count,
            (1, -1) => self.plus_minus += count,
            (-1, 1) => self.minus_plus += count,
            (-1, -1) => self.minus_minus += count,
            _ => unreachable!("exchange eigenvalues are signs"),
        }
    }

    pub fn total(self) -> usize {
        self.plus_plus + self.plus_minus + self.minus_plus + self.minus_minus
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrbitSummary {
    pub product_assignments: usize,
    pub fixed_same_end_assignments: usize,
    pub same_end_nonfixed_orbits: usize,
    pub mixed_end_orbits: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CountError {
    InvalidDimensions,
    UnexpectedFixedMixedAssignment,
    ArithmeticOverflow,
    MissingSector,
}

fn basis_dimension(dimension: usize) -> Result<usize, CountError> {
    dimension.checked_mul(dimension)
        .and_then(|x| x.checked_sub(1))
        .ok_or(CountError::InvalidDimensions)
}

fn product_assignments(slot_dimensions: [usize; 4]) -> Vec<Assignment> {
    let mut products = Vec::new();
    for first_slot in 0..4 {
        for second_slot in (first_slot + 1)..4 {
            for first_letter in 0..slot_dimensions[first_slot] {
                for second_letter in 0..slot_dimensions[second_slot] {
                    let mut slots = [None; 4];
                    slots[first_slot] = Some(first_letter);
                    slots[second_slot] = Some(second_letter);
                    products.push(Assignment { slots });
                }
            }
        }
    }
    products
}

/// The Weyl phase is a sign iff its exponent is d for every basis index j.
/// This is the same finite modular condition used in the Lean source.
fn weyl_phase_is_sign(dimension: usize) -> bool {
    if dimension < 2 {
        return false;
    }
    (0..dimension).all(|j| {
        let exponent = (2 * j + 2 * dimension
            - 2 * ((j + 1) % dimension)) % (2 * dimension);
        exponent == dimension
    })
}

fn choose_two(n: usize) -> Result<usize, CountError> {
    n.checked_mul(n.saturating_sub(1))
        .map(|x| x / 2)
        .ok_or(CountError::ArithmeticOverflow)
}

/// Reconstruct the product exchange orbits and four sector counts for the
/// qutrit–qubit object using basis dimensions [8,3,8,3].
pub fn qutrit_qubit_record_counts()
    -> Result<(SectorCounts, OrbitSummary), CountError>
{
    let qutrit_basis = basis_dimension(3)?;
    let qubit_basis = basis_dimension(2)?;
    let slot_dimensions = [qutrit_basis, qubit_basis, qutrit_basis, qubit_basis];
    let products = product_assignments(slot_dimensions);

    let mut sectors = SectorCounts::default();
    let mut orbits = OrbitSummary {
        product_assignments: products.len(),
        ..OrbitSummary::default()
    };

    for assignment in products {
        let image = assignment.exchanged();
        if assignment == image {
            if !assignment.is_same_end() {
                return Err(CountError::UnexpectedFixedMixedAssignment);
            }
            // An exchange-fixed same-end product is an X-odd record.
            sectors.add(-1, 1, 1);
            orbits.fixed_same_end_assignments += 1;
        } else if assignment < image {
            if assignment.is_same_end() {
                // The pair p <-> Xp gives one antisymmetric (X-even sign)
                // and one symmetric (X-odd sign) eigen-record. Y does not
                // move a same-end occupied set.
                sectors.add(1, 1, 1);
                sectors.add(-1, 1, 1);
                orbits.same_end_nonfixed_orbits += 1;
            } else {
                // For mixed-end products, the read-link exchange moves the
                // far qubit. Its even/odd combinations split between Y signs.
                sectors.add(1, 1, 1);
                sectors.add(1, -1, 1);
                orbits.mixed_end_orbits += 1;
            }
        }
    }

    // The far end is a qubit: its Weyl phase is a sign, so every unordered
    // pair of the three traceless qubit letters contributes one commutator
    // record in the (-1,-1) sector.
    if weyl_phase_is_sign(2) {
        sectors.add(-1, -1, choose_two(qubit_basis)?);
    }

    Ok((sectors, orbits))
}

/// Readout dimension: two read links each contribute (dA*dB)^2, and the
/// unread link contributes (dA^2-1)(dB^2-1).
pub fn qutrit_qubit_readout_dimension() -> Result<usize, CountError> {
    let d_read = 3_usize;
    let d_far = 2_usize;
    let read_link = d_read.checked_mul(d_far)
        .and_then(|x| x.checked_mul(x))
        .ok_or(CountError::ArithmeticOverflow)?;
    let unread_link = basis_dimension(d_read)?
        .checked_mul(basis_dimension(d_far)?)
        .ok_or(CountError::ArithmeticOverflow)?;
    read_link.checked_mul(read_link)
        .and_then(|x| x.checked_mul(unread_link))
        .ok_or(CountError::ArithmeticOverflow)
}

/// Return the exact radical numerator as (radicand, integer coefficient).
/// A term (r,c) means c*sqrt(r); no floating-point fitting is involved.
pub fn qutrit_qubit_radical_terms() -> Result<BTreeMap<usize, usize>, CountError> {
    let (counts, _) = qutrit_qubit_record_counts()?;
    let d_read = 3_usize;
    let d_far = 2_usize;
    let mut terms = BTreeMap::new();

    for (x_sign, y_sign, count) in [
        (1_i8, 1_i8, counts.plus_plus),
        (1, -1, counts.plus_minus),
        (-1, 1, counts.minus_plus),
        (-1, -1, counts.minus_minus),
    ] {
        if count == 0 {
            continue;
        }
        let x_factor = if x_sign == -1 { d_read } else { 1 };
        let y_factor = if y_sign == -1 { d_far } else { 1 };
        let radicand = x_factor.checked_mul(y_factor)
            .ok_or(CountError::ArithmeticOverflow)?;
        *terms.entry(radicand).or_insert(0) += count;
    }
    Ok(terms)
}

pub fn qutrit_qubit_alpha_candidate() -> Result<f64, CountError> {
    let denominator = qutrit_qubit_readout_dimension()? as f64;
    let terms = qutrit_qubit_radical_terms()?;
    let numerator: f64 = terms.iter()
        .map(|(radicand, coefficient)| (*coefficient as f64) * (*radicand as f64).sqrt())
        .sum();
    Ok(numerator / denominator)
}

pub fn qutrit_qubit_inverse_alpha_candidate() -> Result<f64, CountError> {
    Ok(1.0 / qutrit_qubit_alpha_candidate()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_slots_generate_exactly_169_two_letter_products() {
        let assignments = product_assignments([8, 3, 8, 3]);
        assert_eq!(assignments.len(), 169);
        let same_end = assignments.iter().filter(|a| a.is_same_end()).count();
        assert_eq!(same_end, 73);
        assert_eq!(assignments.len() - same_end, 96);
    }

    #[test]
    fn exchange_orbits_reproduce_the_31_48_and_11_decomposition() {
        let (_, orbits) = qutrit_qubit_record_counts().unwrap();
        assert_eq!(orbits.product_assignments, 169);
        assert_eq!(orbits.fixed_same_end_assignments, 11);
        assert_eq!(orbits.same_end_nonfixed_orbits, 31);
        assert_eq!(orbits.mixed_end_orbits, 48);
        assert_eq!(
            orbits.fixed_same_end_assignments + 2 * orbits.same_end_nonfixed_orbits
                + 2 * orbits.mixed_end_orbits,
            169
        );
    }

    #[test]
    fn sector_counts_and_total_match_the_finite_lean_arithmetic() {
        let (counts, _) = qutrit_qubit_record_counts().unwrap();
        assert_eq!(
            counts,
            SectorCounts {
                plus_plus: 79,
                plus_minus: 48,
                minus_plus: 42,
                minus_minus: 3,
            }
        );
        assert_eq!(counts.total(), 172);
    }

    #[test]
    fn qubit_weyl_phase_and_qutrit_non_sign_control_are_distinct() {
        assert!(weyl_phase_is_sign(2));
        assert!(!weyl_phase_is_sign(3));
    }

    #[test]
    fn readout_dimension_is_31104() {
        assert_eq!(qutrit_qubit_readout_dimension().unwrap(), 31_104);
    }

    #[test]
    fn exact_radical_terms_match_the_published_weighted_count() {
        let expected = BTreeMap::from([(1, 79), (2, 48), (3, 42), (6, 3)]);
        assert_eq!(qutrit_qubit_radical_terms().unwrap(), expected);
    }

    #[test]
    fn candidate_inverse_matches_the_published_decimal_arithmetic() {
        let inverse = qutrit_qubit_inverse_alpha_candidate().unwrap();
        assert!((inverse - 137.035_999_189_566_16).abs() < 2e-13);
    }
}
