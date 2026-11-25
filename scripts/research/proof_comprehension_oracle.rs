//! Dependency-free finite oracle for the proof-comprehension pilots.
//!
//! Run with:
//!   rustc --edition=2021 --test scripts/research/proof_comprehension_oracle.rs -o /tmp/proof-comprehension-oracle
//!   /tmp/proof-comprehension-oracle
//!
//! This file is a finite reference oracle only. It is not a Lean proof,
//! a general reduction theorem, or a SAT solver benchmark.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TseitinCounts {
    source_models: usize,
    correct_models: usize,
    mutated_models: usize,
    correct_decision_equivalent: bool,
    mutated_decision_equivalent: bool,
    correct_unique_extension: bool,
}

fn evaluate_tseitin_toy() -> TseitinCounts {
    let mut source_models = 0usize;
    let mut correct_models = 0usize;
    let mut mutated_models = 0usize;
    let mut correct_decision_equivalent = true;
    let mut mutated_decision_equivalent = true;
    let mut correct_unique_extension = true;

    // F(p,q,r) = (p AND q) OR r.
    // u <-> (p AND q), s <-> (u OR r), with root s asserted.
    for p_bit in 0..=1 {
        for q_bit in 0..=1 {
            for r_bit in 0..=1 {
                let p = p_bit == 1;
                let q = q_bit == 1;
                let r = r_bit == 1;
                let source = (p && q) || r;
                if source {
                    source_models += 1;
                }

                let mut correct_extensions = 0usize;
                let mut mutated_extensions = 0usize;

                for u_bit in 0..=1 {
                    for s_bit in 0..=1 {
                        let u = u_bit == 1;
                        let s = s_bit == 1;

                        // Complete biconditional gates:
                        // u <-> (p AND q)
                        // s <-> (u OR r)
                        let correct = (u == (p && q)) && (s == (u || r)) && s; // assert root

                        if correct {
                            correct_extensions += 1;
                        }

                        // Mutation: remove (u OR NOT p OR NOT q), the
                        // reverse implication (p AND q) -> u.
                        // Remaining clauses imply u -> (p AND q), but not
                        // the converse. The outer OR gate remains biconditional.
                        let mutated_and_gate = (!u || p) && (!u || q);
                        let mutated = mutated_and_gate && (s == (u || r)) && s;
                        if mutated {
                            mutated_extensions += 1;
                        }
                    }
                }

                correct_models += correct_extensions;
                mutated_models += mutated_extensions;
                correct_decision_equivalent &= (correct_extensions > 0) == source;
                mutated_decision_equivalent &= (mutated_extensions > 0) == source;
                correct_unique_extension &= correct_extensions == if source { 1 } else { 0 };
            }
        }
    }

    TseitinCounts {
        source_models,
        correct_models,
        mutated_models,
        correct_decision_equivalent,
        mutated_decision_equivalent,
        correct_unique_extension,
    }
}

fn choose(n: u32, k: u32) -> u64 {
    if k == 0 {
        1
    } else if n == 0 {
        0
    } else {
        choose(n - 1, k - 1) + choose(n - 1, k)
    }
}

fn and_iff_left_without_hypothesis(a: bool, b: bool) -> bool {
    (a && b) == a
}

fn and_iff_right_without_hypothesis(a: bool, b: bool) -> bool {
    (a && b) == b
}

fn or_iff_right_without_hypothesis(a: bool, b: bool) -> bool {
    (a || b) == b
}

#[cfg(test)]
fn translation_count_cyclic(n: usize, t2: &[usize], target_column: usize) -> usize {
    (0..n)
        .filter(|translate| t2.iter().any(|z| (*translate + *z) % n == target_column))
        .count()
}

fn translation_count_product_group(
    m: usize,
    n: usize,
    t2_mask: usize,
    target_index: usize,
) -> usize {
    let target = (target_index / n, target_index % n);
    (0..(m * n))
        .filter(|u_index| {
            let u = (*u_index / n, *u_index % n);
            (0..(m * n)).any(|z_index| {
                if (t2_mask & (1usize << z_index)) == 0 {
                    return false;
                }
                let z = (z_index / n, z_index % n);
                ((u.0 + z.0) % m, (u.1 + z.1) % n) == target
            })
        })
        .count()
}

#[cfg(not(test))]
fn main_report() {
    let counts = evaluate_tseitin_toy();
    println!("source_models={}", counts.source_models);
    println!("correct_tseitin_models={}", counts.correct_models);
    println!("mutated_tseitin_models={}", counts.mutated_models);
    println!(
        "correct_decision_equivalent={}",
        counts.correct_decision_equivalent
    );
    println!(
        "mutated_decision_equivalent={}",
        counts.mutated_decision_equivalent
    );
    println!(
        "correct_unique_extension={}",
        counts.correct_unique_extension
    );
}

#[cfg(not(test))]
fn main() {
    main_report();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_tseitin_encoding_is_parsimonious_on_the_toy_formula() {
        let result = evaluate_tseitin_toy();
        assert_eq!(result.source_models, 5);
        assert_eq!(result.correct_models, 5);
        assert!(result.correct_decision_equivalent);
        assert!(result.correct_unique_extension);
    }

    #[test]
    fn removing_reverse_and_gate_implication_breaks_count_preservation() {
        let result = evaluate_tseitin_toy();
        assert!(result.mutated_decision_equivalent);
        assert_eq!(result.mutated_models, 6);
        assert_ne!(result.correct_models, result.mutated_models);
    }

    #[test]
    fn extra_extension_has_a_concrete_witness() {
        let p = true;
        let q = true;
        let r = true;
        let source = (p && q) || r;
        assert!(source);

        // With the reverse AND-gate implication removed, both u=false
        // and u=true are compatible with the asserted outer root s=true.
        let extensions: Vec<(bool, bool)> = [false, true]
            .into_iter()
            .filter_map(|u| {
                let s = true;
                let remaining_and_clauses = (!u || p) && (!u || q);
                let outer_gate = s == (u || r);
                (remaining_and_clauses && outer_gate && s).then_some((u, s))
            })
            .collect();
        assert_eq!(extensions, vec![(false, true), (true, true)]);
    }

    #[test]
    fn choose_positive_bound_is_necessary_at_the_smallest_counterexample() {
        assert_eq!(choose(0, 1), 0);
        assert!(!(0 < choose(0, 1)));
    }

    #[test]
    fn removing_and_left_implication_has_a_boolean_counterexample() {
        assert!(!and_iff_left_without_hypothesis(true, false));
    }

    #[test]
    fn removing_and_right_implication_has_a_boolean_counterexample() {
        assert!(!and_iff_right_without_hypothesis(false, true));
    }

    #[test]
    fn removing_or_right_implication_has_a_boolean_counterexample() {
        assert!(!or_iff_right_without_hypothesis(true, false));
    }

    #[test]
    fn multiset_identity_map_counterfactual_is_false_for_bool_not() {
        let source = vec![false];
        let mapped: Vec<bool> = source.iter().map(|x| !*x).collect();
        assert_ne!(mapped, source);
        assert_eq!(mapped, vec![true]);
    }

    #[test]
    fn translation_average_selects_each_column_exactly_cardinality_t2_times() {
        // Exhaust all nonempty subsets of cyclic groups C_n for 2 <= n <= 6.
        // For each target column s and each z in T2, exactly one translate
        // u = s - z selects s. Distinct z yield distinct translates.
        for n in 2usize..=6 {
            for mask in 1usize..(1usize << n) {
                let t2: Vec<usize> = (0..n).filter(|z| (mask & (1usize << z)) != 0).collect();
                for target in 0..n {
                    let selected_translates = translation_count_cyclic(n, &t2, target);
                    assert_eq!(selected_translates, t2.len());

                    // Add the row indicator v=1 iff chi_q=1.
                    // 2 * count = |T2| * (1 + chi_q) for chi_q in {-1,+1}.
                    for chi_q in [-1i32, 1i32] {
                        let row_selected = chi_q == 1;
                        let lhs = 2 * if row_selected { selected_translates } else { 0 };
                        let rhs = t2.len() as i32 * (1 + chi_q);
                        assert_eq!(lhs as i32, rhs);
                    }
                }
            }
        }
    }

    #[test]
    fn translation_average_holds_for_small_product_cyclic_groups() {
        // Exhaust every nonempty subset of C_m × C_n for several small
        // products, including non-cyclic groups C_2 × C_2 and C_3 × C_3.
        let mut group_target_cases = 0usize;
        let mut row_indicator_cases = 0usize;

        for (m, n) in [(2usize, 2usize), (2, 3), (2, 4), (3, 3), (2, 5), (3, 4)] {
            let group_size = m * n;
            for t2_mask in 1usize..(1usize << group_size) {
                let t2_count = t2_mask.count_ones() as usize;
                for target in 0..group_size {
                    let selected_translates =
                        translation_count_product_group(m, n, t2_mask, target);
                    assert_eq!(selected_translates, t2_count);
                    group_target_cases += 1;

                    for chi_q in [-1i32, 1i32] {
                        let row_selected = chi_q == 1;
                        let lhs = 2 * if row_selected { selected_translates } else { 0 };
                        let rhs = t2_count as i32 * (1 + chi_q);
                        assert_eq!(lhs as i32, rhs);
                        row_indicator_cases += 1;
                    }
                }
            }
        }

        assert_eq!(group_target_cases, 66_447);
        assert_eq!(row_indicator_cases, 132_894);
    }

    #[test]
    fn finite_set_image_can_shrink_without_injectivity() {
        let source = vec![false, true];
        let mut image = Vec::new();
        for _ in &source {
            if !image.contains(&false) {
                image.push(false);
            }
        }
        assert_eq!(source.len(), 2);
        assert_eq!(image.len(), 1);
    }
}
