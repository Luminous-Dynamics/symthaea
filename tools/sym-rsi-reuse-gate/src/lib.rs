#![forbid(unsafe_code)]

// This standalone harness compiles the exact production source files. It does
// not copy or reimplement the allocation or sign-test algorithms.
#[path = "../../../src/consciousness/recursive_improvement/paired_sign_test.rs"]
pub mod paired_sign_test;
#[path = "../../../src/consciousness/recursive_improvement/reuse_confidence_certificate.rs"]
pub mod reuse_confidence_certificate;
#[path = "../../../src/consciousness/recursive_improvement/reuse_error_budget.rs"]
pub mod reuse_error_budget;


#[cfg(test)]
mod adaptive_null_simulation {
    use super::reuse_error_budget::{
        BoundedScoreCandidateEvidence, ReuseCandidateEvidence, ReuseRoundDecision,
        evaluate_bounded_score_reuse_round, evaluate_reuse_round,
    };

    const REPLICATES: usize = 512;
    const ROUNDS: usize = 24;
    const POOL_WIDTH: usize = 4;
    const CANDIDATES_PER_ROUND: usize = 2;
    const INDEPENDENT_UNITS: usize = 128;
    const ALPHA: f64 = 0.05;
    const WILSON_Z_95: f64 = 1.959_963_984_540_054;

    /// Tiny deterministic PRNG for a reproducible stress fixture. This is not
    /// cryptographic and is not used by production candidate/evaluator code.
    fn next_xorshift64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        if *state == 0 {
            *state = 0x9e37_79b9_7f4a_7c15;
        }
        *state
    }

    /// Pre-generate every candidate before the first evaluation decision.
    /// Every candidate has population mean zero: independent Rademacher
    /// differences over the independent evaluation units.
    fn null_candidate_pool(seed: u64) -> Vec<Vec<Vec<i8>>> {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut rounds = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let mut candidates = Vec::with_capacity(POOL_WIDTH);
            for _ in 0..POOL_WIDTH {
                let mut outcomes = Vec::with_capacity(INDEPENDENT_UNITS);
                for _ in 0..INDEPENDENT_UNITS {
                    outcomes.push(if next_xorshift64(&mut state) & 1 == 0 {
                        -1
                    } else {
                        1
                    });
                }
                candidates.push(outcomes);
            }
            rounds.push(candidates);
        }
        rounds
    }

    /// Proposer's next candidate index depends only on the previous coarse
    /// decision and the promotion count. It never receives scores or outcomes.
    fn adaptive_candidate_index(
        round_one_based: usize,
        prior_promotions: usize,
        last_decision: usize,
    ) -> usize {
        (round_one_based * 7 + prior_promotions * 11 + last_decision * 13) % POOL_WIDTH
    }

    /// One-sided 95% Wilson upper limit for a binomial event frequency.
    /// Finite Monte Carlo uncertainty is reported; zero observed events is
    /// never interpreted as proof of zero risk.
    fn wilson_upper_95(events: usize, trials: usize) -> f64 {
        assert!(trials > 0);
        let n = trials as f64;
        let p = events as f64 / n;
        let z2 = WILSON_Z_95 * WILSON_Z_95;
        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let half_width =
            WILSON_Z_95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
        center + half_width
    }

    fn run_pass_fail_campaign(seed: u64) -> bool {
        let pool = null_candidate_pool(seed);
        let mut prior_promotions = 0usize;
        let mut last_decision = 0usize;
        let mut any_false_promotion = false;

        for round_index in 0..ROUNDS {
            let round = round_index + 1;
            let candidate_index = adaptive_candidate_index(round, prior_promotions, last_decision);
            let second_index = (candidate_index + 1) % POOL_WIDTH;
            let candidates = [
                ReuseCandidateEvidence::new(pool[round_index][candidate_index].clone()).unwrap(),
                ReuseCandidateEvidence::new(pool[round_index][second_index].clone()).unwrap(),
            ];

            // The proposer chooses a pair from a pre-generated pool using
            // only previous coarse decisions. Both candidates are evaluated
            // against the same incumbent and independent task population.
            let decision = evaluate_reuse_round(
                ALPHA,
                round,
                prior_promotions,
                CANDIDATES_PER_ROUND,
                0.5,
                &candidates,
            )
            .unwrap();

            match decision {
                ReuseRoundDecision::RejectAll => last_decision = 0,
                ReuseRoundDecision::PromoteCandidate { candidate_number } => {
                    assert!((1..=CANDIDATES_PER_ROUND).contains(&candidate_number));
                    // Every candidate is population-neutral, so every
                    // promotion in this simulation is false by construction.
                    any_false_promotion = true;
                    prior_promotions += 1;
                    last_decision = candidate_number;
                }
            }
        }
        any_false_promotion
    }

    fn run_bounded_score_campaign(seed: u64) -> bool {
        let pool = null_candidate_pool(seed);
        let mut prior_promotions = 0usize;
        let mut last_decision = 0usize;
        let mut any_false_promotion = false;

        for round_index in 0..ROUNDS {
            let round = round_index + 1;
            let candidate_index =
                adaptive_candidate_index(round, prior_promotions, last_decision);
            let second_index = (candidate_index + 1) % POOL_WIDTH;
            let bounded = [candidate_index, second_index].map(|index| {
                pool[round_index][index]
                    .iter()
                    .map(|&value| value as f64)
                    .collect::<Vec<_>>()
            });
            let candidates = [
                BoundedScoreCandidateEvidence::new(bounded[0].clone()).unwrap(),
                BoundedScoreCandidateEvidence::new(bounded[1].clone()).unwrap(),
            ];

            let decision = evaluate_bounded_score_reuse_round(
                ALPHA,
                round,
                prior_promotions,
                CANDIDATES_PER_ROUND,
                0.0,
                &candidates,
            )
            .unwrap();

            match decision {
                ReuseRoundDecision::RejectAll => last_decision = 0,
                ReuseRoundDecision::PromoteCandidate { candidate_number } => {
                    assert!((1..=CANDIDATES_PER_ROUND).contains(&candidate_number));
                    any_false_promotion = true;
                    prior_promotions += 1;
                    last_decision = candidate_number;
                }
            }
        }
        any_false_promotion
    }

    #[test]
    fn pass_fail_reuse_controls_adaptive_null_campaigns() {
        let mut false_campaigns = 0usize;
        for replicate in 0..REPLICATES {
            let seed = 0x9e37_79b9_7f4a_7c15_u64
                .wrapping_add((replicate as u64).wrapping_mul(0x5851_f42d_4c95_7f2d));
            if run_pass_fail_campaign(seed) {
                false_campaigns += 1;
            }
        }

        let upper = wilson_upper_95(false_campaigns, REPLICATES);
        assert!(
            upper <= ALPHA,
            "adaptive pass/fail null: {false_campaigns}/{REPLICATES} campaigns promoted; \
             Wilson 95% upper bound {upper:.4} exceeds alpha {ALPHA}"
        );
    }

    #[test]
    fn bounded_score_reuse_controls_adaptive_null_campaigns() {
        let mut false_campaigns = 0usize;
        for replicate in 0..REPLICATES {
            let seed = 0x243f_6a88_85a3_08d3_u64
                .wrapping_add((replicate as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            if run_bounded_score_campaign(seed) {
                false_campaigns += 1;
            }
        }

        let upper = wilson_upper_95(false_campaigns, REPLICATES);
        assert!(
            upper <= ALPHA,
            "bounded-score null: {false_campaigns}/{REPLICATES} campaigns promoted; \
             Wilson 95% upper bound {upper:.4} exceeds alpha {ALPHA}"
        );
    }
}
