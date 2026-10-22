// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Paired smoke benchmark for sequential detectors under adaptive/contextual alternatives.
//!
//! This integration test is intentionally small and deterministic. It compares (1) a pooled
//! Order/Dispersion PIT e-process, (2) the same fixed-weight portfolio with one predeclared
//! predictable action-context channel, and (3) a predictive-rank martingale over the PIT stream.
//! The alternative is deliberately constructed so directional and dispersion feature means can
//! cancel marginally even though the conditional PIT distribution depends strongly on action.
//!
//! This is a regression fixture, not a general power comparison or a formal proof. The public
//! claims for each detector remain bounded by its own null assumptions, documented in the
//! corresponding source module.

use symthaea_physics_bridge::prequential_pit_martingale::{
    PrequentialPitDecision, PrequentialPitMartingale,
};
use symthaea_physics_bridge::predictive_rank_martingale::{
    PredictiveRankDecision, PredictiveRankMartingale,
};

const ALPHA: f64 = 0.05;
const STREAMS: usize = 256;
const HORIZON: usize = 300;
const REFERENCE_COUNT: usize = 100;

#[derive(Clone, Debug)]
struct AlarmSummary {
    alarms: usize,
    delays: Vec<usize>,
}

impl AlarmSummary {
    fn new() -> Self {
        Self {
            alarms: 0,
            delays: Vec::new(),
        }
    }

    fn record(&mut self, delay: Option<usize>) {
        if let Some(delay) = delay {
            self.alarms += 1;
            self.delays.push(delay);
        }
    }

    fn rate(&self) -> f64 {
        self.alarms as f64 / STREAMS as f64
    }

    fn median_delay(&mut self) -> Option<usize> {
        if self.delays.is_empty() {
            return None;
        }
        self.delays.sort_unstable();
        Some(self.delays[self.delays.len() / 2])
    }
}

#[derive(Clone, Copy, Debug)]
enum SyntheticWorld {
    ConditionalUniformNull,
    MarginallyBalancedContextShift,
}

/// Deterministic xorshift64* generator for repeatable tests only.
fn next_unit_interval(state: &mut u64) -> f64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let value = (*state).wrapping_mul(0x2545_F491_4F6C_DD1D);
    (value >> 11) as f64 / ((1_u64 << 53) as f64)
}

fn collect_alarm_delay(
    pits: &[f64],
    actions: &[f64],
    use_context: bool,
) -> Option<usize> {
    let context_channels = usize::from(use_context);
    let mut monitor = PrequentialPitMartingale::new(ALPHA, context_channels)
        .expect("benchmark alpha and context count are valid");

    for (index, (pit, action)) in pits.iter().copied().zip(actions.iter().copied()).enumerate() {
        let result = if use_context {
            monitor.observe(pit, &[action])
        } else {
            monitor.observe(pit, &[])
        }
        .expect("synthetic PIT and pre-outcome context are valid");
        if result.decision == PrequentialPitDecision::Alarm {
            return Some(index + 1);
        }
    }
    None
}

fn collect_prm_alarm_delay(
    calibration_scores: &[f64],
    calibration_ties: &[f64],
    pits: &[f64],
    pit_ties: &[f64],
) -> Option<usize> {
    let mut monitor =
        PredictiveRankMartingale::new(calibration_scores, calibration_ties, ALPHA)
            .expect("continuous synthetic calibration keys should be unique");

    for (index, (pit, tie_breaker)) in pits
        .iter()
        .copied()
        .zip(pit_ties.iter().copied())
        .enumerate()
    {
        let result = monitor
            .observe(pit, tie_breaker)
            .expect("synthetic PIT/reference keys should be valid");
        if result.decision == PredictiveRankDecision::Alarm {
            return Some(index + 1);
        }
    }
    None
}

fn simulate_world(
    world: SyntheticWorld,
    streams: usize,
    horizon: usize,
    reference_count: usize,
    rng: &mut u64,
) -> (AlarmSummary, AlarmSummary, AlarmSummary) {
    let mut pooled_pit = AlarmSummary::new();
    let mut contextual_pit = AlarmSummary::new();
    let mut rank_martingale = AlarmSummary::new();

    for _ in 0..streams {
        let calibration_scores = (0..reference_count)
            .map(|_| next_unit_interval(rng))
            .collect::<Vec<_>>();
        let calibration_ties = (0..reference_count)
            .map(|_| next_unit_interval(rng))
            .collect::<Vec<_>>();

        let mut pits = Vec::with_capacity(horizon);
        let mut pit_ties = Vec::with_capacity(horizon);
        let mut actions = Vec::with_capacity(horizon);

        for index in 0..horizon {
            // The selected action is known before the current outcome. Under the null, the
            // prequential PIT is conditionally uniform. Under the alternative, the PIT lives in
            // opposite interior bands by action: each band's pooled order effect has opposite
            // sign, while its mean dispersion feature is zero.
            let action = if index % 2 == 0 { 1.0 } else { -1.0 };
            let pit = match world {
                SyntheticWorld::ConditionalUniformNull => next_unit_interval(rng),
                SyntheticWorld::MarginallyBalancedContextShift if action > 0.0 => {
                    0.65 + 0.20 * next_unit_interval(rng)
                }
                SyntheticWorld::MarginallyBalancedContextShift => {
                    0.15 + 0.20 * next_unit_interval(rng)
                }
            };
            pits.push(pit);
            pit_ties.push(next_unit_interval(rng));
            actions.push(action);
        }

        pooled_pit.record(collect_alarm_delay(&pits, &actions, false));
        contextual_pit.record(collect_alarm_delay(&pits, &actions, true));
        rank_martingale.record(collect_prm_alarm_delay(
            &calibration_scores,
            &calibration_ties,
            &pits,
            &pit_ties,
        ));
    }

    (pooled_pit, contextual_pit, rank_martingale)
}

fn report(world: &str, mut summary: (AlarmSummary, AlarmSummary, AlarmSummary)) {
    let (mut pooled, mut contextual, mut prm) = (
        summary.0,
        summary.1,
        summary.2,
    );
    println!(
        concat!(
            "adaptive_monitoring_benchmark,world={world},streams={STREAMS},",
            "horizon={HORIZON},reference_count={REFERENCE_COUNT},alpha={ALPHA},",
            "pooled_pit_fwer={:.4},contextual_pit_fwer={:.4},prm_fwer={:.4},",
            "pooled_pit_median_delay={:?},contextual_pit_median_delay={:?},",
            "prm_median_delay={:?}"
        ),
        pooled.rate(),
        contextual.rate(),
        prm.rate(),
        pooled.median_delay(),
        contextual.median_delay(),
        prm.median_delay(),
    );
}

#[test]
fn paired_adaptive_monitoring_benchmark_separates_null_control_from_context_power() {
    let mut rng = 0x8455_C0DE_u64;
    let null = simulate_world(
        SyntheticWorld::ConditionalUniformNull,
        STREAMS,
        HORIZON,
        REFERENCE_COUNT,
        &mut rng,
    );
    let alternative = simulate_world(
        SyntheticWorld::MarginallyBalancedContextShift,
        STREAMS,
        HORIZON,
        REFERENCE_COUNT,
        &mut rng,
    );

    report("conditional_uniform_null", null.clone());
    report("marginally_balanced_context_shift", alternative.clone());

    // This wide deterministic regression envelope catches gross validity/wealth bugs; it is
    // not a substitute for an independent statistical analysis or confidence intervals.
    assert!(null.0.rate() <= 0.10, "pooled PIT null alarm rate={}", null.0.rate());
    assert!(
        null.1.rate() <= 0.10,
        "contextual PIT null alarm rate={}",
        null.1.rate()
    );
    assert!(
        null.2.rate() <= 0.10,
        "PRM null alarm rate={}",
        null.2.rate()
    );

    // The alternative was chosen specifically so both pooled PIT feature means cancel. The
    // action-context channel has a positive conditional order payoff and should find it quickly.
    assert!(
        alternative.1.rate() >= 0.90,
        "contextual PIT detection rate={}",
        alternative.1.rate()
    );
    assert!(
        alternative.0.rate() <= 0.10,
        "pooled PIT should miss the balanced alternative: rate={}",
        alternative.0.rate()
    );
    assert!(
        alternative.2.rate() <= 0.35,
        "PRM's fixed Order/Dispersion portfolio unexpectedly covered this contrast: rate={}",
        alternative.2.rate()
    );
}
