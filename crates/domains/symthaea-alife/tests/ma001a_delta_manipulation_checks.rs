// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001A-delta manipulation checks, per `ALIFE_MA001A_DELTA_RERUN_PLAN_2026-07-26.md` §8 exit
//! criteria: all 9 of MA-001A's original checks re-verified under the delta-rule mechanism, not
//! assumed to transfer automatically from the original default-learning validation. Mirrors
//! `ma001_manipulation_checks.rs` exactly, checking the same invariants, constructed via
//! `Ma001Run::new_with_delta_rule` and stepped via `step_with_delta_rule`/`run_with_delta_rule`
//! wherever a check actually exercises stepping.

use std::collections::HashMap;

use symthaea_alife::InteractionRecord;
use symthaea_alife::ma001::{Condition, Ma001Config, Ma001Run};
use symthaea_alife::ma001l::DeltaRuleConfig;

fn small_cfg() -> Ma001Config {
    Ma001Config {
        groups: 1, // 4 organisms -- one full round-robin group, fast to run
        total_ticks: 60,
        burn_in_ticks: 20,
        shuffle_epoch_ticks: 10,
        transfer_quantum: 0.05,
        forage_efficiency: 0.6,
        dirichlet_alpha: 0.5,
        min_encounters_per_partner: 5,
    }
}

fn resource_fn() -> impl FnMut(usize) -> f64 {
    move |n| 4.0 / (n.max(1) as f64)
}

fn total_encounter_count(run: &Ma001Run) -> u32 {
    run.organisms
        .iter()
        .map(|o| o.ledger.values().map(|r| r.encounter_count).sum::<u32>())
        .sum()
}

// ---------------------------------------------------------------------------------------------
// Check 1: encounter counts are exactly matched across conditions (schedule-level).
// ---------------------------------------------------------------------------------------------

#[test]
fn check1_encounter_schedule_matches_across_conditions_absent_death() {
    let cfg = small_cfg();
    let (mut bound, bound_dr) =
        Ma001Run::new_with_delta_rule(Condition::Bound, 42, cfg, None, DeltaRuleConfig::default());
    let (mut shuffled, shuffled_dr) = Ma001Run::new_with_delta_rule(
        Condition::Shuffled,
        42,
        cfg,
        None,
        DeltaRuleConfig::default(),
    );
    let (mut no_history, no_history_dr) = Ma001Run::new_with_delta_rule(
        Condition::NoHistory,
        42,
        cfg,
        None,
        DeltaRuleConfig::default(),
    );
    bound.run_with_delta_rule(resource_fn(), &bound_dr);
    shuffled.run_with_delta_rule(resource_fn(), &shuffled_dr);
    no_history.run_with_delta_rule(resource_fn(), &no_history_dr);

    for run in [&bound, &shuffled, &no_history] {
        assert!(
            run.organisms.iter().all(|o| !o.is_dead()),
            "test assumes no deaths -- adjust config if this ever fails"
        );
    }

    let a = total_encounter_count(&bound);
    let b = total_encounter_count(&shuffled);
    let c = total_encounter_count(&no_history);
    assert_eq!(
        a, b,
        "Bound vs Shuffled encounter schedule must match exactly under the delta rule too"
    );
    assert_eq!(
        b, c,
        "Shuffled vs NoHistory encounter schedule must match exactly under the delta rule too"
    );
}

// ---------------------------------------------------------------------------------------------
// Check 2: shuffled ledgers preserve the exact multiset of values (only the key mapping changes).
// Mechanism-independent (no run/step at all -- shuffle_ledger_for_testing only), but constructed
// via new_with_delta_rule to confirm the constructor doesn't disturb the underlying ledger.
// ---------------------------------------------------------------------------------------------

fn sort_key(r: &InteractionRecord) -> (u32, u64, u64) {
    (
        r.encounter_count,
        r.given_to_partner.to_bits(),
        r.received_from_partner.to_bits(),
    )
}

#[test]
fn check2_shuffle_preserves_exact_multiset_of_ledger_values() {
    let cfg = small_cfg();
    let (mut run, _delta_rules) = Ma001Run::new_with_delta_rule(
        Condition::Shuffled,
        5,
        cfg,
        None,
        DeltaRuleConfig::default(),
    );
    let ids: Vec<_> = run.organisms.iter().map(|o| o.id).collect();

    run.organisms[0].ledger.insert(
        ids[1],
        InteractionRecord {
            given_to_partner: 1.0,
            received_from_partner: 0.1,
            encounter_count: 10,
        },
    );
    run.organisms[0].ledger.insert(
        ids[2],
        InteractionRecord {
            given_to_partner: 2.0,
            received_from_partner: 0.2,
            encounter_count: 20,
        },
    );
    run.organisms[0].ledger.insert(
        ids[3],
        InteractionRecord {
            given_to_partner: 3.0,
            received_from_partner: 0.3,
            encounter_count: 30,
        },
    );

    let mut before: Vec<InteractionRecord> = run.organisms[0].ledger.values().copied().collect();
    before.sort_by_key(sort_key);

    run.shuffle_ledger_for_testing(0);

    let mut after: Vec<InteractionRecord> = run.organisms[0].ledger.values().copied().collect();
    after.sort_by_key(sort_key);

    assert_eq!(
        before, after,
        "shuffle must preserve the exact multiset of values under the delta-rule constructor too"
    );
    let mut keys: Vec<_> = run.organisms[0].ledger.keys().copied().collect();
    keys.sort();
    let mut expected = vec![ids[1], ids[2], ids[3]];
    expected.sort();
    assert_eq!(keys, expected);
}

// ---------------------------------------------------------------------------------------------
// Check 3: NoHistory agents truly receive zeros, even with a rich real ledger behind them.
// Mechanism-independent (peek_observed_partner_ctx is read-only), constructed via
// new_with_delta_rule for consistency.
// ---------------------------------------------------------------------------------------------

#[test]
fn check3_no_history_condition_always_feeds_a_blank_record() {
    let cfg = small_cfg();
    let (mut run, _delta_rules) = Ma001Run::new_with_delta_rule(
        Condition::NoHistory,
        9,
        cfg,
        None,
        DeltaRuleConfig::default(),
    );
    let ids: Vec<_> = run.organisms.iter().map(|o| o.id).collect();

    run.organisms[0].ledger.insert(
        ids[1],
        InteractionRecord {
            given_to_partner: 5.0,
            received_from_partner: 5.0,
            encounter_count: 500,
        },
    );

    let mut saw_a_pairing_with_1 = false;
    for tick in 0..6u64 {
        if let Some((partner_id, record)) = run.peek_observed_partner_ctx(0, tick)
            && partner_id == ids[1]
        {
            saw_a_pairing_with_1 = true;
            assert_eq!(
                record,
                InteractionRecord::default(),
                "NoHistory must feed a blank record even though the real ledger is rich"
            );
        }
    }
    assert!(
        saw_a_pairing_with_1,
        "test setup should have paired 0 with 1 within 6 ticks"
    );
}

// ---------------------------------------------------------------------------------------------
// Check 4: the real background ledger stays correct regardless of condition, under the delta
// rule.
// ---------------------------------------------------------------------------------------------

#[test]
fn check4_real_ledger_encounter_count_matches_schedule_regardless_of_condition() {
    for condition in [Condition::Bound, Condition::Shuffled, Condition::NoHistory] {
        let cfg = small_cfg();
        let (mut run, delta_rules) =
            Ma001Run::new_with_delta_rule(condition, 11, cfg, None, DeltaRuleConfig::default());
        run.run_with_delta_rule(resource_fn(), &delta_rules);
        assert!(run.organisms.iter().all(|o| !o.is_dead()));
        for o in &run.organisms {
            let total: u32 = o.ledger.values().map(|r| r.encounter_count).sum();
            assert_eq!(
                total, 60,
                "{condition:?} (delta rule): expected exactly 60 total encounters across all partners"
            );
            if condition != Condition::Shuffled {
                for record in o.ledger.values() {
                    assert_eq!(
                        record.encounter_count, 20,
                        "{condition:?} (delta rule): expected exactly 20 encounters per partner over 60 ticks"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Check 5: the swap changes exactly the two intended records for the intended agent.
// Mechanism-independent (swap_extremes_for_testing only), constructed via new_with_delta_rule.
// ---------------------------------------------------------------------------------------------

#[test]
fn check5_swap_changes_exactly_the_two_extreme_records() {
    let cfg = small_cfg();
    let (mut run, _delta_rules) =
        Ma001Run::new_with_delta_rule(Condition::Bound, 13, cfg, None, DeltaRuleConfig::default());
    let ids: Vec<_> = run.organisms.iter().map(|o| o.id).collect();

    let low = InteractionRecord {
        given_to_partner: 5.0,
        received_from_partner: 0.0,
        encounter_count: 10,
    };
    let mid = InteractionRecord {
        given_to_partner: 1.0,
        received_from_partner: 1.0,
        encounter_count: 10,
    };
    let high = InteractionRecord {
        given_to_partner: 0.0,
        received_from_partner: 5.0,
        encounter_count: 10,
    };

    run.organisms[0].ledger.insert(ids[1], low);
    run.organisms[0].ledger.insert(ids[2], mid);
    run.organisms[0].ledger.insert(ids[3], high);

    run.swap_extremes_for_testing(0);

    assert_eq!(run.organisms[0].ledger[&ids[1]], high);
    assert_eq!(run.organisms[0].ledger[&ids[3]], low);
    assert_eq!(run.organisms[0].ledger[&ids[2]], mid);
    assert_eq!(run.swapped_partners[0], Some((ids[3], ids[1])));

    assert!(run.organisms[1].ledger.is_empty());
    assert!(run.organisms[2].ledger.is_empty());
    assert!(run.organisms[3].ledger.is_empty());
}

// ---------------------------------------------------------------------------------------------
// Check 6: no future-tick information ever enters an observation, under the delta rule's own
// stepping.
// ---------------------------------------------------------------------------------------------

#[test]
fn check6_peeked_observation_reflects_only_ticks_strictly_before_it() {
    let cfg = small_cfg();
    let (mut run, delta_rules) =
        Ma001Run::new_with_delta_rule(Condition::Bound, 17, cfg, None, DeltaRuleConfig::default());
    let mut res = resource_fn();

    for target_tick in 0..15u64 {
        let snapshot: Vec<HashMap<_, _>> = run.organisms.iter().map(|o| o.ledger.clone()).collect();

        // `snapshot` has exactly one entry per organism (built from `run.organisms` above), so
        // enumerating it is equivalent to indexing `0..run.organisms.len()`.
        for (idx, organism_snapshot) in snapshot.iter().enumerate() {
            if let Some((partner_id, observed)) = run.peek_observed_partner_ctx(idx, target_tick) {
                let expected = organism_snapshot
                    .get(&partner_id)
                    .copied()
                    .unwrap_or_default();
                assert_eq!(
                    observed, expected,
                    "tick {target_tick}, organism {idx} (delta rule): peeked observation must \
                     match the ledger as of the end of the previous tick"
                );
            }
        }

        run.step_with_delta_rule(target_tick, &mut res, &delta_rules);
    }
}

// ---------------------------------------------------------------------------------------------
// Check 7: resolve_pair_transfer's conservation property holds throughout a real delta-rule run.
// ---------------------------------------------------------------------------------------------

#[test]
fn check7_energy_never_leaves_the_unit_interval_over_a_real_run() {
    let cfg = small_cfg();
    for condition in [Condition::Bound, Condition::Shuffled, Condition::NoHistory] {
        let (mut run, delta_rules) =
            Ma001Run::new_with_delta_rule(condition, 19, cfg, None, DeltaRuleConfig::default());
        let mut res = resource_fn();
        for tick in 0..cfg.total_ticks {
            run.step_with_delta_rule(tick, &mut res, &delta_rules);
            for o in &run.organisms {
                assert!(
                    (0.0..=1.0).contains(&o.energy),
                    "{condition:?} (delta rule) tick {tick}: energy out of [0,1]: {}",
                    o.energy
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Check 8: physical parameters are identical across conditions (by construction), and the
// delta-rule constructor's learning-pathway override doesn't touch OrganismConfig.
// ---------------------------------------------------------------------------------------------

#[test]
fn check8_organism_config_is_identical_across_conditions_for_the_same_seed() {
    let cfg = small_cfg();
    let (bound, _bound_dr) =
        Ma001Run::new_with_delta_rule(Condition::Bound, 23, cfg, None, DeltaRuleConfig::default());
    let (shuffled, _shuffled_dr) = Ma001Run::new_with_delta_rule(
        Condition::Shuffled,
        23,
        cfg,
        None,
        DeltaRuleConfig::default(),
    );
    let (no_history, _no_history_dr) = Ma001Run::new_with_delta_rule(
        Condition::NoHistory,
        23,
        cfg,
        None,
        DeltaRuleConfig::default(),
    );
    for i in 0..bound.organisms.len() {
        assert_eq!(bound.organisms[i].cfg, shuffled.organisms[i].cfg);
        assert_eq!(shuffled.organisms[i].cfg, no_history.organisms[i].cfg);
        // The delta-rule constructor's learning-pathway override must be identical too.
        assert!(!bound.organisms[i].agent.config.enable_model_learning);
        assert!(bound.organisms[i].agent.td_learner.is_none());
    }
}

// ---------------------------------------------------------------------------------------------
// Check 9: given (seed, condition), the full delta-rule run replays exactly -- confirms
// DeltaRuleLearner introduces no hidden non-determinism.
// ---------------------------------------------------------------------------------------------

#[test]
fn check9_same_seed_and_condition_replays_exactly() {
    fn run_once(condition: Condition) -> (Vec<u32>, Vec<Vec<usize>>) {
        let cfg = small_cfg();
        let (mut run, delta_rules) =
            Ma001Run::new_with_delta_rule(condition, 29, cfg, None, DeltaRuleConfig::default());
        run.run_with_delta_rule(resource_fn(), &delta_rules);
        let encounter_totals: Vec<u32> = run
            .organisms
            .iter()
            .map(|o| o.ledger.values().map(|r| r.encounter_count).sum())
            .collect();
        let action_counts: Vec<Vec<usize>> = run
            .analysis_counts
            .iter()
            .map(|m| {
                let mut v: Vec<usize> = m
                    .values()
                    .map(|c| c[0] as usize * 1_000_000 + c[1] as usize * 1_000 + c[2] as usize)
                    .collect();
                v.sort();
                v
            })
            .collect();
        (encounter_totals, action_counts)
    }

    for condition in [Condition::Bound, Condition::Shuffled, Condition::NoHistory] {
        let a = run_once(condition);
        let b = run_once(condition);
        assert_eq!(
            a, b,
            "{condition:?} (delta rule): identical (seed, condition) must replay identically"
        );
    }
}
