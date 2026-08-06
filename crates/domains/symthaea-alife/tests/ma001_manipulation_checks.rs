// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001 manipulation checks, per
//! `ALIFE_MA001_PARTNER_CONDITIONED_POLICY_PLAN_2026-07-26.md` §7. All nine must pass before any
//! hypothesis test from this experiment is trusted -- a failed check invalidates that run
//! regardless of how the headline metric looks.

use std::collections::HashMap;

use symthaea_alife::InteractionRecord;
use symthaea_alife::ma001::{Condition, Ma001Config, Ma001Run};

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
    let mut bound = Ma001Run::new(Condition::Bound, 42, cfg, None);
    let mut shuffled = Ma001Run::new(Condition::Shuffled, 42, cfg, None);
    let mut no_history = Ma001Run::new(Condition::NoHistory, 42, cfg, None);
    bound.run(resource_fn());
    shuffled.run(resource_fn());
    no_history.run(resource_fn());

    // Confirmed, not assumed: no deaths over this short a run under abundant resources.
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
        "Bound vs Shuffled encounter schedule must match exactly"
    );
    assert_eq!(
        b, c,
        "Shuffled vs NoHistory encounter schedule must match exactly"
    );
}

// ---------------------------------------------------------------------------------------------
// Check 2: shuffled ledgers preserve the exact multiset of values (only the key mapping changes).
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
    let mut run = Ma001Run::new(Condition::Shuffled, 5, cfg, None);
    let ids: Vec<_> = run.organisms.iter().map(|o| o.id).collect();

    // Inject a hand-crafted, clearly-distinguishable ledger for organism 0 across its 3
    // groupmates, bypassing the need to run real ticks to build up history.
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
        "shuffle must preserve the exact multiset of values, only permuting which key holds them"
    );
    // And the keys are unchanged (still exactly these 3 partners).
    let mut keys: Vec<_> = run.organisms[0].ledger.keys().copied().collect();
    keys.sort();
    let mut expected = vec![ids[1], ids[2], ids[3]];
    expected.sort();
    assert_eq!(keys, expected);
}

// ---------------------------------------------------------------------------------------------
// Check 3: NoHistory agents truly receive zeros, even with a rich real ledger behind them.
// ---------------------------------------------------------------------------------------------

#[test]
fn check3_no_history_condition_always_feeds_a_blank_record() {
    let cfg = small_cfg();
    let mut run = Ma001Run::new(Condition::NoHistory, 9, cfg, None);
    let ids: Vec<_> = run.organisms.iter().map(|o| o.id).collect();

    // Inject an obviously non-blank real ledger for organism 0.
    run.organisms[0].ledger.insert(
        ids[1],
        InteractionRecord {
            given_to_partner: 5.0,
            received_from_partner: 5.0,
            encounter_count: 500,
        },
    );

    // Whatever tick organism 0 is paired with organism 1, the observation must still be blank.
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
// Check 4: the real background ledger stays correct regardless of condition.
// ---------------------------------------------------------------------------------------------

#[test]
fn check4_real_ledger_encounter_count_matches_schedule_regardless_of_condition() {
    for condition in [Condition::Bound, Condition::Shuffled, Condition::NoHistory] {
        let cfg = small_cfg();
        let mut run = Ma001Run::new(condition, 11, cfg, None);
        run.run(resource_fn());
        assert!(run.organisms.iter().all(|o| !o.is_dead()));
        // Each of the 4 organisms in one group meets each of its 3 groupmates on an exactly
        // period-3 cadence -- over 60 ticks, exactly 60 total encounters (20 per partner) for
        // every organism. Under Bound/NoHistory (key<->value mapping never disturbed) this holds
        // per-partner too. Under Shuffled it does NOT necessarily hold per-partner -- with
        // shuffle_epoch_ticks=10 not a multiple of the round-robin cycle length (3), a shuffle
        // can fire mid-cycle, when the three partners' encounter_count values aren't yet equal
        // to each other, so permuting them scrambles which key holds which count from then on
        // (the *same* real, intentional property the actual confirmatory config has:
        // shuffle_epoch_ticks=100 isn't a multiple of 3 either). The condition-independent
        // invariant is the *total* across all partners, which this checks for every condition.
        for o in &run.organisms {
            let total: u32 = o.ledger.values().map(|r| r.encounter_count).sum();
            assert_eq!(
                total, 60,
                "{condition:?}: expected exactly 60 total encounters across all partners"
            );
            if condition != Condition::Shuffled {
                for record in o.ledger.values() {
                    assert_eq!(
                        record.encounter_count, 20,
                        "{condition:?}: expected exactly 20 encounters per partner over 60 ticks"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Check 5: the swap changes exactly the two intended records for the intended agent.
// ---------------------------------------------------------------------------------------------

#[test]
fn check5_swap_changes_exactly_the_two_extreme_records() {
    let cfg = small_cfg();
    let mut run = Ma001Run::new(Condition::Bound, 13, cfg, None);
    let ids: Vec<_> = run.organisms.iter().map(|o| o.id).collect();

    let low = InteractionRecord {
        given_to_partner: 5.0,
        received_from_partner: 0.0,
        encounter_count: 10,
    }; // net_balance = -5.0, the lowest
    let mid = InteractionRecord {
        given_to_partner: 1.0,
        received_from_partner: 1.0,
        encounter_count: 10,
    }; // net_balance = 0.0
    let high = InteractionRecord {
        given_to_partner: 0.0,
        received_from_partner: 5.0,
        encounter_count: 10,
    }; // net_balance = +5.0, the highest

    run.organisms[0].ledger.insert(ids[1], low);
    run.organisms[0].ledger.insert(ids[2], mid);
    run.organisms[0].ledger.insert(ids[3], high);

    run.swap_extremes_for_testing(0);

    assert_eq!(
        run.organisms[0].ledger[&ids[1]], high,
        "lowest-balance partner should now hold the highest record"
    );
    assert_eq!(
        run.organisms[0].ledger[&ids[3]], low,
        "highest-balance partner should now hold the lowest record"
    );
    assert_eq!(
        run.organisms[0].ledger[&ids[2]], mid,
        "the uninvolved mid-balance partner must be untouched"
    );
    assert_eq!(
        run.swapped_partners[0],
        Some((ids[3], ids[1])),
        "swapped_partners must record (highest, lowest)"
    );

    // Other organisms' ledgers must be completely untouched.
    assert!(run.organisms[1].ledger.is_empty());
    assert!(run.organisms[2].ledger.is_empty());
    assert!(run.organisms[3].ledger.is_empty());
}

// ---------------------------------------------------------------------------------------------
// Check 6: no future-tick information ever enters an observation.
// ---------------------------------------------------------------------------------------------

#[test]
fn check6_peeked_observation_reflects_only_ticks_strictly_before_it() {
    let cfg = small_cfg();
    let mut run = Ma001Run::new(Condition::Bound, 17, cfg, None);
    let mut res = resource_fn();

    for target_tick in 0..15u64 {
        // Snapshot every organism's ledger exactly as of the end of `target_tick - 1` (i.e.
        // before `target_tick` itself has been stepped).
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
                    "tick {target_tick}, organism {idx}: peeked observation must match the \
                     ledger as of the end of the previous tick, never anything target_tick \
                     itself (or later) would produce"
                );
            }
        }

        run.step(target_tick, &mut res);
    }
}

// ---------------------------------------------------------------------------------------------
// Check 7: resolve_pair_transfer's conservation property holds throughout a real run.
// ---------------------------------------------------------------------------------------------

#[test]
fn check7_energy_never_leaves_the_unit_interval_over_a_real_run() {
    // Conservation itself (Δgiver + Δreceiver = 0 with headroom) is unit-tested directly on
    // `resolve_pair_transfer` in `population.rs`; this is the ma001-side integration check that
    // the driver's own call site never produces an out-of-bounds energy value, which would be
    // the observable symptom of a conservation bug in this driver's own Phase B.
    let cfg = small_cfg();
    for condition in [Condition::Bound, Condition::Shuffled, Condition::NoHistory] {
        let mut run = Ma001Run::new(condition, 19, cfg, None);
        let mut res = resource_fn();
        for tick in 0..cfg.total_ticks {
            run.step(tick, &mut res);
            for o in &run.organisms {
                assert!(
                    (0.0..=1.0).contains(&o.energy),
                    "{condition:?} tick {tick}: energy out of [0,1]: {}",
                    o.energy
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Check 8: physical parameters are identical across conditions (by construction).
// ---------------------------------------------------------------------------------------------

#[test]
fn check8_organism_config_is_identical_across_conditions_for_the_same_seed() {
    let cfg = small_cfg();
    let bound = Ma001Run::new(Condition::Bound, 23, cfg, None);
    let shuffled = Ma001Run::new(Condition::Shuffled, 23, cfg, None);
    let no_history = Ma001Run::new(Condition::NoHistory, 23, cfg, None);
    for i in 0..bound.organisms.len() {
        assert_eq!(bound.organisms[i].cfg, shuffled.organisms[i].cfg);
        assert_eq!(shuffled.organisms[i].cfg, no_history.organisms[i].cfg);
    }
}

// ---------------------------------------------------------------------------------------------
// Check 9: given (seed, condition), the full run replays exactly.
// ---------------------------------------------------------------------------------------------

#[test]
fn check9_same_seed_and_condition_replays_exactly() {
    fn run_once(condition: Condition) -> (Vec<u32>, Vec<Vec<usize>>) {
        let cfg = small_cfg();
        let mut run = Ma001Run::new(condition, 29, cfg, None);
        run.run(resource_fn());
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
            "{condition:?}: identical (seed, condition) must replay identically"
        );
    }
}
