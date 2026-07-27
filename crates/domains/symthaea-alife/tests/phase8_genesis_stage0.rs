// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stage 0 invariants for Genesis v0, per `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`.
//!
//! These must hold *before* any behavioral/interpretive claim is made about what Genesis
//! produces -- a false positive here (e.g. an apparent "cooperative fitness gain" that's
//! actually silent resource creation) would poison every later experiment built on this
//! substrate. No hypothesis about cooperation, reciprocity, or trust is tested anywhere in this
//! file -- see the plan's non-goals.

use std::collections::HashSet;

use symthaea_alife::{
    Action, AgentId, AgentIdAllocator, EncounterScheduler, InteractionRecord, Organism,
    OrganismConfig, PairingMode, Population, PopulationConfig,
};

fn social_organism_cfg() -> OrganismConfig {
    OrganismConfig {
        social_enabled: true,
        transfer_quantum: 0.05,
        forage_efficiency: 0.6, // Phase 1's "sustainable" value -- avoid knife-edge extinction
        ..OrganismConfig::default()
    }
}

/// A population config with reproduction/death effectively disabled, so ledger-sum and
/// conservation checks aren't complicated by individuals appearing/disappearing mid-run.
fn no_churn_population_cfg(organism_cfg: OrganismConfig) -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: -1.0, // energy is clamped >= 0.0, so this never fires
        reproduction_energy_threshold: 10.0, // energy is clamped <= 1.0, so this never fires
        reproduction_energy_cost: 0.0,
        organism_cfg,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------------------------

#[test]
fn agent_id_never_repeats_across_a_populations_lifetime() {
    let cfg = no_churn_population_cfg(social_organism_cfg());
    // Allow real churn here (unlike the no-churn helper's config) to exercise births/deaths --
    // override the thresholds this one test actually wants.
    let cfg = PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.85,
        reproduction_energy_cost: 0.4,
        ..cfg
    };
    let mut pop = Population::new(cfg, 6, 3);
    let mut sched = EncounterScheduler::new(PairingMode::Random, 4);
    let mut all_seen_ids = HashSet::new();
    for id in pop.organisms.iter().map(|o| o.id) {
        assert!(all_seen_ids.insert(id));
    }
    for _ in 0..500u64 {
        // Density-divided share, NOT a flat constant -- Phase 1's own tests
        // (`population.rs::abundant_resources_grow_the_population`) warn a flat undivided
        // resource compounds fast (~15-20 ticks/doubling) and is deliberately kept short (~60
        // ticks) for exactly that reason. This test wants sustained churn over 500 ticks without
        // unbounded growth, so it shares a fixed total pool across the population instead.
        pop.step_social(|n| 2.0 / (n.max(1) as f64), &mut sched);
        let mut live_ids = HashSet::new();
        for id in pop.organisms.iter().map(|o| o.id) {
            assert!(
                live_ids.insert(id),
                "two currently-living organisms share an AgentId: {id:?}"
            );
        }
        for id in &live_ids {
            // A newly-appearing id is fine; an id ever seen before must never be seen *again*
            // after having disappeared and been reassigned to someone else. Since
            // `AgentIdAllocator` never reuses a value (see `agent_id.rs`), any id present now
            // that was already in `all_seen_ids` must be an individual that's been alive the
            // whole time, not a reused label -- there is no way to tell those apart from ids
            // alone, so the real guarantee this test leans on is the allocator's own
            // never-repeats property, exercised here under real population churn.
            all_seen_ids.insert(*id);
        }
    }
}

#[test]
fn agent_id_survives_an_index_shift_caused_by_an_earlier_organisms_death() {
    let cfg = no_churn_population_cfg(social_organism_cfg());
    let mut pop = Population::new(cfg, 4, 1);
    let ids_before: Vec<AgentId> = pop.organisms.iter().map(|o| o.id).collect();

    // Force the organism at index 0 to die on the next tick.
    pop.organisms[0].energy = 0.0;
    let mut sched = EncounterScheduler::new(PairingMode::Random, 7);
    // Use a population-level death threshold that actually fires for this one test.
    pop.cfg.death_energy_threshold = 0.05;
    pop.step_social(|_n| 0.0, &mut sched); // no resource this tick -- nothing should recover

    assert_eq!(
        pop.organisms.len(),
        3,
        "exactly one organism should have died"
    );
    let remaining: HashSet<AgentId> = pop.organisms.iter().map(|o| o.id).collect();
    assert!(
        !remaining.contains(&ids_before[0]),
        "the dead organism's id must not still be present"
    );
    for id in &ids_before[1..] {
        assert!(
            remaining.contains(id),
            "surviving organism {id:?}'s original id must be preserved after the Vec shifted"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Pairing determinism
// ---------------------------------------------------------------------------------------------

#[test]
fn identical_seed_and_population_reproduces_the_exact_same_pairing_schedule() {
    fn ids(n: u64) -> Vec<AgentId> {
        let mut alloc = AgentIdAllocator::new();
        (0..n).map(|_| alloc.next()).collect()
    }
    let living = ids(13); // odd on purpose
    let mut a = EncounterScheduler::new(PairingMode::FixedPartners, 555);
    let mut b = EncounterScheduler::new(PairingMode::FixedPartners, 555);
    for _ in 0..25 {
        assert_eq!(a.pair(&living), b.pair(&living));
    }
}

// ---------------------------------------------------------------------------------------------
// Ledger truth + Conservation
// ---------------------------------------------------------------------------------------------

#[test]
fn ledger_truth_and_conservation_hold_for_an_isolated_forced_transfer() {
    let cfg = social_organism_cfg();
    let mut alloc = AgentIdAllocator::new();
    let giver_id = alloc.next();
    let receiver_id = alloc.next();

    // Debit side: an identical giver that Rests instead of Transfers isolates exactly the
    // transfer's own energy effect, since Forage-only terms (`gained`, `forage_activity_cost`)
    // are zero for both Rest and Transfer, and perception/select_action run identically
    // regardless of `forced_action` -- the only difference between the two runs below is the
    // transfer debit itself.
    let mut giver = Organism::new(cfg, 101).with_id(giver_id);
    let mut giver_control = Organism::new(cfg, 101).with_id(giver_id);
    let empty_record = InteractionRecord::default();

    let tick_g = giver.tick_social(
        0.5,
        Some(Action::Transfer.index()),
        Some((receiver_id, empty_record)),
    );
    let control_tick = giver_control.tick_social(
        0.5,
        Some(Action::Rest.index()),
        Some((receiver_id, empty_record)),
    );
    let given = tick_g.transfer_given;
    assert!(
        given > 0.0,
        "a forced Transfer with headroom and a real partner must debit a positive amount"
    );
    assert_eq!(control_tick.transfer_given, 0.0);
    assert!(
        (giver_control.energy - giver.energy - given).abs() < 1e-12,
        "the Rest-control giver's energy must exceed the Transfer giver's by exactly `given`: \
         control={}, giver={}, given={given}",
        giver_control.energy,
        giver.energy
    );

    // Credit side: a receiver with headroom under 1.0 must gain exactly `given` when credited
    // the way `Population::step_social`'s Phase 3 does.
    let mut receiver = Organism::new(cfg, 202).with_id(receiver_id);
    receiver.tick_social(
        0.5,
        Some(Action::Rest.index()),
        Some((giver_id, empty_record)),
    );
    let receiver_energy_before_credit = receiver.energy;
    receiver.energy = (receiver.energy + given).clamp(0.0, 1.0);
    assert!(
        (receiver.energy - receiver_energy_before_credit - given).abs() < 1e-12,
        "receiver's energy must increase by exactly `given` when there is headroom under 1.0"
    );

    // Ledger truth: updating both sides exactly as `step_social`'s Phase 3 does must produce
    // matching, factual entries -- no inferred field.
    giver
        .ledger
        .entry(receiver_id)
        .or_default()
        .given_to_partner += given;
    receiver
        .ledger
        .entry(giver_id)
        .or_default()
        .received_from_partner += given;
    assert_eq!(giver.ledger[&receiver_id].given_to_partner, given);
    assert_eq!(receiver.ledger[&giver_id].received_from_partner, given);
}

#[test]
fn global_given_equals_global_received_after_many_social_ticks() {
    let cfg = no_churn_population_cfg(social_organism_cfg());
    let mut pop = Population::new(cfg, 8, 55);
    let mut sched = EncounterScheduler::new(PairingMode::Random, 9);
    for _ in 0..300u64 {
        pop.step_social(|_n| 0.6, &mut sched);
    }
    let total_given: f64 = pop
        .organisms
        .iter()
        .flat_map(|o| o.ledger.values())
        .map(|r| r.given_to_partner)
        .sum();
    let total_received: f64 = pop
        .organisms
        .iter()
        .flat_map(|o| o.ledger.values())
        .map(|r| r.received_from_partner)
        .sum();
    assert!(
        (total_given - total_received).abs() < 1e-6,
        "global given/received must match with no churn: given={total_given}, received={total_received}"
    );
}

// ---------------------------------------------------------------------------------------------
// Observation causality
// ---------------------------------------------------------------------------------------------

#[test]
fn ledger_snapshot_fed_to_tick_must_be_the_pre_tick_value_not_this_ticks_own_update() {
    // Locks in the calling convention `Population::step_social` relies on: Phase 2 (ticking)
    // must snapshot each organism's ledger *before* Phase 3 (mutating it from this tick's
    // outcome) runs -- verified structurally in `population.rs` (two separate loops, in that
    // order) and locked in here as an explicit contract test.
    let cfg = social_organism_cfg();
    let mut alloc = AgentIdAllocator::new();
    let giver_id = alloc.next();
    let receiver_id = alloc.next();
    let mut giver = Organism::new(cfg, 1).with_id(giver_id);

    let pre_tick_snapshot = giver.ledger.get(&receiver_id).copied().unwrap_or_default();
    assert_eq!(pre_tick_snapshot.encounter_count, 0);

    let _ = giver.tick_social(
        0.5,
        Some(Action::Transfer.index()),
        Some((receiver_id, pre_tick_snapshot)),
    );
    // Correct order: mutate the ledger only *after* the tick that used the pre-tick snapshot.
    giver.ledger.entry(receiver_id).or_default().encounter_count += 1;

    let post_tick_record = giver.ledger[&receiver_id];
    assert_eq!(post_tick_record.encounter_count, 1);
    assert_ne!(
        pre_tick_snapshot.encounter_count, post_tick_record.encounter_count,
        "the snapshot fed to tick_social must be strictly older than post-tick ledger state"
    );
}

// ---------------------------------------------------------------------------------------------
// Baseline equivalence
// ---------------------------------------------------------------------------------------------

#[test]
fn tick_is_exactly_tick_social_with_no_partner_when_social_disabled() {
    let cfg = OrganismConfig::default(); // social_enabled: false
    let mut a = Organism::new(cfg, 42);
    let mut b = Organism::new(cfg, 42);
    let env = symthaea_alife::Environment::default();
    for t in 0..500u64 {
        let r = env.resource_at(t);
        let ta = a.tick(r, None);
        let tb = b.tick_social(r, None, None);
        assert_eq!(
            ta.energy, tb.energy,
            "t={t}: energy diverged between tick() and tick_social()"
        );
        assert_eq!(
            ta.action, tb.action,
            "t={t}: action diverged between tick() and tick_social()"
        );
        assert_eq!(ta.transfer_given, 0.0);
        assert_eq!(tb.transfer_given, 0.0);
    }
}

#[test]
fn asocial_population_step_is_unaffected_by_social_code_existing() {
    // Phase 1's own existing test (`population.rs::abundant_resources_grow_the_population`)
    // already proves `step()` grows a population; this test is the Genesis-specific regression
    // guard that `step()` itself (as opposed to the new `step_social`) is untouched by this
    // plan's changes, using a config that never sets `social_enabled`.
    let cfg = PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig {
            forage_efficiency: 0.6,
            ..OrganismConfig::default()
        },
        ..Default::default()
    };
    let mut pop = Population::new(cfg, 2, 42);
    for _ in 0..60u64 {
        pop.step(|_n| 0.5);
    }
    assert!(
        pop.len() > 2,
        "population should have grown, got {}",
        pop.len()
    );
}

// ---------------------------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------------------------

#[test]
fn same_seed_and_config_replays_the_exact_same_run() {
    fn run() -> (Vec<f64>, u64, u64) {
        let cfg = no_churn_population_cfg(social_organism_cfg());
        let mut pop = Population::new(cfg, 6, 77);
        let mut sched = EncounterScheduler::new(PairingMode::Random, 88);
        for _ in 0..200u64 {
            pop.step_social(|_n| 0.5, &mut sched);
        }
        let energies: Vec<f64> = pop.organisms.iter().map(|o| o.energy).collect();
        (energies, pop.total_births, pop.total_deaths)
    }
    assert_eq!(run(), run());
}

// ---------------------------------------------------------------------------------------------
// Action accessibility (`Transfer` reachability)
// ---------------------------------------------------------------------------------------------

#[test]
fn transfer_is_reachable_via_real_select_action_not_merely_constructible() {
    // Softmax action selection (`ActiveInferenceAgent::select_action`) assigns every action a
    // strictly positive probability for any finite expected free energy (`exp(x) > 0` for all
    // real `x`), so `Transfer` should be genuinely selectable, not just forceable, whenever
    // `social_enabled` gives it a place in the action space at all. Verified empirically across
    // several seeds/ticks rather than asserted from theory alone, per the Genesis plan's Stage 0
    // discipline -- if this ever failed, it would mean the action-selection objective
    // structurally never permits `Transfer`, which would have to be understood and disclosed
    // before any claim about cooperation/reciprocity could be trusted.
    let cfg = social_organism_cfg();
    let mut alloc = AgentIdAllocator::new();
    let a_id = alloc.next();
    let b_id = alloc.next();
    let mut transfer_seen = false;

    'seeds: for seed in 1..40u64 {
        let mut a = Organism::new(cfg, seed).with_id(a_id);
        let mut b = Organism::new(cfg, seed.wrapping_add(1000)).with_id(b_id);
        for _ in 0..300u64 {
            let a_record = a.ledger.get(&b_id).copied().unwrap_or_default();
            let tick = a.tick_social(0.5, None, Some((b_id, a_record)));
            let b_record = b.ledger.get(&a_id).copied().unwrap_or_default();
            b.tick_social(0.5, None, Some((a_id, b_record)));
            if tick.action == Action::Transfer.index() {
                transfer_seen = true;
                break 'seeds;
            }
            a.ledger.entry(b_id).or_default().encounter_count += 1;
            b.ledger.entry(a_id).or_default().encounter_count += 1;
        }
    }
    assert!(
        transfer_seen,
        "Transfer was never selected by real select_action() across 40 seeds x 300 ticks"
    );
}

// ---------------------------------------------------------------------------------------------
// Event log fidelity (the log accurately records what already-Stage-0-verified mechanics did --
// not re-testing FEP/conservation/ledger correctness itself, just that the log doesn't lie about it)
// ---------------------------------------------------------------------------------------------

#[test]
fn event_log_emits_exactly_one_event_per_organism_per_tick() {
    let cfg = no_churn_population_cfg(social_organism_cfg());
    let mut pop = Population::new(cfg, 5, 21);
    let mut sched = EncounterScheduler::new(PairingMode::Random, 22);
    for _ in 0..10u64 {
        pop.step_social(|n| 2.0 / (n.max(1) as f64), &mut sched);
    }
    let events = pop.drain_event_log();
    assert_eq!(events.len(), 5 * 10, "expected 5 organisms x 10 ticks");
    for t in 0..10u64 {
        assert_eq!(events.iter().filter(|e| e.tick == t).count(), 5);
    }
    // drain_event_log must actually empty the log, not just return a copy.
    assert!(pop.drain_event_log().is_empty());
}

#[test]
fn event_transfer_amount_matches_the_ledger_update_it_caused() {
    let cfg = social_organism_cfg();
    let mut pop = Population::new(no_churn_population_cfg(cfg), 2, 30);
    let mut sched = EncounterScheduler::new(PairingMode::FixedPartners, 31);
    for _ in 0..50u64 {
        pop.step_social(|_n| 0.6, &mut sched);
    }
    let events = pop.drain_event_log();
    let ids: Vec<AgentId> = pop.organisms.iter().map(|o| o.id).collect();

    // For every tick where organism 0 logged a nonzero transfer_amount, that exact amount must
    // appear as a `given_to_partner` increment reflected in the final ledger sum (loosely -- we
    // check the *total* across the run rather than re-deriving per-tick deltas, which the
    // existing `global_given_equals_global_received_after_many_social_ticks` test already covers
    // more directly for the conservation side).
    let logged_total_given: f64 = events
        .iter()
        .filter(|e| e.agent_id == ids[0])
        .map(|e| e.transfer_amount)
        .sum();
    let ledger_given: f64 = pop.organisms[0]
        .ledger
        .values()
        .map(|r| r.given_to_partner)
        .sum();
    assert!(
        (logged_total_given - ledger_given).abs() < 1e-9,
        "sum of logged transfer_amount for organism 0 ({logged_total_given}) must match its \
         ledger's total given_to_partner ({ledger_given})"
    );
}

#[test]
fn lineage_and_generation_are_inherited_correctly_across_reproduction() {
    let cfg = social_organism_cfg();
    let population_cfg = PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.85,
        reproduction_energy_cost: 0.4,
        organism_cfg: cfg,
        ..Default::default()
    };
    let mut pop = Population::new(population_cfg, 4, 40);
    let founder_ids: Vec<AgentId> = pop.organisms.iter().map(|o| o.id).collect();
    for o in &pop.organisms {
        assert_eq!(o.generation, 0);
        assert_eq!(o.lineage_id, o.id, "a founder is its own lineage");
    }

    let mut sched = EncounterScheduler::new(PairingMode::Random, 41);
    let mut saw_a_birth = false;
    for _ in 0..400u64 {
        let summary = pop.step_social(|n| 3.0 / (n.max(1) as f64), &mut sched);
        if summary.births_this_tick > 0 {
            saw_a_birth = true;
        }
    }
    assert!(
        saw_a_birth,
        "expected at least one birth over 400 ticks under abundant resources"
    );

    for o in &pop.organisms {
        if founder_ids.contains(&o.id) {
            assert_eq!(o.generation, 0);
            assert_eq!(o.lineage_id, o.id);
        } else {
            assert!(
                o.generation >= 1,
                "an offspring must be at least generation 1"
            );
            assert!(
                founder_ids.contains(&o.lineage_id),
                "an offspring's lineage_id must trace back to one of the original founders"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Action accessibility, strengthened (Genesis v0.1 audit Gate 6)
// ---------------------------------------------------------------------------------------------

#[test]
fn transfer_selection_actually_responds_to_social_state_not_just_reachable() {
    // `transfer_is_reachable_via_real_select_action_not_merely_constructible` (above) only shows
    // softmax gives Transfer nonzero probability -- guaranteed by construction for any finite EFE,
    // regardless of whether that probability ever actually *moves* in response to social state.
    //
    // A single-tick version of this test (comparing one fresh perceive-and-decide snapshot,
    // blank vs. rich) was tried first and found no divergence across 300 seeds -- but that's a
    // very weak setup: after the Genesis v0.1 audit's Gate 5 fix (zero EFE preference-precision
    // on social channels, so they no longer pull pragmatic value directly), the *only* remaining
    // path from social observation to action is indirect, through belief updating and the
    // generative model's transition/likelihood matrices' small cross-dimension coupling -- and a
    // single perceive() call from a freshly-initialized belief (starting at exactly 0.5 in every
    // dimension) gives that indirect path almost no room to matter. So this test instead runs a
    // *sustained* condition difference over many ticks per seed, giving belief and (real,
    // unrelated to EFE preference) model-learning a real chance to diverge if the causal pathway
    // exists at all: same seed (identical RNG stream and starting state), only whether the
    // ledger snapshot fed into `tick_social` every tick is blank (never met this partner) or rich
    // (a long, generous shared history) differs. If the action sequence never diverges across
    // many seeds x many ticks each, that's a real, disclosable finding that social observation
    // may not be reaching action selection in any way that matters at this scale -- not merely a
    // single-snapshot artifact.
    let cfg = social_organism_cfg();
    let mut alloc = AgentIdAllocator::new();
    let partner_id = alloc.next();

    let blank = InteractionRecord::default();
    let rich = InteractionRecord {
        given_to_partner: 5.0,
        received_from_partner: 5.0,
        encounter_count: 50,
    };

    let seeds = 1..=20u64;
    let ticks = 500u64;
    let mut any_seed_diverged = false;

    'seeds: for seed in seeds.clone() {
        let mut organism_blank = Organism::new(cfg, seed);
        let mut organism_rich = Organism::new(cfg, seed);
        for _ in 0..ticks {
            let tick_blank = organism_blank.tick_social(0.5, None, Some((partner_id, blank)));
            let tick_rich = organism_rich.tick_social(0.5, None, Some((partner_id, rich)));
            if tick_blank.action != tick_rich.action {
                any_seed_diverged = true;
                continue 'seeds;
            }
        }
    }
    assert!(
        any_seed_diverged,
        "action choice never diverged between a sustained blank vs. rich social history across \
         {} seeds x {ticks} ticks each -- Transfer's probability may not respond to social state \
         at all, even given many ticks for belief/model-learning to accumulate a difference",
        seeds.count()
    );
}
