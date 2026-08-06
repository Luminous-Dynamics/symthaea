//! Characterization tests for `ExpectedFreeEnergyComputer`'s term behaviour.
//!
//! These assert what the code CURRENTLY does, and what it currently does is a
//! defect. They exist so that the defect is measured rather than argued, and so
//! that fixing it fails loudly here with a pointer to the write-up.
//!
//! Context: `docs/EFE_DISPATCH_GATE_2026-07-31.md`. A planned 18-day fleet-dispatch
//! experiment was halted because the two terms that distinguish expected free
//! energy from a weighted greedy heuristic — epistemic and novelty — cannot vary
//! across candidate actions. The finding was originally derived statically by
//! composing three functions; these tests turn it into a measurement.
//!
//! **If a test here fails, that is probably good news.** It means someone made the
//! term action-dependent. Delete the characterization test, promote the matching
//! `aspirational_*` test, and update the write-up.

use symthaea_fep::free_energy::ExpectedFreeEnergyComputer;
use symthaea_fep::generative_model::GenerativeModel;
use symthaea_fep::types::HiddenState;

const STATE_DIM: usize = 8;
const OBS_DIM: usize = 8;
const NUM_ACTIONS: usize = 12;

fn fixture() -> (ExpectedFreeEnergyComputer, GenerativeModel, HiddenState) {
    let efe = ExpectedFreeEnergyComputer::new(OBS_DIM);
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
    let state = HiddenState::new(STATE_DIM);
    (efe, model, state)
}

/// Scores every action once against one fixed state, in order.
fn score_all(
    efe: &mut ExpectedFreeEnergyComputer,
    model: &GenerativeModel,
    state: &HiddenState,
    order: impl Iterator<Item = usize>,
) -> Vec<(usize, f64, f64, f64)> {
    order
        .map(|a| {
            let r = efe.compute(a, state, model);
            (a, r.epistemic, r.novelty, r.pragmatic)
        })
        .collect()
}

/// DEFECT. The epistemic term is bit-identical for every action.
///
/// `GenerativeModel::predict_next_state` makes only `next_mean` action-dependent;
/// its precision update `(p*τ)/(p+τ)` has no action term. `HiddenState::entropy`
/// reads `precision` and `mean.len()` only — the mean *values* never enter. So
/// `compute_epistemic_value = predicted_entropy - current_entropy` cannot move.
#[test]
fn characterize_epistemic_term_is_action_invariant() {
    let (mut efe, model, state) = fixture();
    let scored = score_all(&mut efe, &model, &state, 0..NUM_ACTIONS);

    let first = scored[0].1;
    for (action, epistemic, _, _) in &scored {
        assert_eq!(
            *epistemic, first,
            "epistemic value for action {action} differs from action 0 — the defect \
             described in docs/EFE_DISPATCH_GATE_2026-07-31.md may be fixed. If so, \
             delete this test and promote aspirational_epistemic_term_varies_across_actions."
        );
    }
}

/// The pragmatic term DOES vary — but only by action parity, which is the separate
/// (and previously reported) degeneracy in the transition construction:
/// `generative_model.rs` sets `bias_direction = if action_idx % 2 == 0 { -1 } else { 1 }`.
/// Recorded here so the two defects are not conflated: fixing parity would not fix
/// the epistemic term, and vice versa.
#[test]
fn characterize_pragmatic_term_varies_only_by_action_parity() {
    let (mut efe, model, state) = fixture();
    let scored = score_all(&mut efe, &model, &state, 0..NUM_ACTIONS);

    let distinct: std::collections::BTreeSet<u64> =
        scored.iter().map(|(_, _, _, p)| p.to_bits()).collect();

    assert_eq!(
        distinct.len(),
        2,
        "expected exactly 2 distinct pragmatic values across {NUM_ACTIONS} actions \
         (one per action parity), got {}",
        distinct.len()
    );

    for (action, _, _, pragmatic) in &scored {
        let same_parity = scored
            .iter()
            .find(|(a, _, _, _)| a % 2 == action % 2)
            .unwrap()
            .3;
        assert_eq!(
            *pragmatic, same_parity,
            "action {action} broke parity grouping"
        );
    }
}

/// DEFECT. Scoring a candidate set flattens the novelty term.
///
/// `compute` takes `&mut self` and pushes EVERY action it scores into
/// `action_history`. Selection is argmin over all candidates, so all candidates are
/// scored and pushed each epoch, leaving counts near-identical. `compute_novelty`
/// is `1/(1+count)`, so it cannot discriminate.
#[test]
fn characterize_novelty_term_flattens_after_one_enumeration() {
    let (mut efe, model, state) = fixture();

    // Epoch 1: every candidate is unseen, so novelty is uniform at 1/(1+0).
    let epoch1 = score_all(&mut efe, &model, &state, 0..NUM_ACTIONS);
    for (action, _, novelty, _) in &epoch1 {
        assert_eq!(*novelty, 1.0, "action {action} should start unseen");
    }

    // Epoch 2: every candidate has been seen exactly once, so novelty is uniform again.
    let epoch2 = score_all(&mut efe, &model, &state, 0..NUM_ACTIONS);
    let first = epoch2[0].2;
    for (action, _, novelty, _) in &epoch2 {
        assert_eq!(
            *novelty, first,
            "action {action} novelty differs — enumeration no longer flattens the term, \
             which would be a real improvement. See docs/EFE_DISPATCH_GATE_2026-07-31.md."
        );
    }
    assert!(
        first < 1.0,
        "novelty should have decayed after one full enumeration"
    );
}

/// DEFECT. `compute` is impure: novelty counts actions CONSIDERED, not actions TAKEN.
///
/// `compute` takes `&mut self` and pushes into `action_history` while merely
/// *scoring* a candidate. So an action that is evaluated and rejected — every
/// epoch, forever — becomes progressively less "novel" despite never being
/// executed. That inverts the term's purpose: it is supposed to encourage
/// exploring actions you have not tried.
///
/// This is the same defect class as the standing rule that
/// `HdcLtcBridge::train_step` / `predict_forward` must stay pure w.r.t. live state
/// (MASTER_ROADMAP, Signal integrity). That rule should extend to this type.
///
/// Note on scope, recorded because the first version of this test got it wrong:
/// this is NOT enumeration-order dependence within one epoch. `compute_novelty`
/// counts only occurrences of the queried action, and scoring *other* actions does
/// not change that count, so a single-pass enumeration is order-independent.
/// The defect is considered-vs-taken, plus eviction from the 100-entry buffer once
/// enumeration exceeds it (~8.3 epochs at 12 candidates).
#[test]
fn characterize_novelty_counts_considered_not_taken() {
    let (mut efe, model, state) = fixture();

    // Score action 3 repeatedly without ever "taking" it — a rejected candidate.
    let first = efe.compute(3, &state, &model).novelty;
    for _ in 0..4 {
        let _ = efe.compute(3, &state, &model);
    }
    let after_rejection = efe.compute(3, &state, &model).novelty;

    assert_eq!(first, 1.0, "action 3 should start maximally novel");
    assert!(
        after_rejection < first,
        "novelty for a never-taken action should have decayed under the current \
         (defective) implementation: {after_rejection} !< {first}. If it did not, \
         `compute` may have been made pure — see docs/EFE_DISPATCH_GATE_2026-07-31.md."
    );

    // Quantify it, so a fix that only partially addresses this is still visible.
    assert_eq!(
        after_rejection,
        1.0 / 6.0,
        "expected 1/(1+5) after five prior scorings of the same rejected action"
    );
}

// ---------------------------------------------------------------------------
// Aspirational: what a usable dispatch policy needs. Enable when the defect is fixed.
// ---------------------------------------------------------------------------

/// The gate a real EFE arm must clear before any simulation study is worth running:
/// the epistemic term must actually discriminate between candidate actions.
///
/// Threshold from `docs/EFE_DISPATCH_GATE_2026-07-31.md` revival condition 1 —
/// measured on UNWEIGHTED term values, so a tuned `pragmatic_weight` cannot make it
/// pass or fail mechanically.
#[test]
#[ignore = "known defect: epistemic term is action-invariant. See docs/EFE_DISPATCH_GATE_2026-07-31.md"]
fn aspirational_epistemic_term_varies_across_actions() {
    let (mut efe, model, state) = fixture();
    let scored = score_all(&mut efe, &model, &state, 0..NUM_ACTIONS);

    let epistemic: Vec<f64> = scored.iter().map(|(_, e, _, _)| *e).collect();
    let pragmatic: Vec<f64> = scored.iter().map(|(_, _, _, p)| *p).collect();

    let sd = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };

    let ratio = sd(&epistemic) / sd(&pragmatic).max(f64::EPSILON);
    assert!(
        ratio >= 0.02,
        "SD(epistemic)/SD(pragmatic) = {ratio:.4}, below the 0.02 gate — the epistemic \
         term does not meaningfully discriminate between actions"
    );
}
