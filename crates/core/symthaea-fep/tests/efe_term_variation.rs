//! Characterization and regression tests for `ExpectedFreeEnergyComputer`'s term behaviour.
//!
//! The epistemic-term characterization still records a known defect. Novelty-history
//! tests below instead freeze the repaired rule that candidate evaluation and
//! selection are pure and only an explicit action commitment consumes novelty.
//!
//! Context: `docs/EFE_DISPATCH_GATE_2026-07-31.md`. A planned 18-day fleet-dispatch
//! experiment was halted after defects were found in both epistemic and novelty
//! semantics. These tests preserve the still-open epistemic finding while preventing
//! regression of the repaired considered-vs-committed novelty boundary.
//!
//! If the epistemic characterization fails because that term becomes genuinely
//! action-dependent, promote the matching `aspirational_*` test and update the audit.

use symthaea_fep::free_energy::ExpectedFreeEnergyComputer;
use symthaea_fep::generative_model::GenerativeModel;
use symthaea_fep::types::HiddenState;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

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
    efe: &ExpectedFreeEnergyComputer,
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
    let (efe, model, state) = fixture();
    let scored = score_all(&efe, &model, &state, 0..NUM_ACTIONS);

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

/// Generic construction has no action semantics, so the pragmatic term must not
/// differ merely because candidate actions have different integer identities.
/// Equality here records explicit ignorance, not evidence that real actions have
/// identical consequences. Confirmed transition learning or caller-supplied domain
/// priors may legitimately break this equality later.
#[test]
fn generic_action_prior_gives_equal_pragmatic_values() {
    let (efe, model, state) = fixture();
    let scored = score_all(&efe, &model, &state, 0..NUM_ACTIONS);

    let first = scored[0].3;
    for (action, _, _, pragmatic) in &scored {
        assert_eq!(
            *pragmatic, first,
            "generic action {action} acquired an unsupported action-specific pragmatic prior"
        );
    }
}

/// Candidate scoring is pure: enumerating actions cannot make them less novel.
#[test]
fn novelty_does_not_decay_from_candidate_enumeration() {
    let (efe, model, state) = fixture();

    let epoch1 = score_all(&efe, &model, &state, 0..NUM_ACTIONS);
    for (action, _, novelty, _) in &epoch1 {
        assert_eq!(*novelty, 1.0, "action {action} should start unseen");
    }
    assert!(efe.action_history.is_empty());

    let epoch2 = score_all(&efe, &model, &state, 0..NUM_ACTIONS);
    for (action, _, novelty, _) in &epoch2 {
        assert_eq!(
            *novelty, 1.0,
            "merely reconsidering action {action} must not consume novelty"
        );
    }
    assert!(efe.action_history.is_empty());
}

/// Repeated rejected-candidate scoring never enters committed-action history.
#[test]
fn novelty_counts_committed_actions_not_considered_actions() {
    let (mut efe, model, state) = fixture();

    let first = efe.compute(3, &state, &model).novelty;
    for _ in 0..4 {
        let _ = efe.compute(3, &state, &model);
    }
    let after_rejection = efe.compute(3, &state, &model).novelty;

    assert_eq!(first, 1.0, "action 3 should start maximally novel");
    assert_eq!(
        after_rejection, first,
        "a never-committed action must not lose novelty merely from scoring"
    );
    assert!(efe.action_history.is_empty());

    efe.record_committed_action(3);
    assert_eq!(
        efe.compute(3, &state, &model).novelty,
        0.5,
        "one explicit commitment should contribute exactly one novelty-history entry"
    );
    assert_eq!(
        efe.action_history.iter().copied().collect::<Vec<_>>(),
        vec![3]
    );
}

/// Scoring the same candidate set in either order is stable and side-effect free.
#[test]
fn candidate_scoring_is_order_stable_without_history_mutation() {
    let (mut efe, model, state) = fixture();
    efe.record_committed_action(2);
    efe.record_committed_action(2);
    efe.record_committed_action(7);
    let history_before = efe.action_history.clone();

    let mut forward = score_all(&efe, &model, &state, 0..NUM_ACTIONS);
    let mut reverse = score_all(&efe, &model, &state, (0..NUM_ACTIONS).rev());
    forward.sort_by_key(|row| row.0);
    reverse.sort_by_key(|row| row.0);

    assert_eq!(forward, reverse);
    assert_eq!(efe.action_history, history_before);
}

/// Selection alone is not commitment; current compatibility `act()` is.
#[test]
fn selection_does_not_consume_novelty_until_commitment() {
    let config = ActiveInferenceAgentConfig {
        state_dim: STATE_DIM,
        obs_dim: OBS_DIM,
        num_actions: NUM_ACTIONS,
        ..Default::default()
    };
    let mut agent = ActiveInferenceAgent::new(config);

    assert!(agent.efe_computer.action_history.is_empty());
    let selection = agent.select_action();
    assert!(
        agent.efe_computer.action_history.is_empty(),
        "selection must not consume novelty before commitment"
    );

    let _ = agent.act(selection.action);
    assert_eq!(agent.efe_computer.action_history.len(), 1);
    assert_eq!(
        agent.efe_computer.action_history.back().copied(),
        Some(selection.action)
    );
}

/// Novelty history retains the existing bounded 100-entry policy.
#[test]
fn committed_action_history_remains_bounded() {
    let (mut efe, _, _) = fixture();
    for action in 0..105 {
        efe.record_committed_action(action);
    }
    assert_eq!(efe.action_history.len(), 100);
    assert_eq!(efe.action_history.front().copied(), Some(5));
    assert_eq!(efe.action_history.back().copied(), Some(104));
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
    let (efe, model, state) = fixture();
    let scored = score_all(&efe, &model, &state, 0..NUM_ACTIONS);

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
