# Butlin Probe Qualification v1 — Proposal

**Status**: proposal, not implemented. Written as the scoped-down first phase of a larger
"evidence campaign" idea, deliberately cut down from a much larger 5-phase/9-field/9-architecture
vision (dependency graphs, welfare-risk matrix, theory-conflict lanes, multi-architecture
baselines) to the single highest-value question: **do the existing probes actually respond to
anything, and can they tell their intended mechanism apart from an unrelated disruption?**
Everything larger is deferred until this lands and produces a result.

**Explicitly separate from PR #30.** PR #30 is the evidence *language* (tiers, outcomes, the
merge's integrity guarantees) and stays frozen pending arkh-node's review. This is a proposed
follow-up PR/issue that *uses* that language once it exists — it does not reopen PR #30.

## Why this, not a bigger framework, and not yet

The internal motivation is already sufficient without appeal to external literature: this
session's own work on PR #30 found frozen/insensitive measurements, unknown fallback provenance,
a near-miss on misclassifying a genuine null result as a measurement failure, derived flags that
disagreed with raw data, and a benchmark-validity gap that could have manufactured functional
evidence from a broken measurement. Every one of those was found by testing the evidence
*infrastructure*. None of them tested whether the *probes themselves* are trustworthy — that's
what this campaign is for. (Any external citations on the general calibration problem in
artificial-consciousness indicators are deliberately not invoked here until independently
verified — the internal case stands on its own.)

## Scope: 12 of 14 indicators, not 14

Excluded, with reason:
- **GWT-1** — a derived aggregate of the other 11 behavioral signals (`specialization_fraction`
  in `indicators.rs`), not an independent probe. Qualifying its inputs qualifies it by
  construction; a separate "aggregate" campaign doesn't add information here.
- **HOT-4** — already has an internal responsiveness test (the smoothness probe checks
  dissimilarity growth across several perturbation magnitudes within one call). It's the one
  indicator that already does what this whole campaign is trying to add to the other 12.

## What already exists (grounded in `ablation.rs`'s actual 12 specs)

| Indicator | Lever (`config_mutator`) | Downstream benchmark |
|---|---|---|
| RPT-1 | `cfc_config.num_neurons = 1, input_dim = 1` | `WorM::N-back` |
| RPT-2 | `enable_cross_modal_binding = false` | `WorM::ChangeDetection` |
| GWT-2 | `enable_gwt = false` | `WorM::N-back` |
| GWT-3 | `enable_gwt = false` | `WorM::N-back` |
| GWT-4 | `enable_phi_attention = false` | `WorM::SpatialUpdating` |
| HOT-1 | `enable_predictive_processing = false` | `CogBench::TwoStep` |
| HOT-2 | `enable_meta_cognition = false` | `WorM::N-back` |
| HOT-3 | `enable_online_learning = false` | `CogBench::InstrumentalLearning` |
| AST-1 | `enable_attention_schema = false` | `WorM::N-back` |
| PP-1 | `learning_threshold = f32::MAX` | `WorM::N-back` |
| AE-1 | `enable_trajectory_planning = false` | `CogBench::TwoStep` |
| AE-2 | `enable_embodied_cognition = false` | `WorM::SpatialUpdating` |

**Two structural problems this table exposes, both must be resolved before the campaign runs,
not discovered after:**

1. **GWT-2 and GWT-3 use the identical lever** (`enable_gwt = false`). They are not independently
   testable with the current config surface — an "unrelated ablation" sham for one of them cannot
   be "the other GWT indicator's ablation," because it's the same intervention. Either find a
   second, genuinely distinct GWT-capacity-specific knob (e.g. a working-memory-capacity-only
   toggle that leaves broadcast alone, if one exists or can be added), or explicitly pre-register
   that GWT-2/GWT-3 will be reported as a **non-independent pair** — a single combined
   qualification result, not two — rather than silently treating them as separately validated.
2. **Six of the twelve indicators share `WorM::N-back` as their downstream benchmark**
   (RPT-1, GWT-2, GWT-3, HOT-2, AST-1, PP-1). If *any* sufficiently disruptive ablation degrades
   N-back generally (a real risk — N-back needs working memory, attention, and prediction all at
   once), a sham/unrelated ablation for one of these six needs to be checked against N-back too,
   not just against that indicator's own probe. If the sham *also* degrades N-back, that's
   evidence the shared benchmark can't cleanly support `FunctionallySupported` for any of the six
   without a benchmark that isolates the targeted mechanism more specifically — a real, likely
   finding this campaign should expect and report honestly rather than paper over.

## Per-probe design: three conditions, not the full nine-field contract

For each of the 12:

1. **Positive control** — a manipulation expected to move the probe's raw measured quantity
   directly, independent of whether the *targeted* ablation lever is correctly wired. Default
   candidate applicable to most of the 12: force a degenerate/frozen input stream (e.g. the same
   constant `ContinuousHV` every cycle, or an equivalent already-available "no real signal"
   condition) through the full cognitive loop and confirm the probe reads differently than under
   normal varied input. This doesn't have to be theoretically meaningful — it only has to prove
   the measurement pipeline is capable of producing a different number under *some* known
   perturbation. A few probes (anything gated on a discrete flag with a hard on/off, e.g. AST-1)
   may need a bespoke positive control instead of the generic degenerate-input one; decide
   per-probe during implementation, not in this proposal.
2. **Targeted ablation** — the existing lever from the table above, unchanged.
3. **Sham / unrelated ablation** — reuse a *different* indicator's existing lever as the
   disruptive-but-untargeted control (e.g. RPT-1's sham could be `disable_metacognition`;
   HOT-2's sham could be `disable_cross_modal_binding`). Pick pairings during implementation such
   that no indicator's sham is a lever that plausibly shares a mechanism with it (e.g. don't use
   `enable_gwt=false` as anyone's sham, since two real indicators already depend on it).

## Classification rules

| Positive control | Targeted effect | Sham effect | Result |
|---|---|---|---|
| Fails | any | any | `Inconclusive` |
| Passes | no meaningful effect | any | `NotDemonstrated` |
| Passes | expected effect | similar-sized | Causal claim withheld — nonspecific |
| Passes | expected effect | clearly smaller | Candidate `CausallySupported` |
| Passes | opposite/wrong-direction effect | small | `Contradicted` |
| Passes | expected effect | small, plus real downstream-benchmark degradation | Candidate `FunctionallySupported` |

This closes the exact gap PR #30's own audit surfaced: `NotDemonstrated` is only a credible claim
once the positive control proves the probe *could* have shown an effect if there were one to show.

## Pilot before scale

Start with **3-5 deterministic seeds**, not 20+. The pilot answers: is the intervention wired
correctly, is the probe responsive at all, is the effect direction stable, is variance too large
for the design as specified. Only after the pilot should a larger seed count be preregistered,
sized to the dispersion actually measured — not guessed in advance. Committing to a large seed
count before the pilot risks spending significant compute re-confirming that a probe or
intervention is malformed, exactly the failure mode a pilot exists to catch cheaply.

## Correction to the compute-cost concern from the prior discussion

The earlier "75-minute build" number was queue contention on a heavily-loaded shared box, not the
per-seed cost of an experiment — it should not be read as "20 seeds × 75 minutes." The actual
workflow should be:

1. Compile one fixed binary/example (`cargo build --release` once), recording its commit SHA and
   config hash.
2. Run all seeds against that one compiled artifact — no rebuild per seed.
3. Control experiment concurrency (how many seeds run in parallel) separately from compilation
   concurrency (`cargo-gate.sh` governs the latter, not the former) — likely via a simple
   sequential loop or a small job count, sized to what the shared box can absorb without adding to
   the contention this session spent time diagnosing.

## Explicitly deferred (not in this PR/issue)

- Multi-architecture baselines (stripped-down Symthaea, transformer-agent wrapper, randomized
  control, etc.)
- Full dose-response curves (graded intervention strength beyond binary on/off)
- Welfare-risk matrix
- Substrate/theory-uncertainty and theory-conflict lanes
- Dependency-graph infrastructure between indicators/probes/signals/mechanisms
- Any public framework rename/rebrand (e.g. "Butlin Operationalization Matrix") — a naming
  decision for the project owner, not something to pick unilaterally, same pattern as the
  `nix-mind`/`nixward` naming decision elsewhere in this monorepo
- Any publication-grade consciousness claim of any kind

## What "done" looks like for this first pass

- 12 indicators (10 independent + the GWT-2/GWT-3 pair reported honestly as non-independent, or
  as a single combined result if no second lever is found), each with a positive control, targeted
  ablation, and one sham/unrelated ablation.
- 3-5 pilot seeds, with the six N-back-sharing indicators' sham results specifically checked
  against N-back degradation, not just their own probe.
- Raw evidence bundles retained for every result, including negative ones — no silent dropping of
  an indicator that fails qualification.
- A written summary of which of the 12 (or 11, depending on the GWT resolution) probes are
  actually qualified, which are `Inconclusive` for lack of a working positive control, and which
  show non-specific (sham ≈ targeted) effects — this is expected to downgrade some of PR #30's
  current `CausallySupported`/`FunctionallySupported` verdicts, and that's the point: a probe that
  was never shown to be capable of responding shouldn't have been trusted to report a null result
  either.
