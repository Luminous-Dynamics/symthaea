# Predictive Compression Program: does Symthaea learn structure per bit? (Plan + C1 pre-registration)

**Program plan registered 2026-07-17.** Experiment C1's pre-registration (§5) is stamped now,
BEFORE any harness exists or any run has happened. Later experiments (C2–C4) are *scoped* here
but each gets its own registration stamp in this doc before its first run.

**Thesis under test, honestly scoped:** intelligence requires learning compact models that
preserve what matters for prediction — cross-entropy is the bridge between prediction and
compression (`H(p,q) = H(p) + D_KL(p‖q)`; only the mismatch term is reducible). We do NOT claim
compression = intelligence, and we do NOT claim low cross-entropy proves consciousness. The
program's question is narrower and falsifiable: **does the loop's temporal state save predictive
bits over trivial baselines, and can that signal be used to select memories and calibrate
confidence?** Naming restraint: no "adaptive causal compression" branding in VISION.md or
public docs until at least one bits-saved number exists (lesson of the ETHICS 94.5% → 56.2%
retraction).

Companion docs: `KEYSTONE_AB_PROTOCOL_2026-07-17.md` (whose Phase-3 finding — episodic
anticipation ABSENT — this program directly targets), `PHI_SIGNAL_TRACE_2026-07-15.md` (whose
scale-pathology findings dictate this program's metric design).

---

## 1. Ground truth (surveyed 2026-07-17, file:line-verified)

### The prediction path (what exists)
- Predict → observe → error with one cycle of lag: `get_multi_scale_prediction`
  (`src/cognitive_loop/prediction.rs:23-193`) averages `predict_forward` over 3 horizons
  (0.02/0.1/0.2 s); raw per-horizon predictions are retained (`prediction.rs:20-22`).
- The predicted object is a **256-dim point in the compressed CfC space** (sparse Rademacher
  projection from 16,384-D HDC, `predictive_encoder.rs:670-700`), NOT the full HDC state.
- PE = scale-invariant unit-vector distance `0.5·‖â − p̂‖ = sqrt((1−cosθ)/2)`, clamped [0,1]
  (`predictive_encoder.rs:504-544`). Uncorrelated ≈ 0.707. Degenerate cases return a tagged 1.0
  sentinel. Both vectors are simultaneously in hand exactly here — the natural insertion point.
- **No NLL/cross-entropy/bits formulation exists anywhere on this path.** `symthaea-fep`'s
  Gaussian accuracy term (`free_energy.rs:45-93`) is the right shape but is wired to a 4-dim
  summary-statistics observation (`cycle_extracted.rs:728-742`), not the prediction space.
- Scale is NOT trustworthy in this space (PHI_SIGNAL_TRACE Symptom 3: attention-scaled training
  targets vs plain encodings; output-norm degeneracy ~3e-8). This kills naive Gaussian NLL over
  raw residuals — it would measure the scale mismatch, not information.

### Controls that already exist
- Frozen weights: `learning_rate = 0` (provably state-identical to an untrained twin given the
  purity guarantees, `hdc_ltc_bridge.rs:935-938`); also `pause_learning` (`training.rs:55`).
- Frozen state: `snapshot_evolution_state()`/`restore_evolution_state()`
  (`hdc_ltc_bridge.rs:752-760`) — restore a fixed snapshot each cycle.
- Memoryless: `inject()` on the HdcLtc backend is a network reset (`prediction.rs:84-92`) —
  zero-inject each cycle.
- Determinism: `genesis_phrase` seeding + `async_training = false` (keystone pattern).
- Regime battery: `examples/exp_loop_ablation.rs` §E2 — 4 regimes (repetitive / varied /
  alarming / empty) × 500 cycles. Keystone's deterministic 12-sentence learning script is the
  other input source.

### Episodic memory (what the compression gate would change)
- Write gate today (bridge store): `Φ ≥ 0.2 AND (PE > 0.1 OR flow)` (`helpers/parallel.rs:126-142`).
  Replay-PQ gate: coherence ≥ 0.3 (`episodic_replay.rs:491-497`) — the "Φ-weighted PQ" is
  actually **coherence**-weighted (`cycle_phases_memory.rs:353-354` feeds `smoothed_coherence`).
  PE already contributes 0.2 weight to episode priority (`episodic_replay.rs:217-251`).
- **Recall never feeds prediction content.** Recalled episodes only nudge scalars
  (`cycle_extracted.rs:405-455`); the planning path has zero episodic references. On the default
  `HdcLtcUnified` backend the replay-training consumer doesn't even exist
  (`cycle_phases_memory.rs:445` matches only `TemporalNetwork::CfC`). Keystone Phase 3's
  "stores but never predicts" is confirmed and stronger than stated.
- Ordering favors a compression gate: training (Phase 2) completes before the episodic write
  (Phase 3) in the same cycle, and `training_loss` is already plumbed to within reach of both
  write sites (`phase_results.rs:135`, `cycle_phase_feedback.rs:180-183`). Missing: exactly one
  pure post-update eval (loss-before vs loss-after on the same snapshot) — `train_step_impl`
  computes loss only pre-gradient (`hdc_ltc_bridge.rs:658-676`). The async-trainer path returns
  `training_loss = None` — no signal there.

### Calibration (what C4 builds on)
- All ingredients compile in default builds (`magi_loop` is transitively in default-mind via
  `reasoning_engine` — the "not in default" claim was already retracted, AGW_PLAN:529-532):
  `softmax_with_temperature` (`symthaea-core/src/math.rs:37-61`), `analyze_pairs`
  (`calibration_analytics.rs:49-107` — Murphy decomposition + ECE + reliability bins over plain
  `(confidence, correct)` pairs), `BrierScoreTracker` (`calibration.rs:408-693`).
- β-softmax over HDC similarities literally exists (`modern_hopfield.rs:224-245`) but β is a
  retrieval knob, never outcome-calibrated. Perception classification is argmax + affine rescale
  (`src/mind/intent.rs:997-1050`) — no distribution.
- Self-grading precisely located: facade Phase 7.5 resolves its own predictions from a lexical
  self-check (`mod.rs:2186`, `:3034+`); the real resolver machinery (`resolution.rs`) is unused
  on that path. The world-graded pattern to copy is AGW 2.2's ToolUse predictions
  (`mod.rs:2380-2431`, resolved by actual exit codes).
- **Live decorative-metric instance (fix opportunity):** the loop's `knowledge_calibration_ece`
  has no production feeder (`manager.rs:1000-1019` — test-only callers), so `ece()` returns the
  `total_samples == 0` sentinel `0.0` and the `(1 − ece)` factor at
  `cycle_phase_feedback.rs:1061` is always 1.0. Same absent-vs-zero class fixed by `b97ed86042`
  elsewhere; unfixed here.

---

## 2. Metric design (fixed for the whole program)

**Bits saved = relative NLL vs a trivial baseline, under one shared online residual model.**
Absolute per-cycle NLL is declared out of scope until the prediction-space scale pathologies are
resolved — with a shared residual model, Gaussian normalization constants cancel between model
and baseline, so no discretization convention is needed and the quantity is well-defined.

- **Space**: unit-normalized 256-dim compressed encodings (the same normalization PE already
  uses). Residual modeled on the sphere with a single online concentration parameter
  (EMA-estimated, same pattern as `stats.avg_prediction_error_sq`). With one shared κ,
  `bits_saved` reduces to a monotone affine function of `(cos_model − cos_baseline)` scaled by
  the estimated concentration — cheap, per-cycle, honestly interpretable. (A von Mises–Fisher
  log-normalizer is the principled upgrade if ever needed; not required while κ is shared.)
- **Prediction**: the **delta_t-horizon raw prediction alone** (from `raw_predictions`), NOT the
  3-horizon average — the average is not a calibrated one-step forecast.
- **Baselines** (both computed every cycle from data already in hand):
  B-persist: `enc_{t−1}` predicts `enc_t` (the natural one — it is literally the training pair).
  B-zero: zero vector (degenerate-similarity floor; mostly a sanity rail).
- **Recording**: new fields on `CycleResult` (`types/output.rs`) and `CycleMetadata` (via the
  established `symthaea-cognitive-types` flattening pattern): `bits_saved_persist`,
  `bits_saved_zero`, plus the online κ. Measurement-only — no feedback into any control path
  during this program (a bits-saved-driven attention/gating loop is explicitly future work).
- **Cost/energy claims: excluded from the whole program.** Keystone P4 was invalidated by
  ambient load from concurrent sessions; per-cycle µs and joules are not measurable on this box.

---

## 3. Program phases

| Phase | What | Nature | Depends on |
|---|---|---|---|
| **P0** | Instrument `bits_saved` (metric above) + unit tests + harness example `examples/compression_bits.rs` | measurement-only code | nothing |
| **C1** | "Does the liquid state compress?" — live vs frozen-weights vs memoryless, across regimes (§5, registered now) | experiment | P0 |
| **C2** | Compression-gated Chronicle: pure post-update eval (`eval_loss_from` on `HdcLtcBridge`), store `bits_saved_by_update` on `Episode`, A/B the write gate | small mechanism + experiment | C1 positive; **keystone Phase-4 session finished** (touches `hdc_ltc_bridge.rs`) |
| **C3** | Episodic anticipation: minimal content-level recall→prediction path; re-run an order-sensitivity probe (new example, NOT `keystone_ab.rs`) | capability + experiment | C2 |
| **C4** | Calibrated surprise: β-calibrated softmax over intent-class similarities, fit by `analyze_pairs` ECE on held-out outcomes; feed the dead `knowledge_calibration_ece` slot with real records (closing the absent-vs-zero instance) | mechanism + fix | independent of C1-C3 |

**Kill criteria between phases:** C2 proceeds only if C1's H1 is at least SUGGESTIVE — if the
live state saves no bits, gating memory on model-update bits is premature. C3 proceeds
regardless of C2's A/B verdict (the anticipation gap stands on keystone evidence alone), but its
recall machinery should prefer compression-selected episodes only if C2 supported them.

**Non-goals for this program:** absolute NLL; energy terms; value-sensitivity (AGW's
territory); facade/loop unification (Seam C); any change to Broca or the epistemic gates;
bits-saved as a live control signal.

---

## 4. Coordination & hazards (2026-07-17)

- **Another session is running keystone Phase-4 acceptance right now.** Do NOT touch
  `examples/keystone_ab.rs`, `docs/KEYSTONE_AB_PROTOCOL_2026-07-17.md` (its Phase-4 results slot
  belongs to that session), or the snapshot/train-path code in `src/hdc_ltc_bridge.rs` and
  `cycle_phase_dynamics/training.rs` until that run completes. This is why C2 (the only phase
  needing a `HdcLtcBridge` change) is sequenced behind it. P0's insertion points
  (`predictive_encoder.rs`, `types/output.rs`, telemetry) are outside the frozen set.
- **Purity is a hard constraint**: `train_step`/`predict_forward` are pure w.r.t. live state and
  guarded by twin/purity regression tests (`train_step_does_not_perturb_live_evolution_state`,
  `hdc_ltc_bridge.rs:939-985`). C2's post-update eval must use the same scratch-snapshot
  pattern; P0 adds no state at all.
- Determinism check for P0: adding measurement fields must not perturb trajectories — verify by
  bit-identical `CycleResult.prediction_error` streams before/after instrumentation on one seed.
- Commit discipline: explicit pathspecs, commit per phase, this doc updated in place with
  results (house convention).

---

## 5. Experiment C1 — pre-registration (stamped 2026-07-17, before any run)

### Question
Does the loop's evolving temporal state (CfC/HdcLtc) save predictive bits over (a) its own
frozen twin and (b) a memoryless control, at matched inputs and compute — and does the saving
track input structure (regimes) rather than being a constant offset?

### Arms (construction-time config, deterministic per (arm, seed))
| Arm | Definition |
|---|---|
| `live` | default config (HdcLtcUnified backend), `async_training = false` |
| `frozen` | identical, `learning_rate = 0` (weights never update; state still evolves) |
| `memoryless` | identical to `live`, but temporal state zero-injected every cycle (network reset per `prediction.rs:84-92`) |

Declared limitation up front: no ConsciousnessEngine kill-switch exists (keystone limitation
inherited); the measurement spine stays on in all arms. `frozen` isolates *learning*;
`memoryless` isolates *state carryover*; their contrast decomposes the CfC's contribution.

### Tasks & metrics (externally scored — no self-grading)
- **Inputs**: the E2 regime battery (repetitive / varied / alarming / empty), 500 cycles each,
  WARMUP 100 / MEASURE 400, run in a fixed order within one service instance per (arm, seed) —
  plus the keystone 12-sentence learned-script schedule (60 reps) as a fifth block.
- **Primary endpoint**: `Δbits(arm) = mean bits_saved_persist over MEASURE cycles`, per regime.
- **Secondary**: `learning_delta` in bits on the learned-script block = mean bits_saved_persist
  (reps 2–4) minus (reps 55–60) — does saving *grow* with exposure in `live` but not `frozen`?
- **Manipulation check**: B-zero baseline must be beaten by all arms in all non-empty regimes
  (if not, the metric itself is broken — abort and diagnose before interpreting anything).
- **Seeds**: 10, named `compression-c1-seed-{alpha..kappa}-2026-07-17`, hardcoded in the harness.
- **Statistics**: 10-seed sign test per comparison (CONFIRMED ≥9/10, SUGGESTIVE 7–8/10, NOT
  SUPPORTED ≤6/10); per-seed magnitude gate = |difference| > cross-seed spread of the `live` arm.
- **Cost**: not measured (pre-declared exclusion; ambient load).

### Pre-registered predictions
- **P1 (state carries information)**: `live` > `memoryless` on Δbits in the `varied` regime,
  ≥9/10 seeds. This is the CfC-temporal-compression claim made quantitative.
- **P2 (learning adds on top of state)**: `live` > `frozen` on the learned-script
  `learning_delta`, ≥7/10 seeds. (Weaker gate: keystone found the learning effect real but
  carried by a single subsystem; magnitude may be small.)
- **P3 (structure-tracking, not offset)**: Δbits(`live`) in `repetitive` > Δbits(`live`) in
  `empty`, ≥9/10 seeds — bits saved must track available structure. If Δbits is flat across
  regimes, the number is an artifact of the residual model, not a measurement of compression.
- **P4 — the honest risk**: `live` ≈ `memoryless` everywhere (≤6/10 seeds distinguish). If this
  holds, the sentence entering VISION.md is: "The temporal state saves no measurable predictive
  bits over a memoryless control; the 'temporal compression' description of the CfC layer is
  aspirational pending a stronger training signal." C2 is then deferred and the program pivots
  to diagnosing why (training-signal strength, projection loss, or horizon mismatch).

### What would change our minds
- P1+P3 confirmed → the bits-saved ledger is real; proceed to C2 and report the first
  "structure per bit" number.
- P1 confirmed but P3 not → the metric is measuring residual-model artifacts; fix the metric
  before any downstream use (do NOT proceed to C2 on P1 alone).
- P4 (honest risk) → see above; also re-examine whether keystone's PE-based learning effect and
  this bits measure are even measuring the same thing (they should be monotonically related —
  check as an unregistered diagnostic).

### Amendment 1 (registered 2026-07-17, later the same day, still BEFORE any harness or run)

The keystone Phase-4 acceptance completed with a **self-retraction** (see
`KEYSTONE_AB_PROTOCOL_2026-07-17.md` Phase-4 results): the Phases 1-2 suggestive benefit and
the `temporal_consciousness` sole-carrier attribution were artifacts of the state-scrambling
trainer (fixed `e3f1104413`); post-fix arm differences collapsed to ~0.000 and all three
pre-registered acceptance gates failed. Standing truth: **the loop as trained does not learn
sequences in any configuration** — binding constraint is the training signal itself.

Consequences for this registration (predictions and gates UNCHANGED; priors and framing
updated):
1. **P2's rationale is retracted.** The parenthetical in P2 citing "keystone found the learning
   effect real but carried by a single subsystem" no longer holds. P2 is kept exactly as
   registered, but its prior is now strongly negative: the expected outcome is P2 NOT
   SUPPORTED, which would *replicate the keystone retraction in bits units* — independent
   confirmation with a different metric, which is worth having. If P2 unexpectedly succeeds,
   that is a tension with the keystone result and must be investigated, not celebrated.
2. **P1 and P3 are unaffected.** State carryover (live vs memoryless) is not learning; a
   reservoir with frozen or weakly-trained weights can still carry predictive information in
   its evolving state. P1 is now the program's most informative question.
3. **P4 (the honest risk) is more live than at registration.** If both the learning path AND
   state carryover contribute nothing, the "temporal compression" story has no measured support
   at all, and the program pivots to the same binding constraint keystone named: the training
   signal.
4. **C2 is UNBLOCKED** — the keystone session has released `hdc_ltc_bridge.rs`. The C2
   sequencing note in §3/§4 is superseded; the C1-positive precondition still applies.
5. **Metric estimator finalized** (was left as "EMA-estimated" at registration; fixed now,
   before any code exists): shared concentration κ = 1/σ², where σ² is an EMA (α = 0.01) of
   the persistence-baseline squared residual `‖â − b̂‖²` on unit vectors, floored at 1e-6,
   warm-started over the first 20 cycles (bits fields report None/NaN-guarded 0 until then).
   Per-cycle: `bits_saved_persist = κ·(cos_model − cos_persist)/ln 2`;
   `bits_saved_zero = κ·(cos_model − 0)/ln 2` (a zero prediction has undefined direction;
   its cosine is defined as 0). Using the baseline residual for σ² makes "bits saved" read as
   "relative to how predictable the stream already is from persistence" — the deliberately
   conservative choice.
6. **Bonus alignment**: the keystone follow-up ("stronger training signal, same acceptance
   gates") can reuse this program's C1 harness as one of its acceptance instruments —
   bits_saved gives the training-signal work a sharper endpoint than raw PE deltas.

### Amendment 2 (registered 2026-07-17, before any run — sign convention only)

The §5 secondary-endpoint formula wrote the subtraction as "(reps 2–4) minus (reps 55–60)",
which under a growing signal yields a *negative* number — an awkward convention inherited from
keystone's PE-based delta (where learning makes the metric fall). For bits saved, learning
makes the metric RISE. Clarified, no gate change: the harness reports
**`learning_growth = mean bits_saved_persist(reps 55–60) − mean(reps 2–4)`** (positive = saving
grows with exposure), and P2 reads "live > frozen on `learning_growth`". P0 implementation
note: the harness (`examples/compression_bits.rs`), the `bits_saved_*` fields on
`CycleResult`, and two experiment controls (`freeze_cfc_training()`,
`reset_temporal_state()`) were built after Amendments 1–2 and before any run; `CycleMetadata`
telemetry fields are deferred until C1 shows the signal is worth dashboarding.

### Amendment 3 (registered 2026-07-17, after a `--quick` SMOKE run only — before the
registered full run)

The quick smoke (3 seeds, short blocks — explicitly not the registered run; raw log:
session scratchpad `c1_quick.log`) surfaced one metric-validity bug, one early scientific
signal, and one open diagnostic. Fixed/recorded BEFORE the registered run:

1. **κ blow-up on low-variance streams (metric bug, fixed).** On the repetitive regime the
   persistence residual collapses toward 0, the σ² EMA rides the 1e-6 floor, κ reaches 10⁶,
   and bits report ±1.5 *million* — cross-regime comparisons (P3) would measure κ variation,
   not structure-tracking. Fix: (a) σ² gets an **additive regularizer ε = 1e-3**
   (κ ≤ ~1000) instead of a bare floor; (b) the per-cycle κ is exposed
   (`bits_kappa`) so **Δcos = bits·ln2/κ** — dimensionless, bounded — is recoverable;
   (c) **all P1–P4 verdicts are evaluated on Δcos**, with bits reported as the
   secondary bits-denominated view. Directions are unchanged (κ > 0, so Δcos and bits
   always agree in sign); only the ill-conditioned scale leaves the decision criteria.
2. **Early honest signal (recorded, not a verdict).** In the smoke, even the `live` arm's
   shortest-horizon prediction is roughly *uncorrelated* with the next encoding on the
   fully repetitive stream (Δcos ≈ −1 vs a near-perfect persistence baseline) — consistent
   with the keystone Phase-4 retraction, visible already at smoke scale. P4 remains live.
3. **Coverage gaps are input-locked, not arm-specific (open diagnostic).** All three arms
   show identical coverage: alarming 50/150 (exactly 1-in-3 inputs), varied 126/150,
   script 60/72. Certain inputs yield no first-horizon prediction (`raw_predictions`
   empty or the dynamics path skipped), and each such cycle also costs the *following*
   cycle its baseline (deliberate: a missing prediction drops the baseline rather than
   silently pairing â_t with b̂_{t−2}). Mechanism not yet diagnosed — candidates include
   input-similarity memoization and urgency routing. The coverage column keeps this
   visible; diagnosing it is a pre-C1 follow-up, not a blocker, since coverage is
   identical across arms (no differential bias between arms).

### Amendment 4 (registered 2026-07-18, before the registered run) — coverage diagnostic
RESOLVED: the instrumentation found a real pre-existing perception bug

Amendment 3's open diagnostic is closed, and it was not a bits-path artifact — **the loop was
literally blind (exact zero 16,384-D encoding) to a whole class of inputs**, and had been
since before this program existed. Chain, established by `--probe` per-cycle data plus a
direct encoder probe (`symthaea-core/examples/probe_zero_encoding.rs`):

1. `detect_semantic_patterns` (predictive_encoder.rs) matches trigger words as
   **substrings**: "l**if**e" contains "if" → emits `IMPLICATION`; "melt**do**wn" contains
   "do" → emits `ACTION`.
2. `IMPLICATION` and `ACTION` exist **nowhere in the primitive system** (verified by grep of
   `primitive_system/init_tiers.rs`; `BEFORE`/`AFTER`/`CAUSE`/`EFFECT` do exist) — phantom
   names live only in the pattern detector.
3. `apply_attention_in_place` summed weights via `filter_map(get)` (misses contribute 0) but
   divided by the count of ALL detected names → a phantom-only detection computed
   avg attention = 0/1 = **0**, scaling the entire encoding to an exact zero vector.
4. Downstream: PE reports its documented 1.0 `ZeroPrediction`-class sentinel on such cycles
   (probe: "What is the meaning of a life well lived?" → PE=1.0000 on every repetition;
   same for "Critical failure: coolant pressure dropping, meltdown risk rising!"), the bits
   fields correctly report None rather than fabricating numbers, and the deliberate
   baseline-drop rule costs the following cycle too — producing the exact observed coverage
   fractions (alarming 50/150 = 1-in-3 input + follow-on; varied 126/150; script 60/72).

**Fix (landed with this amendment):** average over *matched* weights only, defaulting to 1.0
when nothing matches; regression test pins both implicated sentences to nonzero encodings.
The C1 registered run MUST use the fixed build — coverage should rise to ~full, and the
affected sentences become genuinely visible to all arms for the first time. Smoke numbers
from Amendment 3 predate this fix and are not comparable.

**Also worth recording:** on blind cycles PE=1.0 exceeded the learning threshold, so the
trainer could fire against a *zero* target encoding — a training-corruption vector on any
stream containing phantom-triggering sentences. This existed before the program and is now
closed by the same fix. A separate follow-up (out of scope here): `detect_semantic_patterns`'
substring matching is overly broad even post-fix ("if" in "life" still detects a phantom
name; it just no longer zeroes perception) — word-boundary matching would be the real cure.

### Results (2026-07-18, run complete, exit 0) — PRE-ANNIHILATION-FIX BASELINE

**Binary provenance, stated up front**: this run used the post-Amendment-4 build (phantom-
primitive perception bug fixed, full 400/400 and 72/72 coverage everywhere — confirms that
fix). It did **not** include the d⁻¹·⁹ signal-annihilation fix to `hdc_ltc_bridge.rs`'s output
projection, which a coordinating session (phi-trace/keystone lane) diagnosed and started
implementing in parallel (`docs/PHI_SIGNAL_TRACE_2026-07-15.md`, follow-up 1, 2026-07-18) —
that fix was uncommitted WIP throughout this run and is **not** in this data. This run is
therefore the pre-registered "before" measurement; a post-fix re-run is the natural next step
(see Follow-ups).

**Coverage**: 400/400 (regime blocks) and 72/72 (script block) for every arm×seed×block — the
Amendment 4 perception fix resolved the coverage gap completely (was 50-150/400, 60/72 pre-fix).

#### P1 — live > memoryless, Δcos (varied regime), 10 seeds

| seed | live | memoryless | live wins? |
|---|---|---|---|
| alpha | −0.01856 | −0.00098 | no |
| beta | −0.00048 | −0.00112 | **yes** |
| gamma | −0.01430 | −0.00328 | no |
| delta | −0.02013 | −0.01482 | no |
| epsilon | −0.02847 | −0.01226 | no |
| zeta | −0.01025 | +0.00163 | no |
| eta | −0.00940 | −0.01878 | **yes** |
| theta | −0.00609 | −0.02286 | **yes** |
| iota | −0.00493 | −0.01900 | **yes** |
| kappa | −0.01739 | −0.02477 | **yes** |

**Tally: 5/10 → NOT SUPPORTED** (registered gate: CONFIRMED ≥9/10, SUGGESTIVE 7–8/10, NOT
SUPPORTED ≤6/10). Essentially a coin flip — the live temporal state's shortest-horizon
prediction is not reliably better than a memoryless (zero-inject-every-cycle) control on
naturalistic varied input.

#### P2 — live > frozen, script-block learning_growth (Δcos, late reps − early reps), 10 seeds

| seed | live growth | frozen growth | live wins? |
|---|---|---|---|
| alpha | +0.00157 | −0.00263 | **yes** |
| beta | +0.00971 | −0.00621 | **yes** |
| gamma | −0.00815 | +0.00681 | no |
| delta | −0.00396 | +0.00249 | no |
| epsilon | −0.00446 | +0.00375 | no |
| zeta | +0.00833 | +0.01056 | no |
| eta | +0.01017 | +0.01757 | no |
| theta | +0.00030 | −0.00321 | **yes** |
| iota | −0.00306 | +0.00152 | no |
| kappa | +0.00900 | +0.01009 | no |

**Tally: 3/10 → NOT SUPPORTED, and reversed** — `frozen` (weights latched, no updates) beats
`live` on 7/10 seeds. Weight updates on this pre-fix binary do not measurably improve
predictive-bits growth with exposure; if anything the direction is mildly unfavorable to
training. This is an independent, bits-denominated replication of the keystone Phase-4
self-retraction ("the loop as trained does not learn sequences in any configuration") — same
conclusion, orthogonal metric (compression bits vs. raw PE).

#### P3 — Δcos(live) repetitive > empty, 10 seeds

| seed | repetitive | empty | diff | live wins? |
|---|---|---|---|---|
| alpha | −1.08496 | −0.96460 | −0.12036 | no |
| beta | −0.92832 | −1.07868 | +0.15036 | **yes** |
| gamma | −1.02421 | −1.02473 | +0.00052 | **yes** (razor-thin) |
| delta | −0.92221 | −0.95175 | +0.02954 | **yes** |
| epsilon | −1.12207 | −1.03618 | −0.08589 | no |
| zeta | −0.98213 | −1.04686 | +0.06473 | **yes** |
| eta | −1.04833 | −1.04953 | +0.00120 | **yes** (razor-thin) |
| theta | −0.91840 | −1.01411 | +0.09571 | **yes** |
| iota | −0.97146 | −0.96098 | −0.01048 | no |
| kappa | −1.05373 | −1.01653 | −0.03720 | no |

**Tally: 6/10 → NOT SUPPORTED** (need ≥9/10). Two of the six "wins" (gamma, eta) are inside
noise (~0.0005–0.0012 vs. a ~±0.1 cross-seed spread) — the real win rate against a magnitude
threshold is closer to 4/10. No reliable structure-tracking: mean Δcos is −1.006 (repetitive)
vs. −1.014 (empty) — nearly identical, both near the metric's natural floor because persistence
is trivially near-perfect on both (a constant stream and a genuinely empty one are both maximally
predictable by "repeat last time," so the *comparison itself* has little discriminating power
here — a real design limitation of using repetitive-vs-empty as the structure-tracking pair, see
Follow-ups).

#### Manipulation check — bits_saved_zero > 0 in non-empty regimes, live arm

| regime | positive/10 |
|---|---|
| repetitive | 5/10 |
| varied | 6/10 |
| alarming | 4/10 |
| script | 5/10 |

**Registered response for a failed check**: "the metric itself is broken — abort and diagnose
before interpreting anything." We do NOT take that path here, for three convergent reasons: (1)
11/11 `predictive_encoder` unit tests pass, including a twin test proving the bits path never
perturbs PE/attention; (2) the persistence-relative metric behaves exactly as expected in scale
and sign on repetitive/empty (large negative — persistence dominates a near-constant stream, as
it should); (3) this run deliberately used the **pre-annihilation-fix** binary, and a coin-flip
on bits_saved_zero (cos_model ≈ 0 on average) is precisely what "output norm 3.0e-10, readout
gradients ~1e-13, untrainable" predicts: the shortest-horizon prediction should carry no
reliable directional signal at all, neither positive nor systematically negative. **Read as
convergent cross-validation, not a broken instrument**: two independent diagnostics — the
phi-trace session's direct magnitude probe (`probe_signal_scale`) and this session's bits-saved
cosine metric — reach the same conclusion (numerically dead prediction) via unrelated code
paths and unrelated math.

#### Verdicts against pre-registered predictions

- **P1: NOT SUPPORTED** (5/10, need ≥9/10)
- **P2: NOT SUPPORTED, REVERSED** (3/10 for live; frozen wins 7/10)
- **P3: NOT SUPPORTED** (6/10, ~4/10 above-noise, need ≥9/10)
- **P4 (the honest risk): SUPPORTED.** The live temporal state is not reliably distinguishable
  from a memoryless control (P1), weight training does not measurably help and may mildly hurt
  (P2), and there is no reliable regime-structure-tracking (P3). The sentence entering VISION.md
  per the pre-registered P4 language: *"The temporal state saves no measurable predictive bits
  over a memoryless control on this pre-fix binary; the 'temporal compression' description of
  the CfC layer is aspirational pending the signal-annihilation fix and a stronger training
  signal."* This independently corroborates, via an unrelated metric, the keystone Phase-4
  self-retraction (`KEYSTONE_AB_PROTOCOL_2026-07-17.md`) — two different measurement programs,
  built by different sessions in the same week, agree that this binary does not learn sequences.

#### Unregistered findings (exploratory, flagged as such)

- Mean Δcos magnitude is regime-dependent for a structural reason unrelated to "learning": in
  repetitive/empty regimes persistence is trivially strong (cos_persist ≈ 1), so any real-world
  residual registers as a huge relative gap (mean ≈ −1.0); in varied/alarming persistence is
  weaker so gaps are naturally smaller (mean ≈ −0.01 to −0.06). This is a property of the metric
  design, not a new finding about the loop, but it means repetitive-vs-empty is a weak choice
  for "does structure move the needle" — see Follow-ups.
- `frozen` beat `live` on the manipulation-check-adjacent script block too (bits_saved_zero
  positive 5/10 for live, not separately tabulated for frozen here) — the pattern that
  weight-freezing doesn't cost anything (and may help) recurs across every measure in this run,
  not just P2's specific endpoint.

#### Follow-ups

1. **Re-run C1 post-annihilation-fix** once the phi-trace session's `hdc_ltc_bridge.rs` fix
   lands and passes its own keystone Phase-4 A/B/C acceptance gates. Since Δcos/PE are
   scale-invariant, the fix's effect on C1 should show up primarily through restored
   training-gradient flow — P2 is the prediction most likely to move; P1/P3 test state
   carryover, which normalization alone doesn't create.
2. **Replace the P3 pair.** Repetitive-vs-empty shares a "persistence trivially wins" floor;
   a sharper structure-tracking test would compare `varied` (persistence genuinely weak,
   structure genuinely present) against `alarming` or a shuffled-`varied` control (structure
   removed at matched persistence-difficulty) rather than against `empty`.
3. If the post-fix re-run still shows P1/P2/P3 not supported, treat "does the loop learn
   sequences at all" as answered (no, twice, by two metrics) and redirect effort to the
   training-signal-strength track keystone's own follow-up already names, before spending more
   cycles on compression-specific mechanism (C2/C3/C4).

### Results (2026-07-22, `--quick` rerun, POST-ANNIHILATION-FIX, N=3 seeds)

Follow-up #1 above, executed. **This is a directional smoke-test, not the confirmatory run** —
`--quick` uses 3 seeds (alpha/beta/gamma) and shorter blocks against the pre-registered ≥9/10
acceptance bar, which needs the full N=10 run to actually clear. Binary built from current HEAD
(includes both the keystone-lane annihilation fix and this session's unrelated
`consciousness_level` overflow fix from `SYMTHAEA_COGNITION_IMPROVEMENT_PLAN_2026-07-21.md` —
the latter doesn't touch the HDC-LTC path C1 measures, noted for the record only).

#### P1 — live > memoryless, Δcos (varied regime), 3 seeds

| seed | live | memoryless | live wins? |
|---|---|---|---|
| alpha | 0.17706 | 0.16521 | yes (+0.01185) |
| beta | 0.14191 | 0.16894 | no (−0.02703) |
| gamma | 0.14266 | 0.09043 | yes (+0.05223) |

2/3. Still mixed — consistent with pre-fix (5/10) in direction, underpowered to say more at N=3.

#### P2 — live > frozen, script-block learning growth, 3 seeds

| seed | live growth | frozen growth | live wins? |
|---|---|---|---|
| alpha | +0.23797 | +0.00107 | yes |
| beta | +0.31089 | +0.00067 | yes |
| gamma | +0.28253 | +0.00312 | yes |

**3/3 — full reversal from the pre-fix result** (was 3/10, frozen winning 7/10). `frozen` is now
flat at essentially zero growth in every seed while `live` shows large, consistent growth —
matches the expectation that restoring gradient flow (the annihilation fix) is exactly what P2
tests. Notably `memoryless` also shows large growth (alpha +0.11323, beta +0.34156, gamma
+0.28851), comparable to `live` — so this growth looks driven by weight training, not by state
carryover specifically; P1 (which isolates carryover, holding training constant) stays mixed.

#### P3 — Δcos(live) repetitive > empty, 3 seeds

| seed | repetitive | empty | repetitive wins? |
|---|---|---|---|
| alpha | −0.17004 | −0.19617 | yes (+0.02613) |
| beta | −0.16372 | −0.29183 | yes (+0.12811) |
| gamma | −0.14566 | −0.45444 | yes (+0.30878) |

**3/3 — also reversed from pre-fix** (was 6/10, ~4/10 above noise). Still subject to the
Follow-up #2 caveat below (repetitive-vs-empty shares a trivial-persistence floor) — a real
result, but not yet the sharper test that caveat calls for.

#### Reading this honestly

Two of three predictions moved sharply in the supportive direction after the same fix that made
Keystone Phase 5 pass its gates for the first time — cross-corroboration between two independent
programs again, this time both post-fix and both positive. But N=3 quick-mode seeds cannot clear
the pre-registered ≥9/10 bar, and P1 (the one prediction that isolates temporal-state carryover
from weight training) is still not clearly supported. **Verdict: promising, not confirmed.** The
full N=10 registered run is the next real step before updating the P1-P4 verdicts above or
VISION.md's benchmark table.

### Results (2026-07-24, full N=10 run, POST-ANNIHILATION-FIX, exit 0) — the confirmatory run

The N=10 follow-up the `--quick` smoke test above called for. Same binary generation (HEAD
includes the keystone-lane scale-restoration fix), full registered protocol: 10 named seeds,
full-length E2 regime blocks (400 cycles, 100 warmup) + the 60-rep learned script. Coverage
400/400 and 72/72 everywhere (Amendment 4's perception fix still holding). Run survived three
infrastructure failures before completing cleanly — logged in
`memory/feedback_background_cargo_gets_killed_mystery.md` (Recurrence 3): a `/tmp` scratchpad
wipe lost the first attempt's raw log entirely (only script-block rows, streamed live, were
recoverable — insufficient for P1/P3), and the retry needed both a `symthaea-humanoid`
build-blocker workaround (P0-#6, `[[bin]]` registration, `MASTER_ROADMAP.md`) and a second
mid-run kill before a clean completion. Final data verified via `exit 0` + row-count check
before analysis; raw log archived outside any session-scoped path
(`~/predictive_compression_c1_postfix_backup/`).

#### P1 — live > memoryless, Δcos (varied regime), 10 seeds

| seed | live | memoryless | live wins? |
|---|---|---|---|
| alpha | +0.26917 | +0.32988 | no |
| beta | +0.29055 | +0.29993 | no |
| gamma | +0.27056 | +0.24283 | **yes** |
| delta | +0.29456 | +0.34777 | no |
| epsilon | +0.26274 | +0.20571 | **yes** |
| zeta | +0.17762 | +0.22368 | no |
| eta | +0.24781 | +0.23672 | **yes** |
| theta | +0.33456 | +0.38414 | no |
| iota | +0.29740 | +0.26887 | **yes** |
| kappa | +0.27197 | +0.30435 | no |

**Tally: 4/10 → NOT SUPPORTED** (need ≥9/10). Confirms and sharpens the `--quick` smoke's
"still mixed" reading (2/3) — at full power, `memoryless` is statistically indistinguishable
from `live`, and its mean is if anything marginally *higher* (mean varied Δcos: memoryless
+0.28439 vs live +0.27169, see aggregate table below). **This is the single most informative
result in the post-fix run.** The scale-restoration fix unlocked real learning (P2, below) —
but that learning is being used almost entirely to model the immediate one-step transition
(`enc_{t−1} → enc_t`, which `memoryless` can learn just as well since it trains normally, it
only forgets state *between* cycles), not to exploit cross-cycle temporal context. Training
now works; genuine use of carried state still doesn't show up as an advantage.

#### P2 — live > frozen, script-block learning_growth (Δcos, late reps − early reps), 10 seeds

| seed | live growth | frozen growth | live wins? |
|---|---|---|---|
| alpha | +0.10966 | +0.00338 | **yes** |
| beta | +0.40252 | −0.00945 | **yes** |
| gamma | +0.40043 | −0.00327 | **yes** |
| delta | +0.07810 | +0.00041 | **yes** |
| epsilon | +0.02795 | −0.00279 | **yes** |
| zeta | +0.40718 | −0.00798 | **yes** |
| eta | +0.10767 | −0.00189 | **yes** |
| theta | +0.37770 | +0.00809 | **yes** |
| iota | +0.39459 | +0.00333 | **yes** |
| kappa | +0.31476 | −0.00400 | **yes** |

**Tally: 10/10 → CONFIRMED.** A complete, clean reversal from the pre-fix result (3/10, frozen
actually winning 7/10). `frozen` sits flat at essentially zero growth in every single seed
(mean −0.00142) while `live` shows large, consistent growth in every seed (mean +0.26206,
range +0.028 to +0.407). This is exactly what "restored gradient flow" predicts, and it is now
statistically unambiguous at N=10 — no seed is even close to the boundary.

#### P3 — Δcos(live) repetitive > empty, 10 seeds

| seed | repetitive | empty | diff | live wins? |
|---|---|---|---|---|
| alpha | −0.05315 | −0.11344 | +0.06029 | **yes** |
| beta | −0.05293 | −0.12369 | +0.07076 | **yes** |
| gamma | −0.04643 | −0.28454 | +0.23811 | **yes** |
| delta | −0.03741 | −0.07577 | +0.03836 | **yes** |
| epsilon | −0.03429 | −0.05456 | +0.02027 | **yes** |
| zeta | −0.04978 | −0.12607 | +0.07629 | **yes** |
| eta | −0.05257 | −0.25340 | +0.20083 | **yes** |
| theta | −0.02595 | −0.20828 | +0.18233 | **yes** |
| iota | −0.02260 | −0.28547 | +0.26287 | **yes** |
| kappa | −0.04086 | −0.07023 | +0.02937 | **yes** |

**Tally: 10/10 → CONFIRMED.** Every seed, comfortably above noise (smallest margin +0.020,
nothing like the pre-fix run's razor-thin ties). Both regimes are still net-negative vs.
persistence (repetitive mean −0.0416, empty mean −0.1595) — the model still doesn't fully catch
up to a persistence baseline in either — but it now reliably tracks *which* regime has
learnable structure, which is what P3 actually tests. The Follow-up #2 caveat (repetitive and
empty share a "trivially strong persistence" floor) is still true in principle, but the effect
size here is large enough that it's no longer the dominant explanation the way it was pre-fix.

#### Manipulation check — bits_saved_zero > 0 in non-empty regimes, live arm

| regime | positive/10 |
|---|---|
| repetitive | 10/10 |
| varied | 10/10 |
| alarming | 10/10 |
| script | 10/10 |

**Fully clean pass** — a complete reversal from the pre-fix coin-flip (4-6/10 per regime). The
shortest-horizon prediction now reliably beats the directionless zero floor everywhere,
confirming (retrospectively) that the pre-fix coin-flip really was measuring a numerically dead
signal, not a broken metric — exactly the reading Amendment 3/the pre-fix Results section
argued for at the time, now directly confirmed by the same metric on the fixed binary.

#### Aggregate means (live arm, all 10 seeds)

| regime | mean Δcos | mean PE |
|---|---|---|
| repetitive | −0.0416 | 0.1332 |
| varied | +0.2717 | 0.5988 |
| alarming | +0.4086 | 0.4954 |
| empty | −0.1595 | 0.2570 |

For comparison, arm means on the varied regime + script block:

| arm | mean varied Δcos | mean varied PE | mean script growth |
|---|---|---|---|
| live | +0.2717 | 0.5988 | +0.2621 |
| frozen | −0.0078 | 0.7043 | −0.0014 |
| memoryless | +0.2844 | 0.5840 | +0.4074 |

`frozen`'s varied-regime PE (0.7043) sits right where every arm's PE sat pre-fix (~0.70-0.73)
— essentially unchanged by the scale-restoration fix, because that fix restores *magnitude*,
not *learned structure*, and `frozen`'s weights never update. This is the cleanest read on
mechanism available from this dataset: **the fix bought trainability, not accuracy directly** —
wherever training happens (`live`, `memoryless`) accuracy jumped; wherever it can't (`frozen`)
nothing moved. `memoryless`'s script growth (+0.407) actually exceeds `live`'s (+0.262) — a
second, independent line of evidence for the P1 finding that carried state isn't adding value
on top of what per-cycle training already captures.

#### Verdicts against pre-registered predictions (supersedes the pre-fix verdicts above for
current interpretation — the pre-fix run stays valid as the "before" data point)

- **P1: NOT SUPPORTED** (4/10, need ≥9/10) — and now decisively so, not just underpowered.
  Genuine cross-cycle state carryover is not adding measurable value over per-cycle training on
  a memoryless network, on this task.
- **P2: CONFIRMED** (10/10) — the annihilation fix restored real, unambiguous training signal.
- **P3: CONFIRMED** (10/10) — the loop's predictions now reliably track which input regime has
  learnable structure.
- **P4 (the honest risk, pre-fix): NO LONGER the live reading.** The pre-fix run's "temporal
  compression is aspirational" sentence was correct *for that binary* and remains a true
  historical data point (see the P0-#4/#5 cross-corroboration with the keystone Phase-4
  retraction it enabled). Post-fix, the loop demonstrably does learn to predict better than
  chance and better than a memoryless-in-outcome persistence baseline — training works. What
  the pre-fix run got right in spirit, and what still holds post-fix: the *specific* thing
  "temporal compression via carried state" was supposed to mean — using accumulated context
  across cycles, not just a well-trained one-step map — still has no measured support (P1). The
  updated honest sentence: **"Post-fix, the loop learns to predict (P2, P3 confirmed), but that
  learning is not currently exploiting temporal state carryover beyond what per-cycle training
  alone achieves (P1 not supported) — the system has a trainable one-step predictor, not yet
  demonstrated evidence of the sequence modeling 'temporal compression' implies."**

#### Follow-ups (updated)

1. ~~Re-run C1 post-annihilation-fix~~ — **done, this section.**
2. **P1 is now the sharpest open question in the whole program**, not P2/P3. If the loop learns
   the one-step transition well (confirmed) but doesn't benefit from carried state (confirmed
   null), the natural next probe is whether *any* mechanism in the current architecture can use
   multi-step history — this converges with Keystone's own still-open "episodic anticipation
   absent" finding and C3's original scope (episodic recall → prediction). Recommend
   prioritizing C3 next, ahead of C2, now that P1 gives it a sharper starting hypothesis:
   state-carryover alone doesn't help, so an explicit episodic-recall mechanism would need to
   supply something a plain reservoir update doesn't.
3. Replace the P3 pair (repetitive-vs-empty) — still a valid follow-up in principle, now lower
   priority since P3's effect size is large enough that the trivial-floor concern is no longer
   load-bearing for the conclusion.
4. C2 (compression-gated Chronicle) is unblocked (`hdc_ltc_bridge.rs` is stable post-fix) but
   should incorporate the P1 finding: gating episodic writes on "bits saved by this event's
   model update" is a *training*-quality signal, not a *state-carryover* signal — worth being
   explicit about which capability C2 is actually testing before building it.

---

## 6. C2–C4 sketches (to be individually registered before their runs)

- **C2 (compression-gated Chronicle)**: add `eval_loss_from(snapshot, input, target, dt)` to
  `HdcLtcBridge` (scratch-snapshot pattern, pure); make `train_step_from` report
  `(pre_loss, post_loss)`; store `bits_saved_by_update` on `Episode`; A/B write gates
  (current Φ/coherence gate vs compression gate vs hybrid) on: episode-set overlap, priority-
  rank correlation with later replay utility, and — the honest endpoint — whether the selected
  set differs *at all* from the PE-gated set (PE already gates writes and weights priority 0.2;
  the null result "compression gating ≈ PE gating" is live and must be stated if found).
  Coverage caveats to declare: async-trainer path has no loss signal; non-learning cycles
  produce no update to measure (hybrid fallback preserves current behavior).
- **C3 (episodic anticipation)**: **reprioritized ahead of C2** per the C1 N=10 post-fix
  result (P1: `memoryless` ≈ `live`, no measured value from state carryover alone) — since
  plain reservoir carryover doesn't help, the open question is whether an *explicit* recall
  mechanism can supply something a bare state update can't, independent of C2's
  episode-selection question. Registered in full in §7 below (before implementation).
- **C4 (calibrated surprise)**: replace the intent classifier's affine "confidence" with
  `softmax(β·sim)` over class prototypes; fit β offline by minimizing `analyze_pairs` ECE on a
  held-out labeled set; wire real `(confidence, correct)` records into `CalibrationAudit`
  (`manager.rs`) so `knowledge_calibration_ece` stops reading the 0.0 sentinel — and port the
  `ece_computed` absent-vs-zero guard (`b97ed86042` pattern) to that path regardless. Grading
  must be external (AGW-2.2 pattern), never the facade's lexical self-check.

---

## 7. Experiment C3 — episodic recall → prediction (stamped 2026-07-25, before any
implementation or run)

### Question

Can an *explicit* content-based recall mechanism improve short-horizon prediction beyond what
the CfC's own trained state achieves alone — the question C1's P1 result leaves open (state
*carryover* alone showed no measured benefit; this asks whether *targeted retrieval* of a
specific past episode can do better than carrying an undifferentiated running state).

### Ground truth (surveyed 2026-07-25, file:line-verified, supersedes the §1 note where it
differs)

- `self.memory.episodic_persistence.replay: Option<EpisodicReplay>` is populated
  unconditionally at construction (`constructor.rs:445`, `Some(EpisodicMemory::new(...))` —
  not gated by a config flag) and accumulates real `Episode { input: ContinuousHV, output:
  ContinuousHV, .. }` pairs every cycle the write gate passes (coherence ≥ 0.3,
  `episodic_replay.rs:491-497`), on the default config C1 already used. `episode.input_as_array()`
  / `.output_as_array()` exist (`episodic_replay.rs:269,274`) and always work — no feature gate.
- The only existing similarity-search method, `retrieve_by_embedding_similarity`
  (`episodic_replay.rs:1014-1044`), queries the OPTIONAL `semantic_embedding` field, which is
  only populated behind the non-default `semantic-encoder` feature (confirmed absent from
  `default-mind`) — **not usable for a default-config experiment**. C3 needs a new method
  querying `.input` directly (always populated), mirroring the same cosine-similarity/top-k
  shape.
- The *other* episodic store, `EpisodicMemoryBridge` (`memory_bridge.rs`, `self.fep.episodic_memory`),
  stores only `content: String` + a 64-float `embedding` sample + valence/phi metadata — **no
  output/successor field at all**. It cannot supply a content-based prediction candidate;
  `EpisodicReplay` is the only store with the (input, output) pair C3 needs.
- Confirmed (again) that `EpisodicReplay`'s current *training* consumer only runs for the
  classic CfC backend (`cycle_phases_memory.rs:445`, `if let TemporalNetwork::CfC(...)`) — but
  storage and read-only queries are backend-independent, so a *read* path for HdcLtcUnified
  (the default) is new, unblocked territory, not fighting an existing backend gate.

### Mechanism (minimal, matching the design principle used throughout this program: measure
before you gate, and default the new path to inert)

1. **New read-only method** on `EpisodicReplay`: `retrieve_by_input_similarity(query: &[f32],
   top_k: usize) -> Vec<(Episode, f32)>` — identical shape to
   `retrieve_by_embedding_similarity` but scored against `episode.input_as_array()` (via a
   plain cosine dot-product, compressed to the query's dimension the same way
   `predictive_encoder.rs`'s bits-saved path already does — reuse that projection, don't
   invent a second one).
2. **New config flag** `enable_episodic_recall_prediction: bool` (default `false` — matches
   the program's existing "new capability starts inert" pattern for `training_frozen`/
   `reset_temporal_state`, and means every existing C1 result stays reproducible bit-for-bit
   with this flag off).
3. **Wiring point**: `cycle_phase_dynamics/planning.rs`, immediately after
   `get_multi_scale_prediction` (where `prediction_first_horizon` is already captured for the
   bits-saved diagnostics). When the flag is on: query the top-1 episode by input similarity
   against the *current* compressed encoding (not the prediction — this recalls "have I seen
   something like NOW before", then offers up what happened after it); if similarity ≥
   `RECALL_BLEND_SIM_THRESHOLD` (registered value: **0.5** — chosen as the midpoint between
   "uncorrelated" (~0.0 for random compressed vectors at this dimension, per the bits-saved
   metric's own PE-degenerate-sentinel precedent) and "near-duplicate" (~0.9+); revisit if the
   manipulation check below fails), blend: `prediction' = (1 − w)·prediction + w·recalled_output`
   where `w = clamp((sim − threshold) / (1 − threshold), 0, 0.5)` (recall can influence at most
   half the prediction, never fully override the CfC's own state-based estimate — a
   deliberately conservative cap, not tuned to the outcome).
4. **Purity discipline**: this is the first C-phase that changes *behavior*, not just
   measurement — the twin/no-perturbation test pattern from P0 still applies in reverse: with
   the flag OFF, output must be bit-identical to pre-C3 code (a regression test, mirroring
   `bits_saved_measurement_only_does_not_perturb_pe_or_attention`).

### Tasks & metrics

- **Primary harness**: a new example, `examples/episodic_recall_probe.rs` — NOT an extension of
  `keystone_ab.rs` (per the coordination rule: don't touch that file) and NOT a reuse of
  `compression_bits.rs` wholesale (different independent variable), but sharing its regime/
  script infrastructure by duplication (small, ~200-line harness; the "don't touch keystone_ab"
  rule is about the *file*, not the *pattern*).
- **Arms**: `recall_off` (flag false, the existing default) vs `recall_on` (flag true).
  Deterministic per (arm, seed): same `genesis_phrase` + `async_training=false` pattern as C1.
- **Task 1 — order-sensitivity probe** (keystone Phase-3 design, replicated with the bits-saved
  metric instead of raw PE): 60 reps of the learned 12-sentence script (building an episodic
  store full of real (input, output) pairs across many reps), then 10 probe reps where odd reps
  swap one deterministic adjacent-position pair within fully familiar material. Endpoint:
  `order_sensitivity = mean(bits_saved_persist on swapped-position cycles) − mean(bits_saved_persist
  on matched clean cycles)`. Keystone's finding on this exact design (pre-fix binary, raw PE):
  noise, ±0.01, no arm distinguishable — **the sharpest capability gap this experiment tests.**
- **Task 2 — manipulation check**: recall hit-rate (fraction of cycles where similarity ≥
  threshold) must be > 0 in the `varied`/`script` regimes by rep 30+ (the store needs time to
  fill) — if it's ~0 throughout, the threshold or projection is miscalibrated and C3's result
  would be vacuous (recall never fires) rather than a real test of the mechanism.
- **Seeds**: 10, named `episodic-recall-c3-seed-{alpha..kappa}-2026-07-25`, hardcoded.
- **Statistics**: same 10-seed sign test convention as C1 (CONFIRMED ≥9/10, SUGGESTIVE 7–8/10,
  NOT SUPPORTED ≤6/10).

### Pre-registered predictions

- **P5 (recall creates order sensitivity)**: `recall_on`'s `order_sensitivity` magnitude is
  reliably larger (further from zero, in the direction of "swapped costs more than clean") than
  `recall_off`'s, ≥7/10 seeds. This is keystone's exact capability gap, tested with a different
  mechanism (explicit retrieval) than keystone tried (implicit state only).
- **P6 (recall improves short-horizon prediction generally)**: `recall_on` > `recall_off` on
  mean `bits_saved_persist` across the full `varied` regime (not just the order-probe reps),
  ≥7/10 seeds — the more basic "does recall help at all" question, independent of order.
- **P7 — the honest risk**: recall hit-rate is real (manipulation check passes) but neither P5
  nor P6 clears its bar. If this holds, the sentence entering VISION.md: *"An explicit
  similarity-gated recall-to-prediction path, wired and firing, still does not produce measured
  order-sensitivity or prediction improvement — the gap keystone named is not merely a missing
  wire, and the next candidate cause is the recall *content* itself (single nearest-neighbor
  output may be too noisy/low-resolution a signal) or the blend mechanism (linear blending may
  be the wrong integration rule), not the absence of any recall mechanism."* This would be a
  second, independently-mechanism-tested confirmation of the same capability gap C1's P1 and
  keystone's Phase-3 both already found from different angles.
- **What would change our minds**: P5+P6 confirmed → episodic anticipation is real and
  mechanism-dependent (explicit retrieval succeeds where bare carryover failed) — proceed to
  tune blend weight/threshold and consider promoting the flag toward default-on. P5 without P6
  (or vice versa) → the two capabilities (order-sensitivity, general accuracy) are more
  separable than assumed — treat as two different findings, not one. Neither → P7, and the
  next diagnostic step is registered there rather than guessed now.

### Coordination

No overlap with the keystone/phi-trace lane's files (`hdc_ltc_bridge.rs`,
`hdc_ltc_unified.rs`) — C3 touches only `episodic_replay.rs` (new method, additive),
`planning.rs` (new gated branch), `config/mod.rs` (new flag), and a new example file. Verify
`git status` on those three source files is clean before starting, same discipline as every
prior phase in this program.

### Implementation notes (2026-07-25, mechanism landed, commit `bd19a6cb7e` — before any
experiment run)

Two corrections found while implementing, both caught by the discipline the registration
itself required (regression tests before trusting the mechanism):

1. **Dimension correction.** The registration assumed `Episode.input`/`.output` were
   full 16,384-D HDC vectors (matching the `ContinuousHV` type name) and planned to query
   with the full attended HDV, compressing the recalled `output` via `compress_for_ltc`
   before blending. Wrong: episodes actually store the CfC's own COMPRESSED
   input/output (`cycle_phases_memory.rs:356-358` wraps `compressed_state`/`output` — same
   256-D space as `prediction` — in a `ContinuousHV` purely as a container type). Caught
   immediately: `retrieve_by_input_similarity`'s own regression test panicked "256 vs
   16384" on first run. Fixed by querying with `perception.encoding.compressed_state`
   directly and dropping the now-unnecessary compression step — the recalled `output` is
   already in the right space to blend with `prediction`.
2. **Unrelated pre-existing finding, not a C3 bug.** The purity test
   (`recall_prediction_flag_off_is_bit_identical_to_baseline`) initially asserted bit-exact
   equality between two freshly-constructed, identically-seeded services and failed at
   cycle 1 (cycle 0's cold-start sentinel matched exactly). Diagnosed with a throwaway
   probe (not committed): per-element output differences up to ~1e-7, `prediction_error`
   matching to 6 decimals — benign floating-point non-associativity from the pipeline's
   `rayon::join` parallel post-processing (thread completion order affects summation
   order, not which values get computed). This is the first place in the program a test
   asked for bit-exact cross-construction reproducibility; C1's sign-test methodology
   never needed that property (each seed used exactly one construction) and remains
   unaffected. Both C3 tests now use tolerances (1e-4 noise floor, 1e-3
   meaningful-divergence bar) instead of exact equality.

Both tests pass (`cargo test --lib cognitive_loop::tests::memory_pipeline::recall_prediction`
— 2/2). The mechanism is confirmed wired and firing (mechanism-sanity test), and confirmed
inert with the flag off within noise (purity test). No experiment has been run yet — next
step is the `examples/episodic_recall_probe.rs` harness (§7 Tasks) to test P5/P6/P7.

### Results (2026-07-25, full N=10 run, exit 0) — the registered run

10 named seeds, full-length blocks (varied 400 cycles/100 warmup, script 60 reps + 10 probe
reps), `examples/episodic_recall_probe.rs`. Raw log archived outside any session-scoped path.

#### Manipulation check — recall hit-rate, both arms, both blocks, all 10 seeds

| arm | varied | order |
|---|---|---|
| recall_on | 1.0000 (all 10 seeds) | 1.0000 (all 10 seeds) |
| recall_off | 0.0000 (all 10 seeds) | 0.0000 (all 10 seeds) |

**Fully clean — no ambiguity.** The mechanism is unambiguously wired and firing every single
cycle it's enabled, never when it isn't. (The near-100% hit rate is expected, not a red flag:
both blocks deliberately repeat content — the varied regime cycles a 12-sentence script
repeatedly, the script block repeats the same 12 sentences 60+ times — so once the episodic
store has a few reps of history, nearly every subsequent cycle finds a near-duplicate past
episode above the 0.5 similarity gate. This means the *gating* isn't doing much discriminating
work in this harness design — see Follow-ups.) Any null result below is therefore a genuine
null, not a "recall never fired" artifact.

#### P6 — recall_on > recall_off, varied-regime mean bits_saved_persist, 10 seeds

| seed | on | off | diff | win? |
|---|---|---|---|---|
| alpha | +0.21844 | +0.21815 | +0.00029 | **yes** |
| beta | +0.20485 | +0.19982 | +0.00503 | **yes** |
| gamma | +0.22114 | +0.21710 | +0.00404 | **yes** |
| delta | +0.23656 | +0.23156 | +0.00500 | **yes** |
| epsilon | +0.23600 | +0.24297 | −0.00697 | no |
| zeta | +0.25709 | +0.25327 | +0.00382 | **yes** |
| eta | +0.24746 | +0.25011 | −0.00265 | no |
| theta | +0.22877 | +0.23155 | −0.00278 | no |
| iota | +0.23526 | +0.23456 | +0.00070 | **yes** |
| kappa | +0.23732 | +0.23466 | +0.00266 | **yes** |

**Tally: 7/10 → SUGGESTIVE** (registered gate: CONFIRMED ≥9/10, SUGGESTIVE 7–8/10, NOT
SUPPORTED ≤6/10). But the effect size is tiny: mean diff +0.00091 against a baseline mean of
+0.23137 — under 0.4% relative. This is a directionally-consistent-majority result at a
magnitude that borders on noise; call it a weak signal, not a capability.

#### P5 — |order_sensitivity(recall_on)| > |order_sensitivity(recall_off)|, 10 seeds

| seed | on | off | \|on\| | \|off\| | win? |
|---|---|---|---|---|---|
| alpha | −0.16629 | −0.16740 | 0.16629 | 0.16740 | no |
| beta | −0.23892 | −0.12336 | 0.23892 | 0.12336 | **yes** |
| gamma | −0.15234 | −0.15591 | 0.15234 | 0.15591 | no |
| delta | −0.16746 | −0.16562 | 0.16746 | 0.16562 | **yes** |
| epsilon | −0.19236 | −0.19306 | 0.19236 | 0.19306 | no |
| zeta | −0.20194 | −0.20039 | 0.20194 | 0.20039 | **yes** |
| eta | −0.15376 | −0.16106 | 0.15376 | 0.16106 | no |
| theta | −0.18701 | −0.18663 | 0.18701 | 0.18663 | **yes** |
| iota | −0.17247 | −0.16759 | 0.17247 | 0.16759 | **yes** |
| kappa | −0.16067 | −0.16455 | 0.16067 | 0.16455 | no |

**Tally: 5/10 → NOT SUPPORTED** (need ≥7/10 minimum). A coin flip — recall does not reliably
increase order-sensitivity. Both arms show the SAME underlying pattern keystone found: a
reliably-negative order_sensitivity (mean on = −0.17932, mean off = −0.16856 — direction
consistent with genuine order-sensitivity existing at some baseline level in both arms, likely
carried by the same substrate mechanism keystone's Phase 5 fix unlocked), but recall_on adds
no reliable increment on top of that baseline.

#### Verdicts against pre-registered predictions

- **P5: NOT SUPPORTED** (5/10, need ≥7/10)
- **P6: SUGGESTIVE, not CONFIRMED** (7/10, but effect size <0.4% relative — a weak, likely
  marginal signal rather than a real capability)
- **P7 (the honest risk): effectively confirmed, with a caveat.** The registered honest-risk
  language was: *"an explicit similarity-gated recall-to-prediction path, wired and firing,
  still does not produce measured order-sensitivity or prediction improvement."* P5 matches
  this exactly. P6 is the caveat — there IS a small, majority-direction (7/10) general-prediction
  effect, but it's an order of magnitude too small to call a real capability at this sample
  size. Updated honest sentence for VISION.md: *"An explicit, verified-firing (100% hit rate)
  episodic recall mechanism produces no measured order-sensitivity benefit (P5, 5/10) and only
  a marginal, borderline general-prediction effect (P6, 7/10 seeds but <0.4% relative magnitude)
  — the capability gap keystone named is not resolved by adding explicit retrieval on top of a
  trained one-step predictor; whatever is missing is not simply 'no recall mechanism existed.'"*
- **What this means for the program overall**: C1's P1 (state carryover alone doesn't help) and
  C3's P5 (explicit recall doesn't help order-sensitivity either) are now two independent,
  convergent negative results pointing at the same capability gap from different mechanisms.
  Both C1 and keystone's own Phase-3 design tested this with *implicit* state; C3 tested it
  with *explicit* retrieval — same answer both times.

#### Unregistered findings (exploratory, flagged as such)

- The recall gate essentially never discriminates in this harness (100%/0% hit rate) because
  both test blocks are built from deliberately repeated content — this was necessary to build
  up a populated episodic store cheaply, but it means C3 never actually tested "does recall
  help MORE on high-similarity matches than low-similarity ones" — only "does recall help at
  all, when it always fires." A harness with genuinely varied (non-repeating) content and a
  wider range of realized similarities would be needed to test the gate's own discriminating
  power, not just the mechanism's on/off effect.
- Both arms' order_sensitivity is reliably negative (not near-zero), consistent with genuine
  order-sensitivity existing in the post-annihilation-fix substrate independent of C3's
  mechanism — worth noting as indirect, incidental support that keystone's Phase 5 fix produced
  real capability, even though this experiment's own C3-specific question (does *explicit
  recall* add anything on top) came back negative.

#### Follow-ups

1. **Redesign the harness to test graded similarity**, not just on/off — vary content so recall
   sometimes fires at 0.5–0.6 similarity, sometimes at 0.9+, and check whether the blend weight
   (which scales with similarity, capped at 0.5) produces a dose-response relationship. The
   current 100%/0% hit-rate design cannot distinguish "recall never helps" from "recall only
   helps at high similarity, and this harness only ever tested high similarity."
2. **If P6's weak signal is worth chasing**, a larger N (20+ seeds) or a longer varied-regime
   block would clarify whether it's real-but-small or noise dressed as a majority.
3. Given both C1's P1 and C3's P5 now agree (state carryover and explicit recall both fail to
   produce order-sensitivity), the next diagnostic step this program would recommend is
   examining the blend mechanism itself (linear blending of a single nearest-neighbor's stored
   `output` may be the wrong integration rule — a weighted combination across the top-k, or a
   attention-style soft blend, might behave differently) before concluding the capability gap
   is architectural rather than mechanism-specific.

### Experiment C3b — similarity-gradient probe (stamped 2026-07-25, before any
implementation, same day as C3's main run — Follow-up 1 above, executed)

**Question**: C3's main run saturated at a 100%/0% recall hit-rate (both test blocks used
deliberately repeated content, so nearly every eligible cycle matched above the 0.5 threshold).
This means C3 tested "recall always fires vs never fires" but never tested whether the blend's
effect scales with match quality — a graded mechanism should help more on a near-duplicate
match than a marginal one. C3b asks: **does `bits_saved_persist` show a dose-response
relationship with `recall_similarity`** within the `recall_on` arm, using content designed to
produce a genuine spread of similarities rather than saturating?

**Design**: a new content set with three tiers, cycled in a fixed deterministic pattern so every
seed sees the identical schedule:
- **Prototypes** (4 sentences, each repeated many times): expected to produce high-similarity
  recalls (~0.9+) once the store has a few reps of history.
- **Paraphrases** (4 sentences, same topic/meaning as the 4 prototypes but different wording):
  expected to produce medium-similarity recalls against their prototype's episodes — genuinely
  uncertain in advance where this lands (HDC similarity does not track semantic paraphrase
  similarity in any guaranteed way — this is itself worth observing, not assumed).
  Registered honestly: paraphrase-vs-prototype similarity may cluster anywhere; if it clusters
  near the prototypes' own repeat-similarity, tiers 1 and 2 won't actually separate, and that
  null result (no HDC-level distinction between paraphrase and repeat) is itself informative.
- **Novel one-offs** (12 sentences, each appearing exactly once): expected to produce low
  similarity or no recall at all (nothing to match against on first appearance).

Single arm (`recall_on` only — C3b is not an on/off comparison, it's a within-run stratification
question), same 10 named seeds as C3 (reused, not re-minted — this is an amendment to the same
experiment, not a new one), 400 cycles per seed cycling all three tiers in a fixed round-robin
order with prototypes appearing 4× as often as paraphrases/novel content (to let the store fill
with prototype history before paraphrases are ever tested against it).

**Metric**: for each cycle where `recall_fired` is true, record `(recall_similarity,
bits_saved_persist)`. Bin into 3 similarity bands: `[0.5, 0.7)`, `[0.7, 0.9)`, `[0.9, 1.0]` (the
registered threshold floor is 0.5, so bins start there). Compare mean `bits_saved_persist`
across bins.

**Pre-registered predictions**:
- **P8 (dose-response exists)**: mean `bits_saved_persist` in the `[0.9, 1.0]` bin is reliably
  higher than in the `[0.5, 0.7)` bin, ≥7/10 seeds (same sign-test convention). This is the
  "the blend mechanism itself works correctly, C3's main run just never exercised a low-quality
  match" hypothesis.
- **P9 — the honest risk**: no reliable ordering across bins (or bins are too sparse to compare
  — the manipulation check below must pass before P8/P9 can be evaluated at all). If P9 holds,
  it strengthens C3's original P7 finding: the mechanism doesn't help even in its best case
  (high-similarity matches), which rules out "the harness just always tested worst-case
  matches" as an explanation for C3's null result.
- **Manipulation check**: all 3 similarity bins must have ≥20 samples per seed (aggregated
  across the 400 cycles) or the comparison is underpowered and must be reported as such, not
  forced into a verdict.

**Coordination**: touches only a new example file
(`examples/episodic_recall_gradient_probe.rs`) and reads (never writes) the same
`enable_episodic_recall_prediction`/`retrieve_by_input_similarity`/`recall_similarity` surface
C3 already landed — no changes to `episodic_replay.rs`, `planning.rs`, or any keystone-lane file.

### C3b Amendment 1 (registered 2026-07-26, after the main C3b run but BEFORE the
confound-control run described below)

The main C3b run (10 seeds, full-length) landed a striking result before this amendment: a
**perfectly monotonic 10/10 dose-response** — `bits_saved_persist` mean strictly increases
`[0.5,0.7) < [0.7,0.9) < [0.9,1.0]` in every single seed, well-powered (39–229 samples per
bin, all above the 20-sample floor). Raw numbers, e.g. seed alpha: 0.069 → 0.148 → 0.196;
seed theta: 0.121 → 0.173 → 0.224 — same ordering, every seed.

**Registered BEFORE drawing a conclusion**: this result has an uncontrolled confound. The
similarity tiers are not independent of content repetition — high-similarity recalls come
overwhelmingly from the `prototype` tier (repeated 4x as often as everything else), and a
well-trained CfC would predict heavily-repeated content well from training alone, with zero
causal contribution from the recall blend. The observed "dose-response" could therefore be
entirely a restatement of "repeated content is more predictable," not evidence the recall
mechanism itself adds value in proportion to match quality. P8 cannot be honestly called
CONFIRMED until this confound is controlled.

**Control, registered before running**: rerun the identical schedule with
`enable_episodic_recall_prediction = false` (no recall arm), stratify BOTH arms by CONTENT
TIER (prototype/paraphrase/novel — available regardless of whether recall fires, unlike
similarity) instead of by similarity bin, and compare `recall_on` vs `recall_off` mean
`bits_saved_persist` WITHIN each tier. **P10 (the blend has a real causal effect beyond the
predictability confound)**: `recall_on` > `recall_off` within the `prototype` tier specifically
(where the confound is strongest — if recall adds nothing on top of training there, it adds
nothing anywhere), ≥7/10 seeds. **P11 — the honest risk**: `recall_on` ≈ `recall_off` within
every tier, meaning the clean dose-response above is fully explained by the predictability
confound and the recall blend itself contributes ~nothing causally, despite being correctly
wired, correctly graded by similarity, and correlated with genuine capability differences that
have nothing to do with it.

### C3b Results (main dose-response run)
*(recorded 2026-07-26, N=10, monotonic in all 10 seeds — see Amendment 1 above for why this
alone does not yet support a causal verdict; P10/P11 confound-control results below)*

| seed | [0.5,0.7) n | mean | [0.7,0.9) n | mean | [0.9,1.0] n | mean |
|---|---|---|---|---|---|---|
| alpha | 43 | 0.06903 | 48 | 0.14827 | 216 | 0.19563 |
| beta | 40 | 0.07884 | 48 | 0.15783 | 218 | 0.21329 |
| gamma | 43 | 0.08947 | 48 | 0.15109 | 220 | 0.19999 |
| delta | 42 | 0.06129 | 48 | 0.14056 | 225 | 0.21143 |
| epsilon | 42 | 0.07482 | 42 | 0.12754 | 229 | 0.20233 |
| zeta | 42 | 0.07452 | 48 | 0.13279 | 215 | 0.18660 |
| eta | 40 | 0.09875 | 48 | 0.15018 | 210 | 0.21333 |
| theta | 39 | 0.12070 | 47 | 0.17285 | 214 | 0.22373 |
| iota | 42 | 0.08730 | 46 | 0.14350 | 220 | 0.20492 |
| kappa | 42 | 0.07575 | 47 | 0.15229 | 211 | 0.20346 |

**P8 tally: 10/10 for monotonicity and 10/10 for the registered top-vs-bottom-bin comparison**
— but held pending the confound control per Amendment 1. See C3b's second results block below
for the causal verdict.

### C3b Results (P10/P11 confound-control run, 2026-07-26, N=10, exit 0)

Same schedule as the main C3b run, both arms (`recall_on`/`recall_off`), stratified by content
TIER instead of similarity. This is the causal control the dose-response above needed before
any conclusion could be drawn.

| tier | recall_on wins | mean diff (on − off) |
|---|---|---|
| prototype | 7/10 | +0.00044 |
| paraphrase | **0/10** | **−0.00365** |
| novel | 5/10 | +0.00059 |

Full per-seed data:

**Prototype** (7/10 WIN, mean +0.00044): alpha +0.00077, beta −0.00054, gamma +0.00218, delta
−0.00181, epsilon +0.00338, zeta +0.00121, eta −0.00121, theta +0.00020, iota +0.00013, kappa
+0.00008.

**Paraphrase** (0/10, mean −0.00365 — every seed negative): alpha −0.00319, beta −0.00556,
gamma −0.00076, delta −0.00271, epsilon −0.00259, zeta −0.00546, eta −0.00421, theta −0.00927,
iota −0.00090, kappa −0.00190.

**Novel** (5/10, mean +0.00059 — coin flip): alpha +0.00290, beta +0.00005, gamma +0.00272,
delta −0.00027, epsilon +0.00077, zeta +0.00003, eta −0.00009, theta −0.00001, iota −0.00005,
kappa −0.00016.

#### Verdicts

- **P10 (literal gate): technically met** (7/10 ≥ 7 in the prototype tier) **but the honest
  read is NOT SUPPORTED overall.** The prototype-tier "win" has a mean magnitude
  (+0.00044) that is smaller than the SAME experiment's own clean, opposite-direction,
  larger-magnitude result in the paraphrase tier (0/10, mean −0.00365 — every single seed
  negative, up to −0.00927). P10's registered logic ("if recall adds nothing in the tier where
  the confound is strongest, it adds nothing anywhere") does not license ignoring a clean
  negative result in a different tier; a mechanically-applied single-tier gate would have
  reported "P10 confirmed" while missing the more interesting finding.
- **P11 (the honest risk) is essentially what happened, with a sharper edge than
  registered**: the recall blend does not show a reliable, positive causal effect in any tier.
  Prototype's marginal positive direction is an order of magnitude smaller than paraphrase's
  clean, consistent NEGATIVE effect. **Paraphrase is the theoretically most important case**
  (content genuinely related-but-different from any stored episode — exactly what explicit
  recall is supposed to help with beyond simple repetition), and it is the one tier where the
  mechanism is reliably, unanimously WORSE than not having it at all.
- **What the earlier "perfectly monotonic 10/10 dose-response" (C3b's main run) actually
  was**: a clean demonstration of the registered confound, not evidence of a working graded
  mechanism. High recall-similarity and high content-predictability are correlated for a reason
  that has nothing to do with the recall blend (repeated content trains well on its own); once
  that confound is controlled for, the apparent dose-response has no causal counterpart. This
  is a valuable methodological lesson for the rest of the program and for future work in this
  codebase generally: **a within-arm correlation between a mechanism's own activation strength
  and an outcome metric is not evidence the mechanism causes the outcome** — the C1 pattern
  (always compare against a true counterfactual arm, never just bin by an internal signal) is
  the right default, and C3b's main run should have run the control from the start rather than
  needing a same-day amendment to catch it.
- **Combined with C3's main on/off result** (P6 weakly suggestive, tiny effect; P5 not
  supported) and C1's P1 (state carryover doesn't help): **three independent tests now agree
  that neither implicit state carryover nor explicit similarity-gated recall reliably improves
  prediction in this architecture**, and the one place a mechanism showed a large, clean,
  reliable effect (the paraphrase tier), the effect was *harmful*, not helpful. Updated honest
  sentence for VISION.md: *"Explicit episodic recall, verified correctly wired and graded by
  similarity, shows no reliable causal benefit once content-predictability is controlled for —
  and is reliably WORSE than no recall specifically on paraphrased content, the case it would
  most need to help with to be useful. The apparent similarity-dose-response is fully explained
  by a predictability confound, not by the mechanism's own operation."*

#### Follow-ups

1. **Why is paraphrase content reliably worse with recall on?** One candidate: blending in a
   prototype's stored `output` when the CURRENT input is a paraphrase (not the prototype itself)
   injects a specific, wrong, overconfident prediction — worse than the CfC's own, more
   appropriately uncertain, estimate for genuinely different content. This would mean the blend
   is actively harmful exactly when the recalled match is real-but-imperfect, which is the
   common case for any content that isn't literally identical to something already stored.
2. Any future recall/memory mechanism in this codebase should be evaluated with a true
   counterfactual (on vs off) control from the start, not a within-arm correlation — this
   session's own near-miss (nearly reporting P8 as confirmed from the dose-response alone) is
   itself worth citing as a specific example of the general methodological point.
3. Given three independent negative/harmful results now, this program's remaining lower-priority
   items (C2 compression-gated Chronicle, C4 calibrated surprise) should be re-scoped with this
   lesson in mind — C2 in particular should pre-register an on/off causal control, not just an
   episode-selection comparison, before claiming any benefit from compression-gated writes.

---

### Experiment C3c — root-cause diagnostic for the paraphrase harm (stamped 2026-07-26,
before implementation)

**Question**: C3b Follow-up #1's leading hypothesis was that recall harms paraphrase content
because it matches against a PROTOTYPE episode (not another paraphrase occurrence) and blends
in that prototype's stored output — a specific, confident, but wrong target, since what
actually follows a paraphrase cycle in the schedule is not what followed that prototype cycle.
C3c tests this directly rather than leaving it as a plausible-but-unverified story.

**Mechanism**: `Episode.timestamp` (the cycle number the episode was written at,
`episodic_replay.rs:101`) already exists and is exactly what's needed — if a recalled episode's
timestamp is known, and the harness's own deterministic schedule already records which content
TIER every cycle number belongs to, then "what tier was the MATCHED episode" is a direct lookup,
no new production logic needed beyond exposing the timestamp. New telemetry field
`recall_matched_timestamp: Option<u64>`, threaded through the identical 4-site pattern as
`recall_fired`/`recall_similarity` (measurement-only, mirrors the P0/C3 pattern exactly).

**Prediction P12**: for paraphrase-tier CURRENT cycles where recall fires, the matched episode's
tier (looked up via `recall_matched_timestamp`) is `prototype` in a large majority of cases
(≥80%, single descriptive statistic — this is a diagnostic, not a new A/B experiment, so no
10-seed sign test is needed; one representative seed run is sufficient to confirm or refute the
mechanism). **P13 — the honest risk**: paraphrase recalls match roughly proportionally to store
composition (prototypes are 4x more common, so ~80% prototype matches would be the BASE RATE
even with no preferential mismatch) — if the match-tier distribution is statistically
indistinguishable from the store's own composition, that would mean paraphrases aren't being
mismatched preferentially, and the harm's cause is something else (e.g., the blend weight
formula, or a property of prototype outputs specifically, not a category-confusion story).

**Coordination**: touches the same 4 telemetry-threading sites as `recall_fired`/
`recall_similarity` (`planning.rs`, `cycle_phase_dynamics/mod.rs`, `phase_results.rs`,
`types/output.rs`, `cycle_phase_output/mod.rs`, and the 2 fallback `helpers/*.rs` sites) —
no keystone-lane files, no episodic_replay.rs changes (the timestamp field already exists).

### C3c Results (2026-07-26, single representative seed `alpha`, exit 0)

| current cycle's tier | recalls fired | matched prototype | matched paraphrase | matched novel |
|---|---|---|---|---|
| prototype | 312 | 100.0% | 0.0% | 0.0% |
| paraphrase | 62 | **0.0%** | **100.0%** | 0.0% |
| novel | 0 | — | — | — |

#### Verdict

**P12: REFUTED, cleanly.** The hypothesis was that paraphrase cycles would predominantly match
PROTOTYPE episodes (a category-confusion story: "this paraphrase looks enough like the
prototype it means to be that recall retrieves the prototype's stored outcome, which then
mismatches"). The data shows the exact opposite: **paraphrase recalls match prior PARAPHRASE
occurrences 100% of the time, never a prototype, in all 62 fired recalls.** The compressed
representation apparently distinguishes "prototype N's exact wording" from "paraphrase N's
wording" as genuinely separate regions of the 256-D space — despite both encoding the same
topic — cleanly enough that cross-tier matching essentially never happens in this design. P13
(the honest risk — matches proportional to store composition, i.e. mostly prototype since it's
4x more common) is ALSO not what happened; the actual outcome (perfect within-tier clustering)
was not one of the two registered predictions.

**Novel content never triggers a fired recall** (0/12 sentences, consistent across the run) —
expected, since each appears exactly once and has no prior occurrence of itself to match
against.

#### What this means for C3b's paraphrase-harm finding

The root cause is NOT "recall retrieves the wrong topic's stored outcome." Since paraphrase
recalls reliably retrieve a genuine PRIOR PARAPHRASE occurrence's stored output — which, on the
face of it, should be a reasonable predictor of "what follows a paraphrase now," given the same
sentence recurring in a broadly similar schedule position — the harm must have a different
source. The leading candidate now: **paraphrases are far rarer than prototypes in the schedule
(62 fired-recall opportunities vs 312 — roughly 1:5, matching the registered ~4:1 tier-frequency
design)**, so the CfC's own training signal around paraphrase-adjacent transitions is much
sparser than around prototype-adjacent ones. The recalled paraphrase episode's stored `output`
may therefore itself be a noisier, less-representative snapshot (fewer repetitions to average
over / stabilize against) than a prototype's — meaning the blend isn't injecting a
*wrong-category* target, but a *poorly-estimated same-category* one. This reframes the
program's earlier hypothesis language ("wrong-but-confident") to **"under-trained-but-still-
confident"** — the blend weight formula (scaled only by similarity, never by how many times the
matched episode's content has actually recurred) has no way to discount a recall on the grounds
that its source content is rare and its stored value correspondingly unreliable.

#### Follow-ups

1. **Test the "sparse signal" hypothesis directly**: rerun with paraphrases appearing as
   frequently as prototypes (removing the frequency confound) — if the harm shrinks or
   disappears, that confirms recurrence-frequency (not category) was the real driver. This is a
   cheap, well-scoped next experiment if anyone continues this line.
2. **Blend weight should account for source-episode reliability**, not just similarity — e.g.
   discount by the matched episode's own replay/consolidation count (`Episode.replay_count`
   already exists) as a proxy for how well-estimated its stored `output` is. Worth trying before
   concluding explicit recall is architecturally useless — the current blend formula may simply
   be blind to a signal (source reliability) that's already sitting in the data structure it
   reads from.

---

### Experiment C3d — frequency-equalization test (stamped 2026-07-26, before implementation)

**Question**: C3c refuted the category-confusion hypothesis (paraphrase recalls cleanly match
prior paraphrases, never prototypes) and left the sparse-training-signal hypothesis as the
leading candidate: paraphrases recur ~5x less often than prototypes (62 vs 312 fired-recall
opportunities in the C3c run), so their stored `output` snapshots may be noisier/less
well-estimated, and blending in a noisy-but-confident recall could be what hurts. C3d tests this
directly: **does the paraphrase-specific harm shrink or disappear when paraphrase and prototype
content recur at the SAME frequency**, removing the frequency confound entirely?

**Design**: reuse C3b's confound-control methodology (same schedule, both arms on/off,
stratified by tier) but with a new schedule that alternates prototype and paraphrase content
1:1 (novel content dropped from this specific test — it contributed nothing informative in C3c,
firing zero recalls by construction). 400 cycles, strictly alternating P/X where P cycles
through the 4 prototypes and X cycles through the 4 paraphrases — giving ~200 occurrences each,
comparable-to-larger absolute repetition than either tier had in C3b/C3c, and now EQUAL between
them. Same 10 seeds as C1/C3/C3b (reused, not re-minted).

**Pre-registered predictions**:
- **P14 (sparse-signal hypothesis confirmed)**: the paraphrase-tier harm (on − off, mean
  negative in C3b) shrinks toward zero or reverses sign when frequency is equalized, ≥7/10
  seeds showing a SMALLER magnitude negative (or a non-negative) diff than C3b's own paraphrase
  finding (mean −0.00365). This would confirm recurrence-frequency, not category or blend
  mechanics, was the real driver.
- **P15 — the honest risk**: the harm persists at a similar magnitude even at equal frequency.
  This would refute the sparse-signal hypothesis too, and point toward something else entirely
  — e.g. a structural property of how paraphrase-topic content differs from prototype-topic
  content in this specific 4-topic design (unlikely to generalize, given C3c already showed
  clean separation) or a genuine property of the blend formula itself unrelated to frequency
  (more likely — see C3e below, tested independently regardless of C3d's outcome).

**Coordination**: harness-only change (new example file), no production code touched — the
telemetry and mechanism from C3/C3c are reused as-is.

### C3d Results (2026-07-26, full registered run, N=10 seeds, 400 cycles, exit 0)

| seed | tier | on_mean | off_mean | diff | n |
|---|---|---|---|---|---|
| alpha | prototype | 0.25544 | 0.25617 | −0.00073 | 189 |
| alpha | paraphrase | 0.23010 | 0.23413 | −0.00403 | 190 |
| beta | prototype | 0.23753 | 0.23784 | −0.00031 | 189 |
| beta | paraphrase | 0.22801 | 0.23215 | −0.00414 | 190 |
| gamma | prototype | 0.23783 | 0.23544 | +0.00239 | 189 |
| gamma | paraphrase | 0.21551 | 0.21749 | −0.00197 | 190 |
| delta | prototype | 0.24124 | 0.23620 | +0.00504 | 189 |
| delta | paraphrase | 0.22285 | 0.23053 | −0.00767 | 190 |
| epsilon | prototype | 0.24459 | 0.24440 | +0.00019 | 189 |
| epsilon | paraphrase | 0.21993 | 0.22035 | −0.00042 | 190 |
| zeta | prototype | 0.25698 | 0.25554 | +0.00144 | 189 |
| zeta | paraphrase | 0.24056 | 0.24431 | −0.00375 | 190 |
| eta | prototype | 0.25957 | 0.25556 | +0.00401 | 189 |
| eta | paraphrase | 0.22455 | 0.23073 | −0.00618 | 190 |
| theta | prototype | 0.26829 | 0.26974 | −0.00145 | 189 |
| theta | paraphrase | 0.23402 | 0.24143 | −0.00741 | 190 |
| iota | prototype | 0.25438 | 0.25680 | −0.00242 | 189 |
| iota | paraphrase | 0.20940 | 0.20809 | +0.00131 | 190 |
| kappa | prototype | 0.25768 | 0.26050 | −0.00282 | 189 |
| kappa | paraphrase | 0.23193 | 0.24131 | −0.00938 | 190 |

**Prototype**: 5/10 seeds positive (gamma, delta, epsilon, zeta, eta), mean diff = +0.00053 —
a coin flip, consistent with every prior prototype-tier finding in this program (recall neither
helps nor hurts prototype-tier content).

**Paraphrase**: 9/10 seeds negative (only iota positive), mean diff = **−0.00436** — numerically
*larger in magnitude* than C3b's own paraphrase finding (mean −0.00365) under the old ~4:1
frequency schedule. **P14 tally** (seeds where the diff is less negative than C3b's baseline,
i.e. the harm shrank): only **3/10** (gamma, epsilon, iota) — far short of the ≥7/10 needed to
call P14 supported.

**Off-arm baseline context** (recall disabled, both arms use identical schedules and seeds, so
this isolates inherent content predictability): mean off_mean for prototype = 0.25082, for
paraphrase = 0.23005. Paraphrase content is measurably harder to predict than prototype content
*even with recall entirely off* — a ~0.02-bit gap that has nothing to do with the recall
mechanism.

#### Verdict

**P14: REFUTED.** Equalizing recurrence frequency (paraphrase now recurs exactly as often as
prototype, ~190 vs ~189 opportunities per seed, versus C3b/C3c's ~62 vs ~312) did **not** shrink
the paraphrase-specific harm. If anything the harm is comparable-to-marginally-worse
(−0.00436 vs −0.00365) at equal frequency. This rules out "paraphrase episodes are noisier
because they're rarer" as the driver — C3c already showed the matched episodes are always
same-tier (not a category-confusion story), and C3d now shows the harm isn't a frequency/
under-training story either.

**P15 (the honest risk) is effectively confirmed**: the harm persists at a similar magnitude
regardless of frequency, pointing away from anything about *how often* paraphrase content
recurs and toward something else. The new lead is the off-arm baseline gap noted above:
paraphrase content has a lower off-recall bits-saved mean than prototype content in every
single seed of this run (a first-glance eyeball of the table confirms this holds seed-by-seed,
not just on average) — i.e. paraphrases are inherently harder for the CfC to predict from
context alone, independent of recall. This raises a **new, not-yet-tested hypothesis**: recall's
blend may be disproportionately harmful specifically on *harder-to-predict* content, because
substituting a single point-estimate (the recalled episode's stored output) for the model's own
distribution is a worse trade when the true continuation is more uncertain to begin with —
regardless of category or frequency. This is a property of the blend mechanism interacting with
content difficulty, not a property of paraphrases per se.

#### Follow-ups

1. **Test the content-difficulty hypothesis directly (not yet registered)**: design a schedule
   that varies inherent predictability (e.g. deterministic/formulaic sentences vs. genuinely
   open-ended ones) while holding frequency and surface-level category equal, and check whether
   recall harm tracks off-arm baseline difficulty rather than tier label. This would need new
   content, not a reuse of the existing PROTOTYPES/PARAPHRASES arrays (since tier and difficulty
   are still confounded there — every paraphrase happens to also be the harder-to-predict
   sentence in this design).
2. **C3e (blend-formula fix, mentioned in the C3c follow-ups)** remains worth trying regardless
   of C3d's outcome: discount the blend weight by the matched episode's own `replay_count` (a
   reliability proxy already in the data structure) or, per the new lead above, by some measure
   of the source content's own predictability — before concluding explicit recall is
   architecturally net-harmful rather than just naïvely blended.
3. Three independent hypotheses for the paraphrase harm are now refuted or unsupported
   (category confusion — C3c; recurrence frequency — C3d) or never separated from a confound
   (dose-response — C3b's initial read). The remaining live lead is content-difficulty
   interacting with the blend formula. This is a reasonable checkpoint to pause the diagnostic
   chain and consolidate findings before opening a fourth experiment.

---

### Experiment C3e — content-difficulty test (stamped 2026-07-26, before implementation)

**Question**: C3c refuted category confusion; C3d refuted recurrence-frequency. Both left one
observation unexplained: in every C3d seed, paraphrase content's off-arm (recall-disabled)
bits-saved baseline was measurably lower than prototype's (mean 0.230 vs 0.251) — i.e.
paraphrases are inherently harder to predict from context alone, entirely independent of
recall. C3e asks directly: **does recall harm scale with a content set's inherent
predictability, regardless of category label (prototype/paraphrase) or recurrence frequency
(already equalized by C3d's design)?**

**Design**: two new content tiers, `easy` and `hard`, deliberately built to differ in inherent
predictability while being structurally identical to C3d's tiers in every other respect (4
sentences each, alternating 1:1, 400 cycles, same 10 seeds, same on/off confound-control
harness). Neither tier is a paraphrase of the other — this is a fresh axis, not a re-run of
prototype/paraphrase under new names, specifically to avoid re-confounding difficulty with that
category distinction.
- `EASY` (four short, single-template sentences — same subject-verb-location structure
  repeated with only the nouns varying, deliberately low-surprise): "The cat sat on the mat.",
  "The dog sat on the rug.", "The bird sat on the branch.", "The fish swam in the bowl."
- `HARD` (four longer, structurally-unrelated, information-dense sentences — deliberately
  higher-surprise, no shared template): "Quantum entanglement links particles across arbitrary
  distances instantaneously.", "The committee postponed its decision pending further budgetary
  review.", "Despite forecasts, unexpected turbulence delayed the connecting flight
  significantly.", "Her handwriting, illegible at first glance, revealed a hidden apology."

**Manipulation check (prerequisite for interpreting P16/P17)**: HARD's off-arm bits-saved mean
must be measurably lower than EASY's, confirming the difficulty manipulation worked as intended
— without this, P16/P17 aren't interpretable.

**Pre-registered predictions**:
- **P16 (content-difficulty hypothesis confirmed)**: the HARD tier shows a more negative
  on−off diff (larger recall harm) than the EASY tier in ≥7/10 seeds, mirroring the
  paraphrase-vs-prototype pattern from C3b/C3d but now driven by a fresh, category-free
  difficulty manipulation.
- **P17 — the honest risk**: no reliable difference between EASY's and HARD's harm magnitude.
  This would refute content-difficulty as a general driver too, leaving the paraphrase-specific
  harm as either a property of the blend formula's interaction with THESE SPECIFIC four
  sentences (a narrow, low-generalization finding) or something not yet hypothesized.

**Coordination**: harness-only change (new example file), reuses C3d's confound-control
methodology and telemetry as-is — no production code touched.

### C3e Results (2026-07-26, full registered run, N=10 seeds, 400 cycles, exit 0)

| seed | tier | on_mean | off_mean | diff | n |
|---|---|---|---|---|---|
| alpha | easy | 0.28063 | 0.28689 | −0.00626 | 189 |
| alpha | hard | 0.19766 | 0.20248 | −0.00481 | 190 |
| beta | easy | 0.23437 | 0.23702 | −0.00266 | 189 |
| beta | hard | 0.20209 | 0.20131 | +0.00078 | 190 |
| gamma | easy | 0.22521 | 0.22760 | −0.00239 | 189 |
| gamma | hard | 0.22569 | 0.22055 | +0.00515 | 190 |
| delta | easy | 0.24121 | 0.24109 | +0.00012 | 189 |
| delta | hard | 0.19872 | 0.20081 | −0.00209 | 190 |
| epsilon | easy | 0.24753 | 0.24777 | −0.00023 | 189 |
| epsilon | hard | 0.20321 | 0.20137 | +0.00184 | 190 |
| zeta | easy | 0.21888 | 0.22273 | −0.00384 | 189 |
| zeta | hard | 0.21220 | 0.20879 | +0.00341 | 190 |
| eta | easy | 0.25287 | 0.25373 | −0.00085 | 189 |
| eta | hard | 0.20519 | 0.20761 | −0.00242 | 190 |
| theta | easy | 0.24227 | 0.25085 | −0.00859 | 189 |
| theta | hard | 0.19528 | 0.20050 | −0.00522 | 190 |
| iota | easy | 0.24078 | 0.24228 | −0.00150 | 189 |
| iota | hard | 0.20627 | 0.21038 | −0.00412 | 190 |
| kappa | easy | 0.24601 | 0.24716 | −0.00115 | 189 |
| kappa | hard | 0.20447 | 0.20305 | +0.00142 | 190 |

**Manipulation check — PASSED, decisively**: HARD's off-arm (recall-disabled) bits-saved mean
is lower than EASY's in **10/10 seeds** (mean 0.20568 vs 0.24571, gap +0.04003) — a substantially
larger and more consistent gap than C3d's paraphrase-vs-prototype off-arm difference (0.230 vs
0.251, gap 0.021). The difficulty manipulation worked as intended and worked strongly.

**EASY tier**: 9/10 seeds negative (only delta positive), mean diff = **−0.00274** — a clean,
consistent harm pattern, similar in shape to C3b/C3d's paraphrase-tier finding.

**HARD tier**: 5/10 seeds positive, mean diff = **−0.00061** — a coin flip indistinguishable
from zero, similar in shape to every prior prototype-tier finding.

**P16 tally** (seeds where HARD is more harmed than EASY, i.e. hard's diff more negative): only
**3/10** (delta, eta, iota) — far short of the ≥7/10 needed to support P16.

**Reverse tally** (seeds where EASY is more harmed than HARD): **7/10** (alpha, beta, gamma,
epsilon, zeta, theta, kappa) — the opposite of what P16 predicted, at a magnitude that would
itself clear this program's own SUGGESTIVE bar (7-8/10) if it had been the pre-registered
direction.

#### Verdict

**P16: REFUTED, and not merely null — the effect runs in the opposite direction.** Despite the
difficulty manipulation working cleanly and strongly (harder-to-predict HARD content confirmed
10/10), it is the EASY (more predictable, more template-homogeneous) tier that recall reliably
harms (9/10, mean −0.00274), while HARD (less predictable, structurally heterogeneous) shows
no reliable effect (5/10, mean −0.00061, indistinguishable from a coin flip). **Raw
predictability, in the direction hypothesized, is not the driver of recall harm.**

**A confound this design itself introduced, disclosed honestly**: EASY and HARD were built to
differ in predictability, but by construction they *also* differ in a second, unintended
dimension — **within-tier surface homogeneity**. EASY's four sentences share one rigid
subject-verb-location template (only the nouns vary); HARD's four sentences are maximally
distinct from each other in topic, syntax, and length. This means the compressed 256-D
representations of EASY's four sentences likely sit closer together in that space than HARD's
four do — raising the possibility that recall in EASY sometimes fires against a *different*
EASY sentence's episode (a near-duplicate template match), not only the current sentence's own
prior occurrence, in a way HARD's heterogeneity would rarely allow. This was not measured
directly in this run (no `recall_matched_timestamp`-style cross-tabulation was done here,
unlike C3c) and is a plausible alternative explanation, not yet confirmed. **This means C3e
does not cleanly isolate "difficulty" as a variable** — predictability and within-tier
homogeneity were conflated by this specific choice of content, the same kind of construction
mistake the program has now made and caught twice (C3b's dose-response confound; this one).

#### Follow-ups

1. **Disentangle homogeneity from difficulty directly**: a cleaner C3f design would need four
   tiers (or a 2×2), independently varying "template homogeneity" (do the tier's members
   resemble each other in the compressed space) and "raw predictability" (off-arm baseline),
   rather than letting one imply the other. Cheapest version: reuse HARD's four sentences
   but also test them as more-homogeneous *paraphrase-style* variants of one another (harder
   AND homogeneous) alongside EASY's template kept as-is (easier AND homogeneous), to see
   whether homogeneity alone predicts harm regardless of difficulty.
2. **Directly measure within-tier compressed-space similarity** (pairwise cosine similarity of
   the four stored episodes per tier) as a cheap diagnostic before designing C3f — if EASY's
   four episodes are reliably closer to each other than HARD's, that's suggestive evidence
   for the homogeneity hypothesis without needing a new full run.
3. **Program status**: four hypotheses now tested for the paraphrase-tier/EASY-tier-style
   recall harm — category confusion (C3c, refuted), recurrence frequency (C3d, refuted), raw
   content difficulty in the "harder is harmed more" direction (C3e, refuted and reversed).
   The live, not-yet-tested lead is within-tier surface homogeneity. Given the design-mistake
   rate (2 of 5 sub-experiments so far needed a correction after the fact) and the increasing
   subtlety of the remaining hypothesis space, this is a natural point to check in before
   committing to a sixth registered experiment.

---

### Diagnostic — within-tier homogeneity probe (stamped 2026-07-26, before implementation)

**Question**: C3e's follow-up #2 — does EASY's compressed representation cluster more tightly
than HARD's, giving recall more opportunity to cross-match a *different* tier-mate's stored
episode? This is a cheap pairwise-similarity check, not a full A/B experiment, following C3c's
precedent for single/few-seed descriptive diagnostics rather than a 10-seed sign test.

**Mechanism (no new production code)**: `recall_similarity` is already populated whenever
`retrieve_by_input_similarity` finds any candidate, *whether or not it clears the firing
threshold* (`planning.rs`: `recall_similarity = Some(sim)` runs unconditionally before the
`sim >= threshold` gate). This means a genuine, continuous similarity score is available for
any two pieces of content: store one via a single cycle, then probe with another and read
`recall_similarity` directly off the result — no need to expose the raw 256-D compressed state.

**Design**: pairwise, not aggregate. For each tier (EASY, HARD) and each ordered pair (i, j)
with i ≠ j among the tier's 4 sentences: a **fresh** service (empty store), one cycle storing
sentence i, one cycle probing sentence j, record `recall_similarity`. 12 cross-pairs per tier.
Also 4 self-pairs per tier (store i, immediately re-probe i) as a calibration ceiling — same
text against itself, expected near 1.0, sanity-checks the metric's scale. Repeated across 3
seeds (not 10 — this is a descriptive diagnostic, cheap enough to triangulate a little without
a full registered run).

**Predictions**:
- **P18 (homogeneity hypothesis supported)**: EASY's mean cross-pair similarity is
  substantially higher than HARD's mean cross-pair similarity (clear separation, not just a
  nominal difference) — evidence the template structure really does make EASY's four
  sentences sit closer together in the compressed space.
- **P19 — the honest risk**: EASY's and HARD's cross-pair similarities are comparable. This
  would mean the compressed representation doesn't reflect surface/template similarity the way
  hypothesized, and homogeneity is not a promising lead either — at which point the recurring
  paraphrase/EASY-style harm would have three refuted explanations and one refuted-by-omission
  (homogeneity too weak to test), leaving no clear mechanistic story, only the empirical
  regularity itself.

**Coordination**: harness-only (new example file), no production code touched.

### Diagnostic Results (2026-07-26, 3 seeds, exit 0)

| seed | tier | cross_mean (i≠j, n=12) | self_mean (i=j, n=4) |
|---|---|---|---|
| alpha | easy | 0.32093 | 0.76943 |
| alpha | hard | 0.00077 | 0.65098 |
| beta | easy | 0.32112 | 0.76932 |
| beta | hard | 0.00066 | 0.65099 |
| gamma | easy | 0.32102 | 0.76923 |
| gamma | hard | 0.00068 | 0.65102 |

Values are essentially identical across all 3 seeds (genesis-seed variation barely manifests
over a single 2-cycle fresh-service probe, as expected) — a highly stable, non-noisy signal.

**Self-similarity ceilings are comparable between tiers** (EASY ~0.769, HARD ~0.651, both well
above cross-pair values, confirming the metric behaves sanely and isn't just uniformly higher
for one tier by construction).

**Cross-pair similarity differs by roughly 470x**: EASY's four sentences are mutually similar
at ~0.321 average (individual pairs range 0.206–0.499 — e.g. `gamma easy 2,0 = 0.49899`, right
at the 0.5 firing threshold), while HARD's four sentences are mutually similar at ~0.0007 on
average (individual pairs range from slightly negative to ~0.08, i.e. indistinguishable from
chance).

#### Verdict

**P18: CONFIRMED, decisively.** EASY's within-tier sentences really do sit far closer together
in the compressed representation than HARD's — not a subtle effect, a ~470x gap in mean
cross-pair similarity, with self-similarity ceilings ruling out a trivial global-scale
explanation. The template-homogeneity hypothesis has real structural support: EASY's four
near-identical-template sentences are, to the compressed encoder, meaningfully close to
*each other*, in a way HARD's four maximally-distinct sentences simply are not.

**Important scope caveat, stated honestly**: this confirms the *structural precondition* for
the homogeneity-causes-harm story — it does not yet confirm the causal mechanism itself. C3c
already showed that in the actual full runs, recalls matched the *same sentence's own* prior
occurrence 100% of the time, not a different tier-mate — so literal cross-sentence recall
firing was not observed being responsible for the original paraphrase-vs-prototype effect.
What this diagnostic adds is that EASY's region of compressed space is more "crowded" with
mutually-similar content than HARD's, which is a plausible predisposing factor (nearest-
neighbor search, consolidation dynamics, or blend-weight computation could all behave
differently when a content region is dense) but is not itself a proof that crowding is what
drives the harm. A genuine causal test (the C3f factorial from C3e's follow-up #1) is still
needed to confirm this closes the loop rather than merely being consistent with it.

#### Follow-ups

1. **C3f (not yet built)**: a real 2×2 — cross template-homogeneity (homogeneous vs
   heterogeneous four-sentence sets) against raw difficulty (easy vs hard), all four cells
   independently constructed — would cleanly separate the two properties C3e conflated and
   give a genuine causal test of which one (if either alone) drives the harm.
2. **Program status, six sub-experiments/diagnostics in**: category confusion (refuted),
   recurrence frequency (refuted), raw difficulty in the predicted direction (refuted,
   reversed), and now within-tier homogeneity (structurally confirmed as a real difference
   between the tiers, though not yet shown to be causal). This is a natural point to either
   commit to the C3f causal test or consolidate the program's findings into a final synthesis
   — flagged to the user as a checkpoint rather than unilaterally choosing.

---

### Experiment C3f — homogeneity × difficulty factorial (stamped 2026-07-26, before
implementation)

**Question**: the homogeneity diagnostic confirmed EASY's four sentences cluster ~470x more
tightly than HARD's in the compressed space, but only shows a structural correlation — C3e's
own EASY/HARD tiers conflated homogeneity with difficulty by construction, so it's still
unknown which property (if either alone, or their interaction) actually causes the recall
harm. C3f is the genuine causal test: four independently-constructed cells crossing
homogeneity (one shared rigid template vs four structurally-distinct sentences) against
difficulty (common/predictable vs information-dense/less-predictable), reusing C3e's own EASY
and HARD tiers as two of the four cells (no need to rebuild what's already measured).

**Design**: 2×2, four tiers, round-robin 1:1:1:1 (equal frequency across all four, ~100
occurrences each at 400 cycles — half of C3d/C3e's per-tier count, but every effect measured
so far has been well above the noise floor at that scale). Same 10 seeds, same on/off
confound-control methodology, same 400 cycles.

- **Cell A — Homogeneous+Easy** (= C3e's `EASY`, reused as-is): "The cat sat on the mat.",
  "The dog sat on the rug.", "The bird sat on the branch.", "The fish swam in the bowl."
- **Cell B — Homogeneous+Hard** (new, shares one rigid template, information-dense content):
  "The catalyst exhibits nonlinear behavior under high-pressure conditions.", "The alloy
  exhibits fatigue behavior under cyclic-loading conditions.", "The organism exhibits adaptive
  behavior under resource-scarce conditions.", "The algorithm exhibits divergent behavior under
  adversarial conditions."
- **Cell C — Heterogeneous+Easy** (new, four structurally-distinct but individually common/
  predictable sentences): "It is raining outside today.", "She likes hot tea in the morning.",
  "The store closes at nine tonight.", "He walked his dog around the block."
- **Cell D — Heterogeneous+Hard** (= C3e's `HARD`, reused as-is): "Quantum entanglement links
  particles across arbitrary distances instantaneously.", "The committee postponed its
  decision pending further budgetary review.", "Despite forecasts, unexpected turbulence
  delayed the connecting flight significantly.", "Her handwriting, illegible at first glance,
  revealed a hidden apology."

**Manipulation checks (prerequisite for interpreting P20-P22)**: (1) difficulty — off-arm
bits-saved mean should rank A≈C (easy) above B≈D (hard); (2) homogeneity — a pairwise
`recall_similarity` probe (same method as the prior diagnostic) on cells B and C should show
B's cross-pair similarity comparable to A's (both homogeneous, high) and C's comparable to D's
(both heterogeneous, low/chance).

**Pre-registered predictions** (mutually exclusive readings of the same 4-cell data):
- **P20 (homogeneity is the driver, independent of difficulty)**: both homogeneous cells (A,
  B) show reliable harm (≥7/10 seeds negative diff); both heterogeneous cells (C, D) do not
  (≤6/10).
- **P21 (difficulty is the driver, in C3e's reversed direction, independent of homogeneity)**:
  both easy cells (A, C) show reliable harm; both hard cells (B, D) do not.
- **P22 (interaction only — C3e's finding doesn't generalize along either axis alone)**: only
  cell A (homogeneous-and-easy, the exact condition C3e tested) shows reliable harm; B, C, and
  D do not — meaning both properties are jointly necessary and neither alone suffices.
- **P23 — the honest risk**: no cell (or an uninterpretable mix of cells) shows a clean ≥7/10
  pattern, meaning the harm doesn't cleanly decompose along either axis with this design, and
  C3e's original finding was more likely an idiosyncratic property of that specific 4-sentence
  set than a generalizable difficulty or homogeneity effect.

**Coordination**: harness-only change (new example file), reuses telemetry as-is — no
production code touched.

### C3f Results (2026-07-26, full registered run, N=10 seeds, 400 cycles, exit 0)

| cell | sign>0 | sign<0 | mean diff | mean off-arm baseline |
|---|---|---|---|---|
| A: homog_easy (= C3e `EASY`) | 5/10 | 5/10 | −0.00042 | 0.12795 |
| B: homog_hard (new) | 7/10 | 3/10 | +0.00335 | 0.27453 |
| C: heterog_easy (new) | 8/10 | 2/10 | +0.00070 | 0.19781 |
| D: heterog_hard (= C3e `HARD`) | 3/10 | 7/10 | −0.00392 | 0.10866 |

**Manipulation check 1 — homogeneity — PASSED** (extended the homogeneity-diagnostic harness
to also probe cells B and C, same fresh-service store/probe method, 3 seeds): B's cross-pair
similarity is 0.353 (comparable to A/C3e-EASY's 0.321 — both genuinely homogeneous); C's
cross-pair similarity is 0.032 (comparable to D/C3e-HARD's 0.0007 — both genuinely
heterogeneous, though C's gap to D is itself notable, see below). The homogeneity axis was
realized as intended for both new cells.

**Manipulation check 2 — difficulty — FAILED.** Off-arm baseline ranking came out as **B
(0.275) > C (0.198) > A (0.128) > D (0.109)** — completely different from the intended A≈C
(easy, high) > B≈D (hard, low). Cell B, built with more technical/information-dense
vocabulary, is actually the **most predictable cell of all four**, not the hardest. Root
cause (inferred, not directly measured): B's template is even more rigid than A's ("The [N]
exhibits [N] behavior under [ADJ] conditions" — only two content words vary per slot, arguably
tighter than A's "The [N] [V] on/in the [N]"), and structural/template regularity appears to
dominate this metric's predictability far more than lexical or semantic complexity does — the
CfC seems to learn the templated *shape* of a sentence more than its *content*. This means
C3f's difficulty axis, as built, does not isolate "difficulty" independent of homogeneity —
the two properties remain entangled, just not in the direction intended.

#### Verdict

**P20, P21, and P22 are all REFUTED** by the tallies above: no cell reaches the ≥7/10 harm
bar the way any hypothesis predicted in isolation. Neither homogeneous cell (A: 5/10, B: 7/10
*positive*) is reliably harmed together (P20 refuted); neither easy cell (A: 5/10, C: 8/10
*positive*) is reliably harmed together (P21 refuted); and cell A alone — the exact condition
C3e tested — is no longer reliably harmed either (5/10, a coin flip), so P22 (harm requires
both properties jointly, as in C3e) is also refuted. **P23 (the honest risk — no clean
decomposition) is confirmed**, but the reason is more interesting than "C3e's finding doesn't
generalize along either axis": it points to a bigger, unplanned finding below.

#### Headline finding: recall-harm status is schedule-dependent, not a stable content property

**Cells A and D are bit-for-bit the same sentences C3e used as `EASY` and `HARD`.** Comparing
the same content's tally across the two experiments:

| content (identical text) | C3e (2-tier, ~190/tier) | C3f (4-tier, ~95/tier) |
|---|---|---|
| = `EASY` / cell A | **9/10 negative**, mean −0.00274 (reliable harm) | **5/10 negative**, mean −0.00042 (coin flip) |
| = `HARD` / cell D | 5/10 negative, mean −0.00061 (coin flip) | **7/10 negative**, mean −0.00392 (reliable harm) |

**The pattern essentially inverted.** The content that was reliably harmed in a 2-tier
alternation became a coin flip in a 4-tier round-robin; the content that was a coin flip in
the 2-tier design became reliably harmed in the 4-tier design — using the identical sentences,
identical seeds, and identical on/off mechanism, differing only in how many other tiers were
interleaved and how many occurrences-per-tier resulted (~190 vs ~95). Since these simulations
are deterministic given a seed, this is not sampling noise — it is a genuine, reproducible
consequence of changing the schedule's structure. The mechanistic reason is not mysterious in
retrospect: the CfC's temporal state is ONE continuous trajectory shared across every cycle
regardless of tier, and the episodic store's composition (what else is stored nearby, what a
nearest-neighbor search competes against) also changes when more tiers are interleaved — so a
given piece of content's measured "harm from recall" is a property of *that content in that
specific schedule*, not a portable property of the content alone.

**This reframes every prior finding in this program's C3b–C3e arc**: the paraphrase-specific
harm (C3b), its clean same-tier matching (C3c), its immunity to frequency-equalization (C3d),
and its reversal under a fresh easy/hard axis (C3e) were all measured within *one specific
2-tier schedule design*. They are real, honestly-obtained findings **about that schedule** —
none of this retracts them — but this result means they should not be read as claims about
"paraphrase content" or "EASY content" as portable, context-free properties. A genuinely
different interleaving of the identical text materially changed which tier got harmed.

#### Follow-ups

1. **A new standing methodological lesson for this codebase** (worth its own durable memory
   entry, alongside the existing within-arm-correlation lesson): recall-harm findings from any
   fixed schedule design should be treated as claims about that schedule, not as general
   content properties, until independently re-verified under at least one different
   interleaving/tier-count/per-tier-frequency combination.
2. **The "difficulty via vocabulary" construction method is unreliable** for this metric —
   rigid template structure dominates predictability regardless of lexical content. Any future
   experiment wanting a genuine difficulty axis should either vary structural regularity
   directly (not vocabulary) or measure off-arm baseline empirically before assigning tier
   labels, rather than assuming a label from subjective judgment.
3. **Program status after six sub-experiments/diagnostics**: category confusion (refuted),
   recurrence frequency (refuted), content difficulty (refuted, reversed within one schedule),
   within-tier homogeneity (structurally confirmed but not shown causal, and now shown
   insufficient alone to predict harm across schedules), and finally schedule-dependence
   itself (the strongest and most surprising finding, discovered incidentally rather than
   through a dedicated hypothesis). This is a natural point to close out the diagnostic arc
   with a full program synthesis rather than open a seventh experiment chasing content
   properties that this result suggests may not be the right level of description at all.

---

## 8. Experiment C2 — compression-gated Chronicle (stamped 2026-07-27, before implementation)

### Question

Does gating episodic-memory writes/priority by a compression signal (`bits_saved_by_update` —
how much a single real training update on this episode's own `(input, target)` pair actually
reduces loss) select meaningfully different, and more genuinely replay-useful, episodes than
the current gate — a hard `psi < psi_threshold` reject (§1's "Φ" here is confirmed, again, to
be coherence, not a separate real Φ metric — `cycle_phases_memory.rs:353-354`), followed by
unconditional storage and a `priority_score` (psi + recency − replay-count-penalty +
0.2·prediction_error) used only at eviction/sampling time
(`crates/domains/symthaea-memory/src/episodic_replay.rs:491-523`)? No existing gate uses a
compression/loss-reduction signal at all today.

### Ground truth (surveyed 2026-07-27, file:line-verified)

- `Episode::with_metadata` is constructed every cycle from `current_phi = coherence_summary
  .smoothed_coherence`, `prediction_error`, `valence`, `coherence`
  (`cycle_phases_memory.rs:352-370`), then `replay.store_if_significant(episode)` is called
  unconditionally.
- `store_if_significant` (`episodic_replay.rs:491-523`): reject outright if `episode.psi <
  self.config.psi_threshold`; otherwise ALWAYS store (push to `self.episodes`), updating
  running `average_psi`/`min_psi_in_buffer` stats. Capacity overflow doesn't evict immediately
  — "eviction happens during sampling" (comment at line 517-518) — so the write gate really is
  just the single `psi_threshold` check; `priority_score` only matters for who gets
  sampled/evicted later, not who gets written.
- `Episode::priority_score` = `survival_value` = `psi + recency_bonus·recency_weight −
  replay_count_penalty + 0.2·prediction_error` (`episodic_replay.rs:207-230`) — matches the C2
  sketch's "PE already gates writes and weights priority 0.2" note exactly.
- `HdcLtcBridge::eval_loss_from` (added this session, §6's sketch item 1) is available and
  tested (20/20 `hdc_ltc_bridge` tests pass) — the one piece of new mechanism the sketch called
  for is done.
- `HdcLtcBridge: #[derive(Debug, Clone)]` (confirmed, `hdc_ltc_bridge.rs:177`) — a full,
  independent branch-and-continue (weights + evolution state, everything) is a single
  `.clone()`, no new snapshot/restore plumbing needed for the causal replay-utility check below.

### Standing lessons applied from this session's own history (both directly relevant to C2's
### exact design, not boilerplate)

1. **Within-arm correlation is not causation** (`feedback_within_arm_correlation_is_not_causation.md`,
   discovered in C3b): a within-arm correlation between `bits_saved_by_update` and some later
   outcome is NOT evidence the compression gate causes better replay outcomes. The
   "replay-utility" metric below is therefore designed as a true on/off counterfactual — clone
   the bridge at a branch point, replay the candidate episode in one branch only, compare
   held-out future loss between branches — not an observed correlation between priority score
   and anything.
2. **Recall-harm findings are schedule-dependent** (`feedback_recall_harm_is_schedule_dependent.md`,
   discovered in C3f): any gate-comparison finding from one fixed content schedule is a claim
   about that schedule, not a general property. This experiment runs the identical comparison
   under **two structurally different content schedules** (varying tier count/diversity) before
   drawing any general conclusion — not optional here, given this program's own most recent,
   most surprising finding.

### Mechanism

Deliberately NOT wired into the live `cycle_phases_memory.rs`/`store_if_significant` gate
itself — this is a self-contained harness using `HdcLtcBridge` + the real `Episode`/
`EpisodicMemory` types directly (same pattern as the Hoffman investigation's `Organism`-only
harnesses; matches this program's own "measure before you gate, default new paths to inert"
principle at the most conservative extreme: zero change to production gating behavior at all).

1. Feed a schedule of `(input, target)` pairs through a single `HdcLtcBridge` (varied-content
   tiers, reusing this program's established multi-tier construction pattern).
2. At each step: `pre_loss = eval_loss_from(start, input, target, dt)`, then a real
   `train_step_from(start, input, target, dt, lr)` (the actual, currently-existing training
   consumer this network already runs), then `post_loss = eval_loss_from(start, input, target,
   dt)` — deriving `bits_saved_by_update = log2(pre_loss / post_loss)` when `post_loss > 0`
   (interpretable as "loss-halvings bought by this update," bits-like; clamped/disclosed for the
   `post_loss → 0` edge case).
3. Build a real `Episode` per step (`psi` proxied by `1.0 − normalized pre_loss` — disclosed
   explicitly as a simplified stand-in for the live loop's coherence-derived channel, not a
   literal reproduction; `prediction_error` = `pre_loss` directly) plus the new
   `bits_saved_by_update` value carried alongside (not on the `Episode` struct itself for this
   harness — see Follow-ups for why a real struct field is deferred).
4. **Current-gate selection**: apply the real `psi_threshold` check + `priority_score` exactly
   as production code does, via the actual `Episode`/`EpisodicMemory` types.
5. **Compression-gate selection**: rank by `bits_saved_by_update`, threshold chosen to match the
   CURRENT gate's acceptance *rate* (not an independently-chosen threshold) — so any set
   difference reflects which episodes are selected, not merely how many.
6. **Episode-set overlap**: Jaccard/proportion-overlap between the two selections — the sketch's
   first metric.
7. **Causal replay-utility check (true counterfactual, not correlation)**: for episodes selected
   by exactly one gate but not the other (the genuinely informative disagreement set), clone the
   bridge at that point in the stream, replay the candidate episode in one clone only (an extra
   `train_step_from` on its own `(input, target)`), then run BOTH clones forward through the
   *same* held-out future portion of the schedule, comparing mean loss on that held-out segment.
   This directly measures whether replaying this specific episode helps future prediction — the
   sketch's "priority-rank correlation with later replay utility" metric, redesigned as a
   controlled experiment per the standing lesson above rather than an observed correlation.
8. **The honest endpoint** (sketch's own framing, preserved verbatim as a live possibility): if
   the compression-gate selection differs from the current (PE/psi) gate's selection by only a
   small margin, or if replay-utility doesn't differ between the two gates' disagreement sets,
   that null result must be reported as such, not reframed as a smaller positive finding.
9. Repeat steps 1-8 under **two structurally different schedules** (e.g. a 2-tier alternation vs
   a more diverse round-robin, matching this program's own established content-tier
   constructions) before drawing any conclusion that generalizes beyond "under this one
   schedule."

### Coverage caveats (from the original sketch, still accurate)

Async-trainer path has no loss signal to measure (out of scope for this harness, which trains
synchronously by construction); non-learning cycles produce no update to measure (not modeled
here since this harness always trains every step — a simplification disclosed, not silently
assumed away).

### C2 Results (2026-07-27, both schedules, exit 0)

**First run found a calibration bug, fixed before trusting anything downstream**: the initial
harness used `psi_proxy >= PSI_THRESHOLD` (0.3) directly, matching the real gate's literal
threshold value — but `psi_proxy`'s invented `1/(1+pre_loss)` mapping was never calibrated to
this harness's actual loss scale, and the threshold never rejected anything (399/399 selected,
both schedules — a fully degenerate "current gate"). Fixed by switching both gates to rank-based
top-half selection (`current_gate_selected`'s doc comment records this honestly rather than
silently patching the number) — a cleaner design anyway, since it removes the dependency on an
arbitrary absolute cutoff and directly compares two ranking criteria at an equal, non-degenerate
acceptance rate.

**Manipulation check — FAILED, and the reason is mechanistically clear, not a bug**:
`bits_saved_by_update` (log2 of the pre/post loss ratio from one training step) came back
**nearly constant across every episode, in both schedules** — schedule A: min=0.0290, max=0.0290,
mean=0.0290; schedule B: min=0.0289, max=0.0290, mean=0.0290 — despite `pre_loss` itself varying
genuinely (0.096-0.360 in schedule A). Verified this isn't a measurement bug: `log2(1/0.98) ≈
0.02915`, matching the observed value almost exactly — **a single gradient-descent step at this
learning rate (0.01) buys an approximately constant ~2% RELATIVE loss reduction regardless of
absolute loss magnitude or content**, a well-understood local-linearity property of small-step
gradient descent (not a property specific to this network or a coding error). The metric as
specified — a single step's relative loss delta — was never going to discriminate content
difficulty; it mostly reflects the optimizer's own step size.

**Consequence for the rest of the comparison**: because the compression signal carries almost
no real content-dependent information, its "selection" is close to arbitrary tie-breaking, not
a genuinely different priority criterion. The downstream numbers are reported for completeness
but should NOT be read as evidence about compression-gating specifically:

| | Schedule A (alternating 1:1) | Schedule B (skewed 1:3) |
|---|---|---|
| Jaccard overlap(current, compression) | 0.6583 | 0.1637 |
| disagreement set size (each direction) | 41 | 143 |
| mean causal replay-utility delta, current_only (n=5) | +0.001035 | +0.000024 |
| mean causal replay-utility delta, compression_only (n=5) | −0.001440 | −0.000482 |

All deltas are tiny relative to baseline held-out loss (~0.24-0.26) — consistent with a second,
independent honest finding: a single extra replay of one episode has a near-negligible effect on
held-out generalization loss in this substrate at this scale, regardless of which gate picked it.
The overlap swinging from 0.66 (schedule A) to 0.16 (schedule B) despite the *same*
near-constant `bits_saved` signal in both cases is itself evidence the "compression gate" is
mostly picking up tie-breaking/ordering artifacts that happen to differ by schedule, not a
real content signal — a second echo of this program's schedule-dependence lesson, from an
entirely different mechanism than C3f's.

#### Verdict

**C2, as literally specified in the original sketch (single-step pre/post loss delta as the
compression signal), does not work — not because compression gating is a bad idea, but because
this specific formulation of `bits_saved_by_update` doesn't discriminate content at all** at a
realistic training learning rate. This is a genuine, mechanistically-understood negative result,
not an inconclusive one: the metric's near-constancy is explained, not just observed. The
sketch's own "honest endpoint" (report a null "compression gating ≈ PE gating" result if found)
undersold the actual failure mode found here — the two gates aren't converging because they
measure the same thing, they're failing to differ because the compression signal doesn't
measure anything content-dependent at this calibration.

#### Follow-ups

1. **The natural recalibration**: measure loss reduction over multiple replay steps (or
   accumulated over a full replay batch) rather than a single step, or use a substantially
   larger learning rate for the *measurement* step specifically (distinct from the network's own
   online learning rate) — either could let genuine second-order/content-dependent curvature
   differences show through where a single small step cannot. Not attempted here — a
   sufficiently different redesign to warrant its own registration rather than a quick patch.
2. **`Episode.bits_saved_by_update` struct field** (sketch item 3) deliberately not added in
   this pass — with the signal itself shown uninformative at this calibration, adding a
   permanent field to the real `Episode` type would be premature; revisit only if a recalibrated
   measurement (follow-up 1) shows real discriminating power first.
3. **Program status**: C2 is the second predictive-compression sub-experiment (after the
   C3b-C3f/homogeneity arc) to find its own manipulation check fails for a real, explicable
   reason rather than confirming or refuting the intended hypothesis outright. Given C2's
   original scope is now answered (the sketch's literal design doesn't work, and why), and C4
   (calibrated surprise) remains the one still-unstarted sketch item, this is a reasonable point
   to check in before committing to either C2's recalibration (follow-up 1) or pivoting to C4.

### C2 recalibration — multi-step compressibility (stamped 2026-07-27, before implementation)

**Question**: does `bits_saved_by_update`, redefined as loss reduction achievable after
*sustained* training on one example rather than a single step, actually discriminate content
difficulty where the single-step version failed to?

**Why multi-step over a larger measurement learning rate** (the original follow-up's other
candidate): the single-step near-constancy has a clean first-order explanation --
`ΔL/L ≈ −lr·|∇L|²/L`, and for an MSE loss with a near-linear output projection, `|∇L|²/L` is
plausibly roughly content-invariant (dominated by activation magnitude, not error magnitude), so
a uniformly larger `lr` mostly rescales the same near-constant ratio rather than introducing
content-dependent variation -- it would need to be large enough to leave the linear regime
entirely, risking instability rather than a clean signal. Multi-step training on a *fixed*
`(start, input, target)` triple is a more directly interpretable "compression" measure in the
information-theoretic sense the program's own metric design was reaching for: how much of this
specific transition's loss can be driven out with sustained effort is a genuine complexity/
compressibility property of the content (readily-learnable content should approach a low loss
floor quickly; genuinely complex content should plateau higher even after the same effort),
independent of any single step's local-linearity artifact.

**Mechanism**: for each record already computed in the existing harness, additionally clone the
bridge *before* the real online single-step update (an extra `HdcLtcBridge::clone()`, same
technique as the existing causal replay-utility check), then call `train_step_from(&start,
input, target, dt, lr)` **K=10 times in the clone**, always passing the *same* `start` snapshot
(so only weights accumulate across the 10 calls -- the evaluation point never drifts, isolating
"how learnable is this one transition" from any state-trajectory effect). `bits_saved_by_k_steps
= log2(pre_loss / loss_after_10_steps)`. The clone is discarded after measurement; the real
bridge's actual online training (one real step per record, as before) is completely unaffected.
Report both the original single-step `bits_saved` and this new `bits_saved_by_k_steps` side by
side, so if the recalibration doesn't help either, that's visible directly, not asserted.

**Coordination**: harness-only change (extends the existing `examples/
compression_gated_chronicle_probe.rs`), no other production code touched beyond the
already-landed `eval_loss_from`.

### C2 recalibration Results (2026-07-27, both schedules, exit 0)

Extended `examples/compression_gated_chronicle_probe.rs` with `bits_saved_k_step` (K=10 training
steps on a fixed `(start, input, target)` triple in a disposable clone, per the registration
above), reported side by side with the original single-step `bits_saved`.

**The recalibration also failed to discriminate content, and the reason is the same mechanism
scaled up, not a new one**:

| metric | Schedule A (min/max/mean) | Schedule B (min/max/mean) |
|---|---|---|
| single-step `bits_saved` | 0.0290 / 0.0290 / 0.0290 | 0.0289 / 0.0290 / 0.0290 |
| K-step `bits_saved_k_step` (K=10) | 0.2900 / 0.2901 / 0.2900 | 0.2891 / 0.2900 / 0.2897 |

`10 × log2(1/0.98) ≈ 0.29146` — matches the observed K-step values almost exactly, the same way
the single-step value matched `log2(1/0.98)`. **Ten repeated steps compound the same
~2%-per-step relative reduction multiplicatively, without the network reaching any
content-dependent plateau or convergence floor within that budget** — easy (trivially
repeatable) and hard (structurally varied) content are still both in the same
locally-linear/exponential-compounding regime at K=10, not yet diverging into the
"readily-learnable content converges to near-zero loss, complex content plateaus higher" pattern
the recalibration was designed to expose.

**Consequence, as expected given the near-identical signal**: gate selections built from
`bits_saved_k_step` are nearly identical to the single-step gate's own selections (schedule A:
Jaccard overlap 0.6723 vs the single-step gate's 0.6583; schedule B: 0.1603 vs 0.1637) — a small
amount of reshuffling among near-tied values, not a meaningfully different ranking. Causal
replay-utility deltas are correspondingly unchanged in substance (schedule A: +0.001058/−0.001440
vs the single-step run's +0.001035/−0.001440; schedule B: +0.000024/−0.000707 vs +0.000024/
−0.000482) — the K-step recalibration didn't change the story because it didn't change the
underlying signal in any way that mattered for selection.

#### Verdict

**The K=10 recalibration is also a negative result, mechanistically continuous with the
single-step failure rather than a new, independent finding.** Ten steps was not a large enough
budget to let this substrate's genuinely different content types (a constant vector vs.
structurally varied waveforms) diverge into different loss trajectories — both are still
compounding at essentially the same rate this many steps in. Whether a much larger step budget
(100s-1000s of steps) would eventually reveal a real content-dependent convergence floor, or
whether this substrate's local dynamics stay linear far longer than that, is now the open
question — and it's a different, more expensive kind of question than "recalibrate the
measurement" implied: it would mean running each candidate episode's compressibility probe to
something closer to actual convergence, not just a fixed small step count.

#### Follow-ups

1. **A convergence-floor measure** (run K steps until loss stops improving by some tolerance,
   or a much larger fixed K, then compare the floor reached) is the natural next redesign if
   this line continues — a genuinely different question from either attempt so far, not a minor
   parameter tweak, and meaningfully more expensive to compute (potentially 10-100x more
   training steps per episode than this pass).
2. **Program status**: two independent attempts at a compression-based gating signal (single-step
   and 10-step relative loss reduction) both failed to discriminate content in this substrate,
   for a related, now well-characterized reason (both stay in a locally-linear/exponential
   compounding regime at these step budgets). This is a reasonable point to close out C2 for this
   session rather than open a third, substantially more expensive redesign (convergence-floor
   measurement) without confirming it's worth the cost — C4 (calibrated surprise) remains
   available as a lighter-weight alternative if the program continues.

## 9. Experiment C4 — calibrated surprise (stamped 2026-07-27, before implementation)

### Question

Does replacing the intent classifier's raw affine "confidence" (`similarity + keyword_boost`,
unbounded, not a probability) with a genuine softmax-temperature-calibrated confidence reduce
Expected Calibration Error (ECE) on a held-out labeled set — and can that calibration be
reported honestly (via `analyze_pairs`) rather than left as the `knowledge_calibration_ece`
telemetry field's current 0.0 sentinel?

### Ground truth (surveyed 2026-07-27, file:line-verified — the sketch's own description turned
### out to be stale/imprecise in one important way, corrected here)

- `knowledge_calibration_ece` (`cognitive_loop/types/telemetry.rs:1240`) is fed from
  `KnowledgeManager::telemetry().calibration_ece` (`knowledge/manager.rs:673`), which reads
  `self.calibration_audit.ece()` — and `CalibrationAudit::ece()` (`knowledge/manager.rs:69`)
  returns exactly `0.0` when `total_samples == 0` (`knowledge/manager.rs:70-72`).
  `calibration_audit.record(...)` is only called from `calibrate_from_prediction()`
  (`knowledge/manager.rs:1000-1019`) — confirmed via grep, **`calibrate_from_prediction` has
  zero callers anywhere in `src/`**. The original sketch's "no production feeder → 0.0
  sentinel" diagnosis is confirmed still accurate today.
- **New finding beyond the original sketch**: even if `calibrate_from_prediction` WERE wired
  in, its `correct` label is a coarse, GLOBAL, cycle-level proxy
  (`prediction_error < 0.5`) applied UNIFORMLY to every fact in `last_search_results` —
  conflating "was this cycle's overall CfC prediction good" with "was this specific retrieved
  fact's own confidence well-calibrated." A fact irrelevant to the current prediction, but
  confidently retrieved anyway, would be scored "correct" whenever the cycle's prediction
  happened to be good for unrelated reasons. This is exactly the "grading must be external,
  never a coarse self-check" failure mode the sketch's own C4 text warned about — just located
  in a different, previously-undiagnosed spot than the sketch anticipated.
- **The sketch's "intent classifier" is `SemanticIntentClassifier`** (`language/semantic_intent.rs`),
  confirmed live in the real facade pipeline (`language/mod.rs:996`,
  `self.intent_classifier.classify(input)`) — not a dormant/test-only path. Its `classify()`
  method (`semantic_intent.rs:211-237`) computes `confidence: f32 = similarity + keyword_boost`
  — an unbounded affine score (boost constant `0.05`, `semantic_intent.rs:245`), not a
  probability, exactly matching the sketch's description.
- `analyze_pairs(data: &[(f64, bool)], n_bins) -> Option<CalibrationReport>`
  (`consciousness/recursive_improvement/calibration_analytics.rs:49`) is real, general-purpose
  calibration analytics (ECE, Murphy's Brier decomposition, Wilson CI) — exactly what the
  sketch pointed to, already implemented, unused by anything touching the intent classifier
  today.
- **A labeled validation set does not yet exist at usable size**: `semantic_intent.rs`'s own
  `#[cfg(test)]` module has only 6 example queries total across the 5 `IntentCategory` variants
  (2 NixOS, 1 each of the other 4) — far too small for a non-degenerate calibration fit (this
  program's own `feedback_suspiciously_tight_bootstrap_ci_is_a_red_flag.md` lesson applies
  directly: a tiny sample would give an artificially tight, meaningless ECE either way).

### Scope correction from the original sketch, disclosed rather than silently absorbed

The sketch conflated two architecturally separate concerns under one telemetry field: (a) the
intent classifier's confidence formula (a real, live, fixable thing), and (b) the KNOWLEDGE
MANAGER's `calibration_ece` field specifically (a different subsystem, fed — if it were fed —
by knowledge-graph search results, not intent classification at all). **This pass scopes to
(a) only.** Fixing (b) properly would mean redesigning `calibrate_from_prediction`'s per-fact
correctness signal (a distinct, separately-scoped task, not attempted here) — wiring intent-
classifier calibration into `knowledge_calibration_ece` specifically would misleadingly conflate
two unrelated subsystems' calibration quality under one label, which this pass will not do.

### Mechanism

1. **New, purely additive method** on `SemanticIntentClassifier`: `confidence_calibrated(scores:
   &[(IntentCategory, f32)], beta: f32) -> f32` — softmax over the (already-computed) per-category
   scores at temperature `beta`, returning the probability mass on the top category. The
   existing `IntentClassification.confidence` field (raw affine score) is left completely
   untouched — nothing that currently reads it changes behavior. This mirrors the program's
   established "additive, not breaking" pattern (`eval_loss_from` alongside `train_step_from`).
2. **A modest, hand-authored, explicitly-disclosed labeled validation set**: ~8 example queries
   per `IntentCategory` (5 categories, ~40 total) — large enough to avoid the tiny-sample
   degenerate-ECE trap the 6-example test set would hit, still disclosed as hand-curated, not a
   gold-standard corpus.
3. **Offline β-fit via grid search**: for a range of `beta` values, compute
   `confidence_calibrated` for every labeled example, form `(confidence, was_correct)` pairs,
   call `analyze_pairs` to get each `beta`'s ECE, pick the minimizer. Compare against the raw
   affine score's own ECE (clamped to `[0,1]` for a fair `analyze_pairs` comparison, since the
   unbounded raw score isn't naturally a probability) as the baseline this is meant to beat.
4. **Reporting, not live per-cycle wiring**: given no genuine per-cycle ground-truth signal for
   intent classification exists in production (unlike, say, next-token prediction error), the
   honest design is to report the *offline* calibration quality of the currently-deployed
   classifier (computed once from the labeled set at the fitted `beta`), not force a live
   per-cycle accumulating metric fed by a proxy correctness signal — the exact failure mode
   already found in `calibrate_from_prediction`. No change to `knowledge_calibration_ece`'s
   live wiring in this pass (see scope correction above).

### C4 Results (run 2026-07-27)

Implementation: `SemanticIntentClassifier::confidence_calibrated` (`language/semantic_intent.rs`,
purely additive, 3 unit tests, existing `confidence` field/callers untouched) +
`examples/calibrated_intent_confidence.rs` (40-example held-out labeled set, deliberately
distinct wording from the classifier's own prototype-building examples — verified by inspection,
not just intent — a genuine generalization test, not training-set recall).

**Held-out accuracy**: 25/40 (62.5%) top-1. **Baseline diagnostic** (clamped raw affine
confidence): range **0.5101–0.5980**, a 0.09-wide band — confirms the classifier's own doc
comment ("Character n-gram HDC encodings produce narrow score margins for short queries") in a
concrete, measured number. With 10 `analyze_pairs` bins spanning `[0,1]`, that band sits entirely
inside a single bin: `BASELINE ECE=0.0825 resolution=0.0000` — a **single data point**
(`|mean_conf(0.5425) − bin_accuracy(0.6250)| = 0.0825`), not a genuine multi-bin calibration
curve. `resolution=0.0000` is the tell: the predictor makes essentially the same confidence claim
regardless of instance, so it cannot be miscalibrated in any way `analyze_pairs`'s binning can
detect — the low ECE is a **degenerate-predictor artifact**, the same failure class this
program's own `feedback_suspiciously_tight_bootstrap_ci_is_a_red_flag.md` lesson warns about,
applied here to "suspiciously good ECE" rather than "suspiciously tight CI."

**Grid search**, extended from the registered range up to `beta=1000` specifically to find the
interior turnaround the softmax mechanism predicts (as `beta→∞`, `confidence_calibrated`
converges to 1.0 for every example regardless of margin, which should push ECE back up toward
`|1.0 − 0.625| = 0.375`):

| beta | ECE | Brier | reliability | resolution |
|---|---|---|---|---|
| 0.5 | 0.4225 | 0.4120 | 0.1785 | 0.0000 |
| 5.0 | 0.3987 | 0.3838 | 0.1590 | 0.0000 |
| 20.0 | 0.3050 | 0.2879 | 0.1272 | 0.0689 |
| 40.0 | 0.1767 | 0.2015 | 0.0458 | 0.0725 |
| **80.0** | **0.0949** | **0.1582** | 0.0197 | 0.0879 |
| 100.0 | 0.1446 | 0.1627 | 0.0387 | 0.1080 |
| 250.0 | 0.2763 | 0.2509 | 0.0850 | 0.0612 |
| 1000.0 | 0.3682 | 0.3566 | 0.1393 | 0.0142 |

The predicted turnaround is exactly what happens — ECE and Brier both bottom out around
`beta≈60–100` and rise back toward the baseline-mismatch region by `beta=1000`. This is a real
sanity check that the whole apparatus (softmax mechanism, grid, metric) is behaving as designed,
not noise.

**Honest verdict — the naive top-line ECE comparison and the deeper calibration-quality question
give different answers, and both need reporting:**

- **By raw ECE**, the baseline nominally wins at every beta tested (best calibrated ECE 0.0949 at
  `beta=80` vs. baseline 0.0825). Registering this plainly: **the original question, "does
  calibrated confidence reduce ECE," does not have a clean yes at n=40.** The baseline's
  advantage is not because it is well-calibrated — it is because it is *uninformative*
  (resolution≈0), and a 10-bin ECE estimate at n=40 (≈4 samples/bin) is too underpowered to
  penalize that degeneracy the way it should.
- **By Brier score** (a proper scoring rule that does not require binning and so is not subject
  to the same bin-sparsity artifact), the calibrated confidence wins decisively and consistently
  across a wide beta range: `beta=40..150` all beat the baseline's `0.2295`, bottoming at
  `beta=80`'s `0.1582` — a **~31% reduction**. This is the metric that should be trusted at this
  sample size, and by it the calibrated confidence is a genuinely better probabilistic predictor,
  not just a differently-shaped one.
- **Resolution** confirms this directly: the calibrated confidence spreads meaningfully across
  instances (0.07–0.11 in the beta≈40–150 range) where the baseline has none at all — it is
  actually discriminating between more- and less-confident predictions, which is the entire point
  of a confidence signal, independent of what any single aggregate metric reports.

**Conclusion**: `confidence_calibrated` is a real, substantive calibration improvement over the
raw affine score by the metric that is actually trustworthy at this sample size (Brier +
resolution), but the registered question's literal framing ("reduce ECE") is **not supported** at
n=40/10-bins, because the baseline's apparent ECE advantage is a measurement artifact of its own
degeneracy rather than genuine calibration quality. Reporting both, undiluted, per house
convention — this is a mixed/nuanced result, not an unambiguous win, and should not be summarized
as one.

**Scope closure**: per the registered scope correction, `knowledge_calibration_ece`'s live 0.0
sentinel and `calibrate_from_prediction`'s coarse global-proxy correctness label remain
untouched and undiagnosed-further — a separate task, not attempted here. No production call site
of `SemanticIntentClassifier::classify()` was modified; `confidence_calibrated` exists as an
additive, currently-unused-in-production offline-analysis method only.

---

*Program owner: this session (registered via MASTER_ROADMAP workstream row). Results append in
place; retractions in place with dated bold annotations, per house convention.*
