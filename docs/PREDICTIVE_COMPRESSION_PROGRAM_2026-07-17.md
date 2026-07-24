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
- **C3 (episodic anticipation)**: minimal content path — at high-similarity recall, blend the
  recalled episode's *successor* encoding into the prediction (or offer it as a second
  prediction head scored separately). Endpoint: order-sensitivity probe (keystone Phase-3
  design, new example file) moves from noise (±0.01) to signal in the wired arm only. This
  targets the sharpest capability gap keystone named; success criteria and blend mechanics to
  be registered after C2's data shows which episodes are worth recalling.
- **C4 (calibrated surprise)**: replace the intent classifier's affine "confidence" with
  `softmax(β·sim)` over class prototypes; fit β offline by minimizing `analyze_pairs` ECE on a
  held-out labeled set; wire real `(confidence, correct)` records into `CalibrationAudit`
  (`manager.rs`) so `knowledge_calibration_ece` stops reading the 0.0 sentinel — and port the
  `ece_computed` absent-vs-zero guard (`b97ed86042` pattern) to that path regardless. Grading
  must be external (AGW-2.2 pattern), never the facade's lexical self-check.

---

*Program owner: this session (registered via MASTER_ROADMAP workstream row). Results append in
place; retractions in place with dated bold annotations, per house convention.*
