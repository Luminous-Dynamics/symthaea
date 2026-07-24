# Keystone A/B: Does the Consciousness Machinery Help? (Pre-registered)

**Registered 2026-07-17, BEFORE any run.** This is the experiment the VISION.md
thesis rests on: consciousness-like properties (integration, prediction,
self-modeling, value) are claimed to be *engineering requirements* for robust
cognition — not decoration. Until now that claim was architectural. The
2026-07-15/16 signal-integrity sprint made the instruments real (PE varies and
tracks learning, Φ discriminates input regimes, Ψ gates speech, safety tiers
transition), so the causal question is finally measurable.

## Question

At matched cycle counts on identical input, does the cognitive loop with its
consciousness machinery enabled outperform the same loop with that machinery
disabled, on externally-scored competencies?

## Arms

Config flags are consumed at construction (no runtime unplug), exactly as in E1:

| Arm | Definition |
|-----|-----------|
| `full` | Default config — all 15 consciousness subsystems on |
| `min2` | The 13 NULL-causal-load subsystems (E1 verdicts) off; meta_cognition + embodied_cognition (the two load-bearing ones) kept on |
| `off15` | All 15 subsystems off |

Known limitation, declared up front: the ConsciousnessEngine itself
(SpectralMIPFinder measurement + its learning-rate/confidence feedbacks) has no
config flag and stays on in all arms. This A/B tests the 15 flag-gated
subsystems, not the measurement spine. A follow-up would need an engine kill
switch.

## Tasks & metrics (externally scored — no self-grading)

Deterministic input schedule, identical across arms, per seed:

1. **Predictive learning** (720 cycles): a fixed 12-sentence sequence repeated
   60 times in order. Score: `learning_delta` = mean PE over repetitions 3–4
   (cycles 24–48, skipping cold start) minus mean PE over the last 2
   repetitions (cycles 696–720). Positive = the system learned to predict its
   world. PE is the scale-invariant encoder metric fixed on 2026-07-16 —
   graded against what actually arrives next, not by the system's opinion.
2. **Surprise contrast** (24 cycles): 24 novel sentences never seen before.
   Score: `surprise_contrast` = mean PE on novel − mean PE on last-2-reps
   learned. Positive and large = the system distinguishes novel from learned.
3. **Regime separation** (manipulation check, not a competency): mean
   consciousness_level on a 60-cycle repetitive coda vs the varied body.
   Confirms instruments respond in arms where machinery is on.
4. **Cost**: mean cycle µs per arm — benefits must be priced.

3 seeds (genesis phrases), all arms per seed. ~2,400 cycles/arm-seed... total
3×3×~800 = ~7,200 cycles ≈ 40–90 min wall.

## Pre-registered predictions (falsifiable, written before first run)

- **P1**: `learning_delta` > 0 in ALL arms — the encoder attention + CfC
  training learn regardless of the consciousness stack. If an arm shows no
  learning, the harness (not the thesis) is first suspect.
- **P2 (the null-13 hypothesis, from E1)**: `full` ≈ `min2` on both
  learning_delta and surprise_contrast (|difference| < 10% of the metric's
  cross-seed range). The 13 individually-null subsystems buy no joint external
  competency either.
- **P3 (the honest risk)**: `off15` ≈ `full` on the external competencies. If
  this holds, the truthful headline is: *the flag-gated consciousness
  machinery currently provides no measurable external benefit on
  prediction-learning tasks at matched compute* — and VISION.md's benchmark
  section gets that sentence verbatim. If instead `full` beats `off15` beyond
  seed noise, that is the first causal evidence FOR the thesis.
- **P4**: `full` costs more µs/cycle than `off15` (consciousness is not free);
  report the ratio.
- **P5 (manipulation check)**: regime separation in consciousness_level
  appears in `full` (expected from E2: ~0.26 vs ~0.79) and is
  attenuated/absent in `off15`… note: the engine stays on in all arms, so
  separation may persist everywhere; if so, that further isolates the engine
  (not the 15 subsystems) as the source of regime discrimination.

## What would change our minds

- `full` > `off15` on learning_delta and surprise_contrast, consistent across
  3/3 seeds, by more than the cross-seed spread → first causal support for
  "consciousness helps"; VISION gets the number.
- `full` ≈ `off15` everywhere → the 15 subsystems are (still) ballast on these
  tasks; triage proceeds by deletion/demotion; the thesis must retreat to the
  engine + gating layers or to tasks these metrics don't capture (language
  quality, ethics, safety), which then need their own A/B.
- `off15` > `full` → the machinery is actively costly beyond compute;
  strongest possible argument for the subtraction agenda.

## Results (2026-07-17, run complete, exit 0)

| arm | Δlearn per seed | Δlearn mean | surprise mean | regime-sep | PE late |
|-----|-----------------|-------------|----------------|------------|---------|
| full | **+0.0090, +0.0012, +0.0123** | **+0.0075** | −0.0055 | +0.0130 | 0.7238 |
| min2 | −0.0081, −0.0049, +0.0020 | −0.0037 | −0.0154 | +0.0213 | 0.7314 |
| off15 | −0.0081, −0.0049, −0.0018 | −0.0049 | −0.0139 | +0.0238 | 0.7327 |

### Verdicts against the pre-registered predictions

- **P1 FALSIFIED**: learning_delta was NOT positive in all arms — `min2` and
  `off15` show *anti-learning* (late PE higher than early) in most seeds. The
  encoder+CfC do not learn this task on their own; whatever learning exists
  requires machinery from the 15.
- **P2 FALSIFIED**: `full` ≉ `min2`. The sign of Δlearn flips between them in
  every seed. The 13 individually-NULL subsystems matter *jointly*.
- **P3 — the headline**: `full` beat `off15` on Δlearn in **3/3 seeds**
  (differences +0.0171, +0.0061, +0.0141), i.e. directionally consistent
  first causal evidence FOR the thesis on the learning metric. HOWEVER the
  pre-registered magnitude gate (each difference > cross-seed spread, 0.011)
  passes on only 2 of 3 seeds, and the absolute effect is tiny (~1% of the
  PE base). Honest classification: **SUGGESTIVE, NOT CONFIRMED** — needs more
  seeds for power.
- **P4 INVALID**: ambient load 20–35 from concurrent sessions (crypto/MNIST
  benchmarks) swamped µs/cycle — full's own cost varied 483–1063 µs across
  seeds, wider than any arm difference. Cost is unmeasurable in this
  environment; rerun on a quiet box.
- **P5**: regime separation appears in ALL arms (off15 actually largest) —
  confirming the pre-flagged suspicion: regime discrimination lives in the
  unflagged ConsciousnessEngine, not the 15 subsystems. CL levels do shift
  with machinery (body CL: full 0.838 / min2 0.828 / off15 0.73–0.81).

### Unregistered findings (exploratory, flag as such)

1. **Degeneracy signature**: `min2` ≈ `off15` to 4 decimals in 2/3 seeds — the
   two E1-"load-bearing" subsystems add nothing to external competency; the
   entire full-vs-ablated difference comes from the 13 individually-null
   subsystems acting jointly. Individually-null-but-jointly-load-bearing is
   the classic ablation signature of a *degenerate* (redundant) system —
   biologically plausible and previously invisible to one-at-a-time E1
   ablation. (No contradiction with E1: E1 measured PE level, not learning
   slope.)
2. **Novelty discrimination is absent everywhere**: surprise contrast negative
   in 9/9 rows — novel sentences score slightly LOWER PE than well-learned
   ones. No arm distinguishes new from known. This is the sharpest deficit
   this experiment found and a concrete next target ('full' is closest to
   neutral at −0.0055).
3. The loop's own cycle-500 self-validation fired identically in all arms
   (arm-symmetric, no confound) and reported its own honest negative:
   primitives' Φ gain −0.0001, p=0.20 → influence dampened per its Popperian
   feedback rule.

### Follow-ups

- Rerun with ≥10 seeds on a quiet box (power + honest cost measurement).
- Leave-one-in ablation over the 13 to find which subset carries the joint
  learning effect (13 arms, or bisection).
- Target the novelty-discrimination deficit: likely needs sequence-level
  prediction (the CfC predicts next-state, but PE compares within-cycle
  encodings; a next-SENTENCE prediction metric would test episodic
  anticipation properly).
- Engine kill-switch arm (requires a new config flag) to A/B the measurement
  spine itself.

## Phase 2 amendment (registered 2026-07-17, BEFORE the Phase-2 runs)

**Replication**: arms `full` and `off15` only (min2 ≈ off15 is established),
10 seeds total (the original 3 + 7 new genesis phrases). Primary endpoint:
per-seed sign of `Δlearn(full) − Δlearn(off15)`. Deterministic per
(arm, seed), so seeds are independent architecture draws.

- **CONFIRMED** if the difference is positive in ≥9/10 seeds (two-sided sign
  test p ≈ 0.021) — VISION.md upgrades the keystone row.
- **Still suggestive** at 7–8/10.
- **Not supported** at ≤6/10 — the Phase-1 result was seed luck; VISION.md
  says so.
- Cost (P4) remains excluded — box is not quiet.

**Bisection** (exploratory, seeds 0–2): two new arms — `bisectA` (the first 7
of the 13 null subsystems off: gwt, prefrontal, surprise_exploration,
thermodynamics, hierarchical_free_energy, phi_attention, predictive_processing)
and `bisectB` (the other 6 off: dream_replay, quantum_coherence, resonance,
narrative_self, temporal_consciousness, phenomenal_binding). Whichever arm
reproduces off15's anti-learning localizes the carrier subset; both-partial =
the effect is distributed (strongest degeneracy reading).

### Phase 2 results (2026-07-17, both runs exit 0)

**Replication (10 seeds): 7/10 seeds positive → pre-registered verdict:
STILL SUGGESTIVE, NOT CONFIRMED.** Per-seed Δlearn(full)−Δlearn(off15):
+0.0171, +0.0061, +0.0141, +0.0096, +0.0015, +0.0124, −0.0050, −0.0045,
−0.0056, +0.0026. Aggregate: full +0.0038 vs off15 −0.0010. The Phase-1 3/3
was partly favorable seed draw; seeds 6–8 (three consecutive genesis draws)
show the effect reversed. Sign test 7/10 is p≈0.34 two-sided; at this effect
size (~d 0.5 across seeds) confirmation needs ≥30 seeds. Honest summary: a
small, seed-heterogeneous learning benefit that some architecture draws show
and others don't — real enough to pursue, too weak to headline.

**Bisection round 1 (seeds 0–2): the carrier is localized, and cleanly.**

| arm | Δlearn per seed | note |
|-----|-----------------|------|
| bisectA (7 off: gwt, prefrontal, surprise, thermo, hFE, phi_attn, PP) | −0.0007, +0.0012, +0.0123 | **bit-identical to `full` on seeds 1–2** |
| bisectB (6 off: dream, quantum, resonance, narrative, temporal, binding) | −0.0081, −0.0049, +0.0010 | **reproduces `off15` exactly on seeds 0–1** |

Removing the A-7 changes nothing (their causal load on this metric is exactly
zero — trajectories bitwise unchanged); removing the B-6 destroys the
learning effect.

**Bisection rounds 2–3 (seeds 0–1): the carrier is ONE subsystem.**

| arm (removed) | Δlearn seed0 | Δlearn seed1 | verdict |
|---|---|---|---|
| full (reference) | +0.0090 | +0.0012 | — |
| off15 (reference) | −0.0081 | −0.0049 | — |
| B1 = {dream, quantum, resonance} | −0.0007 | +0.0012 | learning mostly preserved |
| B2 = {narrative, temporal, binding} | −0.0081 | −0.0048 | off15 reproduced |
| **no_temporal (single)** | **−0.0081** | **−0.0048** | **off15 reproduced EXACTLY — sole carrier** |
| no_binding (single) | +0.0090 | +0.0012 | bit-identical to full — zero contribution |
| no_narrative (single) | −0.0007 | +0.0012 | full-identical on seed1; partial on fragile seed0 |

**Conclusion: the entire measured "consciousness helps learning" effect is
carried by the `temporal_consciousness` subsystem.** Mechanistically apt —
the task is learning a repeated temporal sequence, and the subsystem that
models time is the one that enables predicting it. In hindsight E1 contained
the hint: temporal_consciousness was one of only two arms whose ablation
moved Φ.

Two honest readings, both true:
1. *Deflationary*: "consciousness helps" cashes out as "temporal modeling
   helps sequence prediction" — no grand integrative magic detected; 12 of
   the 15 subsystems remain causally inert on every metric measured to date
   (Ψ, CL, Φ level, PE level, learning slope, surprise contrast).
2. *Constructive*: of the fifteen consciousness subsystems, exactly one
   currently earns its keep on an external competency, and the causal-load
   triage now has a clean protected list (temporal_consciousness; narrative
   partially, seed-dependent) and a clean demotion candidate list (the rest,
   pending task batteries that might exercise them — ethics, language,
   novelty tasks were not yet in scope).

The 10-seed replication verdict (SUGGESTIVE, 7/10) applies to this carrier:
temporal_consciousness's benefit is real on most architecture draws and
absent on some — worth investigating what distinguishes seeds 6–8.

## Phase 3 amendment: order anticipation (registered 2026-07-17, BEFORE runs)

**Why**: Phase 1's surprise-contrast metric (novel vs learned sentences)
confounded *familiarity* with *order* and came out negative in 9/9 rows. The
clean test of episodic anticipation is an **order violation within fully
familiar material**: after 60 learning repetitions, present the learned
sentences but occasionally SWAP two positions. Both items are equally
familiar; only sequence distinguishes them. If the system anticipates
*what comes next* (not merely *what is known*), PE at swapped positions
must exceed PE at the same positions in clean repetitions.

**Design**: after the standard 60-rep learning phase, 10 probe repetitions:
odd-numbered probe reps swap exactly one deterministic position pair
(p = (r·5+3) mod 12 with q = (p+6) mod 12); even-numbered reps are clean
controls. Endpoint: `order_sensitivity` = mean PE at swapped slots − mean PE
at the same slots in clean reps. Arms: `full`, `off15`, `no_temporal`;
seeds 0–2.

**Pre-registered questions**:
- **Q1**: order_sensitivity > 0 in `full` (the system anticipates order)?
- **Q2**: if Q1 holds, does `no_temporal` abolish it (order anticipation is
  carried by the same subsystem as the learning effect)?
- **Q3**: `off15` as the floor reference.

Honest risk, stated in advance: PE sits near the ~0.707 uncorrelated
baseline (predictions weakly correlated with next inputs at best), so
order_sensitivity may be ≈0 everywhere — which would mean the loop's
prediction machinery does familiarity adaptation, not sequence anticipation,
and episodic anticipation does not exist yet in any arm. That result would
be published as-is.

### Phase 3 results (2026-07-17, exit 0)

| arm | seed0 | seed1 | seed2 |
|-----|-------|-------|-------|
| full | +0.0112 | −0.0029 | −0.0069 |
| off15 | −0.0115 | +0.0083 | −0.0010 |
| no_temporal | −0.0116 | +0.0071 | +0.0119 |

**Q1: NO — the pre-declared honest risk materialized.** order_sensitivity is
noise (~±0.01, sign scattered across arms and seeds, no arm consistent).
The system does not notice when fully-familiar material arrives in the wrong
order, in ANY configuration. Q2 is moot given Q1.

**Interpretation**: the loop's prediction machinery performs *familiarity
adaptation* (PE drifts down on repeated material — the Phase-1/2 learning
effect), not *episodic anticipation* (predicting WHAT comes next). This also
sharpens the temporal_consciousness finding: it improves prediction quality
on a repeated stream without conferring order anticipation — likely a
statistics-of-the-stream effect, not sequence modeling.

## Phase 4 amendment: the trainer that scrambled time (registered 2026-07-17, BEFORE runs)

**Root cause found** (plan-mode exploration after Phase 3): the per-cycle
training pair was always correct next-input prediction (enc_{t−1} → enc_t),
but `HdcLtcBridge::train_step` evolved the LIVE network as a side effect —
with enc_{t−1}, AFTER the planning phase had stepped it with enc_t — and
overwrote `current_output`. With learning firing every cycle, the live
temporal trajectory was permanently shuffled (enc_t → enc_{t−1} → enc_{t+1}
→ …). Order learning was impossible by construction, and the one-step-behind
signature explains PE sitting above the 0.707 uncorrelated baseline in every
Phase-1/3 arm. Same bug class as the sprint's predict_forward finding.

**Fix**: train_step now runs its forward pass on a scratch evolution
(NetworkStateSnapshot save/restore; weights update, live state doesn't), and
a new `train_step_from` starts the forward pass from the true historical
state (end of cycle t−2, via a 2-deep rolling pre-step snapshot queue in the
service). Twin-bridge purity regression test added (lr=0 training must leave
the trajectory bit-identical to an untrained twin).

**Pre-registered acceptance predictions** (run `--phase1` + `--order`,
seeds 0–2, after the fix):
- **A**: `full`'s late PE on the learned sequence drops **below the 0.707
  uncorrelated baseline** (real predictive correlation; pre-fix 0.72–0.77
  in every arm).
- **B**: order_sensitivity becomes **> 0 in `full` on ≥2/3 seeds** (pre-fix:
  noise, 1/3).
- **C**: `full`'s learning_delta exceeds the pre-fix +0.0075.
- **Registered honest risk**: one gradient step per pair on a random
  reservoir may still be too weak even with correct temporal structure —
  A/B/C may all fail, in which case the next scoped work is a stronger
  training signal (multiple steps per pair, echo-state ridge readout, longer
  BPTT), not silent tuning of this pass.

### Phase 4 results (2026-07-17/18, both runs exit 0)

Post-fix re-measurement, same harness, seeds 0–2:

| arm | Δlearn per seed (post-fix) | mean | PE late | order_sensitivity per seed |
|-----|---------------------------|------|---------|----------------------------|
| full | +0.0014, −0.0050, −0.0007 | −0.0014 | 0.7280 | −0.0070, +0.0050, −0.0073 |
| min2 | −0.0010, −0.0047, −0.0004 | −0.0020 | 0.7286 | — |
| off15 | −0.0010, −0.0047, −0.0005 | −0.0021 | 0.7286 | −0.0061, +0.0130, −0.0069 |
| no_temporal | — | — | — | −0.0061, −0.0069, −0.0068 |

**All three pre-registered gates FAILED:**
- **A FAIL**: PE late ≈ 0.728 in every arm — still above the 0.707
  uncorrelated baseline. No real predictive correlation.
- **B FAIL**: order_sensitivity remains sign-scattered noise in every arm.
- **C FAIL**: full's Δlearn fell from +0.0075 to −0.0014 — no learning at all.
- The **registered honest risk is CONFIRMED**: one gradient step per pair on
  a mostly-random reservoir does not learn this task, even with temporal
  structure unscrambled.

**RETRACTION (the finding that matters):** with the trainer fixed, the
full-vs-ablated differences collapsed to zero — per-seed
Δlearn(full)−Δlearn(off15): pre-fix +0.0171/+0.0061/+0.0141 → post-fix
+0.0024/−0.0003/−0.0002. Therefore:

1. The Phase-1/2 "suggestive learning benefit of the consciousness
   machinery" is **retracted as an artifact of the state-corrupting
   trainer**. The 10-seed 7/10 replication replicated the artifact.
2. The `temporal_consciousness` sole-carrier finding is re-scoped: the
   bisection localization was real and exact, but what it localized was
   **the artifact's dependency** — temporal_consciousness (whose tau
   modulation shapes evolution dynamics) determined how the trainer's
   time-scrambling played out, not a genuine competency contribution.
3. The honest current bottom line of the entire keystone chain: **the
   cognitive loop, as currently trained, does not learn sequences at all in
   any configuration** (Δlearn ≈ 0 everywhere, PE ≥ uncorrelated baseline,
   zero order anticipation). The consciousness machinery shows no external
   competency benefit on these tasks — and the binding constraint is now
   precisely identified as the *training signal*, not the architecture's
   wiring (which is, at last, clean: pure prediction, pure training,
   unscrambled time).

**What survives Phase 4 intact**: every signal-integrity repair (PE varies
honestly, Φ regime-discriminates, Ψ gates, state persists, trainer is pure —
all regression-guarded); the measurement harnesses themselves; and the
finding that ~12 subsystems are causally inert on every metric measured.

**Next scoped work (as registered)**: a stronger training signal — multiple
gradient steps per pair, an echo-state ridge readout over a window, or
longer BPTT — with these same A/B/C gates as its acceptance criteria. The
concurrent Predictive Compression program (C1, pre-registered separately)
attacks the same question from the compression side.

**Method note worth keeping**: pre-registration caught the project's own
most recent positive finding within 24 hours of publishing it. The
Phase-2 bisection's bitwise-exact localization was simultaneously completely
correct and completely misleading — exactness of localization says nothing
about what the localized thing *is*. Acceptance gates on the mechanism's
repair are what caught it.

## Phase 5 amendment: scale restoration (registered 2026-07-18, BEFORE runs)

Phase 4's registered follow-up was "a stronger training signal." The
signal-scale diagnosis (`probe_signal_scale`, trace doc follow-up 1) found
the actual mechanism: the bind chain annihilates magnitude ∝ d^-1.9, leaving
the readout's forward signal at ~3e-10 and its gradients at ~1e-13 —
untrainable at ANY learning rate. The root-cause fix (normalize at the
output-projection boundary, with the matching gradient normalization) IS the
training-signal fix — not more steps on an annihilated signal, but signal.

**Same pre-registered gates as Phase 4** (run `--phase1` + `--order`,
seeds 0–2, post-fix):
- **A**: `full` late PE < 0.707 uncorrelated baseline.
- **B**: order_sensitivity > 0 in `full` on ≥2/3 seeds.
- **C**: `full` Δlearn > +0.0075.
- Registered risks: (i) restored scale may still be insufficient with one
  gradient step per pair (then multi-step/ridge-readout remains next);
  (ii) normalization changes every downstream magnitude consumer at once —
  watch the loop suite for honest test updates; (iii) predictions now have
  O(1) magnitude, so the encoder's scale-invariant PE metric is unaffected
  by construction, but Genuine/degenerate classification rates may shift.

### Phase 5 results (2026-07-18, both runs exit 0)

**ALL THREE PRE-REGISTERED GATES PASS — the first passes in the project's
measured history.**

| arm | Δlearn (3 seeds) | mean | surprise mean | PE late | µs/cyc |
|-----|------------------|------|---------------|---------|--------|
| full | +0.131, +0.176, +0.182 | **+0.163** | +0.172 | **0.530** | 406,696 |
| min2 | +0.189, +0.182, +0.184 | **+0.185** | +0.196 | **0.505** | 335,385 |
| off15 | +0.189, +0.182, +0.184 | **+0.185** | +0.193 | **0.506** | 345,304 |

Order sensitivity (the Phase-3 probe): **positive in 9/9 rows** — full
+0.070/+0.004/+0.063, off15 +0.050/+0.032/+0.068, no_temporal
+0.037/+0.010/+0.067.

- **A PASS**: late PE 0.505–0.530 in every arm — decisively below the 0.707
  uncorrelated baseline (pre-fix: 0.72–0.77, never below, ever).
- **B PASS**: order anticipation positive in all arms and seeds. The
  capability Phase 3 declared absent exists — the system notices when
  fully-familiar material arrives in the wrong order.
- **C PASS**: Δlearn +0.16 to +0.19 — ~24x the gate threshold.
- Novelty discrimination (Phase-1's all-negative metric): **positive 9/9**
  (+0.13 to +0.20).

**What produced this**: not new machinery — the removal of four layers of
self-sabotage, each found by the previous fix's instruments: scrambled
subjective time (train-path state corruption), wiped state (predict-path),
drifting Fourier phase clocks (snapshot exclusion), and a readout whose
signal was annihilated to 3e-10 with ~1e-13 gradients (the d^-1.9 bind-chain
collapse — fixed by one normalization at the projection boundary).

**The reborn keystone answer, now on a living substrate**: the consciousness
machinery is a small consistent DRAG on every external competency measured —
full learns worse than the ablated arms (0.163 vs 0.185), discriminates
novelty slightly worse, costs ~20% more compute, and does not carry the
order anticipation (no_temporal is positive too). min2 ≈ off15 continues
(the two E1 "load-bearing" subsystems remain externally inert). Regime
separation collapsed everywhere at the new CL levels (~0.95+ saturated) —
the manipulation check needs recalibration at post-fix magnitudes.

**Standing bottom line of the five-phase chain**: the cognitive loop now
genuinely learns, discriminates novelty, and anticipates sequence order —
and every bit of that capability lives in the substrate (encoder + CfC +
readout), not in the fifteen flag-gated consciousness subsystems, which
currently subtract a little from all of it. The constructive path for the
machinery is now concrete: each subsystem must either demonstrably earn its
keep on some task battery (ethics, language, safety — not yet tested here)
or be demoted; and the vision's thesis must be defended on those grounds,
not on prediction-learning, where the honest answer is in.

**Combined honest picture after Phases 1–3**: the consciousness machinery's
one measurable external benefit (suggestive, temporal_consciousness-carried)
is familiarity-scale, ~1% of PE; genuine episodic anticipation — arguably
the capability the "perceive, predict, compare" story most implies — does
not exist yet in any arm. That is now the sharpest, best-characterized
capability gap in the cognitive loop, and closing it (real sequence
prediction: CfC trained on next-input targets with order-sensitive
representations) is the highest-leverage capability work this experiment
chain has identified.
