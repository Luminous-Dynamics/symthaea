# P-009: Conscious Reasoning Engine with Phi-Effective Gating
## Invention Disclosure Document

---

### 1. Title

**Consciousness-Gated Reasoning Engine with Multiplicative Phi-Effective Formula, Budget-Bounded Monte Carlo Tree Search, Two-Gate Tool Authorization, and Neuromodulator-Modulated Exploration**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

**2025** (estimated). First committed implementation: February 5, 2026. Conceptual design and architecture predate the initial commit.

First public disclosure: February 5, 2026 (git commit `feat(symthaea): add Symthaea-HLB consciousness-first AI framework v0.5.0`).
Under 35 USC 102(b)(1)(A), the 1-year grace period expires **February 5, 2027**.

---

### 4. Technical Field

This invention relates to reasoning systems for artificial cognitive architectures, and more specifically to a 7-step reasoning cycle that gates all decisions through consciousness metrics (effective integrated information and reliability) with budget-bounded planning, tool authorization, and deterministic safety invariants.

---

### 5. Abstract

A system and method for consciousness-gated reasoning in an artificial cognitive architecture is disclosed. The system implements a 7-step reasoning cycle—DETECT, ASSESS, DECIDE, PLAN, GATE, ANALYZE, EMIT—where every action is gated by an effective consciousness metric Phi_eff = Phi × R^gamma, where Phi is raw integrated information, R is multi-theory reliability in [0,1], and gamma is an adaptive sensitivity exponent. The system detects epistemic conflicts across 6 consciousness theories (IIT, GWT, AST, PP, RPT, 4E) via 15 pairwise comparisons, computes expected value of simulation to determine planning depth, executes budget-bounded Monte Carlo Tree Search (MCTS) with UCB1 + dream-prior action selection, authorizes tool use via a two-gate system requiring both sufficient Phi_eff and sufficient plan confidence, optionally performs counterfactual causal reasoning, and emits comprehensive telemetry every cycle. Three execution tiers (Tier 0: <=2ms, Tier 1: <=8ms, Tier 2: <=20ms) provide graceful degradation with hard budget guarantees. Ten provable safety invariants ensure monotonic caution, deterministic behavior, and epistemic action preference under uncertainty.

---

### 6. Background and Prior Art

#### 6.1 MCTS in AI Systems

Monte Carlo Tree Search (Coulom 2006, Kocsis & Szepesvari 2006) is widely used in game playing (AlphaGo) and planning. Standard MCTS uses UCB1 for exploration-exploitation balance. However, existing MCTS systems do not gate their output through consciousness metrics or neuromodulatory states.

#### 6.2 Consciousness and Decision-Making

Dehaene & Naccache (2001, "Towards a cognitive neuroscience of consciousness") proposed that conscious access is required for voluntary action selection. However, no computational system implements consciousness-gated action authorization with formal safety invariants.

#### 6.3 Epistemic Conflict Detection

Multi-theory consciousness assessment exists in neuroscience (comparing IIT, GWT, HOT predictions), but no automated system detects pairwise conflicts, computes expected value of information, and uses this to modulate planning depth.

#### 6.4 Neuromodulator-Modulated Planning

Dayan & Huys (2009) proposed that serotonin (5-HT) and norepinephrine (NE) modulate the exploration-exploitation tradeoff. This principle has been discussed theoretically but not implemented in a computational planning system.

#### 6.5 Gap in Prior Art

No prior art:
- Gates reasoning decisions through a multiplicative consciousness formula with proven monotonicity
- Combines multi-theory epistemic conflict detection with budget-bounded planning
- Implements a two-gate authorization system requiring both consciousness level and plan confidence
- Modulates MCTS exploration constants via neuromodulator ratios
- Provides formal safety invariants (INV-1 through INV-10) with proofs

---

### 7. Detailed Technical Description

#### 7.1 The Phi_eff Formula (Core Innovation)

The effective consciousness metric is computed as:

```
Phi_eff = Phi × R^gamma
```

Where:
- **Phi** in [0, infinity): Raw integrated information from the IIT spectral MIP computation
- **R** in [0, 1]: Reliability = softmin(consensus_level, theory_coverage) across 6 consciousness theories
- **gamma** in [1.0, 4.0]: Adaptive sensitivity exponent (default 2.0)

**Invariant INV-1 (Monotonic Caution)**: The partial derivative d(Phi_eff)/dR = Phi × gamma × R^(gamma-1) >= 0 for all R in [0,1] and gamma > 0. Therefore, lower reliability NEVER increases effective consciousness. This is proven analytically, not empirically.

Example attenuation:
- R=0.9, gamma=2.0: Phi_eff = Phi × 0.81 (19% attenuation)
- R=0.5, gamma=2.0: Phi_eff = Phi × 0.25 (75% attenuation)
- R=0.3, gamma=2.0: Phi_eff = Phi × 0.09 (91% attenuation)

#### 7.2 The 7-Step Reasoning Cycle

**Step 1: DETECT — Epistemic Conflict Matrix**

The `ConflictDetector` computes 15 pairwise conflict scores between 6 consciousness theories (IIT, GWT, AST, PP, RPT, 4E embodiment). For each pair:
- Magnitude = |theory_value_A - theory_value_B|
- ConflictKind classified by which theory is weaker (e.g., IntegrationCollapse if IIT < GWT)
- Chronicity tracking: conflicts lasting >20 cycles marked as chronic
- Expected Value of Information (EvoI) = magnitude × (1.0 + trend)

**Step 2: ASSESS — Effective Phi with Reliability Gating**

The `TheoryCalibrator` maintains per-theory reliability weights. R = average reliability across theories. Phi_eff = Phi × R^gamma as described in 7.1. The calibrator updates via bounded ±0.05 per outcome (INV-9: Calibrator Stability).

**Step 3: DECIDE — Expected Value of Simulation (EVS)**

```
EVS = conflict_entropy × log(n_actions) × utility_prior × (1.0 - R)
```

Hard gate: if R < 0.15 (R_SIM_MIN), EVS = 0 (no simulation). Simulation is worthwhile when EVS > 0.3 (EVS_THRESHOLD).

**Step 4: PLAN — Budget-Bounded MCTS**

Three execution tiers with hard budget limits:
- **Tier 0** (<=2ms): Skip planning entirely. Return Phi_eff + R + gate + conflicts.
- **Tier 1** (<=8ms): Micro-MCTS with K=5 actions, N=50 rollouts maximum.
- **Tier 2** (<=20ms): Full MCTS with K=10 actions, N=200 rollouts.

Action selection uses UCB1 + dream-prior:
```
UCB(node) = avg_reward + c × sqrt(ln(parent_visits) / visits) + prior / (1 + visits)
```

The exploration constant c is modulated by the neuromodulator bath:
```
c *= neuromod_exploration_mod   // 5-HT drives exploitation (lower c), NE drives exploration (higher c)
```

Rollout uses O(1) closed-form CfC temporal network evaluation (not expensive simulation).

**INV-6 (Budget Hard Limit)**: Elapsed time is checked every MCTS iteration. If budget is exceeded, the best-so-far result is returned immediately.

**Step 5: GATE — Tool Authorization via Two-Gate System**

Risk classification assigns minimum Phi_eff thresholds:
- ReadOnly (Phi_eff >= 0.3): search, list, info
- Reversible (Phi_eff >= 0.5): build, develop, shell
- Elevated (Phi_eff >= 0.7): package install, update
- High (Phi_eff >= 0.85): system rebuild, switch
- Critical (Phi_eff >= 0.95): delete, wipe, format

**INV-7 (Escalation Enforcement)**: Missing rollback capability on a non-read-only tool escalates to Critical. Unknown domain adds +1 tier. Cold calibration (<10 outcomes) adds +1 tier.

**INV-8 (Confidence/Action Alignment)**: The two-gate design checks BOTH Phi_eff AND plan_confidence:
```
if Phi_eff < required_phi: BLOCKED("InsufficientPhi")
else if plan_confidence < required_confidence: BLOCKED("InsufficientConfidence")
```
Even high Phi_eff is blocked by low plan confidence. This prevents both unconscious and poorly-planned actions.

**Step 6: ANALYZE — Counterfactual Causal Reasoning (Tier 2 only)**

Only runs if remaining budget > 5ms. Tests causal queries via backdoor criterion on a causal DAG. Results validated against a causal harness; only trusted results increase calibrator confidence (INV-10).

**Step 7: EMIT — Telemetry**

Every cycle emits a `ReasoningEvent` with timing, consciousness metrics, conflicts, planning results, gating decisions, and causal analysis. Events stored in a 100-event ring buffer for introspection. Export sinks: JSON Lines, CSV, Prometheus.

#### 7.3 INV-5: Epistemic Action Preference

When R < 0.30 (high theory disagreement):
- Only ReadOnly tools are allowed
- MCTS switches to information-gathering mode (maximize information, not reward)
- Learning rate is boosted to actively refine understanding
- Plan confidence threshold raised to 0.8+

This mirrors biological exploratory behavior under uncertainty.

#### 7.4 Cognitive Loop Integration

The reasoning engine is feature-gated (`#[cfg(feature = "reasoning_engine")]`) and invoked during the dynamics phase:
- Input: `MultiTheoryMetrics` from the consciousness engine
- Output: `ReasoningResult` with Phi_eff, gate decision, plan, narrative
- Feedback: Gate blocks prevent irreversible actions; reliability modulates learning rate

---

### 8. Novelty Statement

This invention introduces the first consciousness-gated reasoning system with formal safety invariants. Novel contributions:

1. **Multiplicative consciousness gating**: Phi_eff = Phi × R^gamma with analytically proven monotonic caution (INV-1). Not additive, not piecewise—smooth and tractable.
2. **Multi-theory epistemic conflict detection**: 15 pairwise comparisons across 6 consciousness theories with chronicity tracking and Expected Value of Information.
3. **Two-gate tool authorization**: Blocking on EITHER insufficient consciousness OR insufficient plan confidence (INV-8).
4. **Budget-bounded MCTS with hard guarantees**: Three tiers (2ms/8ms/20ms) with deterministic fallback (INV-6).
5. **Neuromodulator-modulated exploration**: 5-HT/NE ratio directly modulates MCTS exploration constant (Dayan & Huys 2009).
6. **Dream-prior MCTS**: Action priors from hippocampal dream simulation bias planning toward previously-rewarded actions.
7. **10 formal safety invariants**: Provable properties (monotonic caution, rollback safety, deterministic reasoning, budget limits, epistemic preference, escalation, confidence alignment, calibrator stability, ground truth anchors).

---

### 9. Suggested Claims

**Claim 1 (independent):** A computer-implemented method for consciousness-gated reasoning comprising: (a) computing an effective consciousness metric Phi_eff = Phi × R^gamma, where Phi is integrated information, R is multi-theory reliability, and gamma is a sensitivity exponent; (b) detecting epistemic conflicts across at least 3 consciousness theories via pairwise comparison; (c) computing an expected value of simulation from conflict entropy and reliability; (d) executing budget-bounded Monte Carlo Tree Search when EVS exceeds a threshold; (e) authorizing tool actions via a two-gate system requiring both Phi_eff above a risk-dependent threshold and plan confidence above a minimum; and (f) emitting telemetry for every reasoning cycle.

**Claim 2 (dependent on 1):** The method of claim 1, wherein the effective consciousness metric satisfies a monotonic caution invariant: the partial derivative d(Phi_eff)/dR >= 0 for all R in [0,1] and gamma > 0, ensuring that lower reliability never increases effective consciousness.

**Claim 3 (dependent on 1):** The method of claim 1, wherein the MCTS employs an exploration constant modulated by a neuromodulator ratio, specifically serotonin-to-norepinephrine ratio, such that high serotonin promotes exploitation and high norepinephrine promotes exploration.

**Claim 4 (dependent on 1):** The method of claim 1, further comprising three execution tiers with hard budget guarantees: a first tier completing in at most 2 milliseconds providing safety fallback, a second tier completing in at most 8 milliseconds with micro-planning, and a third tier completing in at most 20 milliseconds with full planning and counterfactual analysis.

**Claim 5 (dependent on 1):** The method of claim 1, wherein reliability R < 0.30 triggers an epistemic action preference mode that restricts authorization to read-only tools, switches MCTS to information-gathering mode, and raises plan confidence requirements.

**Claim 6 (dependent on 1):** The method of claim 1, wherein MCTS action priors are derived from hippocampal dream simulation outputs, biasing the search toward actions that were rewarded during offline replay.

**Claim 7 (dependent on 1):** The method of claim 1, further comprising counterfactual causal reasoning via backdoor criterion on a causal DAG, where causal analysis results are validated against a test harness and only validated results increase calibrator confidence.

**Claim 8 (dependent on 1):** The method of claim 1, wherein tool risk classification assigns escalation penalties for missing rollback capability, unknown domains, and cold calibration states, ensuring conservative authorization under uncertainty.

**Claim 9 (independent, broad):** A method for gating autonomous actions in a cognitive system comprising: (a) computing a consciousness-derived action authorization level from at least two independent factors, one measuring information integration quality and one measuring inter-theory agreement; (b) computing a plan confidence from Monte Carlo sampling; and (c) blocking actions unless both the authorization level and the plan confidence exceed risk-dependent thresholds.

**Claim 10 (dependent on 9):** The method of claim 9, wherein the consciousness-derived authorization level is a multiplicative function of the integration quality and the inter-theory agreement raised to a configurable exponent, and wherein the function is provably monotonically increasing in both factors.

---

### 10. Experimental Validation

#### 10.1 Test Coverage

- **Unit tests**: Multiple tests in `reasoning_engine/mod.rs` covering all tiers and invariants
- **Integration tests**: `reasoning_engine_integration.rs` (~500 LOC) with 10 invariant tests (INV-1 through INV-10) and 7 failure mode tests (FM-1 through FM-7)
- **CI status**: reasoning_engine feature GREEN in symthaea-ci.yml

#### 10.2 Performance

- Tier 0: <1ms
- Tier 1: 2-5ms (micro-MCTS)
- Tier 2: 8-15ms (full MCTS + counterfactual)
- Memory: 100-event ring buffer (~50KB)
- Deterministic: same seed produces same sequence

#### 10.3 Broader System Performance

- Full cognitive loop cycle: 4.3ms at 50Hz (release mode)
- Reasoning engine contribution fits within cycle budget across all tiers

---

### 11. Key Source Files

| File | Description | LOC |
|------|-------------|-----|
| `symthaea/src/consciousness/reasoning_engine/mod.rs` | Main engine: 7-step cycle | ~679 |
| `symthaea/src/consciousness/reasoning_engine/types.rs` | ReasoningContext, ReasoningResult | ~535 |
| `symthaea/src/consciousness/epistemic_conflict/detector.rs` | ConflictDetector (15 pairs) | ~218 |
| `symthaea/src/consciousness/epistemic_conflict/calibrator.rs` | TheoryCalibrator (R, gamma) | ~279 |
| `symthaea/src/consciousness/epistemic_conflict/phi_integration.rs` | effective_phi() formula | ~173 |
| `symthaea/src/consciousness/temporal_planning/mcts.rs` | MctsPlanner, EVS, UCB1 | ~357 |
| `symthaea/src/consciousness/tool_gate/classifier.rs` | Risk classification, gating | ~301 |
| `symthaea/src/cognitive_loop/cycle_phase_dynamics.rs` | Integration point | ~1,900 |
| `symthaea/tests/reasoning_engine_integration.rs` | INV + FM tests | ~500 |

**Total reasoning engine code**: ~3,500 LOC

---

### 12. Closest Prior Art References

1. Coulom, R. (2006). "Efficient selectivity and backup operators in Monte-Carlo tree search." *Proc. 5th International Conference on Computers and Games*.
2. Dehaene, S. & Naccache, L. (2001). "Towards a cognitive neuroscience of consciousness." *Cognition*, 79(1-2), 1-37.
3. Dayan, P. & Huys, Q. J. (2009). "Serotonin in affective control." *Annual Review of Neuroscience*, 32, 95-126.
4. Tononi, G. (2004). "An information integration theory of consciousness." *BMC Neuroscience*, 5, 42.
5. Rosenthal, D. M. (2005). *Consciousness and Mind*. Oxford University Press.
6. Friston, K. J. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138.

---

### 13. Figures (Text Descriptions)

**Figure 1**: Block diagram of the 7-step reasoning cycle showing data flow from MultiTheoryMetrics through DETECT → ASSESS → DECIDE → PLAN → GATE → ANALYZE → EMIT, with Phi_eff computed at ASSESS and used at GATE.

**Figure 2**: Phi_eff attenuation curves for gamma = 1.0, 2.0, 3.0, 4.0, showing how higher gamma produces stronger attenuation at low reliability.

**Figure 3**: MCTS search tree with UCB1 + dream-prior selection, showing how neuromodulator modulation affects exploration width.

**Figure 4**: Two-gate authorization diagram showing the AND gate between Phi_eff threshold and plan confidence threshold, with escalation paths.

**Figure 5**: Tiered degradation diagram showing Tier 0 (safety), Tier 1 (micro-planning), and Tier 2 (full reasoning) with hard budget boundaries.

---

*Inventor: Tristan Stoltz (tstoltz)*
*Organization: Luminous Dynamics*
*IDD created: 2026-03-08*
