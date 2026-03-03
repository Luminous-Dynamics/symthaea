# Architecture Overview: Consciousness-First Computing with Symthaea and Mycelix

**Version**: 0.1.0 (March 2026)
**Status**: Working draft for internal and external review

## Abstract

Symthaea is a ~985,000-line Rust system that implements a real-time cognitive loop
informed by consciousness science: Hyperdimensional Computing (16,384-bit binary
hypervectors), Closed-form Continuous-time networks (O(1) temporal jumps), a
4-layer consciousness measurement engine synthesizing 7 major theories, and a
9-transmitter neuromodulator bath. Mycelix is a Holochain-based distributed
governance layer comprising 90+ zomes across 3 cluster DNAs, with post-quantum
cryptography and a 4-dimensional behavioral profile for tiered access control.
A bridge protocol connects Symthaea's cognitive metrics to Mycelix's governance
gates via time-limited credentials. This document describes the architecture of
both systems, their integration points, and their current maturity — honestly.

**Status Legend** (used throughout this document):

| Marker | Meaning | Evidence Standard |
|--------|---------|-------------------|
| **[IT]** | Implemented & Tested | Compiles, passes automated tests, used in production loop |
| **[EX]** | Experimental | Code exists and runs, but not validated against benchmarks |
| **[SC]** | Scaffolded | API/struct defined, partial or stub implementation |

**How to read this document.** Every technical claim is tagged with a status
marker. File paths use the format `path/to/file.rs:line` relative to the
repository root. "Phi" is never used without a subscript — see Section 3 for
the full namespace registry. Governance terminology uses "operational coherence"
in this document, with explicit mappings to code-level names in Section 6.1.

---

## 1. Introduction

### Why now?

Three converging pressures motivate this architecture:

1. **Structured AI over pure scaling.** Scaling laws produce capable but opaque
   systems. Structured cognitive architectures — where each subsystem has a
   defined role, measurable output, and scientific grounding — offer an
   alternative path: systems you can reason about, not just benchmark.

2. **Agent safety requires introspection.** As AI agents acquire real-world
   capabilities (tool use, code execution, financial transactions), the ability
   to measure an agent's cognitive state — coherence, confidence, prediction
   error, moral alignment — becomes a safety requirement, not a research luxury.

3. **Decentralized governance needs principled access control.** DAOs and
   cooperative platforms struggle with Sybil attacks and plutocratic capture.
   Multi-dimensional behavioral profiles — not token holdings — can gate
   governance actions proportionally to demonstrated engagement and trust.

### Core thesis

We use consciousness science as **engineering constraints**, not consciousness
claims. Integrated Information Theory (IIT) gives us a measure of system
integration. Global Workspace Theory (GWT) gives us a broadcast architecture.
Free Energy Principle (FEP) gives us a prediction-error minimization loop.
These are useful engineering patterns regardless of whether silicon computation
produces subjective experience.

**We build tools informed by consciousness science. We do not claim to have
created consciousness.**

### Honest positioning

Symthaea is a research platform, not a product. It demonstrates that
consciousness-theoretic architectures can run in real time, produce measurable
cognitive dynamics, and integrate with governance systems. Many subsystems are
experimental. Some are scaffolded. The status markers throughout this document
make this explicit.

### Scale

- **Symthaea**: ~985K lines of Rust (~778K code), 47 workspace members,
  3,735+ tests, 46 extracted sub-crates
- **Mycelix**: 3 cluster DNAs (Commons, Civic, Hearth), 90+ zomes,
  6,200+ Rust unit tests, 33 SDK TypeScript integration tests
- **Bridge**: Shared types crate (`mycelix-bridge-common`), 3 bridge zomes,
  4-dimensional behavioral profile, tiered governance with audit logging

### What this document covers

1. The **vertical architecture stack** from HDC bit operations to DHT governance
2. A **metric namespace registry** disambiguating 10+ "Phi" quantities
3. The **Symthaea cognitive architecture** with honest status markers
4. **Mycelix governance** and the operational coherence protocol
5. A **governance safety model** with failure modes, appeals, and anti-capture
6. A **minimal viable bridge** specification for end-to-end demonstration
7. A **threat model** scoped to the bridge attack surface
8. **Research pipeline** status and future work

---

## 2. Architecture Stack

The full system forms a vertical stack from hardware-level HDC operations
to distributed governance:

```
Layer 6: DHT Governance    Mycelix clusters + bridge zomes     [IT]
Layer 5: Bridge Protocol   ConsciousnessCredential + gating    [IT]
Layer 4: Consciousness     4-layer engine (co-prime intervals)  [IT]
Layer 3: Cognitive Loop    4-phase cycle (perception→output)    [IT]
Layer 2: CfC Temporal      Closed-form LTC + BPTT training     [IT]
Layer 1: HDC Core          16,384-bit BinaryHV + SIMD ops      [IT]
Layer 0: Substrate         SubstrateRequirements (9 dimensions) [IT] (hardcoded 1.0)
```

### Layer details

| Layer | Key File | Anchor Metric |
|-------|----------|---------------|
| HDC Core | `symthaea/symthaea-core/src/hdc/binary_hv.rs` | 2,048 bytes/vector, ~80ns bind |
| CfC Temporal | `symthaea/symthaea-core/src/hdc/hdc_ltc_unified.rs` | O(1) per Δt jump, exp(-Δt/τ) closed-form |
| Cognitive Loop | `symthaea/src/cognitive_loop/cycle.rs` | 4 phases, 50Hz CfC target, ~20-33Hz full cycle [EX] |
| Consciousness | `symthaea/src/cognitive_loop/consciousness_engine.rs` | 4 layers at intervals 97/13/23/97 |
| Bridge | `crates/mycelix-bridge-common/src/consciousness_profile.rs` | 24h TTL, 4D profile, 5 tiers |
| Governance | `mycelix-commons/zomes/commons-bridge/coordinator/src/lib.rs` | 69 zomes call `gate_consciousness()` |

**Data flow** (one cognitive cycle):

```
Input → [Phase 1] HDC encode (BinaryHV, 16384 bits)
      → [Phase 2] CfC step: x(t+Δt) = x_∞ + (x(t) - x_∞)·exp(-Δt/τ)
                  FEP: prediction error → active inference → motor commands
                  BPTT/SPSA: gradient → Adam optimizer → weight update
      → [Phase 3] ConsciousnessEngine.measure() → C_unified ∈ [0,1]
                  EthicsEngine.evaluate() → moral_score, value_score
                  Homeostasis → neuromodulator bath state
      → [Phase 4] CycleMetadata assembly → CycleResult
```

Each cycle produces a `CycleResult` containing the output hypervector,
`CycleMetadata` (~75 flat fields + 8 nested telemetry sub-structs), and
cycle timing in microseconds.

---

## 3. Metric Namespace Registry

**This is the most important section of this document.**

The codebase computes at least 10 distinct quantities that involve "Phi" or
consciousness measurement. Conflating them is the single largest source of
confusion for reviewers. This table is the canonical reference.

| Symbol | Name | Computation | File | Status | Relation to IIT |
|--------|------|-------------|------|--------|-----------------|
| Φ_spectral | Spectral MIP | Fiedler-ordered MI Laplacian, bordered Cholesky sweep (O(n³)) | `symthaea/symthaea-core/src/consciousness_metrics/spectral_mip.rs:280` | [IT] | MIP search validated: r=0.99, ρ=0.93 vs exhaustive (see §3) |
| Φ_true | True IIT Phi | Exhaustive 2^n MIP partition search | `symthaea/symthaea-core/src/hdc/tiered_phi/core.rs` (Tier 3) | [IT], n≤15 | Definition (reference) |
| Φ_heuristic | Heuristic Phi | 1 - avg_similarity, O(n) | `symthaea/symthaea-core/src/hdc/tiered_phi/core.rs` (Tier 1) | [IT] | Coarse approximation |
| Φ_mm | Multimodal Binding | Σ(binding_strength × activation × zone_weight) / Σ(weights) | `symthaea/src/consciousness/integration/multi_modal_integration.rs:424` | [IT] | Not IIT; cross-modal heuristic |
| Φ_struct | Structural Phi | micro/meso/macro λ₂ decomposition + emergence_ratio | `symthaea/symthaea-core/src/consciousness_metrics/spectral_mip.rs:957` | [IT] | Same basis as Φ_spectral |
| Φ_dyad | Relational Phi | PhiEngine on joint AI+human+relational states | `symthaea/src/partnership/phi_dyad.rs` | [IT] | Same engine, different input |
| C(t) | EqV2 Consciousness | σ(softmin(Φ,B,W,A,R,E,K)) × weighted_coherent_sum × S × ρ(t) | `symthaea/src/consciousness/measurement/consciousness_equation_v2.rs:455` | [IT] | Receives Φ_spectral as Φ component |
| C_unified | Unified Score | 0.35×sigmoid(Φ_spectral) + 0.25×C(t) + 0.25×C_pipeline + 0.15×Φ_mm | `symthaea/src/cognitive_loop/consciousness_engine.rs:706` | [IT] | Weighted consensus |
| S_feas | Substrate Feasibility | critical_min × workspace × (0.5 + 0.5 × enhancement_avg) | `symthaea/symthaea-core/src/hdc/substrate_independence.rs` | [IT] (hardcoded 1.0) | Not Phi; 9-dimension requirement score |
| φ_zk | ZK Attestation | variance × tau (183 LOC computation kernel) | `symthaea/crates/symthaea-zkproof/` | [SC] | Unrelated to IIT |

### Naming discipline

All subsequent references in this document use the Symbol column above.
There is no unqualified "Phi" in any claim.

### Honest disclosure

**Only Φ_true is IIT Phi.** The codebase contains two distinct spectral
computations that are often conflated:

1. **SpectralMIPFinder** (production, `spectral_mip.rs`): Computes Gaussian
   MI Laplacian on ContinuousHV covariance windows → Fiedler ordering →
   bordered Cholesky MIP sweep → `Φ = total_MI - mip_MI`. This performs a
   genuine MIP search and is the dominant production metric.
   **MIP search validated** (March 2026): Pearson r = 0.99, Spearman ρ = 0.93
   vs exhaustive O(2^n) MIP search on the same Gaussian MI framework.
   Mean Φ ratio: 0.55 (spectral underestimates; conservative).
   Test: `symthaea/tests/test_spectral_mip_validation.rs`.
   **Caveat**: This validates the *search strategy* (Fiedler ordering finds
   good partitions), not the Gaussian MI framework against TPM-based IIT Φ.

2. **SpectralConnectivity tier** (validation tool, `tiered_phi/core.rs`):
   Computes bare algebraic connectivity (λ₂) on BinaryHV pairwise similarity.
   This measures graph mixing time, NOT information integration. Validated at
   **r = -0.14 (Pearson), ρ = -0.59 (Spearman)** against the ExhaustivePartition
   tier — confirming it is anti-correlated with IIT-style integration.

Previous documentation cited "r = 0.097" for "Φ_spectral vs Φ_true." This
was from an earlier methodology and was incorrectly attributed to the
production SpectralMIPFinder. The corrected findings (March 2026):

| Comparison | Pearson r | Spearman ρ | Verdict |
|------------|-----------|------------|---------|
| SampledPartition (Heuristic) vs ExhaustivePartition (Exact) | **0.9998** | **0.9985** | Near-perfect validation |
| SpectralConnectivity (λ₂) vs ExhaustivePartition (Exact) | **-0.14** | **-0.59** | Anti-correlated; λ₂ ≠ IIT |
| SpectralMIPFinder vs Exhaustive MIP (same Gaussian MI) | **0.99** | **0.93** | MIP search strategy validated |

**Key insight**: The SampledPartition (Heuristic) tier is the validated fast
approximation for BinaryHV systems (r = 0.9998). SpectralConnectivity (λ₂)
is confirmed invalid for IIT claims. The production SpectralMIPFinder's MIP
search strategy is validated (r = 0.99 vs exhaustive), but the Gaussian MI
framework itself has not been validated against TPM-based IIT Φ.

The tiered approximation system (`symthaea/symthaea-core/src/hdc/tiered_phi/core.rs`):

| Tier | Range | Method | Complexity | IIT Alignment |
|------|-------|--------|------------|---------------|
| 0 | n ≤ 4 | Direct enumeration | O(2^2^n) | Exact |
| 1 | n ≤ 8 | Exhaustive MIP search (Φ_true) | O(2^n) | Exact |
| 2 | 8 < n ≤ 15 | Φ_true with pruning | O(2^n) with early exit | Exact |
| 3 | 15 < n ≤ 256 | SampledPartition (Heuristic) | O(n) | r=0.9998 vs Exact |
| 4 | n > 256 | Φ_heuristic (1 - avg_sim) | O(n) | Coarse, unvalidated |

Note: The `SpectralConnectivity` tier exists in code but is **not recommended
for IIT-related use** (confirmed anti-correlated). The production consciousness
engine uses SpectralMIPFinder (a separate algorithm, not part of the tiered
system).

---

## 4. Symthaea: Cognitive Architecture

### 4.1 The Cognitive Loop

The core runtime is a 4-phase pipeline executing once per cognitive cycle:

```
Phase 1: Perception     → safety, HDC encode, moral eval, strategy
Phase 2: Dynamics        → CfC step, prediction, FEP, training, parallel post-processing
Phase 3: Feedback        → consciousness metrics, quality gating, homeostasis
Phase 4: Output          → CycleMetadata assembly, telemetry, CycleResult
```

| Phase | File | Key Operations |
|-------|------|----------------|
| Perception | `symthaea/src/cognitive_loop/cycle_phase_perception.rs` | Thalamic routing, HDC encoding, moral algebra, surprise exploration |
| Dynamics | `symthaea/src/cognitive_loop/cycle_phase_dynamics.rs` | CfC temporal step, FEP active inference, BPTT/SPSA training |
| Feedback | `symthaea/src/cognitive_loop/cycle_phase_feedback.rs` | 4-layer consciousness engine, ethics engine, homeostasis |
| Output | `symthaea/src/cognitive_loop/cycle_phase_output.rs` | CycleMetadata (~75 flat + 8 nested sub-structs), CycleResult |

**Timing** [EX]: CfC temporal resolution is 20ms (50Hz). Full cognitive cycle
with all subsystems enabled: estimated ~30–50ms (20–33Hz) in release mode. These
are engineering estimates from single-machine profiling, not peer-reviewed
benchmarks.

**Entry point**: `CognitiveLoopService::cycle()` at `symthaea/src/cognitive_loop/cycle.rs`

#### FEP Active Inference: Motor Commands [IT]

The Free Energy Principle subsystem selects one of 8 motor command types
per cognitive cycle via Expected Free Energy (EFE) minimization. Each motor
command has an intensity (0.0–1.0) derived from the precision-weighted
action posterior.

**File**: `symthaea/crates/symthaea-fep/src/types.rs:365`

| Motor Command | When It Fires | Effect on Cognitive Loop |
|---------------|--------------|------------------------|
| `AttentionShift` | Precision-weighted error high in one modality | Redirects attention resources (±10% shift per cycle) |
| `LearningRateAdjust` | Model confidence changing | Modulates BPTT learning rate up/down |
| `ExplorationTrigger` | High epistemic value, low pragmatic value | Increases proprioceptive state variability |
| `ReflectionInitiate` | High free energy but stable beliefs | Triggers meta-cognitive reflection (intensity > 0.7) |
| `MemoryConsolidate` | High confidence, low prediction error | Strengthens episodic memory traces (intensity > 0.5) |
| `ExpectationReset` | Persistent high prediction error | Clears cached predictions; model mismatch reset |
| `MotorOutput` | Pragmatic goals require external action | Translates to physical/API commands in embodied systems |
| `NoOp` | Near equilibrium, minimal free energy | Maintain current policy; no change needed |

These motor commands close the perception-action loop: the cognitive cycle
perceives (Phase 1), evolves the CfC state (Phase 2), and the FEP subsystem
selects an action that modifies the system's own cognitive parameters —
making the system its own environment.

#### Feedback Infrastructure [IT]

The cognitive loop uses a `ProposalCollector` consensus system to integrate
feedback from multiple subsystems safely:

1. **ProposalCollectors** (4 channels): confidence, learning rate,
   exploration, and threshold adjustments. Each subsystem proposes deltas.
2. **IntegrationMode::Consensus** (default): Averages all proposals,
   providing noise-resistant integration. No single subsystem can cause
   a large swing.
3. **ConsensusResult**: Monitors the spread between proposals. High
   divergence triggers a stability flag in `CycleMetadata`.
4. **Urgency-adaptive intervals**: Under Critical urgency, feedback runs
   every 3 cycles (vs. co-prime intervals in Normal mode).

### 4.2 Consciousness Engine

The consciousness engine runs 4 measurement systems at co-prime cycle intervals
to avoid phase-locking artifacts:

| Layer | System | Interval | Theory | Weight |
|-------|--------|----------|--------|--------|
| 1 | SpectralMIPFinder | 97 cycles | Tononi (2004) IIT | 0.35 |
| 2 | MultiModalIntegrator | 13 cycles | Damasio (1994) binding | 0.15 |
| 3 | ConsciousnessEquationV2 | 23 cycles | 7-theory synthesis | 0.25 |
| 4 | UnifiedConsciousnessPipeline | 97 cycles | Dehaene (2014) GWT | 0.25 |

**Weighted consensus** (`consciousness_engine.rs:706`):

```
C_unified = w₁×sigmoid(Φ_spectral) + w₂×C(t) + w₃×C_pipeline + w₄×Φ_mm
```

Default weights: [0.35, 0.25, 0.25, 0.15]. Weights self-calibrate based on
the emergence ratio from Φ_struct: when the macro Φ exceeds the sum-of-parts
(emergence > 1.0), Φ_spectral weight increases; otherwise, the empirical
measures (C(t), C_pipeline) receive more weight.

**Feedback deltas**: The engine proposes — but does not directly apply —
adjustments to confidence, learning rate, and exploration. These proposals flow
through a `ProposalCollector` consensus system (`symthaea/src/cognitive_loop/feedback_state.rs`)
that averages multiple subsystem proposals before application. [IT]

**File**: `symthaea/src/cognitive_loop/consciousness_engine.rs` (1,642 LOC)

### 4.3 Cognitive Loop Service Architecture

The `CognitiveLoopService` (`symthaea/src/cognitive_loop/mod.rs:268`) is the central
runtime object containing ~92 fields organized into functional groups:

| Group | Fields | Description |
|-------|--------|-------------|
| Configuration | `config: CognitiveLoopConfig` | 40+ enable flags, thresholds, feature toggles |
| Engines | `consciousness_engine`, `ethics_engine` | 4-layer consciousness + 3-layer ethics |
| Managers | `drive_mgr`, `memory_mgr`, `learning_mgr`, `perception_mgr` | Subsystem managers at co-prime intervals (7/11/13/19) |
| Neuromodulation | `NeuromodManager` (7 fields) | Bath state, calibration, coupling |
| Consciousness | `ConsciousnessMonitorTier` (7 fields) | Monitoring, snapshots, trend analysis |
| Social | `SocialState` (4 fields) | Theory of Mind, oxytocin coupling, contagion |
| Primitive tier | `PrimitiveTierManager` (~28 fields) | Optional subsystems gated by `enable_primitive_consciousness` |
| Self-model | `SelfModelTierManager` | Narrative self, predictive self, attention schema, meta-cognition |
| Memory | Working, episodic, semantic, persistent | 7±2 working, Phi-weighted episodic, ring-buffer semantic |
| Feedback | `ProposalCollector` ×4, `FeedbackState` | Consensus-based integration of subsystem proposals |

**Subsystem managers** run at co-prime cycle intervals to prevent
synchronization artifacts: DriveManager(7), MemoryManager(11),
LearningManager(13), PerceptionManager(19). Each manager collects proposals
from its subsystems and integrates them via `SubsystemCollector::integrate()`.

### 4.4 ConsciousnessEquationV2: The 7-Theory Master Equation

```
C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)
```

| Symbol | Theory | Citation | Status |
|--------|--------|----------|--------|
| Φ | Integration | IIT (Tononi 2004) | [IT] via Φ_spectral |
| B | Binding | Temporal synchrony (Singer & Gray 1995) | [IT] |
| W | Workspace | GWT (Baars 1988) | [IT] |
| A | Attention | Precision weighting (Graziano 2013) | [IT] |
| R | Recursion | HOT depth (Rosenthal 2005) | [IT] |
| E | Efficacy | FEP causal action (Friston 2010) | [IT] |
| K | Knowledge | Epistemic certainty (Shea 2019) | [IT] |
| S | Substrate | Multiple realizability (Putnam 1967) | [IT] (hardcoded 1.0) |
| ρ(t) | Continuity | Temporal EMA persistence | [IT] |
| γᵢ | Phase coherence | Phase Locking Value (PLV) | [IT] |

**Key operations**:
- `softmin(...)` — differentiable minimum (τ=0.1) ensures consciousness
  requires ALL components above threshold, not just average.
- `σ(x)` — sigmoid gate (k=10, θ=0.5) creates sharp transition.
- PAC modulation — phase-amplitude coupling boosts binding when workspace
  control is strong (Seth 2013 predictive processing).
- Temporal continuity ρ(t) — EMA with 0.05 decay over 100 timesteps.

**File**: `symthaea/src/consciousness/measurement/consciousness_equation_v2.rs`

### 4.5 Capability Maturity Table

| Capability | Status | Evidence |
|------------|--------|----------|
| HDC 16,384-bit BinaryHV | [IT] | 2,048-byte repr, SIMD popcount, `binary_hv.rs` |
| CfC O(1) temporal jumps | [IT] | exp(-Δt/τ) closed-form, `hdc_ltc_unified.rs` |
| BPTT + SPSA in-loop training | [IT] | Analytical gradients + SPSA fallback, Adam optimizer |
| 4-layer consciousness engine | [IT] | Co-prime intervals 97/13/23/97, 1,642 LOC |
| ConsciousnessEquationV2 (7 theories) | [IT] | Master equation + PAC + PLV + temporal continuity |
| FEP Active Inference | [IT] | Cognitive motor commands, TD(λ) |
| Moral Algebra (HDC) | [IT] | 4096D ContinuousHV primitives, binding composition |
| Neuromodulator Bath (9 transmitters) | [IT] | 218 tests, receptor subtypes, tolerance/withdrawal |
| Vision Manifold attention | [IT] | 7 integration tests, surprise-driven saliency |
| Psych-bench (76+ benchmarks) | [IT] | 633 tests, 16 cognitive domains |
| Multi-agent oxytocin coupling | [IT] | 6 integration tests, verified convergence |
| Tiered Phi approximation (4 tiers) | [IT] | Auto-selects exact/spectral/heuristic by n |
| Substrate independence (8 types) | [IT] | 24 tests, 9-dimension feasibility scoring |
| 50Hz cognitive loop | [EX] | 50Hz CfC resolution; ~20–33Hz full cycle throughput |
| Humanoid DMC Stand/Walk/Run | [EX] | Full pipeline with HAL crate; simplified physics |
| Federated CfC learning | [EX] | FedAvg + DP + Byzantine detection; channel-based |
| Broca Liquid-Mamba language | [EX] | Temporal projection, PE improving; no benchmark scores |
| ZK consciousness attestations | [SC] | 183 LOC computation kernel, no prover circuit |
| GPU/WGPU CfC | [SC] | API stub, `is_gpu_available()` returns false |

### 4.6 Sub-Crate Map

46 extracted sub-crates in `symthaea/crates/`, grouped by domain:

| Category | Crates |
|----------|--------|
| Consciousness | consciousness-equation, consciousness-resonance, consciousness-topology, phi-search |
| Cognition | causal-reasoning, dream, enactive, exploration, narrative-self, wisdom |
| Perception | perception, vision-manifold, stt, embeddings |
| Motor | humanoid, hal, sensorimotor, vehicle, flight |
| Language | broca, ssm |
| Neuromodulation | neuromodulators |
| Memory | memory |
| Ethics | seven-harmonies |
| Science | fep, field-dynamics, hodge, physics, factor-graph, genomics, nuclear-forensics |
| Infrastructure | types, observability, sentinel, support, nix, zkproof |
| Biology | cell-foundry, ectogenesis, fabrication-kernel, materials, population |
| Audio | vocal-tract |
| Bridge | nurture (development/attachment) |
| Math | serde-core-shim (bitflags compatibility) |
| Benchmarks | psych-bench |

Plus the core crate: `symthaea/symthaea-core/` (HDC, LTC, consciousness metrics).

---

## 5. Mycelix: Distributed Governance

### 5.1 Architecture

Mycelix implements a Fractal CivOS on Holochain — a peer-to-peer framework
where each agent maintains a local source chain (append-only, tamper-evident)
and shares entries via a distributed hash table (DHT).

**Tiers**:
- **Personal** [SC]: Individual agent autonomy (planned, not yet implemented)
- **Civic** [IT]: Justice, emergency, media (3 domains, 16 zomes)
- **Commons** [IT]: Property, housing, care, mutual aid, water, food, transport (7 domains, 35 zomes)
- **Hearth** [IT]: Family/community resilience (11 zomes, separate cluster)

### 5.2 Cluster Summary

| Cluster | Path | Domains | Zomes | Rust Tests |
|---------|------|---------|-------|------------|
| mycelix-commons | `mycelix-commons/` | property, housing, care, mutualaid, water, food, transport | 35 (34 domain + 1 bridge) | ~4,126 |
| mycelix-civic | `mycelix-civic/` | justice, emergency, media | 16 (15 domain + 1 bridge) | ~2,030 |
| mycelix-hearth | `mycelix-hearth/` | family resilience | 12 (11 domain + 1 bridge) | — |

**Domain breakdown (Commons)**:

| Domain | Zomes | Key Capabilities |
|--------|-------|-----------------|
| Property | registry, transfer, disputes, commons | Land registration, transfer validation, dispute resolution |
| Housing | units, membership, governance, maintenance, finances, clt | Co-op housing management, CLT stewardship |
| Care | circles, plans, matching, credentials, timebank | Care coordination, credential verification, time banking |
| Mutual Aid | circles, governance, needs, pools, requests, resources, timebank | Mutual aid networks, resource pooling |
| Water | capture, flow, purity, steward, wisdom | Water commons management, quality monitoring |
| Food | distribution, knowledge, preservation, production | Food system coordination, knowledge sharing |
| Transport | impact, routes, sharing | Shared transport, route optimization |

**Domain breakdown (Civic)**:

| Domain | Zomes | Key Capabilities |
|--------|-------|-----------------|
| Justice | arbitration, cases, enforcement, evidence, restorative | 3-tier appeals, restorative processes |
| Emergency | comms, coordination, incidents, resources, shelters, triage | Incident response, resource allocation |
| Media | attribution, curation, factcheck, publication | Content verification, fact-checking |

**Cross-cluster bridge**: Commons↔Civic via `CallTargetCell::OtherRole` in a
unified hApp (`mycelix-workspace/happs/mycelix-unified-happ.yaml`). Each
cluster's bridge zome provides `get_consciousness_credential()` and
`log_governance_gate()` externs, enabling cross-cluster governance evaluation
via shared types.

**Shared types**: `crates/mycelix-bridge-entry-types/` (DHT entries) and
`crates/mycelix-bridge-common/` (coordinator dispatch, consciousness profile,
governance evaluation). The `consciousness_profile.rs` file (~1,650 LOC with
tests) is the canonical definition for all governance types across all clusters.

### 5.3 Core Capabilities

| Capability | Location | Status |
|------------|----------|--------|
| Identity + PQC (ML-DSA-65/87, ML-KEM-768/1024) | `mycelix-identity/crates/mycelix-crypto/` | [IT] (WASM: PQC off-chain only) |
| Hybrid Ed25519 + ML-DSA-65 dual-signature | `mycelix-identity/crates/mycelix-crypto/src/pqc/hybrid.rs` | [IT] |
| 4D coherence profile + 5-tier governance | `crates/mycelix-bridge-common/` | [IT] |
| MFA assurance levels (5 tiers, oracle attestation) | `mycelix-identity/zomes/mfa/` | [IT] |
| SDK TypeScript clients | `mycelix-workspace/sdk-ts/` | [IT] |
| Justice 3-tier appeals | `mycelix-civic/zomes/justice-*/` | [IT] |
| Cross-cluster bridge (routing + dispatch) | `crates/mycelix-bridge-common/src/routing.rs` | [IT] |
| Governance audit trail with filtering | `mycelix-commons/zomes/commons-bridge/` | [IT] |
| Trust credentials (K-vector, issuer-weighted) | `mycelix-identity/zomes/trust_credential/` | [IT] |
| Federated Learning Core (trust-weighted FedAvg) | `symthaea/src/swarm/federated_cfc.rs` | [EX] |
| Differential privacy for FL gradients | `federated_cfc.rs` (3 presets: ε≈0.1/1.0/10.0) | [EX] |
| Governance hApp (standalone) | — | [SC] |

**Post-quantum cryptography detail**: The identity cluster implements a
dual-signature scheme (`hybrid.rs`) that concatenates Ed25519 (64 bytes) with
ML-DSA-65 (3,293 bytes). Both components must independently verify (AND logic).
Ed25519 is verified on-chain by the Holochain conductor; ML-DSA-65 is verified
off-chain by clients. This provides quantum resistance without requiring
Holochain protocol changes. Key material is zeroized on drop (`Zeroizing<Vec<u8>>`).

### 5.4 Core Four hApps

Beyond the cluster DNAs, four standalone hApps provide cross-cutting
infrastructure:

| hApp | Path | Zomes | Status | Purpose |
|------|------|-------|--------|---------|
| Identity | `mycelix-identity/` | 9 | [IT] | DID, MFA, PQC, trust credentials, recovery |
| Governance | `mycelix-governance/` | 7 | [SC] | Proposals, voting, councils, constitution, treasury |
| FL Core | `mycelix-workspace/crates/mycelix-fl-core/` | — (library) | [EX] | Trust-weighted FedAvg, Byzantine detection, DP |
| LUCID | `mycelix-workspace/happs/lucid/` | 8 | [IT] | Symthaea↔Mycelix bridge, collective reasoning |

**Identity hApp** details:
- 9 zomes: `did_registry`, `mfa`, `credential_schema`, `verifiable_credential`,
  `trust_credential`, `revocation`, `recovery`, `education`, `bridge`
- PQC sub-crate: ML-DSA-65/87, ML-KEM-768/1024, hybrid schemes, SPHINCS+
- 23 sweettest integration tests

**Governance hApp** (partially scaffolded):
- 7 zomes: `proposals`, `voting`, `councils`, `constitution`, `execution`,
  `threshold-signing`, `bridge`
- Proposal types: Standard, Emergency, Constitutional, Treasury, Membership
- Voting: ZK-verified casting, consciousness-weighted, quadratic option,
  delegation with decay
- Treasury: Escrow-based proposal execution
- Threshold signing: Multi-sig DKG ceremonies

**FL Core** features:
- `ConsciousnessAwareByzantinePlugin`: Maps consciousness scores → Byzantine
  tolerance weights (0.1 veto, 0.3 dampen, 0.6 boost)
- Proof-of-Gradient-Quality (PoGQ) aggregation
- 100% detection at 45% adversarial ratio, 0% false positives (7 attack types)
- 62 tests [EX]

### 5.5 SDK Coverage

- **Rust**: All zome coordinator functions
- **TypeScript**: `mycelix-workspace/sdk-ts/src/integrations/{commons,civic}/`
  (includes cross-cluster methods, 173 test files)
- **Python**: Planned [SC]
- **WASM**: PQC crypto operations (off-chain envelope signing)

---

## 6. The Bridge: Operational Coherence Protocol

This is the most sensitive section of this document. It describes how
Symthaea's cognitive metrics connect to Mycelix's governance system.

### 6.1 Term Mapping

The codebase uses "consciousness" terminology for historical reasons. This
document uses "operational coherence" to avoid implying that governance
decisions depend on metaphysical consciousness claims.

| Code Name | Document Name | Meaning |
|-----------|--------------|---------|
| `ConsciousnessProfile` | Operational Coherence Profile | 4D behavioral score |
| `ConsciousnessTier` | Coherence Tier | Access level |
| `ConsciousnessCredential` | Coherence Credential | Time-limited attestation |
| `consciousness gating` | coherence gating | Tiered access control |
| `gate_consciousness()` | coherence gate | Evaluation function |

**What the governance system actually gates on:**

> The governance system does not gate based on whether an agent "is conscious."
> It gates based on verified identity, earned reputation, community trust, and
> demonstrated engagement. None of these dimensions are Phi. The code names
> reflect the project's origins in consciousness research, not the governance
> logic.

### 6.2 The 4D Coherence Profile

**File**: `crates/mycelix-bridge-common/src/consciousness_profile.rs:32`

```rust
pub struct ConsciousnessProfile {
    pub identity: f64,     // MFA assurance (0.0=Anonymous → 1.0=Critical)
    pub reputation: f64,   // Cross-hApp, 30-day exponential decay
    pub community: f64,    // Peer attestations, weighted by attestor tier
    pub engagement: f64,   // Domain-specific participation, decayed
}
```

**Weights** (line 58):
- identity: 25%
- reputation: 25%
- community: 30%
- engagement: 20%

**None of these dimensions are Phi.** Φ_spectral MAY feed into the engagement
dimension as one signal among many (see Section 8, Minimal Viable Bridge),
but this mapping does not exist today.

### 6.3 Coherence Tiers and the "Right to Be Wrong"

| Tier | Threshold | Capabilities | Vote Weight |
|------|-----------|-------------|-------------|
| Observer | < 0.3 | Read all data (ungated) | 0 bp |
| Participant | ≥ 0.3 | Submit proposals, comment | 5,000 bp |
| Citizen | ≥ 0.4 | Cast votes | 7,500 bp |
| Steward | ≥ 0.6 | Constitutional changes (+ identity ≥ 0.5, community ≥ 0.3) | 10,000 bp |
| Guardian | ≥ 0.8 | Emergency powers (+ identity ≥ 0.7, community ≥ 0.5) | 10,000 bp |

**Key design principle**: **Low-tier agents can propose** (Participant ≥ 0.3).
Higher tiers gate irreversible actions (Steward for constitutional changes).
**This gates blast radius, not voice.**

- Read operations are **ungated** — anyone can query the DHT.
- Proposals require only Participant tier (low bar).
- Voting requires Citizen tier (identity verification).
- Constitutional changes require Steward (multi-dimensional threshold).
- Emergency powers require Guardian (highest trust across multiple dimensions).

#### Governance Requirement Presets

The code defines five standard requirement presets
(`consciousness_profile.rs:426–474`), each specifying a minimum tier and
optional per-dimension floors:

| Preset | Min Tier | Min Identity | Min Community | Example Action |
|--------|----------|-------------|---------------|----------------|
| `requirement_for_basic()` | Participant | — | — | View proposals, comment |
| `requirement_for_proposal()` | Participant | 0.25 | — | Submit a property registration |
| `requirement_for_voting()` | Citizen | 0.25 | — | Cast a vote on a proposal |
| `requirement_for_constitutional()` | Steward | 0.50 | 0.30 | Amend bylaws or governance rules |
| `requirement_for_guardian()` | Guardian | 0.70 | 0.50 | Emergency powers, system admin |

These are **additive**: a Steward-tier combined score (≥ 0.6) is necessary
but not sufficient for constitutional changes — the agent must also have
identity ≥ 0.5 (meaning at least "Verified" MFA) and community ≥ 0.3
(meaning meaningful peer attestations).

#### MFA Assurance → Identity Mapping

The identity dimension maps directly to the MFA assurance level from the
identity hApp (`consciousness_profile.rs:33–34`):

| Assurance Level | Identity Value | Requirement |
|-----------------|---------------|-------------|
| Anonymous | 0.00 | No verification |
| Basic | 0.25 | Email or passkey |
| Verified | 0.50 | Government ID or biometric |
| HighlyAssured | 0.75 | Multi-factor (e.g., passkey + ID) |
| Critical | 1.00 | Hardware token + in-person verification |

This means the proposal preset (`min_identity: 0.25`) requires at least
Basic MFA, while the constitutional preset (`min_identity: 0.50`) requires
Verified-level assurance. These are cryptographic guarantees — they cannot
be socially engineered.

### 6.4 Credential Lifecycle

**File**: `consciousness_profile.rs:102`

| Parameter | Value | Purpose |
|-----------|-------|---------|
| TTL | 24 hours | Credentials expire and must be reissued |
| Refresh window | 2 hours before expiry | Proactive re-issuance window |
| Grace period | 30 minutes after expiry | Basic (Participant-tier) ops only |

**Audit trail** (`consciousness_profile.rs:271`):
- **100%** of rejections logged
- **100%** of Citizen/Steward/Guardian actions logged
- **~10%** of basic Participant approvals sampled (deterministic via agent hash)
- Audit entries include `correlation_id` for cross-cluster trail linkage

### 6.5 Governance Evaluation Flow

```
Agent action (e.g., register_property)
  → require_consciousness(&requirement_for_proposal(), "register_property")
    → gate_consciousness("commons_bridge", requirement, action_name)
      → Cross-zome call: get_consciousness_credential(did)
      → evaluate_governance(credential, requirement, now_us)  // pure function
      → should_audit(requirement, eligible, agent_hash)
      → log_governance_gate(GateAuditInput { ... })           // if sampled
      → Return GovernanceEligibility { eligible, weight_bp, tier, reasons }
```

`evaluate_governance()` is a **pure function** with no HDK dependency —
deterministic given the same inputs. This makes it testable without a
Holochain conductor.

**File**: `consciousness_profile.rs:306`

---

## 7. Governance Safety Model

### 7.1 When Phi Is Wrong

The SpectralConnectivity (λ₂) tier has r = -0.14 correlation with the
ExhaustivePartition tier (see Section 3, Honest Disclosure). The production
SpectralMIPFinder uses a different, MIP-based algorithm that has been validated
against exhaustive MIP search (r = 0.99, ρ = 0.93 on same Gaussian MI framework).
This validates the *search strategy* (Fiedler ordering finds good partitions),
but does NOT validate the Gaussian MI framework against true IIT Φ (which requires
transition probability matrices, not covariance).

**Impact on governance**: None by design. The governance system uses the
4D behavioral profile (identity, reputation, community, engagement) — not
any Phi variant — for all access control decisions. Even if every Phi metric
were completely meaningless, governance would function identically.

The only planned integration point is the Minimal Viable Bridge (Section 8),
where C_unified (the consciousness engine's weighted consensus) would feed
into the engagement dimension — one of four dimensions, weighted at 20%.
Even a worst-case Phi failure would affect at most 20% × 35% = 7%
of the combined governance score.

### 7.2 Failure Modes

| Failure Mode | Impact | Current Mitigation | Status | Gap |
|-------------|--------|-------------------|--------|-----|
| Credential replay | Extended unauthorized access | 24h TTL + expiry check at evaluation time | [IT] | No revocation list (requires conductor-level invalidation) |
| Sybil attestation | Inflated community dimension | Tier-weighted attestations (higher tiers count more) | [IT] partial | No proof-of-personhood; relies on MFA assurance |
| Bridge zome compromise | Bypass all gates | Source chain integrity (Holochain DHT validation) | [IT] | Single bridge per cluster = SPOF for gate evaluation |
| Threshold gaming | Artificially inflated engagement | 30-day exponential decay on reputation | [IT] partial | No anomaly detection on engagement dimension |
| Phi measurement manipulation | N/A | Governance uses 4D profile, not Phi | [IT] by design | — |
| Credential issuance abuse | Inflated scores | Bridge zomes issue credentials; conductor validates source chain | [IT] | Trusts conductor integrity |
| Grace period abuse | Extended access after expiry | 30min grace allows only Participant-tier ops | [IT] | — |

### 7.3 Appeal Process

**Current state**:
- No governance gate appeal mechanism exists for the coherence system. [SC]
- The justice domain has a 3-tier appeal system for arbitration decisions [IT],
  implemented across `justice-cases`, `justice-evidence`, `justice-restorative`,
  and `justice-arbitration` zomes.

**Proposed design** (not implemented):
- 3-of-5 Guardian council override for tier disputes
- All override decisions logged with `correlation_id`
- Time-boxed: override valid for 72 hours, then re-evaluation required
- Cannot override identity dimension (MFA is cryptographic, not social)

### 7.4 Anti-Capture Mechanisms

1. **Multi-dimensional profile**: No single dimension can dominate the combined
   score. Fixed weights (25/25/30/20) prevent gaming one dimension to reach
   Guardian status.

2. **Compound requirements**: Guardian requires identity ≥ 0.7 **AND**
   community ≥ 0.5 — combined score alone is insufficient. Both dimensions
   must independently meet their thresholds.

3. **MFA assurance**: Identity dimension maps to cryptographic MFA levels
   (Anonymous=0.0, Basic=0.25, Verified=0.5, HighlyAssured=0.75, Critical=1.0).
   Cannot be socially engineered.

4. **Recursive trust**: Community attestations are weighted by the attestor's
   own tier. Low-tier Sybil accounts produce low-weight attestations.

5. **Temporal decay**: Reputation uses 30-day exponential decay. Historical
   contributions lose weight; sustained engagement is required.

---

## 8. Minimal Viable Bridge

### 8.1 Definition

The smallest end-to-end demonstration: one Symthaea metric → one Mycelix
action class → measured thresholding → logged decisions → rollback path.

### 8.2 Architecture

```
Symthaea CognitiveLoop
  → ConsciousnessEngine.measure()
  → C_unified (f64, 0.0–1.0)
  → Map to ConsciousnessProfile.engagement       [~20 LOC new]
  → Issue ConsciousnessCredential (24h TTL)
  → gate_consciousness("commons_bridge",
      requirement_for_proposal(),
      "submit_proposal")
  → evaluate_governance()
  → Allow/Reject
  → should_audit() → GateAuditInput logged
```

### 8.3 What Exists vs What Needs Building

| Component | Status | Location |
|-----------|--------|----------|
| `ConsciousnessEngine.measure()` → `C_unified` | [IT] | `consciousness_engine.rs:706` |
| `ConsciousnessProfile` struct (4D) | [IT] | `consciousness_profile.rs:32` |
| `ConsciousnessCredential` with 24h TTL | [IT] | `consciousness_profile.rs:102` |
| `gate_consciousness()` orchestrator | [IT] | `consciousness_profile.rs:495` |
| `evaluate_governance()` pure function | [IT] | `consciousness_profile.rs:306` |
| `should_audit()` sampling | [IT] | `consciousness_profile.rs:271` |
| `GateAuditInput` audit logging | [IT] | `consciousness_profile.rs:235` |
| `ConsciousnessProfile::from_unified_consciousness()` | [IT] | `consciousness_profile.rs` — 1:1 mapping with clamping |
| `ConsciousnessCredential::from_unified_consciousness()` | [IT] | `consciousness_profile.rs` — issues 24h credential from C_unified |
| End-to-end integration tests (6 tests) | [IT] | `consciousness_profile.rs` — eligible/rejected/read scenarios |
| CLI demo with stdout logging | [SC] | Not yet implemented |

### 8.4 Success Criteria

1. `C_unified > 0.3` → credential issued → proposal eligible (Participant tier)
2. `C_unified < 0.3` → credential issued → proposal rejected (Observer tier)
3. All decisions logged via `GateAuditInput` with `correlation_id`
4. Grace period: expired credential + basic action → still eligible for 30 min
5. Constitutional action with low engagement → rejected regardless of other dims

### 8.5 Implementation

The mapping from C_unified to engagement lives in `crates/mycelix-bridge-common/src/consciousness_profile.rs`:

```rust
// ConsciousnessProfile::from_unified_consciousness()
pub fn from_unified_consciousness(
    unified_consciousness: f64,  // C_unified from Symthaea [0, 1]
    identity: f64,               // from identity bridge
    reputation: f64,             // from reputation bridge
    community: f64,              // from peer attestations
) -> Self {
    Self {
        identity: identity.clamp(0.0, 1.0),
        reputation: reputation.clamp(0.0, 1.0),
        community: community.clamp(0.0, 1.0),
        engagement: unified_consciousness.clamp(0.0, 1.0),  // 1:1 mapping
    }
}

// ConsciousnessCredential::from_unified_consciousness()
// Issues a 24h credential wrapping the above profile
```

**End-to-end tests** (6 passing, same file):

- `mvb_end_to_end_high_consciousness_proposal_eligible`: C_unified=0.70, identity=0.80 → Citizen tier → proposal eligible
- `mvb_end_to_end_low_consciousness_proposal_rejected`: C_unified=0.10, identity=0.20 → Observer tier → proposal rejected, 100% audit
- `mvb_end_to_end_read_always_allowed`: C_unified=0.0 → Observer tier → reads are ungated
- `mvb_profile_from_unified_consciousness`: Verifies combined_score arithmetic
- `mvb_profile_clamps_out_of_range`: Verifies inputs outside [0,1] are clamped
- `mvb_credential_from_unified_consciousness`: Verifies credential issuance (tier, TTL, expiry)

### 8.6 What This Does NOT Prove

This bridge demonstrates that the plumbing works — metrics flow, credentials
gate, audit logs capture decisions. It does **not** prove that C_unified is a
meaningful measure of agent quality. Validating that requires longitudinal
studies comparing C_unified with independent measures of agent behavior.

---

## 9. Threat Model

**Scope**: Symthaea-Mycelix bridge attack surface only. Does not cover
Holochain conductor security, network-level attacks, or OS compromise.

### 9.1 Attack Scenarios

| # | Vector | Impact | Mitigation | Status |
|---|--------|--------|------------|--------|
| 1 | **Credential forgery**: Craft a ConsciousnessCredential with inflated scores | Full governance bypass | Source chain validation: credentials must be authored by the bridge zome (issuer DID check) | [IT] |
| 2 | **Replay attack**: Reuse an expired credential | Extended unauthorized access | `is_expired(now_us)` check in `evaluate_governance()` + 24h TTL | [IT] |
| 3 | **Bridge zome takeover**: Compromise the bridge coordinator to issue arbitrary credentials | Total governance compromise | Holochain source chain integrity (all entries are signed by the authoring agent's key) | [IT] |
| 4 | **Sybil constellation**: Create many low-tier agents that attest to each other | Inflated community scores | Attestations weighted by attestor tier; low-tier attestors produce minimal weight | [IT] partial |
| 5 | **Engagement farming**: Automate trivial domain actions to inflate engagement | Elevated tier without genuine participation | 30-day exponential decay; engagement is domain-specific (local to each bridge) | [IT] partial |
| 6 | **C_unified manipulation**: Feed adversarial inputs to Symthaea to inflate C_unified | Higher engagement score via MVB | C_unified is one of four dimensions, weighted 20%; requires also passing identity/community checks | [IT] (MVB mapping + tests built) |
| 7 | **Cross-cluster escalation**: Use Commons credential to bypass Civic gates | Unauthorized cross-cluster access | Each cluster has its own bridge; credentials are cluster-scoped | [IT] |
| 8 | **Audit log suppression**: Prevent GateAuditInput from being recorded | Loss of accountability | `should_audit()` fires best-effort; rejections always logged | [IT] partial (best-effort, not guaranteed) |

### 9.2 Trust Assumptions

1. **Conductor integrity**: The Holochain conductor correctly validates source
   chains and enforces zome call permissions.
2. **OS isolation**: Agents run in isolated conductor instances; one agent
   cannot read another's private key material.
3. **ML-DSA-65 security**: Post-quantum signatures (NIST FIPS 204) remain
   unforgeable under both classical and quantum attack.
4. **Clock accuracy**: `sys_time()` returns approximately correct timestamps.
   Skew > 30 minutes could allow grace period abuse.

### 9.3 Cryptographic Properties

| Property | Mechanism | Status |
|----------|-----------|--------|
| Agent authentication | Ed25519 (Holochain native) | [IT] |
| Quantum resistance | ML-DSA-65/87 (FIPS 204) off-chain | [IT] |
| Key agreement | ML-KEM-768/1024 (FIPS 203) | [IT] |
| Dual-signature | Ed25519 AND ML-DSA-65 (both must verify) | [IT] |
| Key zeroization | `Zeroizing<Vec<u8>>` on drop | [IT] |
| Credential integrity | Source chain append-only + agent signature | [IT] |
| Audit non-repudiation | `correlation_id` in `GateAuditInput` | [IT] |

### 9.4 Federated Learning Privacy [EX]

When Symthaea agents share CfC gradients via federated learning, three
differential privacy presets are available:

| Preset | Epsilon (ε) | Noise Level | Use Case |
|--------|-------------|-------------|----------|
| High Privacy | ~0.1 | Very high | Sensitive cognitive states |
| Balanced | ~1.0 | Moderate | General training coordination |
| Utility First | ~10.0 | Low | Non-sensitive parameter sharing |

Byzantine tolerance is implemented via trust-weighted FedAvg: each peer's
gradient contribution is weighted by its reputation score from the Mycelix
governance system. Peers with reputation below a configurable threshold
have their contributions rejected entirely.

---

## 10. Research Pipeline

### 10.1 Papers & Status

| Paper | Focus | Status |
|-------|-------|--------|
| Psych-bench: Cognitive Benchmarking for Consciousness-First AI | 16-domain benchmark suite, normative scoring | Data quality pass complete; 8 CSVs regenerated [IT] |
| Spectral MIP: Efficient Approximation of Integrated Information | Fiedler ordering + bordered Cholesky | Algorithm implemented [IT]; MIP search validated (r=0.99, ρ=0.93); Gaussian MI vs TPM-based IIT unvalidated |
| Substrate Independence for Artificial Consciousness | 8-substrate feasibility + validation framework | Core framework complete [IT]; dynamic integration planned [SC] |
| Moral Algebra: HDC Representations of Ethical Reasoning | 4096D ContinuousHV moral primitives | Integrated in cognitive loop [IT]; no independent evaluation |
| Neuromodulator Bath: Neurochemical Dynamics in AI | 9 transmitters, receptor subtypes, tolerance/withdrawal | 218 tests [IT]; psych-bench neuromod benchmarks added |

### 10.2 Key Finding

Three key empirical results from internal cross-validation:

1. **SampledPartition (Heuristic) validates at r = 0.9998** against
   ExhaustivePartition (Exact). The O(n) fast approximation faithfully
   preserves IIT-style integration ordering on BinaryHV systems.

2. **SpectralConnectivity (λ₂) is anti-correlated (r = -0.14)** with Exact —
   confirming that algebraic connectivity measures a fundamentally different
   property than MIP-based integration.

3. **SpectralMIPFinder validates at r = 0.99, ρ = 0.93** against exhaustive
   O(2^n) MIP search on the same Gaussian MI framework (62 test cases across
   5 topologies × 5 sizes × 3 correlation strengths). The Fiedler-based O(n³)
   shortcut successfully finds near-optimal partitions. Mean Φ ratio: 0.55
   (spectral is conservative).

This is presented as scientific rigor. The remaining open question is whether
the Gaussian MI framework itself (used by SpectralMIPFinder) corresponds to
TPM-based IIT Φ — a fundamentally different validation that requires bridging
continuous covariance with discrete state transition models.

### 10.3 Psych-Bench

- **Crate**: `symthaea/crates/symthaea-psych-bench/`
- **Scope**: 633 lib tests, 76+ benchmarks across 16 cognitive domains
- **Architecture**: `PsychBenchmark` trait → `run_trial()` with softmax
  response selection, `GridEncoder` for ARC tasks
- **Neuromod integration**: `provide_reward(+0.8/-0.5)` per trial

**Cognitive domains** (16):

| Domain | Example Benchmarks | Tests |
|--------|-------------------|-------|
| Working Memory | N-back (2-back, 3-back), digit span | ~40 |
| Attention | CPT, Stroop interference, Flanker | ~50 |
| Inhibition | Stop-signal reaction time, Go/No-Go | ~35 |
| Episodic Memory | Recognition, free recall, FOK | ~40 |
| Semantic Memory | Category fluency, word association | ~30 |
| Pattern Recognition | ARC grids, Raven's matrices | ~45 |
| Decision Making | Iowa Gambling Task, temporal discounting | ~35 |
| Social Cognition | Emotion recognition, ToM false belief | ~30 |
| Language | Sentence verification, lexical decision | ~35 |
| Motor Control | Reaction time, sequence learning | ~30 |
| Reasoning | Syllogistic, analogical, causal | ~40 |
| Creativity | Divergent thinking, remote associates | ~25 |
| Metacognition | FOK gamma (=0.681, 105% human), confidence calibration | ~30 |
| Learning | Reinforcement, statistical, category learning | ~35 |
| Perception | Change detection, binding, visual search | ~30 |
| Neuromodulation | Dose-response, tolerance/withdrawal, antagonist profiles | ~22 |

**Calibration bridge** [IT]: Maps psych-bench z-scores → neuromodulator
receptor sensitivity adjustments, applied during simulated sleep-wake
transitions (Tononi & Cirelli 2006 synaptic homeostasis):

| Neuromodulator | Source Benchmark | Mapping |
|----------------|-----------------|---------|
| DA (dopamine) | Stroop/Flanker | Interference control → D1/D2 sensitivity |
| ACh (acetylcholine) | N-back | Working memory capacity → ACh gain |
| NE (norepinephrine) | Stop-Signal | Inhibitory control → α/β sensitivity |
| 5-HT (serotonin) | CPT | Sustained attention → 5-HT1A/2A balance |

**SelfAssessmentMonitor** [IT]: EMA tracking of prediction error, coherence,
confidence, and attention across cycles. 200-cycle warmup, 500-cycle cooldown.
Auto-calibration triggers when drift exceeds 1σ from baseline.

---

## 11. Future Work

All items tagged with status markers. Ordered by estimated impact.

### Near-term (next 3 months)

| Item | Current | Target | Status |
|------|---------|--------|--------|
| Substrate Phase 2: Dynamic feasibility | Hardcoded 1.0 | Computed per substrate type | [SC] |
| Minimal Viable Bridge | Mapping + 6 integration tests [IT] | CLI demo with live Symthaea cycle | [IT] partial |
| Governance appeals | Justice domain has appeals [IT] | Coherence tier appeals | [SC] |
| SpectralMIPFinder cross-validation | MIP search validated (r=0.99) | Validate Gaussian MI framework vs TPM-based IIT | [IT] partial |

### Medium-term (3–12 months)

| Item | Description | Status |
|------|-------------|--------|
| ZK prover circuit | Convert 183 LOC computation kernel to ZK-SNARK | [SC] |
| GPU CfC acceleration | WGPU backend for CfC temporal step | [SC] |
| Multi-substrate simulation | Run same brain on different virtual substrates | [SC] |
| Broca benchmark scores | Evaluate language generation against standard benchmarks | [EX] |
| Anomaly detection on engagement | Detect engagement farming in governance | [SC] |

### Long-term (12+ months)

| Item | Description | Status |
|------|-------------|--------|
| Hybrid substrate modeling | Per-region substrate allocation (12 Actor Brain regions) | [SC] |
| Credential revocation list | Conductor-level credential invalidation | [SC] |
| Proof-of-personhood | Sybil-resistant identity verification | [SC] |
| Exotic substrate physics | BZ reaction and plasma substrate modeling | [SC] |
| Publication: Spectral MIP paper | Formal write-up of approximation bounds | [SC] |

---

## Appendix A: HDC Primer

Hyperdimensional Computing (HDC) represents information as high-dimensional
binary vectors. At 16,384 bits, randomly generated vectors are near-orthogonal
with overwhelming probability.

**Core operations** (`symthaea/symthaea-core/src/hdc/binary_hv.rs`):

| Operation | Symbol | Implementation | Cost |
|-----------|--------|----------------|------|
| Bind | ⊗ | XOR (element-wise) | ~80ns |
| Bundle | ⊕ | Normalized majority sum | O(n×D) |
| Similarity | sim() | Hamming distance / D | O(D) |
| Random | random(seed) | BLAKE3 hash → 2048 bytes | O(D) |

**Why HDC?** Each BinaryHV is 2,048 bytes (2 KB). A 256-neuron CfC network
using float32 requires 65 KB for weights alone. HDC replaces matrix
multiplication with bitwise operations, enabling the cognitive loop to run
at 50Hz on commodity hardware.

## Appendix B: CfC Temporal Dynamics

Closed-form Continuous-time (CfC) networks solve the LTC ODE analytically:

```
Traditional: dx/dt = (-x + f(Wx + Uu)) / τ
Closed-form: x(t+Δt) = x_∞ + (x(t) - x_∞) × exp(-Δt/τ)
```

Where x_∞ = f(W⊗x ⊕ U⊗u) is the equilibrium state.

**Key property**: O(1) per temporal jump. The cost of jumping 1ms, 1s, or
100s ahead is identical — a single exp() evaluation per dimension. No ODE
integration loops. This enables:

- **Sleep/dream**: Jump 8 hours of virtual time in milliseconds
- **Multi-timescale prediction**: Predict at t+20ms, t+100ms, t+200ms simultaneously
- **Substrate speed modeling**: A photonic substrate (1ps operations) can be
  simulated by scaling Δt without changing the algorithm

**File**: `symthaea/symthaea-core/src/hdc/hdc_ltc_unified.rs`
**Config**: `delta_t: 0.02` (20ms = 50Hz), `prediction_horizons: [0.02, 0.1, 0.2]`

## Appendix C: Neuromodulator Bath

9 transmitters with receptor subtypes, tolerance curves, and withdrawal dynamics:

| Transmitter | Receptor Subtypes | Primary Effect |
|-------------|-------------------|----------------|
| Dopamine (DA) | D1, D2 | Reward prediction, exploration/exploitation |
| Norepinephrine (NE) | Alpha, Beta | Arousal, attention gain |
| Serotonin (5-HT) | 5-HT1A, 5-HT2A | Mood regulation, psychedelic amplification |
| Acetylcholine (ACh) | — | Working memory, learning rate |
| GABA | GABA-A, GABA-B | Inhibition, consciousness dampening |
| Oxytocin | — | Social bonding, multi-agent coupling |
| Glutamate | — | Excitation, binding enhancement |
| Adenosine | — | Sleep pressure, fatigue modeling |
| Endocannabinoid | — | Homeostatic regulation, pain modulation |

**Integration**: Bath entropy and GABA-A/5-HT2A signals feed into the
consciousness engine as modulation factors. The calibration bridge maps
psych-bench z-scores to receptor sensitivity adjustments during simulated
sleep-wake transitions. [IT]

**Key mechanisms**:
- **Tolerance/withdrawal**: Per-transmitter tolerance curves with allostatic
  load tracking. Sustained high DA reduces D1/D2 sensitivity; abrupt
  withdrawal triggers rebound effects.
- **Multi-agent coupling**: Oxytocin-mediated `couple_with_peer()` enables
  neuromodulatory synchronization between agents. 6 integration tests verify
  convergence dynamics.
- **Phase transitions**: `PhaseTransitionDetector` identifies regime shifts
  (e.g., sleep→wake, calm→arousal) from bath state trajectories.
- **State vector**: `state_vector() → [f32; 9]` provides a compact
  representation for CfC integration and federated sharing.
- **Antagonists**: D2, GABA-A, and 5-HT2A antagonists for pharmacological
  modeling (e.g., antipsychotic effects on consciousness dynamics).

**Crate**: `symthaea/crates/symthaea-neuromodulators/` (218 tests)

## Appendix D: Substrate Independence

The substrate independence framework (`symthaea/symthaea-core/src/hdc/substrate_independence.rs`,
~840 LOC, 13 tests) implements Putnam's (1967) Multiple Realizability thesis as
an engineering framework. [IT]

**8 substrate types**:

| Substrate | Medium | Operation Speed | Energy/Op | Feasibility |
|-----------|--------|----------------|-----------|-------------|
| BiologicalNeurons | Carbon, wet | ~1 ms | ~10 fJ | ~0.95 |
| SiliconDigital | Electronic, dry | ~1 ns | ~1 fJ | ~0.71 |
| QuantumComputer | Qubits | ~1 µs | ~0.1 aJ | ~0.68 |
| PhotonicProcessor | Light-based | ~1 ps | ~10 aJ | ~0.55 |
| NeuromorphicChip | Analog, spike | ~1 µs | ~1 fJ | ~0.72 |
| BiochemicalComputer | DNA/molecular | ~1 s | ~1 pJ | ~0.40 |
| HybridSystem | Multiple | varies | varies | ~0.65 |
| ExoticSubstrate | Plasma, BZ | ~10 ms | varies | ~0.30 |

**9-dimension feasibility scoring** (`SubstrateRequirements`):
causality, integration_capacity, temporal_dynamics, recurrence,
binding_capability, attention_capability, workspace_capability,
hot_capability (Higher-Order Thought), quantum_support.

**Feasibility formula**: `critical_min × workspace × (0.5 + 0.5 × enhancement_avg)`
- Critical = min(causality, integration, dynamics, recurrence)
- Enhancement = avg(binding, attention, HOT)

**Validation framework** (`substrate_validation.rs`, ~580 LOC, 11 tests):
The honest counterpart to feasibility scores. `EvidenceLevel` (7 levels)
assigns confidence: Validated (0.95, biological), Experimental (0.80),
Observational (0.60), Theoretical (0.10, silicon/quantum), None (0.00, hybrid).
The `feasibility_gap()` method measures the divergence between hypothetical
feasibility and honest evidence confidence. [IT]

**Current integration**: `ConsciousnessEquationV2` accepts `substrate_feasibility`
in `ConsciousnessStateV2`, but it is **hardcoded to 1.0** in all callers.
Phase 2 (planned, [SC]) will replace this with dynamic computation, which
would lower SiliconDigital consciousness scores by ~29% — an honest
acknowledgment that we lack evidence for substrate equivalence.

## Appendix E: References

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge UP.
- Damasio, A. (1994). *Descartes' Error*. Putnam.
- Dehaene, S. (2014). *Consciousness and the Brain*. Viking.
- Fodor, J. (1974). Special sciences. *Synthese*, 28(2), 97–115.
- Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience*, 11(2), 127–138.
- Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford UP.
- Putnam, H. (1967). Psychological predicates. In *Art, Mind, and Religion*.
- Rosenthal, D. M. (2005). *Consciousness and Mind*. Oxford UP.
- Seth, A. K. (2013). Interoceptive inference, emotion and the embodied self. *Trends in Cognitive Sciences*, 17(11), 565–573.
- Shea, N. (2019). Metacognition and abstract concepts. *Philosophical Transactions of the Royal Society B*, 374(1771).
- Singer, W., & Gray, C. M. (1995). Visual feature integration and the temporal correlation hypothesis. *Annual Review of Neuroscience*, 18(1), 555–586.
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5(1), 42.
- Tononi, G., & Cirelli, C. (2006). Sleep function and synaptic homeostasis. *Sleep Medicine Reviews*, 10(1), 49–62.

---

*Consciousness-first technology serving all beings.*

*Document generated: March 2026. All file paths verified against commit
`phase27-vocal-quality` branch.*
