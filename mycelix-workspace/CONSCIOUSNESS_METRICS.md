# Consciousness Metrics: What We Actually Compute

This document provides honest documentation of the three distinct consciousness-adjacent
metrics used across Mycelix and Symthaea. None of these metrics currently compute true
IIT Phi (Integrated Information Theory's Φ).

---

## 1. Spectral Connectivity (formerly "Integration", originally "Phi" in the bridge)

**Type**: `SpectralConnectivityAssessment` (in `symthaea-mycelix-bridge/src/lib.rs`)

**What it computes**: The algebraic connectivity (Fiedler value) of the network's
connectivity graph, computed via Symthaea's PhiEngine.

**What it does NOT compute**: True IIT Phi (Φ). The SpectralConnectivity (lambda2) tier has
Pearson r = -0.14, Spearman rho = -0.59 vs ExhaustivePartition (Exact IIT Phi) -- a weak
negative correlation. Lambda2 measures graph mixing time, not IIT integration. The production
SpectralMIPFinder (MI Laplacian + Fiedler + MIP sweep) is a distinct algorithm whose
correlation with Exact is unknown. This metric captures network connectivity structure,
not integrated information.

**Fields**:
- `connectivity_before`: Fiedler value before applying an FL update
- `connectivity_after`: Fiedler value after applying an FL update
- `connectivity_gain`: Difference (after - before)

**Used by**:
- `SymthaeaQualityPlugin` — FL anomaly detection (connectivity drop = anomaly signal)
- `ConsciousFlRound` — Per-node quality assessment in FL rounds
- `SymthaeaBackend` — Configurable drop thresholds (`connectivity_drop_threshold`)
- `ConsciousnessVector` — One of 6 dimensions in the C-Vector (see Section 5)

**Known limitations**:
- Weak negative correlation with true IIT Phi (lambda2 measures graph mixing time, not integration)
- Can be trivially gamed by maintaining graph connectivity without genuine integration
- Sensitive to network topology but blind to causal structure

**Deprecated alias**: `IntegrationAssessment` (type alias kept for backward compatibility)

---

## 2. Coherence (formerly "Phi" in mycelix-fl)

**Type**: `CoherenceGateConfig` / `CoherenceScore` (in `mycelix-fl/src/coherence.rs`)

**What it computes**: A lightweight gradient quality proxy combining:
- L2 norm of the gradient vector
- Shannon entropy of the gradient distribution
- Cosine similarity between consecutive gradients (output consistency)

**What it does NOT compute**: Any form of integrated information or consciousness
measurement. This is a pure signal-quality metric.

**Used by**:
- `CoherenceGate` — FL pipeline stage that filters low-quality gradients
- `ConsciousnessAwareByzantinePlugin` — Score-based boost/dampen/veto
- K-Vector `k_coherence` dimension — Agent output consistency tracking

**Known limitations**:
- Measures gradient consistency, not consciousness
- Can be trivially gamed by submitting consistent but poisoned gradients
- Does not detect sophisticated model poisoning attacks

**Time-series tracking**: `CoherenceTimeSeries` (behind `coherence-series` feature)
provides trend analysis, anomaly detection, and statistics over coherence measurements.

---

## 3. Consciousness Level (formerly "Governance Phi")

**Type**: `ConsciousnessAttestationData` (in `symthaea-mycelix-bridge/src/lib.rs`)

**What it computes**: An attested scalar [0.0, 1.0] derived from the **ConsciousnessVector
composite** (see Section 5). The composite is a weighted average of all populated C-Vector
dimensions. Signed with Ed25519 and stored on-chain for governance gating.

**What it does NOT compute**: A verified measure of consciousness. This is a composite
quality score that we label "consciousness level" as a governance gate, not a scientific
claim. However, the C-Vector now includes True IIT Phi when available, making it
significantly more honest than the previous single-metric approach.

**Used by**:
- `GovernanceConsciousnessConfig` — Action-type gates (Basic >= 0.2, Proposal >= 0.3,
  Voting >= 0.4, Constitutional >= 0.6)
- `ConsciousnessAttestation` — On-chain DHT entry with optional `ConsciousnessVectorEntry`
- `AdaptiveThreshold` — Dynamic voter consciousness requirements per proposal type
- Holistic vote weight formula: `Reputation^2 x (0.7 + 0.3 x ConsciousnessLevel) x
  (1 + 0.2 x HarmonicAlignment)`, capped at 1.5
- Per-dimension gates (optional): `min_true_phi_constitutional`, `min_coherence_voting`

**Known limitations**:
- True IIT Phi is only computed for small component counts (n <= 4 by default) due to O(2^n) cost
- C-Vector composite still collapses to a scalar for gate checks (but full vector is stored)
- Attestation signature binds to the scalar composite, not the full vector

---

## 4. Canonical Thresholds

All threshold values live in `crates/mycelix-bridge-common/src/consciousness_thresholds.rs` as
`ConsciousnessThresholds`:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `consciousness_gate_basic` | 0.2 | Basic governance participation |
| `consciousness_gate_proposal` | 0.3 | Proposal submission |
| `consciousness_gate_voting` | 0.4 | Voting |
| `consciousness_gate_constitutional` | 0.6 | Constitutional changes |
| `fl_veto` | 0.1 | FL update vetoed below this |
| `fl_dampen` | 0.3 | FL update dampened below this |
| `fl_boost` | 0.6 | FL update boosted above this |

---

## 5. Consciousness Vector (C-Vector) — IMPLEMENTED

**Type**: `ConsciousnessVector` (in `symthaea-mycelix-bridge/src/lib.rs`)
**Governance parallel**: `ConsciousnessVectorEntry` (in `mycelix-governance/zomes/bridge/integrity/src/lib.rs`)

The C-Vector replaces the single misleading scalar with a multi-dimensional profile.
Each dimension is `Option<f64>` for incremental population. Implemented dimensions:

| Dimension | Field | Method | Complexity | Source |
|-----------|-------|--------|------------|--------|
| Spectral Connectivity | `spectral_connectivity` | Fiedler value (algebraic connectivity) | O(n²) | Symthaea PhiEngine |
| True Phi | `true_phi` | IIT Phi via minimum information partition | O(2^n), n<=4 default | Symthaea TruePhiCalculator |
| Fast Phi | `phi_fast` | Effective information approximation | O(n²) | Symthaea TruePhiCalculator |
| Entropy | `entropy` | Shannon entropy, normalized to [0,1] | O(n) | Symthaea TruePhiCalculator |
| Coherence | `coherence` | 1 - mean pairwise distance | O(n²) | Bridge computation |
| Epistemic Confidence | `epistemic_confidence` | Validation accuracy mapping | O(1) | Bridge computation |

### Composite Scoring

`ConsciousnessVector::composite()` produces a backward-compatible scalar [0,1]:

| Dimension | Weight |
|-----------|--------|
| True Phi / Fast Phi (fallback) | 0.35 |
| Coherence | 0.20 |
| Entropy | 0.15 |
| Epistemic Confidence | 0.15 |
| Spectral Connectivity | 0.15 |

Weights are renormalized over populated dimensions only. If no dimensions are populated,
composite returns 0.0.

### Fallback Chain

`ConsciousnessVector::best_phi()` returns the best available phi estimate:
`true_phi` → `phi_fast` → `spectral_connectivity` → `0.0`

### Performance Budget

| Dimension | Cost | When |
|-----------|------|------|
| spectral_connectivity | ~2ms (n≤8) | Always |
| phi_fast | ~5ms | Always |
| true_phi | ~100ms for n=8 | Only n≤4 by default (configurable) |
| entropy | O(n) | Always |
| coherence | O(n²) | Always for n≤64 |
| epistemic_confidence | ~0 (derived) | Always |

Total additional cost per assessment: ~10ms (without true_phi) or ~110ms (with true_phi for n≤4).

### Configuration

`ConsciousnessVectorConfig` (in bridge crate):
- `max_components_true_phi: usize` — Maximum components for true phi computation (default: 4)
- `compute_phi_fast: bool` — Whether to compute fast phi approximation (default: true)
- `max_entropy: f64` — Maximum entropy for [0,1] normalization (default: 4.0)

### Spectral-Phi Divergence Detection

A novel anomaly signal in the FL plugin: when `spectral_connectivity > 0.5` but
`best_phi < 0.1` (and true_phi or phi_fast was actually computed), the update is flagged.
This detects networks that look connected (high Fiedler value) but lack genuine integration
(low Phi) — a pattern consistent with adversarial graph manipulation.

Triggered in `SymthaeaQualityPlugin::adjustment_for()` as Rule 3, resulting in a 0.5x
weight multiplier with source `"symthaea_quality_spectral_phi_divergence"`.

### Where C-Vector Is Used

- **FL pipeline**: `QualityScore.consciousness_vector` — per-node quality assessment
- **FL plugin**: `adjustment_for()` uses `best_phi()` for boost, divergence for anomaly
- **Unified round**: `NodeRoundScore.consciousness_vector` — per-round tracking
- **Attestation**: `ConsciousnessAttestationData.consciousness_level` = `composite()`
- **Governance DHT**: `ConsciousnessAttestation.consciousness_vector` (Optional<ConsciousnessVectorEntry>)
- **Governance gates**: Per-dimension optional gates (`min_true_phi_constitutional`, `min_coherence_voting`)
- **Holistic voting weight**: `calculate_from_vector()` uses composite when vector available

---

## 6. Rename Reference

For developers migrating code, here are the key renames (Feb 2026):

| Old Identifier | New Identifier | Crate |
|---------------|---------------|-------|
| `PhiAssessment` | `IntegrationAssessment` | symthaea-mycelix-bridge |
| `PhiAttestationData` | `ConsciousnessAttestationData` | symthaea-mycelix-bridge |
| `phi.rs` | `coherence.rs` | mycelix-fl |
| `phi_series.rs` | `coherence_series.rs` | mycelix-fl |
| `PhiThresholds` | `ConsciousnessThresholds` | mycelix-bridge-common |
| `GovernancePhiConfig` | `GovernanceConsciousnessConfig` | governance-bridge-integrity |
| `PhiAttestation` | `ConsciousnessAttestation` | governance-bridge-integrity |
| `phi_config.rs` | `consciousness_config.rs` | governance-bridge-coordinator |
| `phi_bridge.rs` | `coherence_bridge.rs` | mycelix-sdk/agentic |
| `phi_integration.rs` | `coherence_integration.rs` | mycelix-sdk/agentic |
| `k_phi` | `k_coherence` | mycelix-sdk/matl/kvector |
| `phi_threshold` | `dampen_threshold` | mycelix-fl-core |
| `phi_boost_threshold` | `boost_threshold` | mycelix-fl-core |
| `default_phi` | `default_score` | mycelix-fl-core |
| `phi_scores` | `consciousness_scores` | mycelix-fl-core |
| `set_phi_scores()` | `set_consciousness_scores()` | mycelix-fl-core |
| `phi_for()` | `consciousness_score_for()` | mycelix-fl-core |
| `phi_thresholds.rs` | `consciousness_thresholds.rs` | mycelix-bridge-common |
| `gov_basic` | `consciousness_gate_basic` | mycelix-bridge-common |
| `gov_proposal` | `consciousness_gate_proposal` | mycelix-bridge-common |
| `gov_voting` | `consciousness_gate_voting` | mycelix-bridge-common |
| `gov_constitutional` | `consciousness_gate_constitutional` | mycelix-bridge-common |

### Phase 2 Renames (SpectralConnectivity + C-Vector, Feb 2026)

| Old Identifier | New Identifier | Crate |
|---------------|---------------|-------|
| `IntegrationAssessment` | `SpectralConnectivityAssessment` | symthaea-mycelix-bridge |
| `integration_before` | `connectivity_before` | symthaea-mycelix-bridge |
| `integration_after` | `connectivity_after` | symthaea-mycelix-bridge |
| `integration_gain` | `connectivity_gain` | symthaea-mycelix-bridge |
| `QualityScore.integration` | `QualityScore.spectral` | symthaea-mycelix-bridge |
| `integration_drop_threshold` | `connectivity_drop_threshold` | symthaea-mycelix-bridge |
| `NodeState.last_integration` | `NodeState.last_connectivity` | symthaea-mycelix-bridge |
| `classify_from_integration()` | `classify_from_connectivity()` | symthaea-mycelix-bridge |
| `integration_gain_boost_threshold` | `connectivity_gain_boost_threshold` | symthaea-mycelix-bridge (fl_plugin) |
| `NodeRoundScore.integration` | `NodeRoundScore.spectral` | symthaea-mycelix-bridge (unified_round) |

Backward-compatible aliases exist only in `mycelix-bridge-common` (`PhiThresholds` type alias,
`phi_thresholds()` function alias, and `phi_thresholds` module alias), `mycelix-fl`
Cargo.toml (`phi-series` feature alias for `coherence-series`), and `symthaea-mycelix-bridge`
(`IntegrationAssessment` type alias for `SpectralConnectivityAssessment`).
