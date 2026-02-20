# Consciousness Metrics: What We Actually Compute

This document provides honest documentation of the three distinct consciousness-adjacent
metrics used across Mycelix and Symthaea. None of these metrics currently compute true
IIT Phi (Integrated Information Theory's Φ).

---

## 1. Integration (formerly "Phi" in the bridge)

**Type**: `IntegrationAssessment` (in `symthaea-mycelix-bridge/src/lib.rs`)

**What it computes**: SpectralConnectivity — the algebraic connectivity (Fiedler value)
of the network's connectivity graph, computed via Symthaea's PhiEngine.

**What it does NOT compute**: True IIT Phi (Φ). The correlation between SpectralConnectivity
and true IIT Phi is r=0.097 (near zero). This metric captures network connectivity structure,
not integrated information.

**Fields**:
- `integration_before`: Fiedler value before applying an FL update
- `integration_after`: Fiedler value after applying an FL update
- `integration_gain`: Difference (after - before)

**Used by**:
- `SymthaeaQualityPlugin` — FL anomaly detection (integration drop = anomaly signal)
- `ConsciousFlRound` — Per-node quality assessment in FL rounds
- `SymthaeaBackend` — Configurable drop thresholds (`integration_drop_threshold`)

**Known limitations**:
- Near-zero correlation with true IIT Phi
- Can be trivially gamed by maintaining graph connectivity without genuine integration
- Sensitive to network topology but blind to causal structure

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

**What it computes**: An attested scalar [0.0, 1.0] derived from Symthaea's assessment
pipeline. Currently composed of the Integration metric above, epistemic confidence, and
anomaly detection. Signed with Ed25519 and stored on-chain for governance gating.

**What it does NOT compute**: A verified measure of consciousness. This is a composite
quality score that we label "consciousness level" as a governance gate, not a scientific
claim.

**Used by**:
- `GovernanceConsciousnessConfig` — Action-type gates (Basic >= 0.2, Proposal >= 0.3,
  Voting >= 0.4, Constitutional >= 0.6)
- `ConsciousnessAttestation` — On-chain DHT entry for governance verification
- `AdaptiveThreshold` — Dynamic voter consciousness requirements per proposal type
- Holistic vote weight formula: `Reputation^2 x (0.7 + 0.3 x ConsciousnessLevel) x
  (1 + 0.2 x HarmonicAlignment)`, capped at 1.5

**Known limitations**:
- Currently relies on SpectralConnectivity (r=0.097 with true Phi)
- Single scalar loses dimensional information
- Attestation can be replayed if signatures aren't properly scoped

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

## 5. Future: Consciousness Vector (C-Vector)

The planned replacement for the single misleading scalar. Each dimension has known
limitations documented:

| Dimension | Method | Complexity | Source |
|-----------|--------|------------|--------|
| Integration | True IIT Phi (bounded approx for n>12) | O(2^n) / O(n^2) | Symthaea TruePhiCalculator |
| Differentiation | Shannon entropy of state space | O(n) | Symthaea entropy module |
| Composition | Count of mechanisms with phi > threshold | O(2^n) | Symthaea IIT4 module |
| Exclusion | Compare Phi across system boundaries | O(n * 2^k) | New |
| Temporal Depth | Integration persistence over time | O(window) | CoherenceTimeSeries |
| Self-Modeling | Prediction error of internal model | O(1) | Symthaea active inference |

The C-vector replaces the single scalar with an honest multi-dimensional profile.
Each dimension feeds into:
- K-vector (`k_coherence` dimension becomes C-vector summary)
- FL gating (multi-signal instead of single threshold)
- Governance (dimensional requirements per action type)

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

Backward-compatible aliases exist only in `mycelix-bridge-common` (`PhiThresholds` type alias,
`phi_thresholds()` function alias, and `phi_thresholds` module alias) and `mycelix-fl`
Cargo.toml (`phi-series` feature alias for `coherence-series`).
