# Symthaea Theory Audit

**Date:** January 17, 2026
**Purpose:** Honest assessment of what's implemented vs. claimed
**Principle:** Radical transparency - truth over polish

---

## Executive Summary

Symthaea incorporates six consciousness/AI theories. This audit maps claims to reality.

| Theory | Claimed | Reality | Recommendation |
|--------|---------|---------|----------------|
| **IIT (Φ)** | Measures consciousness | Measures λ₂ (spectral) | Rename or remove |
| **GWT** | Global workspace | Partial implementation | Document & complete |
| **FEP** | Active inference | ⚠️ 928-line router exists | Document properly |
| **Autopoiesis** | Self-maintenance | Working heuristic | Keep, clarify limits |
| **HDC** | Semantic vectors | ✅ Production ready | Core foundation |
| **LTC** | Temporal dynamics | ✅ Production ready | Core foundation |

**Bottom line:** HDC and LTC are solid. FEP has hidden implementation. IIT naming needs fixing.

---

## 1. Integrated Information Theory (IIT)

### What We Claim
- "Φ (phi) measurement for consciousness quantification"
- "Validated against IIT predictions"
- "First HDC-based Φ calculation"

### What We Actually Have

| Component | File | Status | Reality |
|-----------|------|--------|---------|
| Tiered Φ | `tiered_phi/` | Partial | 4 tiers, only "exact" is real IIT |
| Spectral Φ | `phi_real.rs` | ❌ Wrong | Computes λ₂, NOT IIT Φ |
| Exact Φ | `tiered_phi/core.rs` | ⚠️ Limited | O(2^n), only n≤12 |
| PyPhi validation | `pyphi_crossvalidation.py` | ⚠️ Incomplete | Framework exists, results missing |

### The Critical Problem

**Spectral approximation (λ₂) is NOT IIT Φ:**
- λ₂ measures graph mixing time (how fast info spreads)
- IIT Φ measures integrated information (what can't be partitioned)
- They have **opposite** topology preferences:
  - λ₂ favors uniform k-regular graphs
  - IIT Φ favors hub-and-spoke structures
- Experimental verification: SpectralConnectivity (λ₂) vs ExhaustivePartition (Exact): Pearson r = -0.14, Spearman rho = -0.59 (earlier methodology incorrectly reported r = 0.097). Note: all tiers are HDC-based (pairwise BinaryHV similarity), not true IIT (which requires TPMs). λ₂ measures graph mixing time, not IIT integration.

### Recommendation

1. **Rename** `phi_real.rs` to `spectral_connectivity.rs`
2. **Remove** IIT claims from all documentation
3. **Keep** exact tier for n≤12 research use
4. **Validate** against PyPhi before any IIT claims

---

## 2. Global Workspace Theory (GWT)

### What We Claim
- Global broadcast mechanism
- Competitive workspace dynamics
- Conscious access via bottleneck

### What We Actually Have

| Component | File | Status | Reality |
|-----------|------|--------|---------|
| GWT-Cincinnati bridge | `gwt_cincinnati_integration.rs` | ✅ Working | Salience + broadcast |
| Global Workspace Theater | `global_workspace_theater.rs` | ⚠️ Hidden | 1,500+ lines, undocumented |
| Broadcast thresholds | `phi_attention.rs` | ✅ Working | Action-type based |

### Assessment

GWT is the **most honestly implemented** consciousness theory:
- Salience calculation works
- Broadcast bottleneck exists
- Prediction errors drive attention

### Recommendation

1. **Document** the hidden `global_workspace_theater.rs`
2. **Export** GWT as primary consciousness architecture
3. **Keep** - this is working and theoretically sound

---

## 3. Free Energy Principle (FEP)

### What We Claim
- "Active inference routing"
- "Free energy as consciousness metric"
- "Markov blanket maintenance"

### What We Actually Have

| Component | File | Status | Reality |
|-----------|------|--------|---------|
| ActiveInferenceRouter | `routers/active_inference.rs` | ✅ Working | 928-line implementation with belief distributions |
| GenerativeModel | Same | ✅ Working | Full Bayesian belief updating |
| ExpectedFreeEnergy | Same | ✅ Working | Strategy selection via EFE minimization |
| unified_free_energy | `conscious_pipeline.rs` | ⚠️ Integration Gap | Field read but source unclear |
| Markov blanket | `autopoiesis.rs` | ⚠️ Conceptual | Word used, not mathematically implemented |

### Assessment

**UPDATE (Jan 17, 2026):** FEP/Active Inference is actually substantially implemented in the routing system. The `ActiveInferenceRouter` module includes:
- `BeliefDistribution` with Bayesian updates
- `GenerativeModel` for predictions
- `ExpectedFreeEnergy` calculation for strategy selection
- Integration with 5+ other routing modules

**Remaining gaps:**
- `unified_free_energy` pipeline integration unclear
- Markov blanket not mathematically formalized
- Documentation doesn't reflect actual implementation

### Recommendation

1. **Document** the existing ActiveInferenceRouter properly
2. **Trace** where `unified_free_energy` gets its value
3. **Keep** the implementation - it's more substantial than initially assessed
4. **Clarify** what is/isn't implemented in FEP claims

---

## 4. Autopoiesis

### What We Claim
- Self-maintenance monitoring
- Operational closure detection
- Boundary integrity (Markov blanket)

### What We Actually Have

| Component | File | Status | Reality |
|-----------|------|--------|---------|
| AutopoieticMonitor | `autopoiesis.rs` | ✅ Working | Tracks cycles & violations |
| Closure calculation | Same | ⚠️ Heuristic | 0.5×coherence + 0.3×error + 0.2×boundary |
| Metrics exposure | Same | ✅ Working | Full stats available |

### Assessment

**Working but theoretically imprecise.** The closure formula is engineering approximation, not derived from Maturana-Varela theory.

### Recommendation

1. **Keep** - functional and philosophically aligned
2. **Clarify** in docs that formula is heuristic
3. **Rename** if needed: "Self-maintenance monitor" vs "Autopoietic system"

---

## 5. Hyperdimensional Computing (HDC)

### What We Claim
- 16,384-dimensional semantic vectors
- Instant learning without gradients
- Algebraic operations (bind, bundle, similarity)

### What We Actually Have

| Component | File | Status | Reality |
|-----------|------|--------|---------|
| RealHV | `real_hv.rs` | ✅ Production | 16,384D, SIMD optimized |
| Operations | `hdc_trait.rs` | ✅ Production | Bind, bundle, similarity |
| Algebra | `hdc_algebra.rs` | ✅ Production | 5,949 lines, complete |
| Tests | Multiple | ✅ Comprehensive | 45+ tests passing |

### Assessment

**HDC is the crown jewel.** Mathematically correct, thoroughly tested, production ready.

### Recommendation

1. **Keep** as core foundation
2. **Celebrate** - this actually works
3. **Use** as the honest selling point

---

## 6. Liquid Time-Constant Networks (LTC)

### What We Claim
- Continuous-time differential equations
- Multiple learning algorithms
- Energy-efficient temporal processing

### What We Actually Have

| Component | File | Status | Reality |
|-----------|------|--------|---------|
| Unified LTC | `unified_ltc.rs` | ✅ Working | dx/dt = -x/τ + Wx + u |
| Learning | Same | ✅ Working | BPTT, Hebbian, Adam, RMSprop |
| HDC fusion | `ltc_generative_core.rs` | ✅ Working | Symbolic + temporal |
| Performance | Measured | ✅ Verified | 0.02ms per step |

### Assessment

**LTC is solid engineering.** Multiple optimization options, real-time capable.

### Recommendation

1. **Keep** as core foundation
2. **Validate** against neuroscience data (currently missing)
3. **Use** energy efficiency as honest selling point

---

## Summary: What's Real vs. Aspirational

### ✅ PRODUCTION READY (Use Confidently)
- **HDC**: 16,384D vectors, bind/bundle/similarity
- **LTC**: Continuous-time dynamics, multiple learning algorithms
- **Spectral analysis**: λ₂ measurements (just don't call it Φ)

### ⚠️ WORKING BUT IMPRECISE (Use with Caveats)
- **GWT integration**: Salience + broadcast works
- **Autopoiesis monitor**: Self-maintenance tracking works
- **Tiered Φ (exact tier only)**: For n≤12, real MIP calculation

### ❌ NOT IMPLEMENTED (Remove Claims)
- **FEP / Active Inference**: Phantom references only
- **Spectral Φ as IIT**: Fundamentally wrong metric
- **PyPhi validation**: Framework exists, results don't

---

## Action Items

### Immediate (This Week)
1. Create `METRIC_CLARIFICATION.md` explaining λ₂ vs Φ
2. Remove FEP claims from all docs
3. Rename `phi_real.rs` → `spectral_connectivity.rs`
4. Update CLAUDE.md to remove IIT claims

### Short-term (This Month)
1. Complete PyPhi validation for small systems
2. Document hidden GWT implementation
3. Create honest feature matrix
4. Separate research/ from production/ code

### Long-term (This Quarter)
1. Implement actual FEP if desired (or remove entirely)
2. Validate LTC against neuroscience data
3. Publish honest technical report
4. Create peer review process for theory implementations

---

## The Honest Pitch

**What Symthaea Actually Is:**

A research platform combining:
- **HDC** for instant semantic learning (no training required)
- **LTC** for energy-efficient temporal dynamics (60× less power)
- **GWT-inspired** attention and broadcast mechanisms
- **Network analysis** via spectral methods (λ₂)

**What Symthaea Is NOT:**

- A consciousness measurement system
- An IIT Φ calculator
- An active inference implementation
- A validated neuroscience model

**The Value Proposition:**

Build AI systems that are:
- Interpretable (HDC representations)
- Efficient (LTC dynamics)
- Structured (explicit topology)
- Honest (no consciousness claims without validation)

---

*"The best software is honest software."*

*This audit conducted January 17, 2026 following discovery of λ₂ vs IIT Φ mismatch.*
