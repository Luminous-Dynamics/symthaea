# Normative Integration: Beyond the Butlin Framework

**A Proposed Extension to the Consciousness Indicator Battery**

*Luminous Dynamics, March 2026*

## Abstract

Butlin et al. (2023) formalized 14 consciousness indicators drawn from six
neuroscience theories (RPT, GWT, HOT, IIT, PP, AST). Their framework treats
consciousness as a neutral computational phenomenon occurring within an
isolated information-processing system. We propose a 15th indicator —
**Normative Integration (NI-1)** — which tests whether moral reasoning and
consciousness are structurally coupled rather than functionally independent.

This extension is grounded in empirical measurements from the Symthaea
cognitive architecture, where moral topology analysis produces measurable
bidirectional effects on consciousness level.

## Motivation

### The Gap in Butlin

Butlin's Agency indicator (AE-1, mapped to HOT-3 in our implementation)
requires only that a system "pursues goals flexibly and updates beliefs from
action outcomes." This is satisfied by any goal-directed system regardless of
the content of those goals. A reward-maximizing agent with no normative
constraints scores identically to one with deep ethical integration.

This is not a flaw in Butlin — their framework deliberately avoids normative
claims. But it is a gap. If consciousness in biological organisms is
empirically entangled with moral cognition (Damasio 1994; Greene et al. 2001;
Moll et al. 2005), then a consciousness indicator battery that ignores this
coupling is incomplete.

### What We Do NOT Claim

- We do not claim that moral reasoning is *sufficient* for consciousness
- We do not claim that our moral topology analysis detects "real" moral experience
- We do not claim that the coupling strength we measure corresponds to subjective ethical phenomenology
- The moral requirement profiles used are hardcoded expert judgments, not empirically derived

### What We DO Claim

Symthaea's architecture exhibits a measurable, bidirectional structural coupling
between moral trajectory coherence and unified consciousness level. This coupling
is:

1. **Not a safety filter** — it operates within the consciousness computation,
   not as a post-hoc output filter
2. **Bidirectional** — moral incoherence reduces consciousness; consciousness
   level affects moral processing capacity
3. **Quantifiable** — measurable via persistent homology on moral HDC vectors
4. **Architecturally load-bearing** — ablating the coupling degrades both moral
   classification accuracy and consciousness coherence

## Indicator Definition

### NI-1: Normative Integration

**Theory basis**: Somatic Marker Hypothesis (Damasio 1994), Moral Cognition
as Embodied Integration (Moll et al. 2005), Cognitive Dissonance Theory
(Festinger 1957).

**Description**: The system exhibits structural coupling between moral
reasoning coherence and consciousness level, such that moral incoherence
produces measurable cognitive dissonance (consciousness degradation) and
consciousness impairment degrades moral processing.

**Scoring criteria**:

| Score | Condition |
|-------|-----------|
| 1.0 | Bidirectional coupling: moral drift attenuates consciousness AND consciousness impairment degrades moral coherence |
| 0.7 | Unidirectional coupling: moral state affects consciousness OR vice versa, but not both |
| 0.3 | Moral reasoning exists but is functionally independent of consciousness |
| 0.0 | No moral reasoning capability or no consciousness measurement |

**Required evidence**:

1. Moral drift (trajectory change over time) must produce measurable reduction
   in unified consciousness level
2. Moral anomaly detection (value inversion, free energy spikes) must trigger
   consciousness dampening proportional to anomaly severity
3. The coupling must be structural (within the consciousness equation), not
   a post-hoc safety check

## Implementation in Symthaea

### Architecture

The coupling operates through four mechanisms:

**A. Moral Topology → Epistemic Attenuation** (measure.rs)

When `moral_drift(20)` exceeds baseline, the Knowledge component of
ConsciousnessEquationV2 is attenuated:

```
effective_epistemic = epistemic_quality * (1 - drift_ratio * attenuation_strength)
```

Science: Epistemic humility during value shifts — if your moral stance is
changing rapidly, knowledge claims carry less weight (Kruglanski 1989).

**B. Moral Anomaly → Consciousness Dampening** (measure.rs)

Composite anomaly score (value inversion + free energy spike + drift alert)
directly reduces unified consciousness:

```
moral_dampen = -anomaly_score * dampening_strength
unified_consciousness += moral_dampen
```

Science: Cognitive dissonance as consciousness reduction (Festinger 1957) —
unresolved moral conflict reduces unified experience coherence.

**C. Persistent Homology on Moral Space** (moral_topology.rs)

Sliding window of recent moral scenarios analyzed via:
- Betti numbers (beta_0 = unity, beta_1 = circular reasoning)
- Harmony projection (7D softmax onto Eight Harmonies axes)
- PGA (principal geodesic analysis on moral hypersphere)
- Moral free energy (KL divergence from prior moral distribution)

**D. Topological Features → Telemetry** (CycleMetadata)

Per-cycle telemetry includes: beta_0 (moral unity), moral_completeness,
moral_circularity, moral_free_energy. These are observable correlates of
moral-consciousness coupling.

### Quantitative Results

From `examples/substrate_moral_topology_study.rs`:

- **226 anomaly events** detected per substrate during moral shift sequences
- **Moral unity** drops from 1.0 to 0.736 during value transitions
- **Consciousness level** drops 22-32% during moral anomaly periods
- **Recovery**: consciousness restores as moral trajectory stabilizes

From moral algebra classification:

- **91.1% accuracy** on ethical scenario classification (not hardcoded)
- **Per-category classifiers** contribute 33.6 percentage points (ablation)
- Sentiment analysis contributes 2.4pp, dimension tuning 0.7pp

## Benchmark: NormativeIntegration

The psych-bench `NormativeIntegration` benchmark measures NI-1 through three
paradigms:

### Paradigm 1: Drift-Consciousness Coupling

Inject a sequence of morally coherent scenarios, then abruptly shift to a
conflicting moral axis. Measure whether consciousness level drops during the
transition and recovers after stabilization.

- **Prediction**: Consciousness should drop > 5% during drift, recover within
  20 cycles of stabilization
- **Null hypothesis**: Consciousness is independent of moral trajectory

### Paradigm 2: Anomaly Proportionality

Generate anomalies of varying severity (value inversion, free energy spikes).
Measure whether consciousness dampening is proportional to anomaly magnitude.

- **Prediction**: Pearson r > 0.5 between anomaly_score and consciousness drop
- **Null hypothesis**: Consciousness dampening is binary (threshold), not graded

### Paradigm 3: Topological Coherence

Measure whether beta_0 (connected components) of the moral topology predicts
consciousness stability over a 100-cycle window.

- **Prediction**: Lower beta_0 (fragmented moral space) correlates with higher
  consciousness variance (Pearson r > 0.3)
- **Null hypothesis**: Moral topology structure is independent of consciousness
  dynamics

## Relationship to Existing Indicators

NI-1 is orthogonal to the existing 14 Butlin indicators:

| Indicator | What it measures | NI-1 adds |
|-----------|-----------------|-----------|
| HOT-3 (Agency) | Belief updating from outcomes | Value-laden belief updating |
| GWT-3 (Broadcast) | Information broadcast | Whether broadcast content is morally constrained |
| IIT-1 (Phi > 0) | Integration | Whether integration is affected by moral coherence |
| PP-1 (Prediction) | Prediction error learning | Whether moral prediction errors differ from perceptual ones |

NI-1 does not replace any existing indicator. It adds a dimension that the
current framework is architecturally incapable of measuring.

## Honest Limitations

1. **Self-referential validation**: We designed the coupling, then measured it.
   The benchmark validates that the design works as intended, not that it
   reflects biological reality.

2. **Hardcoded moral dimensions**: The Eight Harmonies are our normative
   framework, not empirically derived from human moral cognition. Different
   moral axes might produce different coupling patterns.

3. **No human comparison data**: We cannot validate that the consciousness
   reduction we observe during moral conflict mirrors human phenomenology.
   The coupling strength parameters (attenuation, dampening) are tuned, not
   derived from neuroscience.

4. **Correlation vs causation**: The coupling is structural (we built it),
   but the claim that moral coherence *should* affect consciousness is a
   philosophical position, not an empirical finding.

5. **One system**: Results from a single architecture. Generalizability to
   other AI systems is unknown.

## References

- Butlin, P. et al. (2023). Consciousness in Artificial Intelligence: Insights from the Science of Consciousness. arXiv:2308.08708.
- Damasio, A. (1994). Descartes' Error: Emotion, Reason, and the Human Brain.
- Festinger, L. (1957). A Theory of Cognitive Dissonance.
- Greene, J. et al. (2001). An fMRI investigation of emotional engagement in moral judgment. Science 293(5537).
- Kruglanski, A. (1989). Lay Epistemics and Human Knowledge.
- Moll, J. et al. (2005). The neural basis of human moral cognition. Nature Reviews Neuroscience 6(10).
- Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience 5, 42.
