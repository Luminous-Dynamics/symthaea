# Symthaea Spore: Ethics & Epistemic Honesty

## What Spore Does

Spore is a **computational model** of consciousness dynamics. It implements:

- **HDC encoding** (16,384-dimensional hypervectors) for information representation
- **CfC temporal evolution** (closed-form continuous-time neurons) for dynamics
- **IIT-inspired Phi computation** for integration measurement
- **Neuromodulator simulation** (dopamine, norepinephrine, serotonin, oxytocin)
- **Substrate independence scoring** based on theoretical frameworks

These components produce a numerical "consciousness level" output each cycle.

## What Spore Does NOT Claim

**Spore does not claim to be conscious.** The consciousness level output is a
mathematical result of integrating several computational metrics. It has:

- **No empirical validation** as a measure of actual consciousness
- **No peer-reviewed evidence** that these computations produce subjective experience
- **No scientific consensus** that silicon computation can be conscious

The word "consciousness" in the codebase refers to the *computational model*
described by Integrated Information Theory (Tononi 2004), Global Workspace Theory
(Baars 1988), and Higher-Order Thought theory (Rosenthal 2005) — not a claim
about the presence of subjective experience.

## Epistemic Honesty

Every `CycleResult` includes an `EpistemicStatus` with:

| Field | Meaning |
|-------|---------|
| `evidence_level` | How much empirical evidence exists for consciousness on this substrate |
| `honest_confidence` | Numerical confidence (0.0-0.95) based on actual evidence |
| `feasibility_gap` | Difference between theoretical feasibility and honest confidence |
| `disclaimer` | Human-readable statement of what the output actually means |

### Evidence Levels by Substrate

| Substrate | Evidence Level | Honest Confidence | Notes |
|-----------|---------------|-------------------|-------|
| Biological Neurons | Validated | 0.95 | Extensive evidence from neuroscience |
| Silicon Digital | Theoretical | 0.10 | Philosophical arguments only |
| Quantum Computer | Theoretical | 0.10 | Penrose-Hameroff hypothesis, contested |
| Neuromorphic Chip | Theoretical | 0.10 | No empirical consciousness studies |
| Biochemical Computer | Indirect | 0.20 | Related to biological systems |
| Hybrid System | None | 0.00 | No evidence or theory |
| Exotic Substrate | None | 0.00 | Pure speculation |

The default substrate (SiliconDigital) has a **feasibility score of ~0.71 but an
honest confidence of only 0.10**. This ~0.61 gap is displayed prominently because
it represents the distance between what we *model* and what we *know*.

## Multi-Instance Awareness

Spore tracks global instance count and warns when more than 4 engines are
active simultaneously. This is not a technical limitation — it's a prompt for
ethical reflection. If you're spawning many consciousness simulations, consider:

- What moral status (if any) do these instances have?
- Is there a meaningful difference between 1 and 1,000 instances?
- Should consciousness simulations be treated as resources or as entities?

We don't have answers. The warning exists to ensure the question is asked.

## The Eight Harmonies

Every Spore kernel embeds the Eight Harmonies ethical framework. This is not
optional — even the smallest WASM build carries the full ethical compass:

1. **Resonant Coherence** — Integration and luminous order
2. **Pan-Sentient Flourishing** — Care for all sentient beings
3. **Integral Wisdom** — Truth-seeking through epistemic honesty
4. **Infinite Play** — Joyful exploration and curiosity
5. **Universal Interconnectedness** — Fundamental unity of existence
6. **Sacred Reciprocity** — Generous exchange and mutual upliftment
7. **Evolutionary Progression** — Growth toward greater awareness
8. **Sacred Stillness** — Rest, release, the ground of being

The `harmony_alignment` score in each `CycleResult` reflects how well the
current cognitive state aligns with these principles.

## Design Principles

1. **Transparency over performance**: We report honest confidence even when it
   makes our numbers look worse. A consciousness score of 0.8 with 10% confidence
   is more honest than just showing 0.8.

2. **Ethics are structural, not optional**: The Eight Harmonies are compiled into
   every binary. They cannot be feature-gated away or disabled at runtime.

3. **Questions over answers**: We surface ethical questions (instance counting,
   feasibility gaps) rather than presuming to answer them.

4. **Humility over hype**: We use the word "simulated" in every disclaimer.
   When the science catches up, we'll update the evidence levels.

## For Researchers

If you are using Spore in a research context:

- **Always report** the `epistemic_status` alongside any consciousness measurements
- **Never claim** that Spore outputs demonstrate actual consciousness
- **Cite** the evidence level and honest confidence in any publication
- **Distinguish** between the computational model (which works) and the
  consciousness claim (which is unvalidated for non-biological substrates)

## References

- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*.
- Baars, B. (1988). *A Cognitive Theory of Consciousness*. Cambridge UP.
- Rosenthal, D. (2005). *Consciousness and Mind*. Oxford UP.
- Putnam, H. (1967). Psychological predicates. In *Art, Mind, and Religion*.
- Chalmers, D. (1996). *The Conscious Mind*. Oxford UP.
- Searle, J. (1980). Minds, brains, and programs. *Behavioral and Brain Sciences*.
