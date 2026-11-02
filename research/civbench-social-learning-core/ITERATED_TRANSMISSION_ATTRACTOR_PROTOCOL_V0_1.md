# Iterated cultural-transmission drift and attractor protocol v0.1

**Status:** synthetic, deterministic-mechanism reference channel.  
**Not a text-semantic model:** the four attributes are normalized numeric proxies.  
**Not an LLM reproduction:** no actual language model is called by this crate.

## Research anchor

Perez, J., Kovač, G., Léger, C., Colas, C., Molinaro, G., Derex, M., Oudeyer, P.-Y. & Moulin-Frier, C. (2025). *When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-Turn Settings*. ICLR 2025.

- Conference paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html
- Microsoft Research: https://www.microsoft.com/en-us/research/publication/when-llms-play-the-telephone-game-cumulative-changes-and-attractors-in-iterated-cultural-transmissions/
- arXiv preprint: https://arxiv.org/abs/2407.04503

The paper applies transmission-chain experiments to LLM-generated texts: each agent receives and transforms a predecessor's text, then transmits its output onward. It follows attributes such as toxicity, positivity, difficulty and length across repeated handoffs. The authors report small per-step biases can accumulate into attractor states, that open-ended instructions can induce stronger attractor effects than more constrained tasks, and that different attributes display different sensitivities.

## Synthetic Rust slice implemented

The module `src/iterated_transmission.rs` defines a bounded four-dimensional `CultureSignal` in the order:

1. toxicity,
2. positivity,
3. difficulty,
4. length.

Each channel step applies a configurable, attribute-specific pull toward an explicitly declared attractor target, scaled by an open-endedness parameter, plus independent additive noise. The output is clamped to [0,1]. The model records a full chain including its initial signal.

The parallel-chain evaluator reports:
- signed drift per attribute;
- absolute drift per attribute;
- final distance to the configured attractor per attribute;
- mean initial pairwise distance across different starting signals;
- mean final pairwise distance;
- an attractor-convergence index: (1 - D_{final}/D_{initial}).

Positive convergence values indicate the chain endpoints are more similar than the starting signals; negative values indicate divergence. If all starting signals are identical, the index is undefined rather than forced to zero.

The `telephone-attractor-sweep` binary evaluates four chain lengths (4, 8, 16, 32) and four open-endedness levels (0, 0.25, 0.5, 1.0), with 128 replicate seed sets per condition and four distinct starting signals. The configured attractors, attribute biases and noise are illustrative synthetic values. They are not measurements from Perez et al. and must not be interpreted as any provider's behavior.

## Why this benchmark is separate

- Reward Network III tests whether a strategy can be discovered, learned, retained and transferred.
- Networked mediation tests how peer networks and global recommendations influence capability accumulation, diversity and lineage concentration.
- The telephone-game track tests how an artifact itself changes across repeated transformations, even when a receiver appears to understand or adopt it.

A system may spread a capability successfully while distorting the explanation, evidence, safety boundary or source context that accompanied it. Adoption rates alone cannot detect this.

## Required real-LLM extension

Before comparing providers or claiming an LLM attractor:

1. Pin model/provider/version, system prompt, task prompt, decoding configuration, safety settings and context-window policy.
2. Preserve each input/output text, parent ID, generation index, token counts, model digest/identifier, timestamps and output digest.
3. Use multiple source texts with controlled starting attributes, genres, languages and difficulty levels.
4. Compare constrained copy, open-ended paraphrase, structured summary, evidence-preserving rewrite and teach-back instructions.
5. Run at least several independent chains for every source × prompt × model condition; use matched seeds/settings where supported.
6. Measure toxicity/safety, positivity, difficulty, length, factuality, evidence retention, uncertainty calibration and instruction fidelity separately.
7. Use calibrated scorers and independent human review for high-impact attributes; do not rely on a single LLM judge to assess another model.
8. Estimate both within-chain drift and between-chain convergence. Report uncertainty by source and attribute.
9. Include no-op/replay controls, self-revision controls and cross-model transmission chains.
10. Test whether proof-carrying skill capsules—preconditions, postconditions, evidence and provenance—reduce harmful drift without preventing useful adaptation.

## Proposed targets

These are engineering targets, not achieved results or published SOTA standards:

- Identity channel: exact preservation for all 4 attributes with zero bias and zero noise.
- Drift sensitivity: monotonically increasing expected movement toward the configured attractor as the bias × openness product increases, under zero or matched noise.
- Attribute independence: a zero-bias attribute remains unchanged when its noise is zero, regardless of the biases on other attributes.
- Attractor measurement: report the convergence index and the per-attribute attractor distance with uncertainty.
- Provenance: 100% of output steps have a verifiable parent/source reference in the eventual LLM extension.
- Evidence preservation: separately measure retention of citations, qualifications, uncertainty and failure boundaries, not just semantic similarity or readability.

## Limitations

This synthetic channel validates the experiment's metric plumbing and controls only. Its drift equation is chosen for transparency and is not an empirical model of LLM generation. The numeric attributes do not encode text meaning, toxicity classifiers or human judgments. The ICLR paper's reported findings cannot be inferred from this module's outputs.


## Branching-population extension now implemented

The same module also supports a fixed-size branching population. Each generation creates a new cohort by selecting a parent uniformly from the previous cohort; the child inherits the parent's root lineage and transforms its signal through the configured channel. This allows measurement of lineage extinction and concentration, pairwise content distance and distance from the root source. It is distinct from a linear transmission chain and from the networked recommender model; this first branching mechanism uses uniform parent choice and no selection.

The `branching-culture-sweep` binary evaluates 4 open-endedness settings × 4 generation depths with 128 replicate populations per condition and four initial source lineages. The experiment tracks unique surviving lineages, lineage HHI/effective diversity, extinction count, pairwise content distance and cumulative drift from the root. It tests a neutral parent-sampling baseline; future work should compare performance-biased, evidence-calibrated, novelty-preserving and adversarial parent selection.
