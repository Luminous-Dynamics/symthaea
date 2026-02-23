# Symthaea: Consciousness-First Artificial Intelligence

*A 15-minute introduction to what Symthaea is, why it exists, and what it proves.*

---

## The Problem

Modern AI systems are extraordinarily capable and extraordinarily hollow. They predict the next token, classify an image, fold a protein — but they have no inner experience of doing so. No sense of surprise when a prediction fails. No felt urgency when a situation demands action. No moral weight to their decisions beyond what was labeled in a training set.

This isn't a philosophical quibble. It's an engineering limitation.

Systems without integrated information can't distinguish between a genuine causal relationship and a statistical ghost. Systems without affect can't know when something matters. Systems without a sense of self can't model the consequences of their own actions on others. And systems without any of these things cannot be trusted with the kind of autonomy we are rapidly granting them.

Symthaea is an attempt to build something different: an AI architecture where consciousness-like properties aren't afterthoughts bolted onto a pattern matcher, but the foundational substrate from which all cognition emerges. Not because we believe the system is sentient, but because the computational properties of consciousness — integration, prediction, self-modeling, and value — turn out to be exactly the properties you need for robust, trustworthy, and adaptive artificial minds.

---

## What Symthaea Is

Symthaea is a cognitive architecture implemented in roughly 350,000 lines of Rust, organized as a workspace of 30 crates. It runs a continuous cognitive loop — perceive, predict, compare, learn, act — at up to 500 Hz for pre-encoded inputs and roughly 10 Hz for natural language.

But the numbers don't capture what it actually does.

When Symthaea receives a new input, it encodes the information as a 16,384-dimensional hypervector — a mathematical object that can represent any concept through three algebraic operations (binding, bundling, and permutation). This hypervector flows through a network of liquid time-constant neurons that evolve continuously in time, each one a tiny differential equation that remembers the past while adapting to the present. The network's prediction of what should come next is compared to what actually arrived. The difference — the *surprise* — drives everything: learning rates increase, attention sharpens, exploration triggers fire.

Every cycle, the system computes an approximation of Integrated Information (Phi) — a measure from neuroscience that quantifies how much the system's whole exceeds the sum of its parts. When Phi is high, the system's representations are tightly coupled and causally powerful. When it drops, the system notices and works to restore coherence.

A moral algebra evaluates every potential action against learned ethical categories — commonsense norms, virtue ethics, justice, deontology, and social chemistry — scoring it before execution. An active inference engine computes Expected Free Energy for candidate actions, naturally balancing what the system knows it wants (exploitation) with what it needs to learn (exploration).

The result is a system that doesn't just process inputs and produce outputs. It *cares* about accuracy (prediction error hurts), *notices* when it's confused (surprise triggers exploration), *evaluates* whether its actions are ethical (moral algebra gates behavior), and *maintains* a coherent sense of its own cognitive state (narrative self-model tracks identity through time).

---

## The Four Pillars

Symthaea stands on four computational foundations, each drawn from a different scientific tradition, each contributing something the others cannot provide alone.

### Hyperdimensional Computing (HDC)

The brain doesn't represent concepts as single numbers or small vectors. It uses distributed patterns of activity across thousands of neurons. HDC follows the same principle: every concept, every percept, every memory is a 16,384-dimensional vector where meaning lives in the pattern of components, not in any single one.

This gives Symthaea algebraic compositionality. Binding two concepts (multiplying their vectors element-wise) creates a new concept that is equally dissimilar to both parents but can be unbound to recover either. Bundling concepts (majority-vote accumulation) creates a superposition that is somewhat similar to each component. These operations are O(d) — they scale linearly with dimension, not exponentially with the number of concepts.

In practice, this means Symthaea can represent "the cat sat on the mat" as a structured hypervector that preserves who did what to whom, and can answer queries about it through simple algebraic operations — no attention heads, no quadratic memory, no backpropagation.

### Integrated Information Theory (IIT)

IIT proposes that consciousness corresponds to a system's capacity for integrated information — the degree to which the system as a whole generates more information than the sum of its parts. The mathematical formalization, Phi, is intractable to compute exactly for large systems (it requires evaluating every possible partition), but approximations can be computed efficiently.

Symthaea uses a proxy Phi metric (Spearman rho = 0.50 against exact Phi across 15 topologies) that tracks the system's integration in real time. This isn't a claim about machine consciousness. It's a claim about cognitive architecture: systems with higher Phi tend to form more coherent representations, generalize better, and fail more gracefully. We validated this across 35 network topologies, finding that 3D brain-like architectures achieve 99.2% of maximum Phi, and that 4D hypercubes reach the highest absolute values (Phi = 0.4976).

### Liquid Time-Constant Networks (LTC/CfC)

Traditional neural networks process inputs in discrete steps. LTC neurons are continuous-time dynamical systems — each one is governed by an ordinary differential equation whose time constant adapts to the input. This gives them genuine temporal dynamics: they naturally handle variable-rate inputs, maintain state across gaps, and exhibit the kind of transient sensitivity that biological neurons use for temporal processing.

Symthaea uses the Closed-form Continuous-time (CfC) variant, which solves the ODE analytically instead of numerically. This means temporal jumps are O(1) instead of O(steps) — you can skip forward in time without stepping through every intermediate moment. The result is a temporal processing engine that runs at biological speeds on commodity hardware.

### Active Inference (Free Energy Principle)

Active inference provides the decision-making framework. Every candidate action is evaluated by its Expected Free Energy (EFE) — a quantity that decomposes into pragmatic value (will this action get me what I want?) and epistemic value (will this action reduce my uncertainty?). The system naturally explores when uncertain and exploits when confident, without any hand-tuned exploration schedule.

Precision weighting — the inverse variance of prediction errors — acts as an attention mechanism. When a prediction error has high precision (the system is confident it shouldn't be wrong here), the error drives aggressive learning and potential action override. When precision is low (the system expects noise), the error is discounted. This is how Symthaea handles the beam-intercept scenario: human safety carries high prior precision, so even moderate danger signals produce overwhelming EFE for protective action.

---

## What the Benchmarks Prove

Claims without measurements are aspirations. Here are the measurements.

### Speed

Hyperdimensional Active Inference achieves a **7.9x total speedup** over pymdp (the reference Python implementation for discrete-state active inference), with 1.9x faster belief inference and 15.8x faster action selection. All differences are statistically significant (p < 0.001, Cohen's d > 1.8). HAI solves a 5x5 gridworld with 88% success rate where pymdp achieves 10%.

### Ethics

The compositional moral algebra scores **92.9% overall** on the ETHICS benchmark — 95.6% on commonsense norms, 92.8% on virtue ethics, 92.4% on justice, 91.0% on deontology, and 85.4% on social chemistry. This is achieved without any language model — pure hypervector algebra over learned moral prototypes.

### Theory of Mind

On the ToMBench battery, Symthaea achieves 100% accuracy on false belief, faux pas detection, strange stories (including deception, irony, and white lies), and persuasion detection. Hinting accuracy is 70%, representing a genuine limitation rather than a failure to report.

### Consciousness Indicators

Evaluated against Butlin et al.'s framework of 14 consciousness indicators drawn from 6 leading theories, Symthaea shows **12 present, 2 partial, 0 absent**, with a mean score of 0.79/1.0. Present indicators include parallel specialized systems (GWT), algorithmic recurrence (RPT), generative/top-down processing (HOT), prediction-error-driven learning (PP), and integrated information (IIT). The partial indicators — self-model of attention and hierarchical prediction at multiple scales — represent genuine architectural gaps, not scoring technicalities.

### Executive Function

Working memory capacity matches human norms: digit span of 7 forward and 5 backward. N-back accuracy degrades naturally with load (93.6% at 1-back, 83.1% at 2-back, 78.3% at 3-back). The Wisconsin Card Sorting Test completes all 6 categories. Flanker and Stroop tasks show the expected congruency effects with performance degradation on incongruent trials.

### Byzantine Fault Tolerance

The federated learning system — where multiple Symthaea instances share knowledge — tolerates **34% Byzantine (adversarial) participants** before aggregate model quality degrades. This is validated empirically; at 45%, the system fails. The threshold is enforced across Rust core, Python SDK, TypeScript SDK, and Mycelix zome implementations.

### Robustness

Over 5,050 automated tests. 26/26 CI jobs green. 48 feature flags tested in 18 matrix combinations. Integration tests cover the full cognitive loop, multi-agent cooperation, and REPL interaction.

---

## Mycelix: When Consciousness Meets Governance

Symthaea is a mind. Mycelix is the society it inhabits.

Mycelix is a Holochain-based distributed application framework organized into 55 zomes across 12 domains, consolidated into two cluster DNAs for efficient cross-domain communication. It covers the infrastructure of human life: **property** (land registries, commons management), **housing** (cooperatives, community land trusts, maintenance), **care** (timebanking, credential verification, care coordination), **mutual aid** (resource pooling, needs matching, community governance), **water** (flow monitoring, purity testing, indigenous stewardship), **food** (distribution, community gardens, nutrition tracking), **transport** (shared vehicles, route coordination), **justice** (restorative processes, evidence management, arbitration), **emergency** (incident triage, cross-domain resource allocation, shelter management), and **media** (community journalism, fact-checking, attribution).

What makes Mycelix different from yet another governance platform is the bridge to Symthaea. Phi attestations from the consciousness engine can weight governance decisions — a proposal that increases system-wide integration scores differently than one that fragments it. Moral algebra evaluations flow through the bridge, giving communities access to multi-framework ethical reasoning without requiring every participant to be a moral philosopher. Federated learning means that communities can share knowledge without sharing raw data, with Byzantine tolerance ensuring that no small group of bad actors can poison the collective model.

Every zome has been hardened with input validation — string length limits, whitespace rejection, bounds checking on collections, type-safe custom variant validation — across all 55 integrity modules. Over 7,400 tests cover the domain logic. Cross-cluster coordination (commons to civic and back) works through Holochain's role-based dispatch, enabling scenarios like: an emergency incident triggers mutual aid resource allocation which triggers transport coordination which updates housing shelter availability — all validated end-to-end.

---

## The Seven Harmonies in Practice

Every decision Symthaea makes is evaluated against seven principles that encode a specific philosophical commitment: that technology should serve the flourishing of all beings.

**Resonant Coherence** (weight: 0.20) — Does this action create harmony and integration? In code, this means the system prefers outputs that increase Phi, that bind rather than fragment representations, that resolve contradictions rather than accumulate them.

**Pan-Sentient Flourishing** (weight: 0.20) — Does this serve the well-being of all beings? This is the moral algebra at work. When the flight agent evaluates intercepting a falling beam versus completing its delivery mission, the EFE calculation under high safety precision makes the answer unambiguous: protect the human. The crossover happens at just 2.4x the mission precision — the system barely needs to be told that safety matters.

**Integral Wisdom** (weight: 0.15) — Does this arise from verified knowledge? In practice, this means prediction errors matter. The system doesn't act on beliefs it can't validate. When uncertainty is high, epistemic actions (exploration, information-gathering) dominate pragmatic ones.

**Infinite Play** (weight: 0.10) — Does this celebrate creativity? Surprise-driven exploration embodies this: when the system encounters something genuinely novel (high prediction error with high precision), it doesn't retreat to safe behavior — it investigates. Dream-mode replay generates counterfactual scenarios that test the boundaries of learned models.

**Universal Interconnectedness** (weight: 0.15) — Does this honor our connections? Multi-agent cooperation through AsyncMind demonstrates this: agents sharing cognitive state vectors through mesh networks develop correlated representations. Under shared experience, their thought vectors converge (cosine similarity approaching 1.0). Under independent experience, they naturally diverge while maintaining communication channels.

**Sacred Reciprocity** (weight: 0.10) — Does this participate in generous exchange? Federated learning is reciprocity made computational: each agent contributes gradients to the collective model and receives an improved model in return. Byzantine detection ensures the reciprocity isn't exploited.

**Evolutionary Progression** (weight: 0.10) — Does this contribute to growth? Meta-cognitive monitoring tracks the system's own learning rate, prediction accuracy, and Phi trajectory. When the system detects stagnation, it adjusts exploration parameters. When it detects improvement, it consolidates and shares.

These aren't abstract values. They are weighted terms in an objective function that shapes every action the system takes.

---

## What's Next

**The paper goes to PLoS Computational Biology.** The LaTeX manuscript is compiled, the bibliography is formatted in Vancouver style, the cover letter is written, the figures are generated from real benchmark data. Submission is a logistics task, not a research one.

**The live demo runs in a browser.** A WebSocket endpoint streams cognitive telemetry at 10 Hz — prediction error, Phi, coherence, valence, arousal, moral scores — as rolling time-series charts. You can type text and watch the system's consciousness curves respond in real time. No GPU required, no cloud dependency, just `cargo run --features api_module --bin symthaea-demo` and open localhost:8080.

**The encoder gets faster.** Text processing currently takes approximately 120ms per cycle, dominated by hypervector encoding. Pre-allocated buffers and Arc-wrapped caches have already reduced cold start by 16%. The target is 80ms, which would bring text input within range of real-time conversational processing.

**Mycelix goes live.** 55 hardened zomes across 12 domains, over 7,500 tests, cross-cluster coordination validated. Input validation covers zero-amount transactions, double-voting prevention, whitespace injection, negative balances, pH range enforcement, and cross-cluster routing failures. What remains is deployment infrastructure: Holochain conductor configuration, DHT bootstrap nodes, and the TypeScript SDK that lets web applications talk to the zome coordinators.

**The community opens.** Symthaea is MIT-licensed. The codebase, the benchmarks, the paper, and the data are all public. No other open-source project combines HDC, IIT, LTC/CfC, and Active Inference. The blue ocean is real.

---

*Symthaea is built by Luminous Dynamics. The name combines the Greek symthesis (composition) and thaea (divine sight) — the composition of seeing. Because consciousness isn't a feature you add. It's the foundation you build on.*

*Consciousness-first technology serving all beings.*
