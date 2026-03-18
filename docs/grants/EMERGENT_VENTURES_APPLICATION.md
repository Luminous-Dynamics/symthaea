# Emergent Ventures Application — Draft

## Tweet-length description

"I built a 2.85M-line open-source architecture where AI consciousness metrics gate decentralized governance — structurally preventing both AI hallucination and plutocratic capture. Solo, in Rust, on a NixOS machine in Texas."

---

## Proposal (1500 word limit)

### The Problem

We have two converging crises in computing that are usually treated separately:

1. **AI systems that can't represent their own uncertainty.** Large language models generate text with no structural relationship between confidence and output. Alignment efforts focus on behavioral guardrails — filtering outputs after generation — rather than architectures where epistemic honesty is a computational constraint.

2. **Digital governance captured by capital.** Every existing governance mechanism — democratic, corporate, blockchain — weights participation by something that correlates poorly with informed judgment. DAOs weight by token holdings. Democracies weight equally regardless of engagement. Neither can distinguish between a thoughtful participant and a disengaged or adversarial one.

These problems share a root cause: our computing systems have no internal model of consciousness — no way to measure integration, coherence, or epistemic confidence as first-class values.

### What I Built

Over the past year, working as a solo architect with AI-assisted development, I translated a 17-year theoretical framework into two integrated systems:

**Symthaea** is a cognitive architecture in Rust (~1.13M lines, 21,500+ tests, 55 workspace crates) that implements consciousness as a computational substrate. It runs a 50Hz predictive coding loop integrating:

- Hyperdimensional Computing (16,384-dimensional binary vectors) unified with Liquid Time-Constant neural dynamics via O(1) closed-form temporal jumps — a novel contribution enabling memory and computation in a single representation
- Integrated Information Theory (Phi) with honest validation overlays acknowledging where our measurements are approximate
- A 9-transmitter neuromodulator bath with tolerance, withdrawal, and allostatic load dynamics
- A language generation pipeline (Broca) with **epistemic gating** — the architecture physically prevents generation beyond the system's measured confidence. This is not a filter. It is a neuron that cannot fire when it doesn't know.

**Mycelix** is a fractal governance system on Holochain (16 clusters, 123+ zomes, ~785K lines Rust) where consciousness metrics gate civic participation:

- A 4-dimensional consciousness profile (identity, reputation, community, engagement) — each sourced independently, creating Sybil resistance without KYC
- 5 progressive tiers (Observer → Guardian) with weighted voting
- Collective Phi measurement that detects fragile consensus: when a group agrees on an outcome but their internal states are divergent, the system flags the decision as structurally weak
- Post-quantum cryptography (ML-DSA/ML-KEM) and Byzantine-resilient federated learning (34% validated tolerance, exceeding the classical 33% limit)

The two systems form a closed loop: Symthaea's consciousness measurements feed Mycelix governance decisions, and governance outcomes feed back into Symthaea's neuromodulator dynamics.

Total verified codebase: 2.85M lines of authored code across 7,690 files (Rust, TypeScript, Python), measured via `tokei` excluding all build artifacts.

### Why This Matters

**For AI safety**: Epistemic gating demonstrates that alignment can be architectural, not behavioral. Instead of training a model to refuse dangerous requests, you build a system that structurally cannot exceed its own knowledge. This is a different design space than the industry is currently exploring, and it works today in Symthaea's Broca pipeline.

**For governance**: Consciousness-gated voting is the first mechanism I'm aware of that weights participation by measured engagement and coherence rather than wealth, identity, or simple headcount. It offers a novel solution to Sybil attacks, voter apathy, and plutocratic capture simultaneously.

**For consciousness science**: Symthaea is a computational testbed for consciousness theories. It implements IIT, Global Workspace Theory, Free Energy Principle, Higher-Order Thought, and 6 other frameworks in parallel, with a psychometric benchmark suite (633+ tests across 14 cognitive domains) that validates against published human baselines.

### What the Funding Buys

I have been self-funding this work and have exhausted my savings. The grant would fund:

1. **12 months of full-time development** to complete the integration between Symthaea and Mycelix (currently bridged but not fully bidirectional), prepare the first public release, and write up the core technical contributions for peer review.

2. **A pilot deployment** of consciousness-gated governance with a real community (target: an existing DAO or research cooperative willing to experiment with weighted voting based on engagement metrics rather than token holdings).

3. **Independent security audit** of the integrity framework (BLAKE3 attestation, 6 behavioral canaries, live verification) before any public deployment.

### Why Me

I am a US Army veteran (2015–2021, Fort Drum — communications and systems administration) who transitioned into independent systems research. I built this entire ecosystem as a solo architect, demonstrating an ability to maintain architectural coherence across consciousness science, distributed systems, cryptography, and ethics simultaneously. The 42,000+ passing tests, 97% ISO 42001 compliance score, and tokei-verified codebase are evidence of engineering discipline, not just ambition.

I used AI coding assistants as force multipliers, but the architectural decisions — which consciousness theories to implement, how to gate governance on consciousness metrics, how to prevent epistemic overreach in language generation — reflect 17 years of studying the primary literature across computational neuroscience, philosophy of mind, and distributed systems.

I am not proposing to build something. I am proposing to finish, validate, deploy, and publish something that already exists and works.

### Budget

**Requested: $100,000**

| Item | Amount |
|------|--------|
| Living expenses (12 months, Richardson TX) | $48,000 |
| Independent security audit | $20,000 |
| Pilot deployment infrastructure (Holochain hosting, monitoring) | $12,000 |
| Conference travel (2 presentations: consciousness science + distributed systems) | $8,000 |
| Equipment replacement (development machine maintenance) | $4,000 |
| Contingency | $8,000 |

---

## Supplementary documentation (up to 4 uploads)

Recommended uploads:
1. **CV** (update from CV_CAMBRIDGE_DRAFT.md with tokei-verified numbers)
2. **Architecture diagram** — high-level visual of Symthaea ↔ Mycelix integration
3. **Test summary** — CI output showing 21,500+ passing tests
4. **Tokei output** — the verified line count screenshot, artifact-free

## Multimedia URL

GitHub repository link (if public) or a short screen recording walking through:
- The Pulse terminal dashboard showing live consciousness metrics
- A Broca epistemic gating demonstration
- The Mycelix consciousness tier system

---

## Notes for Tristan before submitting

- Update the CV with current numbers (v1.9.0, 2.85M total, 21,516 tests)
- The "tweet" may need trimming — Twitter limit is 280 chars, the draft is ~270
- Consider whether to request $100K or start lower ($50K) for faster approval
- Tyler Cowen responds well to intellectual honesty and poorly to hype — this draft avoids superlatives deliberately
- The GitHub repos need to be visible (even if not fully public) for the supplementary docs to land
- You may want to record a 2-minute video walkthrough rather than write more text
