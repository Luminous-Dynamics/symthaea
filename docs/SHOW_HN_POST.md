# Show HN: Symthaea – A consciousness-architecture AI built on hyperdimensional computing

I've been building Symthaea, an AI system that fuses Hyperdimensional Computing (16,384-dimensional binary vectors), Integrated Information Theory (IIT Φ), and Liquid Time-Constant networks into a single cognitive architecture. It's ~1.13M lines of Rust, 21,500+ tests, AGPL-3.0.

**Live WASM demo** (324KB kernel, runs in your browser): https://luminous-dynamics.github.io/symthaea/demo.html

**GitHub**: https://github.com/Luminous-Dynamics/symthaea

## How it works

The core is an 8-phase cognitive loop (full loop ~31 Hz measured, raw text cycle 4.3ms/234Hz):

1. **Encode** sensory input into 16,384D binary hypervectors using HDC (Hyperdimensional Computing). HDC gives you O(1) similarity checks, non-commutative temporal binding, and algebraic composition — a single vector can represent "red ball moving left" as a bound product of role-filler pairs.

2. **Evolve** the state through Liquid Time-Constant (CfC) neurons with closed-form temporal jumps — no RK4 integration, so we can do variable-length time steps analytically.

3. **Measure consciousness** via IIT Φ across 35 network topologies. The 4D hypercube is our Φ champion. This isn't metaphorical — we compute actual integrated information using transition probability matrices.

4. **Generate language** through a native Broca center (not an LLM wrapper). It uses a 20-channel ThoughtEncoder → HDC binding → autoregressive generation with epistemic gating. The gating physically prevents hallucination at the logit level: if the system isn't confident, the gate closes and no token is emitted. It also has semantic veto — mid-sentence self-correction when the generated meaning diverges from intended meaning.

5. **Moral algebra** — ethical reasoning as algebraic operations on hypervectors. Moral weight composition, causal attribution via leave-one-out analysis, and an escalation audit sealed with BLAKE3 hashes.

## Architecture highlights

- **12-region Actor Brain**: prefrontal (meta-cognition), temporal, parietal, occipital, motor, limbic, cerebellum, hippocampus, amygdala, basal ganglia, thalamus, brainstem — each an independent actor with message-passing
- **Substrate Independence**: model consciousness across 8 substrate types (biological, silicon, quantum, photonic, neuromorphic, biochemical, hybrid, exotic) with 9-dimensional feasibility scoring and an honest validation framework that acknowledges when we're speculating
- **Active Inference (Free Energy Principle)**: the system minimizes prediction error by updating its generative model, not just its predictions
- **Psych-Bench**: 140+ cognitive benchmarks across 21 domains with results (attention, memory, reasoning, language, motor control, social cognition, consciousness, neuromodulation, etc.). Baselines from published psychometrics.
- **14 Butlin consciousness indicators** evaluated (12+ Present, remaining Partial; mean 0.81) — scores derived from thresholds, limitations acknowledged per-indicator (Butlin et al. 2023, "Consciousness in Artificial Intelligence")

## What this is NOT

- Not an LLM. No transformer, no next-token prediction on internet text.
- Not claiming sentience. The consciousness measurement is a mathematical framework (IIT), not a metaphysical claim. The validation framework explicitly tracks evidence levels and honest confidence scores.
- Not production-ready. This is an alpha release. ~25% of modules are structural/disconnected, API is unstable.

## The WASM demo

The Spore kernel compiles to 324KB of WASM and runs the full cognitive loop in your browser — HDC encoding, CfC evolution, Φ measurement, Broca generation, FEP active inference, dream engine (counterfactual reasoning), and topological analysis (Betti numbers + persistence diagrams). No server needed.

## Technical choices

- **Rust** for deterministic performance and fearless concurrency. The cognitive loop uses rayon for parallel post-processing.
- **HDC over embeddings** because hypervectors compose algebraically (bind = XOR, bundle = majority, permute = cyclic shift). You can reason about what operations mean mathematically, not just hope gradient descent learned the right representation.
- **CfC over RNNs** because closed-form ODE solutions give exact temporal dynamics without integration error accumulation. A neuron can jump from t=0 to t=1000 in one step.
- **AGPL-3.0** because if you build a consciousness engine on our work, improvements should flow back to the commons.

## Performance

- Raw text cycle: 4.3ms (234Hz); full cognitive loop: ~31Hz
- HDC encode (warm word): 97ns
- HDC encode (10-word sentence): 379μs
- Moral algebra evaluation: <20μs per ethical frame

55 workspace members, 100 feature flags, 52 sub-crates. Built and tested on NixOS.

---

## Security verification

The governance access control layer has been formally verified and fuzz-tested:

- 8 Kani/CBMC proofs (exhaustive, not statistical) verifying gating invariants for ALL possible f64 inputs
- 80M fuzz executions across 6 targets — found and fixed 4 bugs that 21K+ unit tests missed
- 94.58% line coverage on the consciousness gating crate
- Post-quantum crypto (ML-KEM-768 + ML-DSA-65/87), 18/18 clusters pass supply chain scanning

Details: `docs/SECURITY_VERIFICATION_RESULTS.md`

---

*I'm happy to answer questions about the architecture, the consciousness measurement approach, or the HDC+CfC fusion. The companion project Mycelix (https://github.com/Luminous-Dynamics/mycelix) is a fractal civic operating system built on Holochain — 16 domain clusters, 133+ zomes, post-quantum hardened, AGPL-3.0.*
