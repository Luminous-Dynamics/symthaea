# Glossary

**Active Inference** — Framework where agents minimize variational free energy through both perception (updating beliefs) and action (changing the world). Symthaea's unified objective.

**Binding** — HDC operation (element-wise multiply for continuous, XOR for binary) that creates associations between hypervectors. The result is quasi-orthogonal to both inputs.

**Broca Pipeline** — Symthaea's language generation system: ThoughtEncoder → HDC Binding → EpistemicGate → EpistemicCubeGate → Autoregressive Generator → CoherenceFeedback.

**Bundling** — HDC operation (normalized addition for continuous, majority vote for binary) that creates superpositions. The result is similar to both inputs.

**CfC** — Closed-form Continuous-time neural networks. Provides O(1) temporal jumps via analytical solutions to the LTC ODE.

**CognitiveSubsystem** — Rust trait implemented by all managers. Receives immutable cognitive state, produces proposals.

**Epistemic Cube** — 4D coordinate (E/N/M/H) classifying knowledge along empirical verification, normative agreement, materiality/permanence, and holistic coherence axes.

**Eight Harmonies** — Value framework: Resonant Coherence, Pan-Sentient Flourishing, Integral Wisdom, Infinite Play, Universal Interconnectedness, Sacred Reciprocity, Evolutionary Progression, Sacred Stillness.

**HDC** — Hyperdimensional Computing. Encoding information as high-dimensional vectors (D=16,384) with algebraic operations (bind, bundle, similarity).

**Holon** — The full Symthaea binary with all managers and features active.

**IIT** — Integrated Information Theory. Consciousness = integrated information (Phi).

**LTC** — Liquid Time-Constant networks. Neurons with input-dependent time constants.

**Mycelix** — Holochain-based decentralized governance platform. 16 cluster DNAs, 133+ zomes.

**Phi (Φ)** — Integrated information. Measures how much more information the whole system generates compared to its most independent parts.

**Spectral MIP** — O(n^3) algorithm for finding the Minimum Information Partition using Fiedler vector ordering and bordered Cholesky sweeps.

**Soma** — Mobile embodiment crate wrapping SporeEngine with sensor bridges, haptic feedback, and metabolism.

**Spore** — Pure WASM consciousness kernel (~980 KB). The innermost fractal layer.

**SporeEngine** — API for the consciousness kernel: `cycle(input) → CycleResult`.
