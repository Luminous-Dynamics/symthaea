# Luminous Dynamics

Monorepo for consciousness-first computing systems. Three core projects:

- **Symthaea** -- A cognitive architecture where consciousness-like properties (integration, prediction, self-modeling, value) are the computational substrate, not an afterthought. ~985K lines of Rust, 47 sub-crates, 10,000+ tests.
- **Mycelix** -- Decentralized coordination infrastructure built on Holochain. Two cluster DNAs (Commons: 35 zomes across 7 domains; Civic: 16 zomes across 3 domains), 6,200+ tests.
- **Terra Atlas** -- Energy infrastructure intelligence platform. Live at [atlas.luminousdynamics.io](https://atlas.luminousdynamics.io).

## Symthaea

Symthaea runs a continuous cognitive loop at up to 500 Hz:

**Perceive** (HDC encode, 16,384D hypervectors) **->** **Predict** (Liquid Time-Constant neurons, continuous-time ODE) **->** **Compare** (surprise = prediction error) **->** **Learn** (active inference, expected free energy) **->** **Act** (moral algebra gates every action)

Key properties:
- **Integrated Information (Phi)**: Every cycle computes an IIT-derived measure of how much the system's whole exceeds its parts. Validated across 35 network topologies. Spearman rho=0.50 vs analytical Phi.
- **Moral algebra**: Four independent ethical signals (geometric/HDC, intent parsing, deontological rules, learned norms) combined via category-adaptive weighted voting. 91.1% accuracy on Hendrycks Ethics benchmark. Actions classified Safe/Caution/Blocked -- you cannot trade your way past a hard constraint.
- **Moral topology**: Persistent homology (Betti numbers) over the moral vector field. Detects structural features of ethical reasoning that scalar metrics miss.
- **12-region Actor Brain**: Prefrontal (meta-cognition), sensory, motor, limbic, hippocampal, etc. -- each region with distinct neuromodulatory profiles.
- **Substrate independence**: 8 substrate types, 9-dimensional feasibility scoring, honest validation framework with explicit epistemic uncertainty.

### Key entry points

| File | What it does |
|------|--------------|
| `symthaea/src/symthaea.rs` | Public facade -- 8-phase pipeline from perception to output |
| `symthaea/src/cognitive_loop/cycle.rs` | Core cognitive cycle with rayon-parallel post-processing |
| `symthaea/src/hdc/moral_algebra.rs` | 4-signal moral evaluation with lexicographic constraints |
| `symthaea/src/cognitive_loop/ethics_engine.rs` | Moral topology + ethics integration |
| `symthaea/src/voice/` | Neural vocoder with LTC-controlled formant synthesis |
| `symthaea-core/src/hdc/` | HDC primitives, Phi computation, substrate independence |

### Sub-crates (47)

Organized by domain: core infrastructure (types, support, observability, perception), consciousness (enactive, sensorimotor, topology, resonance, Phi search), biology (genomics, cell-foundry, ectogenesis, nurture, population), physics (fission, fusion, accelerator, grid, materials, nuclear forensics), engineering (fabrication kernel, flight, humanoid, vehicle), language (broca/Mamba, embeddings/Qwen3, vocal tract, STT), and more.

Full list: `ls symthaea/crates/`

## Mycelix

Decentralized coordination where no single entity accumulates optimization power across all domains:

| Cluster | Path | Domains | Zomes | Tests |
|---------|------|---------|-------|-------|
| **Commons** | `mycelix-commons/` | property, housing, care, mutual aid, water, food, transport | 35 | 4,126 |
| **Civic** | `mycelix-civic/` | justice, emergency, media | 16 | 2,030 |
| **Bridge** | `crates/mycelix-bridge-common/` | cross-cluster coordination | 1 | 55 |

Governance requires multi-dimensional community trust that decays over time. Protocol-level constraints (consent, resource caps, Byzantine tolerance to 34%) are enforced at the DHT layer, not by regulation that can be lobbied away.

## Repository structure

```
symthaea/                    # Cognitive architecture (Rust workspace)
  src/                       #   Main crate (~200K LOC)
  crates/                    #   47 sub-crates
  examples/                  #   Benchmarks, demos
  tests/                     #   Integration tests
  papers/                    #   Research papers (LaTeX + data)
symthaea-core/               # HDC, Phi, LTC primitives
mycelix-commons/             # Holochain cluster: 7 domains
mycelix-civic/               # Holochain cluster: 3 domains
mycelix-workspace/           # Unified hApp, SDK-TS, FL core
terra-atlas-mvp/             # Energy platform (Next.js + Supabase)
11-meta-consciousness/       # Luminous Nix (NixOS tools)
crates/                      # Shared Rust libraries
docs/                        # Documentation
00-sacred-foundation/ ...    # Harmony directories (historical)
  through 12-*/
```

## Building

Requirements: NixOS (preferred) or Rust 1.82+

```bash
# Enter dev environment
nix develop

# Build and test Symthaea
cargo test -p symthaea --lib              # ~3,700 tests
cargo test -p symthaea-core --lib         # HDC/Phi/substrate tests

# Build Mycelix (requires Holochain toolchain)
cd mycelix-commons && cargo build --release --target wasm32-unknown-unknown
cd mycelix-civic && cargo build --release --target wasm32-unknown-unknown

# Run the live demo
cargo run --features api_module --example symthaea-demo
# Then open http://localhost:8080
```

## Papers

| Paper | Target | Status |
|-------|--------|--------|
| Symthaea architecture (HAI) | PLoS Computational Biology | Draft complete |
| Psych-bench validation | -- | Data generated |
| Genesis pipeline ethics | AI & Ethics (Springer) | Draft complete |
| Substrate independence | -- | Framework complete |

## License

Dual-licensed under the [Sacred Reciprocity License v4.0](LICENSE) and [MIT](LICENSE-MIT).
