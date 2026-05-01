# Symthaea Workspace Dependency Map

55 workspace members. Dependency graph is a strict DAG (zero circular dependencies).

## Dependency Tiers

### Tier 0: Zero Internal Dependencies (Foundational)

```
symthaea-types          — canonical enums (Harmony), constants, types
symthaea-neuromodulators — 9-transmitter bath (DA, NE, 5-HT, ACh, GABA, Oxy, Glu, Ade, ECB)
symthaea-consciousness-equation — master C(t) equation facade
serde-core-shim         — serialization compatibility
```

### Tier 1: Core Foundation

```
symthaea-core (202K LOC)
  ├── depends on: types, neuromodulators
  ├── provides: HDC (BinaryHV, ContinuousHV), Phi engine, consciousness metrics
  └── depended on by: 31 crates (hub of the ecosystem)

symthaea-fep
  ├── depends on: (none internal)
  └── provides: Free Energy Principle active inference primitives
```

### Tier 2: Domain Crates (depend on Tier 0-1 only)

```
symthaea-harmonies       — Eight Harmonies (core, types)
symthaea-broca           — CfC-HDC language (core)
symthaea-nix             — NixOS consciousness (core, tree-sitter)
symthaea-psych-bench     — psychometric benchmarks (core, fep, neuromodulators, ssm)
symthaea-memory          — working/episodic/semantic/persistent (core)
symthaea-dream           — counterfactual dream engine (core)
symthaea-exploration     — surprise-driven exploration (core)
symthaea-ssm             — state-space models (core)

Genesis crates (all depend on core + fep):
  symthaea-genomics      — DNA assembly, damage modeling
  symthaea-cell-foundry  — iPSC, IVG, SCNT
  symthaea-ectogenesis   — artificial womb
  symthaea-nurture       — Bowlby attachment (+ neuromodulators)
  symthaea-population    — population genetics (+ types)

Physics & embodiment (all depend on core):
  symthaea-physics       — tokamak plasma encoding
  symthaea-physics-bridge — HDC semantic physics
  symthaea-multirotor    — multirotor FEP control
  symthaea-humanoid      — bipedal DMC benchmark
  symthaea-hal           — hardware abstraction
  symthaea-vehicle       — ground vehicle control

Perception & language:
  symthaea-perception    — multimodal sensor fusion (core)
  symthaea-vision-manifold — patch-based video + temporal (core)
  symthaea-foveation     — active vision (core)
  symthaea-stt           — speech-to-text (core)
  symthaea-vocal-tract   — articulatory synthesis (core)
  symthaea-embeddings    — Qwen3/BGE text + image (core)
  symthaea-narrative-self — narrative self-model (core)

Consciousness:
  symthaea-consciousness-resonance  (core)
  symthaea-consciousness-topology   (core)
  symthaea-causal-reasoning          (core)
  symthaea-factor-graph              (core)
  symthaea-field-dynamics            (core)
  symthaea-hodge                     (core)

Other:
  symthaea-observability   (core)
  symthaea-sentinel        (core)
  symthaea-crucible        (core)
  symthaea-support         (core)
  symthaea-zkproof         (core)
  symthaea-materials       (core)
  symthaea-nuclear-forensics (core)
  symthaea-fabrication-kernel (core)
  symthaea-phi-search      (core)
  symthaea-phi-oracle      (core)
  symthaea-pulse           (core)
  symthaea-spore           (core, consciousness-equation, harmonies)
  symthaea-wisdom          (re-export only)
```

### Tier 3: Main Crate (composition layer)

```
symthaea (main crate, v1.9.0)
  ├── depends on: all Tier 0-2 crates via optional features
  ├── re-exports: pub use symthaea_X as module_name in consciousness/mod.rs
  └── provides: CognitiveLoopService, 8-phase pipeline, public facade
```

## Dependency Fan-Out (most depended upon)

| Crate | # Dependents |
|-------|-------------|
| symthaea-core | 31 |
| symthaea-fep | 9 |
| symthaea-types | 5 |
| symthaea-neuromodulators | 3 |
| symthaea-harmonies | 2 |

## Integration Pattern

All sub-crates integrate via **trait abstraction + re-export**, never via circular callbacks:

1. Sub-crate defines types and logic
2. Main crate re-exports via `pub use symthaea_X as module_name`
3. CognitiveLoopService pulls data one-way from sub-crate instances
4. Feature gates prevent compilation of unused crates

## Adding a New Sub-Crate

1. Create `crates/symthaea-new/` with `Cargo.toml` depending on `symthaea-core`
2. Add `"crates/symthaea-new"` to workspace members in root `Cargo.toml`
3. Add feature flag: `new-feature = ["dep:symthaea-new"]`
4. Add `pub use` in `src/consciousness/mod.rs` (behind `cfg`)
5. Add to CI feature matrix in `.github/workflows/ci.yml`
6. Add proptest coverage for safety-critical functions
