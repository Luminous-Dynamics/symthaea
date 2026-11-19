# GGL / Symthaea integration status — 2026-11-19

**Owner:** [GGL-VISUAL-001 #12859](https://github.com/Luminous-Dynamics/symthaea/issues/12859)  
**Draft PR:** [#12862](https://github.com/Luminous-Dynamics/symthaea/pull/12862)  
**Branch:** `research/ggl-visual-integration-2026-11-18`  
**Inspected head:** `015886af18f1c7cad96242b9340be160908910b0`

## Architecture

GGL is a typed authoring / compiled-geometry / generative-grammar layer over existing owners. It is not a second renderer, scene graph, geometry engine, critic, gallery, QD archive, GPU fabric or evidence ontology.

- Canvas: 2D SceneNode and SVG renderer.
- Atelier: procedural generators and explore/mutate.
- Art Eye: rasterization and pixel-derived percepts.
- Gallery: persistence and opted-in human feedback.
- Symtropy: N-dimensional math and projections.
- #7933: Creative-QD and human-evaluation owner.
- #8919: shared grammar-recovery and falsification methodology.

## Rust adapter authored

New crate: `crates/bridges/symthaea-ggl-canvas`.

It consumes compiled GGL 0.1 JSON (not GGL source text) and maps circle, segment, polygon and circular arc to the existing Canvas SceneNode. It converts y-up to y-down, wraps each primitive in a stable SHA-256-derived Group ID, and emits an identity sidecar and adapter manifest.

Input validation includes a 32 MiB cap, primitive/vertex/ID bounds, required and matching primitive count, finite/f32-representable coordinates, bounded unit metadata, duplicate-ID rejection, unsupported-type rejection, zero-length segment/radius/arc rejection, namespace validation, and digest-shape validation. No implicit unit conversion is performed; style is a separate layer.

## Tests authored (not yet Rust-qualified)

The branch includes:
- original four-primitive GGL source and compiled-scene fixtures;
- Canvas SceneNode JSON, identity sidecar and manifest;
- Canvas-side serde/render test checking geometry conversion and stable SVG IDs;
- bridge tests checking four primitive mappings, source SHA binding, repeat determinism, duplicate/unsupported/invalid inputs and resource limits.

The initial fixture placeholder-digest issue was corrected: the source digest matches the actual GGL program file.

## Verification boundary

- Python GGL reference tests: 95/95 passed locally (2026-11-19, 4.33 s).
- Visual-QD pilot manifest validator: 18 briefs / six families / eight-candidate budget passed structural checks.
- Rust adapter and tests are authored but were not compiled or run locally because this environment has no Cargo/Rust/rustfmt toolchain.
- PR #12862 remains draft. Inspected GitHub checks were skipped, not passed. No exact-head Rust or CI pass is claimed.

Required next gate: exact-head `cargo fmt --check`, `cargo test -p symthaea-ggl-canvas -p symthaea-canvas`, scoped Clippy and review of numeric/arc edge cases.


## Atelier production integration

A second Rust adapter now wraps the existing `symthaea_atelier::generate(config, snapshot, rng)` seam:

- Explicit seed, namespaced stable call-site ID and request/scene digest receipt.
- Reuses the existing Canvas `SceneNode`; it does not call Atelier's aesthetic score or automatic selection.
- Registers single-family Atelier styles with conservative complete-output node ceilings: LSystem 3, ParametricCurve 11, PersistenceTexture 130, ColorField 15, ReactionDiffusion 1,027, StrangeAttractor 6,002, HofstadterButterfly 1,593, PersistenceDiagram 263. Composite remains excluded pending a separately qualified aggregate budget.
- Preflights requested scene-node budget before generation; also caps viewport at 8,192 units, generator elements at 5,000, output nodes at 10,000, path bytes at 512 KiB and scene depth at 128.
- Rejects any snapshot that would require Canvas sanitization/truncation rather than silently repairing it.
- Validates output Canvas geometry/style/transform/resource bounds and wraps the resulting subtree in a stable GGL group.

Rust adapter tests now include deterministic same-seed generation, stable wrapper ID, Canvas SVG rendering, Composite rejection, snapshot sanitation rejection, and a preflight rejection when a ReactionDiffusion request asks for fewer nodes than its registered worst-case bound.

The per-family ceilings were derived from the current source bounds inspected on 2026-11-19: reaction-diffusion's 64×64 grid and 32×32 sample lattice; strange-attractor's 8,000 maximum iterations and integer step-downsampling; Hofstadter q≤13, three phase samples and 1,590 eigenvalues; and Canvas snapshot defaults of 64 persistence pairs / 64 Betti counts. They are conservative source-derived limits, not measured runtime benchmarks.

## Visual QD pilot

- [Protocol](GGL_VISUAL_QD_BENCHMARK_V0.1.md)
- [18-brief metadata manifest](fixtures/ggl-visual-qd-pilot-v0.1.json)

This is metadata/contract only. It does not include SVG/PNG stimuli, reference target programs, licensing clearance, human annotations, or a sequestered holdout. It is not sealed, confirmatory or a basis for SOTA claims.

## External benchmark landscape

- VectorGym (EMNLP 2026): Sketch2SVG, complex human-authored SVG editing, Text2SVG and SVG captioning; authors report Qwen3-VL 8B Gym as open-source SOTA on their benchmark and matching GPT-4o in their evaluation. https://aclanthology.org/2026.emnlp-main.820/
- Vector-Bench (July 2026): 40 SVG surgical-repair tasks; strongest reported endpoint among 34 reaches 15.0% full specification success and 43.7% mean repair progress; includes protected-object and unintended-change measures. https://arxiv.org/abs/2607.19056
- VectorEdits (LREC 2026): over 270,000 SVG/instruction pairs; review license before ingestion. https://aclanthology.org/2026.lrec-1.868/
- SVG-Score (September 2026): warns generic CLIP-like metrics miss vector-specific count, color and spatial errors. https://arxiv.org/abs/2609.03806

GGL's competitive hypothesis is that typed programs and edit transactions improve correctness, preservation, controllability and inverse-grammar recoverability when paired with strong visual models and Atelier—not that deterministic geometry alone creates more beautiful art.

## Next steps

1. Exact-head Rust build/test/Clippy.
2. Register existing Atelier systems as bounded, versioned GGL productions.
3. Extend #7933 with visual task adapters and matched candidate budgets.
4. Materialize rights-cleared stimuli and expert targets.
5. Use #8919's semantic-equivalence and heterogeneous-grammar methodology for held-out program recovery.
6. Conduct blinded, powered artist evaluation before any preference/superiority claim.
