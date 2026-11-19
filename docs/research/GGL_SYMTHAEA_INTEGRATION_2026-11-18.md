# GGL × Symthaea × Symtropy integration decision

**Status:** research / design record, 2026-11-18  
**Owner work item:** [GGL-VISUAL-001 #12859](https://github.com/Luminous-Dynamics/symthaea/issues/12859)

## Decision

GGL is a typed authoring language, generative-grammar IR, inverse-program hypothesis format, and constrained-edit plan format. It is **not** a second renderer, scene graph, geometry kernel, perceptual critic, gallery, GPU fabric, or provenance ontology.

GGL compiles into existing owner-native systems. Each existing system remains authoritative for its current responsibility.

## Existing owners to compose

| Existing owner | Current responsibility | GGL boundary |
|---|---|---|
| `symthaea-canvas` | 2D `SceneNode`, `NodeKind`, 2D transforms, SVG renderer, scene features and bounded validation | Primary 2D compiler target. Do not fork its scene graph or renderer. |
| `symthaea-atelier` | L-systems, parametric curves, color fields, reaction-diffusion, strange attractors, persistence/Hofstadter art and explore/mutate loop | Register bounded, versioned generator adapters as GGL productions; do not copy generators. |
| `symthaea-art-eye` | SVG rasterization via resvg and perceptual features from actual rendered pixels | Render/percept adapter. Current static rasterization ignores CSS keyframe animation; creative-surprise remains neutral until memory-backed integration. |
| `symthaea-gallery` | Artwork persistence, curation, ratings/history and 16D style embedding | Artifact/feedback sink. Existing score-derived style embedding is not a universal human-preference model. |
| `symthaea-vision-manifold` | Visual perception and HDC representations | Candidate retrieval and hypotheses only; similarity is not geometric proof. |
| `symtropy-math` | N-dimensional points, bivectors, rotors and transforms | Higher-dimensional semantics through an adapter; do not fork the math kernel. |
| `symtropy-render-bridge` | 2D/3D/4D projection/render integration | Optional higher-dimensional backend; not a second renderer. |
| `symthaea-projection::grammar` | Semantic evidence/state visualization roles (color/line/opacity/motion) | Keep separate. This is not a generative art grammar or universal aesthetic vocabulary. |
| [GPU Fabric #6968](https://github.com/Luminous-Dynamics/symthaea/issues/6968) | Backend-neutral GPU operations and existing Canvas WebGPU direction | Reuse if needed; no GGL-specific GPU fabric. |
| [Creative QD #7933](https://github.com/Luminous-Dynamics/symthaea/issues/7933) | Intent contracts, quality/diversity descriptors and human evaluation | Add visual-art tasks here; keep dimensions separate. |
| [SCI-GRAMMAR #8919](https://github.com/Luminous-Dynamics/symthaea/issues/8919) | Cross-world grammar recovery and falsification methods | Reuse equivalence classes, held-out compositions, heterogeneous controls and abstention; do not duplicate its evaluator. |
| [Constructed glyph notation #9694](https://github.com/Luminous-Dynamics/symthaea/issues/9694) | Visual semantic notation / script design | Related but distinct: a glyph notation encodes structured meaning; GGL describes executable geometry/creative programs. |

## Stable identity without breaking Canvas

Current Canvas `SceneNode` primitives do not carry semantic IDs. `NodeKind::Group` has an optional ID, while Circle/Ellipse/Line/Polygon/Rect/Path do not.

Initial adapter decision:

1. Wrap each compiled primitive in a `Group { id: Some(stable_id) }`.
2. Emit an identity sidecar mapping GGL semantic ID → Canvas wrapper ID → source span/program node.
3. Emit a manifest binding source/program digest, canonical geometry digest, Canvas-scene digest, adapter version, unit/frame metadata and ID-map digest.

This avoids a breaking change to Canvas. Before production, a Rust-side conformance test must deserialize the emitted scene JSON into the actual Rust type, render it through `symthaea_canvas::render_svg`, and verify wrapper IDs survive the SVG output and rasterization.

**Numeric caveat:** Canvas's SVG renderer formats many geometry values to one decimal place. That output is not a lossless geometry serialization. Exact geometry round-trip must be checked against the canonical scene representation; SVG pixel/render differences are a separate metric under a pinned renderer and tolerance.

## GGL semantic layers

1. **Source language:** versioned syntax, source spans, imports and explicit deterministic seeds.
2. **Typed AST/IR:** dimensions/units, coordinate frames, 2D/3D/4D transforms, primitive/curve types, constraints and bounded productions.
3. **Generative program:** motif definitions, composition, repeat, transform-group actions, bounded recursion, registered Atelier adapters and seeded stochastic operators.
4. **Scene compilation:** target adapters (initially Canvas 2D; later Symtropy 3D/4D where supported), stable identity sidecar and explicit unsupported-feature handling.
5. **Edit transactions:** expected source/program/scene digests, target IDs, protected IDs, selection cardinality, pre/postconditions, reversible diff and receipt.
6. **Inverse hypotheses:** candidate grammar sets/equivalence classes, residuals, uncertainty and abstention. An inferred program is not automatically the original artist's construction process.

Hard mathematical invariants, soft composition objectives and human preference observations must be different types. Aesthetic scores must not silently become hard validity constraints.

## Evidence boundary

- Observed pixels, vector source, inferred primitive, generated geometry, rendered percept and human rating are separate evidence types.
- A relation may be exact for generated geometry and uncertain for a photograph-derived observation.
- HDC/vision similarity can retrieve candidate motifs; it cannot prove equality, symmetry, provenance or semantic meaning.
- Art Eye's current documented limits (static raster; neutral creative-surprise without memory) must remain visible in downstream claims.
- Gallery's style embedding is a learned engineering descriptor, not a universal theory of taste.
- No message or semantic payload should be inferred from geometric resemblance alone.

## Local reference adapter

A research adapter is included in the GGL starter package at `integrations/symthaea_canvas_adapter.py`. It maps the current 56-primitive Resonant Garden program into the serde-shaped Canvas scene form and emits an identity sidecar/manifest. Thirteen Python contract tests pass. Initial supported mappings are circles, segments, polygons and circular arcs (arcs are represented as SVG paths); units/frame metadata are preserved; duplicate IDs, non-finite values, invalid radii and unsupported constructs are rejected.

**Qualification ceiling:** the current environment has no Rust/Cargo toolchain. These tests do not establish Rust deserialization, Canvas rendering, SVG ID survival or CI qualification.

## Benchmark plan

### GGL-CANVAS-ROUNDTRIP
Measure supported construct coverage, canonical geometry agreement, identity-map preservation, serde agreement, rendered SVG validity and unsupported/opaque counts. Keep geometry error distinct from raster pixel difference. Pin toolchain, renderer, adapter and target.

### GGL-GRAMMAR-RECOVERY
Use known-source synthetic programs across at least eight families (repeat, rotation, reflection, translation, nested scale, L-system, bounded recursion and mixed/heterogeneous systems). Hold out production compositions and depth. Score evaluator-defined semantic equivalence classes, not source strings. Include non-identifiable pairs and no-common-grammar controls. Report recovery, MDL, residual, overfit, false-shared-grammar, abstention and cost.

### GGL-EDIT-PRESERVATION
Adapt Vector-Bench/VectorGym tasks only where licensing and task semantics permit. Report full specification success, protected-object identity, valid-output UCR, structural/render preservation, abstention and latency. Match model/tool/candidate budgets; synthetic transaction success is not comparable to arbitrary-SVG model success.

### GGL-VISUAL-QD (extension of #7933)
Add visual briefs for geometric/ornamental, icon/poster, editorial illustration, organic/branching, abstract composition and ambiguous/adversarial instructions. Compare current Atelier, disclosed best-of-N, intent filtering, GGL-conditioned generation and QD archive at matched budgets. Separate intent adherence, structural correctness, coherence, novelty, diversity, controllability and human preference. Use Art Eye for pixel measurements and blinded human/artist studies for preference. Do not use the Birkhoff-inspired Canvas score as ground-truth beauty.

## External comparison set

- [VectorGym (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.820/): Sketch2SVG, editing, Text2SVG and captioning; includes human-authored tasks and a Qwen3-VL 8B multi-task baseline.
- [VectorEdits (LREC 2026)](https://aclanthology.org/2026.lrec-1.868/): 270k+ instruction/SVG edit pairs.
- [Vector-Bench](https://arxiv.org/abs/2607.19056): 40 difficult surgical SVG repair tasks with protected-object/UCR evaluation; published best full-specification success was 15% on its evaluated endpoints.
- [SVG-Score](https://arxiv.org/abs/2609.03806): human-preference/semantic evaluation; generic CLIP can miss structural errors such as changed counts or spatial relations.
- [SVG leave-one-out structural evaluation](https://arxiv.org/abs/2604.08809): already studies per-element contribution plus purity, coverage, compactness and locality over >19k edits. GGL's current changed-pixel influence score is a prototype; adapt this prior work rather than claiming LOO novelty.
- [Rapidata Text-to-SVG human feedback](https://huggingface.co/datasets/Rapidata/text-to-svg-human-feedback-benchmark): multi-million human comparisons; dataset card states CC-BY-4.0.
- [DuetSVG (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_DuetSVG_Unified_Multimodal_SVG_Generation_with_Internal_Visual_Guidance_CVPR_2026_paper.pdf): image-to-SVG reconstruction baseline; compare only on matched task/data/protocol.

Published scores are context, not matched GGL results. No superiority claim without frozen tasks, evaluator, rendering environment, candidate budget, baseline version and paired uncertainty estimates.

## Provisional targets (not results)

- 100% of supported synthetic constructs round-trip within declared tolerances.
- 100% of invalid/unsupported fixtures reject or remain opaque; zero silent drops.
- 100% deterministic output for identical source, seed, compiler and renderer versions.
- 100% protected-ID preservation in structured transaction fixtures.
- ≥90% held-out synthetic grammar recovery only after independent evaluator qualification.
- External benchmark scores reported by task, not one aggregate.
- Human-preference superiority only after preregistered power analysis, matched candidate budget and independent replication.

## Next phases

1. Freeze AST/IR and Canvas adapter conformance contract.
2. Implement a small Rust parser/validator/compiler slice targeting Canvas; no new renderer.
3. Register selected Atelier generators as bounded GGL productions.
4. Wire Art Eye and Gallery as perception/feedback adapters, keeping machine features separate from preference.
5. Extend #7933 with visual QD; reuse #8919's grammar-recovery methodology.
6. Add Symtropy higher-dimensional adapters after 2D qualification.



## Implementation update — 2026-11-19

A bounded Rust adapter crate now exists at `crates/bridges/symthaea-ggl-canvas`. It consumes compiled GGL 0.1 JSON (it is not a source parser) and maps circles, segments, polygons and circular arcs into the existing Canvas `SceneNode`. It creates deterministic Group wrapper IDs and an identity sidecar; no Canvas public type or renderer was forked.

The adapter rejects unsupported primitive kinds, duplicate semantic IDs, invalid/non-finite/out-of-`f32` geometry, zero-length segments, invalid arcs, malformed digests, malformed units, mismatched/missing primitive counts, unsafe namespaces and input over 32 MiB. It preserves units as metadata and performs no silent unit-to-pixel conversion. Its manifest states the digest-encoding version explicitly.

The branch also contains Rust tests for the raw GGL source SHA, all four primitive mappings, Canvas SVG rendering and stable-ID survival, deterministic repeated adaptation, duplicate IDs, unsupported types, invalid arcs, malformed count/units, namespace errors, non-finite numbers, large finite arc-angle normalization and oversized input. A second Canvas-side test exercises the actual serde/render contract against the frozen four-primitive fixture.

**Execution boundary:** these Rust source files and tests are authored, but no Cargo/Rust compiler is available in the current authoring environment. They have not been compiled or executed locally. The draft PR's GitHub checks are skipped, not passed. The earlier 95/95 Python reference tests do not qualify the Rust implementation. Exact-head `cargo fmt --check`, `cargo test -p symthaea-ggl-canvas -p symthaea-canvas` and scoped Clippy remain outstanding.

A visual QD pilot has also been defined under #7933:
- `docs/research/GGL_VISUAL_QD_BENCHMARK_V0.1.md`
- `docs/research/fixtures/ggl-visual-qd-pilot-v0.1.json`

The manifest has 18 brief/contract records across six families and fixes a candidate budget of eight per arm for later matched-budget comparisons. It is explicitly **not** an executable or confirmatory corpus yet: SVG/PNG stimuli, target programs, licensing clearance and human annotations are not materialized. No model scores or artist-preference results are claimed.

The updated local Python reference suite remains 95/95 passing (4.19 s). This is separate from the unexecuted Rust qualification.

## Claim ceiling

GGL's Python interpreter, Canvas reference adapter, and Rust bridge are research-stage. The Rust bridge is written but uncompiled/unexecuted here; skipped checks are not passes. The 18-entry visual QD manifest is a brief metadata pilot without art stimuli or human labels. No SOTA superiority, human-preference improvement, universal theory of beauty, historical authoring recovery, or real crop-formation decoding claim is made.