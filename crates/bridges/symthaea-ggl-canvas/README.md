# symthaea-ggl-canvas

A Rust bridge from **compiled GGL 0.1 geometry** into Symthaea's existing `symthaea_canvas::SceneNode` representation.

This crate is deliberately not a second renderer, scene graph, or GGL source parser. It consumes the versioned output of the GGL compiler and delegates SVG rendering to `symthaea-canvas`.

## Supported mappings

| GGL primitive | Canvas representation | Coordinate rule |
|---|---|---|
| `circle` | `NodeKind::Circle` | negate `cy` |
| `segment` | `NodeKind::Line` | negate both endpoint y coordinates |
| `polygon` | `NodeKind::Polygon { closed: true }` | negate every vertex y coordinate |
| `arc` | `NodeKind::Path` with SVG `A` commands | reflect y and reverse sweep orientation |

Every primitive is wrapped in a Canvas `Group` with a deterministic SHA-256-derived XML-safe ID. `IdentityMap` preserves the GGL semantic ID, source line and path to the Canvas group/leaf without changing Canvas's existing public `SceneNode` type.

## Example

```rust
use symthaea_ggl_canvas::adapt_compiled_scene;
use symthaea_canvas::render_svg;

let output = adapt_compiled_scene(compiled_ggl_json, "my-artwork-v1")?;
let svg = render_svg(&output.scene, 0.5);
// Persist output.identity_map and output.manifest beside the artwork.
```

The output manifest binds the source/program/geometry/scene digests to the generated Canvas scene and identity-map digests. The digest encoding is explicitly versioned as `serde-json-compact-struct-order-v1`; digest values are not assumed to be interchangeable with Python's JSON serialization.

## Safety and limits

- Maximum compiled JSON input: 32 MiB.
- Maximum primitive count: 100,000; the declared count is required and must match.
- Maximum polygon vertices: 256.
- Maximum semantic ID: 256 UTF-8 bytes.
- Units: absent/null or a non-empty string of at most 64 bytes. Unit labels are retained; there is no implicit unit-to-pixel conversion.
- Coordinates and generated arc path endpoints must be finite and representable by Canvas `f32`.
- Zero-length segments, non-positive radii, zero/over-one-revolution arcs, duplicate IDs, unknown primitives, malformed digests and unsafe namespaces are rejected.
- Arc start angles are reduced modulo one revolution before trigonometric evaluation.
- Styles are intentionally not translated in this first geometry adapter; creative profiles remain an independent layer.


## Atelier production adapter

The `atelier` module calls the existing `symthaea_atelier::generate` API and returns a Canvas `SceneNode`. It is not a second generator. Calls use an explicit `StdRng` seed, stable GGL instance ID, and a receipt binding request and scene digests. Atelier's aesthetic score and selection loop are not invoked; human/Creative-QD selection remains with existing owners.

Registered single-family styles and conservative complete-output node ceilings (including the GGL wrapper):

| Atelier style | Output ceiling |
|---|---:|
| LSystem | 3 |
| ParametricCurve | 11 |
| PersistenceTexture | 130 |
| ColorField | 15 |
| ReactionDiffusion | 1,027 |
| StrangeAttractor | 6,002 |
| HofstadterButterfly | 1,593 |
| PersistenceDiagram | 263 |

The all-in-one `Composite` style is not registered because its aggregate compute/output budget has not been independently qualified. The request must declare a node cap at least as large as the selected style's conservative ceiling; the preflight happens before generation.

Additional bounds include an 8,192 scene-unit viewport, max_elements 5,000, max 10,000 visited nodes, depth 128, 256 polygon vertices, 256 gradient stops, 512 KiB per path, bounded scene strings and transforms, and a snapshot check that rejects rather than silently sanitizing/truncating inputs. Negative scale is allowed for reflection, while near-zero and unbounded transforms are rejected.


## Visual-fidelity benchmark update (2026-11-19)

The visual-QD pilot is being extended with a **separate faithfulness diagnostic**, not a single aesthetic leaderboard:

- VectorGym (EMNLP 2026) is the broad multi-task baseline: Sketch2SVG, expert-authored complex SVG edits, Text2SVG and captioning. Its authors report a Qwen3-VL 8B GRPO baseline as open-source SOTA on their tasks, matching GPT-4o in their evaluation. [Paper](https://aclanthology.org/2026.emnlp-main.820/)
- Vector-Bench (July 2026) is the surgical edit/preservation reference: 40 difficult repair tasks, 60.55 protected objects on average, and a reported best endpoint with 15.0% full-specification success and 43.7% mean repair progress. [Paper](https://arxiv.org/abs/2607.19056)
- SVG-Score (September 2026) adds human-aligned semantic evaluation: its project page reports 8,671 SVGs, 1,858 captions and held-out human ratings; the authors show that generic CLIP barely responds to count and spatial-relation errors. [Paper](https://arxiv.org/abs/2609.03806) · [Project/results](https://potpov.github.io/svg-score-webpage/)
- SVGFaith (November 4, 2026) is the newest relevant meta-evaluation: 1,842 controlled contrastive pairs over eight capability axes plus 1,327 human-annotated Arena comparisons. It reports that several embedding metrics are much stronger on object identity (at least 0.789 pairwise accuracy) than on spatial relations / attribute binding (at most 0.672), and proposes dependency-aware requirement scoring. These are author-reported preprint results, not results from GGL. [Paper](https://arxiv.org/abs/2611.06048) · [Code/data](https://github.com/Yukinonooo/SVGFaith)

**GGL's response:** add a requirement-dependency graph whose leaves are atomic visual claims. Each requirement is independently classified as hard-geometric, render-observable, uncertain-perceptual, or human-judged. Hard geometry is checked by deterministic GGL/Canvas relations; rendered semantics use a separately calibrated visual evaluator; aesthetic preference remains human feedback. Parent requirements may depend on prerequisite nodes, but a failed prerequisite makes dependent claims fail or become not-evaluable—it must not silently disappear inside a scalar average. Keep the eight GGL diagnostic axes explicitly versioned and distinct from SVGFaith's axis labels.

Targets should be reported separately: exact relation pass rate, pairwise faithfulness accuracy by axis/direction, spatial/attribute-binding accuracy, false acceptance of hard negatives, protected-object UCR, abstention calibration, human agreement and evaluator cost. Do not use CLIP score as the geometry verifier, and do not use an LLM judge as the hard-contract authority.

## Qualification

`tests/adapter.rs` exercises all four supported primitives, raw source SHA binding, deterministic output, actual Canvas SVG rendering, stable IDs and malformed-input controls. `symthaea-canvas/tests/ggl_adapter_conformance.rs` independently checks the serialized Canvas fixture against the Rust types and renderer.

The code and tests are authored, but a Rust compiler is unavailable in the current authoring environment. A passing `cargo fmt --check`, `cargo test -p symthaea-ggl-canvas -p symthaea-canvas`, and scoped Clippy run must be recorded before the bridge is described as compiled or qualified.
