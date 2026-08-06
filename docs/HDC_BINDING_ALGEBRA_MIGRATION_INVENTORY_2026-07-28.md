# HDC Binding Algebra — Workspace Migration Inventory

**Status: inventory only. No caller code changed in this document/commit.**
Phase 3 / Commit D of the HDC Binding Algebra Qualification and Migration
Plan (see `SYMTHAEA_UAL_FROZEN_EVIDENCE_2026-07-27.md`, and `symthaea-core`'s
`docs/BINDING_ALGEBRA_CHARACTERIZATION_REPORT.md` for the measurements this
inventory is judged against).

**Scale reality check, honestly reported per the plan's own warning against
this consuming the whole campaign**: broadening the search from concrete
`.inverse()` call sites to keyword mentions of "unbind"/"self-inverse"/
"preserves similarity" in comments and identifiers surfaces **80+ files**
across `symthaea-core`, `symthaea-causal-reasoning`, `symthaea-psych-bench`,
`symthaea-quantum-comp`, `symthaea-vision-manifold`, `symthaea-morphogenesis`,
and others. Individually reading and judging all of them would consume this
entire work unit and then some — exactly what the plan's risk-tiering exists
to prevent. Tier A below is judged individually and completely. Tier B is
named by module/crate with representative counts, not exhaustively judged —
each is flagged as its own future review, not silently cleared. Tier C is
inventory-only.

## Tier A — inspected individually, complete

All 19 workspace `.inverse()` call sites (the concrete, unambiguous signal
for "this code claims to unbind/recover via `ContinuousHV`'s inverse", as
opposed to noisy keyword-only comment mentions), receiver type resolved by
reading each call site directly, not inferred from the method name alone.

| File:line | Receiver type | Semantic category | Risk | Disposition |
|---|---|---|---|---|
| `symthaea-core/src/hdc/cross_modal_binding.rs:529,535,537,542` | `ContinuousHV` | Multi-strategy cross-modal unbinding (Symmetric/Hierarchical/Asymmetric/Attentional binding types, each with its own unbind strategy; Attentional's own comment already says "unbinding is approximate") | **Highest priority for future dedicated review** — this module already thinks carefully about approximate vs. exact unbinding across 4 strategies; it should be re-read against Commit B's real numbers (esp. the inverse-magnitude-grows-with-dimension finding) before anyone treats its "Symmetric"/"Hierarchical" unbind as exact | Flag: dedicated follow-up review, not fixed here (no evidence yet it's *wrong*, just that its assumptions should be re-checked against real data) |
| `symthaea-core/src/hdc/cantor_pyramid.rs:780,893` | `ContinuousHV` | Role-filler memory retrieval (`ContinuousHV::from_slice(data).bind(&role.inverse())`) inside a hierarchical search/codebook-matching structure | Medium — real production unbinding-based retrieval; dimension used there should be checked against the inverse-instability finding | Flag: check the actual dimension this runs at against Commit B's per-dimension inverse-magnitude table |
| `symthaea-core/src/physics/standard_model.rs:536-537` | `ContinuousHV` | `same_generation()`: unbind a "generation" component from two particle HVs, compare similarity | Low-medium — physics-domain demo/research code, not obviously safety- or correctness-critical for any live system | Inventory + note; no urgent action |
| `symthaea-fep/src/markov_blanket.rs:795` | `ContinuousHV` (explicit type in signature) | Cross-agent coordinate-anchor alignment: unbind foreign anchor, bind with native anchor to get a "coordinate rotation delta", bundle into a composite Translation Operator HV | **High** — this is `symthaea-fep`, a core-tier crate; a cross-agent alignment mechanism relying on `inverse()`'s accuracy is exactly the kind of thing that could be silently degraded by the numerical-instability finding | Flag: dedicated follow-up review, prioritize alongside `cross_modal_binding.rs` |
| `symthaea-psych-bench/src/benchmarks/executive/ravens.rs:343,344,350,351` | `ContinuousHV` | Raven's Progressive Matrices: extracts row/column transformation rules via TRUE inverse-based unbinding (`feat[2].bind(&feat[0].inverse())`), then *applies* the extracted rule to predict a held-out cell | **Notable positive precedent, not a risk** — this benchmark already does, with proper `inverse()`-based unbinding, structurally the same "extract a compositional rule, apply it to a novel case" pattern P4a was attempting with the double-bind heuristic. Directly relevant empirical comparison for the eventual P4a redesign (Phase 5, separate work unit) | No fix needed; **cross-reference from the future P4a redesign work unit** — check whether Ravens' proper-unbinding approach actually predicts well, as a real before/after case study |
| `symthaea-vision-manifold/src/encoder.rs:424` | `ContinuousHV` | Position-invariant template matching (`unbind_position`): the doc comment **explicitly cites `ContinuousHV::inverse`'s own doc** as justification for correctness | **High** — a direct, explicit dependency on the exact claim Commit C just corrected; this consumer's own confidence was built on the pre-correction doc | Flag: dedicated follow-up review — re-verify this still works acceptably given the real (not idealized) inverse-based recovery quality (~0.92 mean similarity, not exact) |
| `symthaea-morphogenesis/src/morpho_topology.rs:262` | `ContinuousHV` | Estimate a tissue's membrane-voltage state by unbinding a spatial coordinate from a composite tissue HV, optionally cleaned up via an associative memory | Low-medium — research/simulation code | Inventory + note; no urgent action |
| `symthaea-morphogenesis/examples/planarian_tas_sim.rs` (example, not library) | `ContinuousHV` | Example/demo construction, not a load-bearing library consumer | Low (example code) | No action |
| `symthaea-core/src/hdc/unified_hv.rs:608,1185,1187,1209` | `ContinuousHV` | The primitive's own doc comment + internal unit tests for `inverse()` itself | N/A — this is the definition, already corrected in Commit C and covered by Commit B's hard contract tests | Closed by Commits B/C |
| `symthaea-core/src/hdc/binding_algebra_audit.rs`, `examples/binding_algebra_characterization.rs`, `symthaea-psych-bench/.../ual/hdc_binding_properties.rs` (2 sites) | `ContinuousHV` | This audit's own code | N/A | N/A |
| `symthaea-core/src/hdc/complex_analysis.rs:548`, `src/physics/physics_numerical_validation.rs:2540`, `src/hdc/riemannian_geometry.rs:226,623,642`, `src/hdc/linear_algebra.rs:2191,2211,2670`, `src/physics/tensor_algebra.rs:282,312,494,509,788` | **NOT `ContinuousHV`** — matrix/tensor/metric inverses in unrelated linear-algebra and general-relativity physics code | Out of scope entirely | None | Confirmed out of scope by reading each site; excluded from this audit |
| `symthaea-chronicle/src/allen.rs`, `symthaea-group-theory/src/permutation.rs` | **NOT `ContinuousHV`** — Allen's interval-algebra relation inverse; group-theoretic permutation inverse | Out of scope entirely | None | Confirmed out of scope |

**Bind-then-bind-again (double-bind self-squared) pattern search**: beyond
UAL's own P2/P4a (already fixed), no other clean instances of this specific
pattern were found in the Tier A sites above — every other located
`.inverse()`-based consumer already uses proper inverse-based unbinding, not
the double-bind heuristic. This is a genuinely reassuring finding: UAL's P2/
P4a appear to have been the outlier, not the workspace norm.

**Persisted/serialized hypervector risk**: `ContinuousHV` derives
`Serialize`/`Deserialize`, but Commits B/C made **no behavioral change** to
`random`/`bind`/`inverse`/`normalize` — only documentation and new
measurement code. There is therefore no serialization-format risk from this
work unit specifically. This becomes relevant again only if a future Commit
E (or the deferred Phase 4/5/6 work) changes actual computation.

## Tier B — named by module/crate, NOT individually judged (flagged for future review)

Grep-surfaced via broader "unbind"/"self-inverse"/"preserves similarity"
keyword matches (80+ files total). Grouped by module/crate with rough counts
and a one-line characterization; **none of these have been read in detail
for this inventory** — each is a candidate for its own future, appropriately-
scoped review, not cleared or condemned here:

| Module/crate | Approx. file count | Likely category | Recommended future disposition |
|---|---|---|---|
| `symthaea-core/src/hdc/` (encoders: `sequence_encoder`, `grid_encoder`, `binary_grid_encoder`, `text_encoder`, `hierarchical_bundle`, `cantor_recursive_hv`, `resonator`, `sdm`, `semantic_decoder`, `semantic_bridge`, `binding_problem`, `ucl_cross_domain_frames`, `primitive_system/*`, etc.) | ~25 | Representation/encoding infrastructure — likely mostly bind/bundle for representation, not necessarily unbinding-dependent | Own audit pass: distinguish "uses bind for representation only" (low risk) from "relies on unbind for retrieval correctness" (needs the same scrutiny as Tier A) |
| `symthaea-causal-reasoning/src/counterfactual/*` (`hdc_surgery.rs`, `semantic_roles.rs`, `composer.rs`, `mod.rs`) | 4 | Counterfactual "surgery" on causal structure via HDC role/filler manipulation — name alone suggests genuine unbinding-dependent semantics | Priority follow-up — "surgery" implies precise manipulation, which is exactly where approximate unbinding could silently produce wrong counterfactuals |
| `symthaea-psych-bench/src/benchmarks/{reasoning/arc_*, binding/*, spatial/*, causal_reasoning/*, institutional_reasoning/*}` | ~20 | Other cognitive benchmarks structurally similar to Ravens/UAL — ARC reasoning, feature conjunction, temporal order, landmark binding, causal chains | Priority follow-up, same rationale as Ravens: these may have the identical "does this benchmark's HDC mechanism measure what it claims" question UAL P2/P4a had, just never audited |
| `symthaea-quantum-comp/*` | ~9 | Already the subject of an independent, closed 5-probe audit (per MASTER_ROADMAP: "no general phase/quantum-inspired-encoding advantage found") | Likely low incremental risk — that audit already calibrated noise-model and encoding comparisons carefully; a light re-check against this audit's findings is still worthwhile but not urgent |
| `symthaea-vision-manifold/src/{manifold,predictive}.rs` (encoder.rs already in Tier A) | 2 | Likely related to the Tier A `encoder.rs` finding | Bundle with the `encoder.rs` follow-up |
| `symthaea-perception/src/multi_modal.rs`, `symthaea-geodesic/src/{manifold,topology}.rs`, `symthaea-cell-foundry/src/cell_encoder.rs`, `symthaea-embeddings/src/lib.rs`, `symthaea-logparse/src/{fixtures,encoder}.rs`, `symthaea-broca/src/decoder.rs`, `symthaea-phone-embodiment/src/bridge.rs`, `symthaea-atelier/src/art_protocol.rs` | ~9 | Varied domain crates, one or two mentions each | Lowest priority within Tier B; inventory now, revisit if any of the higher-priority reviews above surface a systemic pattern worth checking everywhere |

## Tier C — inventory only

`ContinuousHV::random` call sites not captured above (351 total minus the
Tier A/B sites already named): the large remaining majority. Per the plan,
these get no individual disposition unless Tier A/B evidence surfaces a
systemic risk pattern that would warrant grep-checking for it workspace-wide
(e.g., if the `cross_modal_binding.rs` or `causal-reasoning` follow-ups find
a real bug, re-grep Tier C for the same anti-pattern specifically, not for
general risk).

## What this inventory does NOT do

- Does not modify any of the flagged files.
- Does not conclude any Tier A/B site is broken — only that several (marked
  "High"/"Highest priority") make assumptions worth re-verifying against
  Commit B's real measurements, which this document flags but does not
  itself perform.
- Does not attempt Tier B's "inspect by representative pattern" exhaustively
  — given the scale (80+ files), doing so honestly would require its own
  dedicated work unit(s), which this document explicitly recommends rather
  than simulates.

## Recommended next steps (not started here)

In priority order, based on the "High"/"Highest priority" Tier A flags:
1. `symthaea-fep/src/markov_blanket.rs` — core-tier crate, cross-agent
   correctness.
2. `symthaea-core/src/hdc/cross_modal_binding.rs` — most sophisticated
   existing unbinding module, re-check its 4 strategies against real data.
3. `symthaea-vision-manifold/src/encoder.rs` — explicitly cited the
   pre-correction doc claim as justification.
4. `symthaea-causal-reasoning/src/counterfactual/*` — "surgery" semantics
   are precision-sensitive by name.
5. The `symthaea-psych-bench` reasoning/binding benchmark cluster — same
   audit pattern that found UAL's P2/P4a defects, not yet applied elsewhere.

Per the plan's Phase 4 scope: only items where this inventory (or the above
follow-ups) find a **demonstrated** dependency on the false contract
warrant a Commit E fix. Nothing here yet meets that bar — this document
identifies where to look, not confirmed defects.
