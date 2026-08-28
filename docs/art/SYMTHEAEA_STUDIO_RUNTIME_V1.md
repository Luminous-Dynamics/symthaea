# Symthaea Studio Runtime v1

Status: implementation tranche 1

## Goal

Symthaea should not have a separate art generator for every application. She should have one artistic cognition/runtime that can inhabit multiple creative environments.

```text
Symthaea artistic cognition
        |
        v
Host-neutral Creative World API
        |
   +----+---------+---------+
   |              |         |
 Canvas          Bevy    Blender
 sketchbook   embodied    atelier
              studio
```

The host-neutral layer owns semantic actions, authority, exact revision binding, counterfactual branches, and causal receipts. Hosts own their native scene graphs, renderers, physics, tools, and file formats.

## Existing foundations

The repository already has `symthaea-atelier`, `symthaea-canvas`, Vision Manifold, and a Bevy dashboard. The sibling Symtropy repository has `symthaea-bevy-brain`, Symthaea/Symtropy embodiment adapters, Bevy scene/physics bridges, and a Bevy 0.19 application stack. Studio Runtime v1 is therefore an integration architecture, not a replacement for those systems.

## Tranche 1 ownership

### `symthaea-art-world`

Host-neutral protocol and invariants:

- read-only snapshots bound to exact revisions;
- host-advertised affordances;
- semantic artistic operations rather than raw API strings;
- `Observe`, `Propose`, and `Author` authority modes;
- first-class abstention;
- reversible counterfactual branches;
- explicit commit permits;
- append-only host receipts.

This crate must never contain a universal beauty/reward function.

### `symthaea-blender-bridge`

Transport-neutral Blender protocol:

- versioned handshake and capability discovery;
- observation and affordance requests;
- proposal/preview/commit/reject lifecycle;
- newline-delimited JSON framing;
- a closed symbolic operation-key mapping;
- no arbitrary Python, `eval`, shell, or script payload channel.

A later thin Blender add-on should implement these symbolic operations with `bpy` while keeping the Rust side authoritative for artistic reasoning and provenance.

## Authority model

`Observe` permits perception and critique only.

`Propose` permits reversible candidate branches but requires explicit acceptance before a committed mutation.

`Author` permits autonomous mutation only inside an explicitly designated workspace. It does not remove provenance: every committed operation still needs a permit record and commit receipt.

Authority is orthogonal to artistic quality. Accept/reject is a decision record, not a universal preference label.

## Counterfactual rule

Every nontrivial host should support this shape:

```text
committed revision R
    |
    +-- proposal A -> preview only
    +-- proposal B -> preview only
    +-- proposal C -> preview only
    +-- abstain

Only a separately authorized proposal may advance R -> R+1.
```

A preview must never mutate the committed base revision.

## Bevy direction

Bevy should become the embodied/living studio rather than merely another renderer. The first integration should attach a typed art port beside `CognitiveBrain`, preserving semantic scene observations and proposals instead of collapsing them into one generic motor vector.

Follow-on Bevy stages:

1. typed `ArtPerceptionFrame` and `ArtActionProposal` ECS components;
2. read-only scene observation and capability discovery;
3. proposal ghosts/temporary entities;
4. revision-bound accept/reject flow;
5. Vision-Manifold re-observation of counterfactual renders;
6. physical tools/materials for motor learning;
7. accelerated practice worlds for technique acquisition.

The Symtropy-side v1 port is intentionally shadow/proposal-only. It must not change existing game or physics behavior.

## Blender direction

Blender should become the professional digital atelier:

- mesh and sculpt;
- materials/shader graphs;
- cameras and lighting;
- Grease Pencil;
- geometry nodes;
- animation and compositing;
- high-quality counterfactual renders.

The Blender add-on should be thin. Symthaea should reason in semantic actions such as `Deform`, `ApplyMaterial`, `MoveCamera`, or `CreateStroke`; the add-on maps those into whitelisted Blender operations.

## Artistic-development direction

Studio hosts are embodiments of one developing artist. They should eventually share:

- artistic questions and intentions;
- visual/perceptual memory;
- technique memory;
- portfolio history;
- unresolved problems;
- artistic identity;
- internal-world developmental evidence.

Do not directly map internal variables to visual style (for example, uncertainty -> blue). Internal dynamics should change what artistic problems attract attention, not act as a decorative color lookup table.

## Qualification gates

Tranche 1 is not considered qualified until all of the following pass on the target Rust toolchain:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-art-world --all-targets
cargo test -p symthaea-art-world
cargo clippy -p symthaea-art-world --all-targets -- -D warnings
cargo check -p symthaea-blender-bridge --all-targets
cargo test -p symthaea-blender-bridge
cargo clippy -p symthaea-blender-bridge --all-targets -- -D warnings
```

In addition, conformance tests must establish:

- Observe cannot propose or commit;
- Propose cannot commit without explicit acceptance;
- Author commits still carry an explicit permit record;
- stale-revision proposals cannot be committed;
- preview does not advance the committed revision;
- no protocol field can carry executable host code;
- abstention is representable and preserved in receipts;
- host capability claims are versioned and testable.

## Next tranche

After Rust qualification, build **Studio Runtime v1.1** in this order:

1. Atelier intent/question adapter into `symthaea-art-world`;
2. Symtropy Bevy typed art port;
3. deterministic scene snapshot/revision hashing;
4. proposal ghosts and counterfactual render capture;
5. Blender Python add-on implementing only the whitelisted v1 protocol;
6. Vision-Manifold whole-scene/counterfactual consequence comparison;
7. VART-STUDIO-001 preregistration comparing Canvas vs Blender API vs Bevy embodied practice.

The research question is not which host produces the highest scalar score. It is whether a persistent artistic system develops more coherent perception, technique, intention, revision, restraint, and cross-medium transfer when the same artistic self can inhabit increasingly rich studios.
