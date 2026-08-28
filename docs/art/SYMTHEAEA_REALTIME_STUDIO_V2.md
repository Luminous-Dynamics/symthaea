# Symthaea Real-Time Studio v2 — Living 3D Art and Cinematic Direction

Status: architecture + host-neutral temporal/practice substrate implemented

## Thesis

Symthaea should not compete as another prompt-to-video endpoint. The stronger target is an autonomous developing artist that can inhabit a persistent simulated studio, perceive consequences over time, acquire technique, direct cameras and worlds, remember its own work, and decide when not to intervene.

```text
artistic question
      |
      v
working intention
      |
      v
whole-world perception  <------------------------------+
      |                                                   |
      v                                                   |
counterfactual candidates + abstain                       |
      |                                                   |
      v                                                   |
revision/frame-bound preview renders                      |
      |                                                   |
      v                                                   |
separate consequence evidence                             |
      |                                                   |
      v                                                   |
chosen intervention -> host/world -> temporal consequence-+
      |
      v
portfolio + technique + unresolved-question memory
```

The differentiator is not maximum photorealism. It is persistent artistic causation.

## Host split

- **Canvas**: low-cost sketchbook and fast compositional studies.
- **Bevy/Symtropy**: living embodied studio, real-time world, physics, tools, interactive cinema, accelerated practice.
- **Blender**: professional high-fidelity atelier for mesh/sculpt/material/camera/lighting/animation/compositing.

All three remain embodiments of one artistic self through host-neutral revision, temporal, intent, and developmental contracts.

## New v2 crates

### `symthaea-art-temporal`

Adds host-neutral cinematic time without changing the v1 mutation contract:

- rational frame rates;
- exact frame spans;
- validated camera poses/keyframes;
- revision-bound shot plans;
- proposal scheduling;
- bounded render-observation requests;
- color/depth/normal/object-id/motion channels;
- exact revision/frame render receipts;
- aligned baseline + counterfactual render sets;
- temporal consequence samples;
- no aggregate aesthetic score.

Semantic and pixel evidence must refer to the same committed revision and time coordinate. A visually excellent render from the wrong revision is invalid evidence.

### `symthaea-art-practice`

Adds persistent developmental memory:

- artistic questions;
- working intentions;
- works, studies, sketches, experiments, failures, and abandoned works;
- committed revision lineage;
- retained rejected proposals;
- technique attempts and observed prediction errors;
- discoveries and transfer targets;
- unresolved questions preserved without ranking.

This is intentionally not a style vector. It records what Symthaea was trying to understand and what happened when she acted.

## Real-time Bevy architecture

```text
                        SYMTHAEA
                           |
              +------------+-------------+
              |                          |
      artistic/practice memory      cognitive state
              |                          |
              +------------+-------------+
                           |
                           v
                    Art World API
                           |
                           v
                 Bevy Studio Adapter
       +-------------------+-------------------+
       |                   |                   |
 scene snapshot       studio timeline      capture queue
       |                   |                   |
       +-------------------+-------------------+
                           |
                  counterfactual branches
                           |
                 +---------+----------+
                 |                    |
             baseline             candidate(s)
                 |                    |
                 +---------+----------+
                           |
                     render evidence
                           |
                           v
                    Symthaea's eye
                           |
                           v
                     intervention
                           |
                           v
                    physical world
```

Bevy is the medium, not merely the renderer.

## Real-time video / cinema boundary

Bevy owns world state, rendering, camera motion, simulation, temporal composition, and interactive direction. Encoding remains outside artistic cognition:

```text
Bevy rendered frames + audio
          |
          +--> FFmpeg / GStreamer --> file
          |
          +--> WebRTC -------------> live stream
```

No codec-specific decision should become an artistic reward signal.

## Artistic eye

The perception stack should evolve in four layers:

1. **pixels** — color/value/edge/texture/focal hierarchy;
2. **scene** — stable entities/material/light/camera/spatial relationships;
3. **motion** — trajectories, optical flow, rhythm, timing, recurrence;
4. **causality** — which interventions changed which perceptual/world variables.

Whole-scene perception should be multi-scale. A director needs to notice local detail without losing composition or temporal structure.

## Artistic hand

Bevy should eventually expose both semantic tools and physical tools.

Semantic API operations are useful for composition and high-level direction. Physical tools are necessary for technique acquisition:

```text
intention
   -> motor trajectory
   -> tool/material contact
   -> resulting mark/deformation
   -> re-observation
   -> prediction error
   -> skill update
```

Candidate media include brush, palette knife, charcoal, clay/sculpting tools, light rigs, camera rigs, procedural materials, particle fields, reaction-diffusion surfaces, fluids, destructible matter, topology-changing forms, and higher-dimensional projections.

## Counterfactual direction

Every nontrivial intervention should preserve a do-nothing baseline when affordable:

```text
revision R @ frame F
  |- abstain / baseline
  |- candidate A
  |- candidate B
  `- candidate C
```

Each branch is rendered from the same revision/frame contract. Evidence stays multidimensional: intention advancement, preservation damage, uncertainty, visual structure, motion coherence, duplication, technical feasibility, and unforeseen structure remain separate channels.

No universal beauty scalar is introduced.

## Cinematic memory

The artist should remember temporal motifs as first-class history:

- camera grammars used recently;
- recurring shapes and visual motifs;
- unresolved spatial or narrative tensions;
- long-range recurrence;
- shot durations and transitions;
- previous rejected camera alternatives;
- relation between music, motion, and world events;
- discoveries about causal media.

This enables a motif introduced forty minutes earlier—or months earlier in a persistent installation—to recur because of artistic history rather than prompt coincidence.

## Persistent living works

A work may have no fixed duration. The same contracts support:

- a 12-second cinematic study;
- a 90-minute autonomous film;
- an interactive installation;
- a world evolving continuously for months;
- a collaborative work changed by viewers and other agents.

Portfolio identity therefore binds to world/revision history, not just exported files.

## Industry advantage hypothesis

Do not aim first at areas where giant visual models have structural advantages: raw photorealism, encyclopedic visual priors, one-shot text-to-video, or engine rendering breadth.

Test instead whether Symthaea can outperform conventional generators on:

- causal consequence prediction;
- persistent world consistency;
- long-horizon motif continuity;
- learned physical technique;
- counterfactual revision quality;
- explicit restraint/abstention;
- cross-medium skill transfer;
- provenance of artistic decisions;
- interactive direction;
- continuity of artistic questions across works.

The product category is closer to **autonomous artist/director in a causal world** than **video generator**.

## Performance architecture

Real-time cognition must not destabilize rendering. Treat work as three timing classes:

- **frame-critical**: render, transform interpolation, camera, physics stepping;
- **cycle-critical**: bounded perception/capture bookkeeping and already-decided action execution;
- **deferred**: expensive visual analysis, counterfactual batch rendering, portfolio consolidation, critique, long-horizon planning.

Use bounded queues and explicit drop receipts rather than unbounded backlogs. Candidate previews may use reduced resolution or reduced simulation fidelity, but their fidelity class must be recorded so evidence from unlike render classes is not silently compared.

## Implementation sequence

### RT1 — temporal identity and capture

Implemented host-neutral contract. Bevy side should add deterministic studio frame identity, exact revision/frame capture requests, bounded capture queue, and receipts.

### RT2 — branch-isolated preview worlds

Create disposable preview entities/world state. A preview never advances the committed art revision. Record branch lineage and cleanup.

### RT3 — cinematic planning

Add shot/sequence plans, camera keyframes, temporal observations, and deterministic replay. Keep planning separate from scene mutation authority.

### RT4 — whole-scene eye

Render baseline/candidates from the same camera/revision/frame and pass them through the same visual pipeline. Measure consequence-prediction calibration rather than aesthetic acceptance.

### RT5 — embodied tools

Introduce a small physical medium first (recommended: deformable clay or brush-on-surface) and test technique acquisition against direct API control.

### RT6 — audiovisual direction

Bind Muse/scene timing through shared temporal receipts. Do not hard-code emotion-to-color or beat-to-motion mappings.

### RT7 — persistent interactive cinema

Allow viewers/agents to perturb the world. Symthaea observes and may revise direction while preserving artistic identity and full provenance.

## Research gates

### VART-RT-001 — temporal/revision integrity

Render observations must be rejected if revision or frame mismatches. Repeated deterministic scene playback should reproduce semantic revision hashes and scheduled action order.

### VART-RT-002 — counterfactual isolation

Previewing N candidate branches must leave committed world state byte/semantic-hash equivalent to a no-preview control until a separately authorized commit occurs.

### VART-RT-003 — consequence prediction

On held-out interventions, predict multidimensional perceptual/world changes better than persistence and shuffled-action baselines.

### VART-RT-004 — technique acquisition

Compare embodied practice against random practice and direct semantic operations. Measure prediction calibration, sample efficiency, held-out gesture/material transfer, and recovery from failed attempts.

### VART-RT-005 — long-horizon artistic continuity

Compare full portfolio/question memory against shuffled-history and no-history controls. Test recurrence of unresolved questions and motifs without collapsing toward exact duplication.

### VART-RT-006 — interactive direction

Perturb the world/viewer trajectory while holding artistic question fixed. Test whether revisions remain causally linked to the question and world evidence rather than becoming arbitrary reactive effects.

## Qualification

Host-neutral crates:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-art-world -p symthaea-art-temporal -p symthaea-art-practice --all-targets
cargo test -p symthaea-art-world -p symthaea-art-temporal -p symthaea-art-practice
cargo clippy -p symthaea-art-world -p symthaea-art-temporal -p symthaea-art-practice --all-targets -- -D warnings
```

Additional invariants:

- no render receipt can silently cross revision or frame;
- camera keyframes are finite, in-range, and ordered;
- counterfactual sets retain an abstention baseline;
- portfolio lineage retains rejected proposals and failures;
- no new beauty/reward/fitness scalarization API;
- no host code execution channel is introduced;
- no Studio Runtime type grants mutation authority by observation alone.

## North-star demo

One self-generated artistic question follows Symthaea through three media:

1. Canvas composition studies;
2. Bevy embodied/causal world exploration and a short real-time cinematic sequence;
3. Blender high-fidelity realization.

The final exhibit includes not only the rendered work but the question lineage, rejected candidates, counterfactuals, technique discoveries, causal receipts, and cross-medium transfer history.

The goal is not to prove that the final image scores highest. The goal is to demonstrate a developing artist whose work has memory, causation, technique, and temporal intention.
