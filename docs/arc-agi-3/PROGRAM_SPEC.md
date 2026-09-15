# Symthaea ARC-AGI-3 Program Specification

**Status:** Draft foundation

**Tracker:** #3458

**Scope:** Parallel ARC-AGI-3 program. This document does not modify the existing ARC-AGI-2 experimental subject, preregistration, scorer, or evidence lineage.

## Purpose

ARC-AGI-2 remains Symthaea's controlled laboratory for static/few-shot reasoning mechanisms. ARC-AGI-3 is the interactive program for testing whether the same domain-general architecture can acquire world models from sparse interaction, infer latent goals, choose information-bearing actions, plan, transfer learned mechanics across levels, and minimize costly actions without benchmark-specific memorization.

The scientific target is not a public-demo score. It is evidence that a frozen generic subject learns and acts efficiently on unseen interactive environments under a source-blind execution boundary.

## Architectural principle

ARC-3 must exercise Symthaea rather than create a second benchmark-specific intelligence stack. Benchmark-facing code translates observations/actions, records evidence, and maintains game-scoped episodic state. Reusable cognitive authority should come from existing or general-purpose Symthaea components.

Current reuse targets include `crates/core/symthaea-core/src/hdc/grid_encoder.rs`, the cognitive loop, `src/consciousness/recursive_improvement/world_prediction.rs`, calibration, `magi_integration.rs`, and active-inference / expected-free-energy machinery.

Game IDs, fixture paths, recording names, source paths, level identifiers, and evaluator metadata are provenance only. They must not influence semantic representations, hypothesis priority, action selection, or planning.

## Cognitive loop

```text
canonical observation
        |
        v
scene + frame-delta decomposition
        |
        v
HDC relational representation
        |
        v
episodic transition ledger
        |
        v
world-model hypothesis bank
        |
        v
retrodictive qualification
        |
        +-----------------------------+
        |                             |
        v                             v
goal hypotheses               uncertainty / information value
        |                             |
        +--------------+--------------+
                       v
               active meta-controller
                       |
                       v
          explore / verify / plan / act
                       |
                       v
                canonical action
                       |
                       v
                  environment
                       |
                       +----> next observation
```

## Evidence model

An evidence-bearing step must commit the pre-action observation, selected action, ordered intermediate frames when present, post-action observation, derived scene/delta, lifecycle state, monotonic step index, and prior receipt root. Observation facts are immutable. Hypotheses, confidence, goal beliefs, planner state, and later explanations are derived authority and may reference observations but may not rewrite them.

A future causal receipt should be equivalent to:

```text
R_t = H(R_{t-1} || O_t || S_t || B_t || A_t || O_{t+1})
```

where `O` is canonical environment observation, `S` is scene/delta evidence, `B` is the frozen policy's belief-state commitment, and `A` is the canonical action. The independent verifier must be able to establish prefix causality: action `t` may depend only on evidence available through `t`.

## Memory boundaries

Within one game, raw episodic evidence and qualified mechanics may persist across levels because ARC-3 explicitly tests learning across an interactive sequence. Starting a new game creates a new episodic root and fresh game-rule state. Cross-game retained knowledge must be domain-general cognition or separately qualified consolidation; copying raw game-specific episodes/rules into a new game is forbidden for blind-generalization evidence.

## World-model authority

A plausible rule is not yet a planning rule. Every world-model hypothesis must carry semantic identity, scope, support, contradiction, predicted consequences, confidence/uncertainty, and lifecycle state. Planning authority is granted only after retrodictive qualification against the immutable transition ledger.

Retrodiction must report support, contradiction, abstention, and out-of-scope counts separately. Model breadth and prediction quality are distinct quantities. A narrow accurate model must not be confused with a broad one.

Probabilistic confidence should reuse Symthaea's existing world-prediction/calibration machinery where applicable rather than inventing ARC-specific confidence semantics.

## Goal inference

World dynamics and objective inference are separate unknowns. Later work must maintain explicit goal hypotheses rather than folding goals into transition rules. Goal evidence may include lifecycle transitions, level progression, preserved/invariant structures, repeated completion motifs, and environment feedback, but public-game-specific goals must not be encoded as generic constants.

## Active exploration

Actions may be pragmatic, epistemic, or mixed. A later policy should evaluate both expected progress and expected information gain while charging every environment interaction to the benchmark action budget. Exploration is successful when a small number of actions discriminate among competing hypotheses, not when the agent wanders until a known public level happens to complete.

## Source-blind execution

Environment and agent run across a narrow process boundary. The agent receives only canonical protocol-authorized messages. It must not inspect environment source, fixture files, interpreter objects, stack traces containing game logic, evaluator-only state, parent/sibling process memory, or arbitrary filesystem locations. Evidence-bearing execution must fail closed rather than silently falling back to an in-process mode.

## Evaluation hierarchy

Public-demo performance is engineering evidence only. A strong claim requires a frozen generic subject and an untouched evaluation population under an exposure/query budget.

```text
synthetic mechanism qualification
        -> public-demo engineering baseline
        -> frozen generic subject
        -> untouched evaluation
        -> sequestered replication
```

Benchmark feedback is epistemic consumption. Once task-level or detailed evaluation information influences a subsequent subject, that descendant belongs to a later knowledge epoch and must not be represented as blind to that evidence.

## Initial work packages

| ID | Issue | Exit concept |
|---|---|---|
| ARC3-001 | #3459 | Canonical protocol types and semantic/provenance separation |
| ARC3-002 | #3460 | Source-blind environment/agent boundary with negative controls |
| ARC3-003 | #3461 | Deterministic scene and frame-delta semantics |
| ARC3-004 | #3462 | HDC relational scene/transition encoding using existing HDC primitives |
| ARC3-005 | #3464 | Append-only temporal transition ledger with fault-safe replay |
| ARC3-006 | #3466 | Explicit falsifiable world-model hypothesis bank |
| ARC3-007 | #3467 | Retrodictive qualification before planning authority |

Later authorized work remains ARC3-008 through ARC3-018 as defined in #3458: goal inference, epistemic exploration, planner, active-abstraction controller, cross-level memory, action-efficiency integration, causal replay verifier, public baseline, architecture ablations, offline competition closure, and blind candidate freeze.

## First milestone gate

The first milestone is reached when ARC3-001 through ARC3-007 are qualified together on synthetic/recorded fixtures and can demonstrate a complete source-blind loop:

```text
observe -> canonicalize -> perceive -> record -> hypothesize -> retrodict -> qualify
```

No claim of intelligent ARC-3 action selection is authorized at that point. The purpose of the milestone is to establish trustworthy perception, temporal evidence, and world-model authority before scarce benchmark actions are spent by a planner.

## Non-goals for the foundation tranche

Do not implement game-specific rule tables, hard-coded public-demo solutions, per-game prompts, benchmark score thresholds as cognitive features, hidden evaluator access, ARC-2 subject changes, or a leaderboard-optimized planner. Those would either weaken scientific interpretability or contaminate the parallel ARC-2 program.
