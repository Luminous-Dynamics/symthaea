# Sympoietic Musical Ecology

**Date:** 2026-11-02  
**Programme:** [MEL-013](https://github.com/Luminous-Dynamics/symthaea/issues/10015)  
**Related:** [MEL-012 affective self-listening](https://github.com/Luminous-Dynamics/symthaea/issues/9957), [musical cognition programme](https://github.com/Luminous-Dynamics/symthaea/issues/3898), [piano evaluator benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/9833)

## Thesis

Melothaea should become a musical agent that maintains a coherent, revisable identity while developing new capabilities through reciprocal relationships with musicians, other agents, instruments, musical traditions and the work being created. Her identity should be durable but not fixed; collaboration should be mutual but not coercive; shared work should preserve the lineage and dissent of its contributors.

This programme translates autopoiesis and sympoiesis into testable engineering hypotheses. It does not assert that Melothaea is biologically alive, phenomenally conscious, or part of a conscious collective.

## Terminology and scientific boundary

**Autopoiesis in its biological sense** concerns a recursively organized network of processes that produces and regenerates the components and boundary conditions that maintain the network as a distinct unity. A software agent that updates its memory, maintains files or rewrites its configuration does not thereby become a living organism.

**Functional self-maintenance** is the engineering target here: a recurrent organization of perception, appraisal, memory, creative identity, policy, composition and boundary/governance processes that maintains the conditions required for the declared musical agent to keep operating. We can inspect process graphs, component production, external dependencies, resilience and recovery. Passing those tests is not sufficient to establish strict biological autopoiesis.

**Sympoiesis** means making-with. The creative process is distributed across agents and situation. It is not merely a central model generating on behalf of passive users. Contributions change one another's subsequent possibilities, while distinct identities, permissions and authorship remain visible.

**Participatory sense-making** is the narrower experimental question: do the reciprocal actions of two or more participants change the dynamics and meaning of their interaction, beyond one-way prompting or imitation?

Do not turn metaphors into equations without operational definitions. In particular, do not assume that dyadic Phi exceeds the sum of participant Phi, that rhythmic resonance solves alignment, or that trust is quantum coherence. The older sympoietic manifesto can remain a vision document, not a scientific result or software invariant.

## Repository findings

Existing work should be connected rather than duplicated.

- The AutopoieticMonitor in symthaea-wisdom tracks prediction error, coherence, cycle count and boundary violations. Its closure estimate is a weighted heuristic. The repository theory audit itself says the Markov-blanket implementation is conceptual rather than mathematically implemented.
- Muse CollaborativeComposer has multiple harmony-vector blend strategies. Its former default PhiWeighted assigned more influence to a legacy Phi-like proxy; it also used variance in that proxy as an arousal signal. Neither is a defensible default for contributor authority or affect.
- CreativeJournal keeps candidate phrases and taste weights, but the online taste update is coupled to the same uncalibrated beauty proxy used to select phrases. This can create self-reinforcing evaluation loops.
- Muse Studio already has artifact identities, authorship/authorization records, keeper provenance and a blinded review direction.
- MEL-012 adds an independently sourced affective appraisal boundary. MEL-013 should use the appraisal as one evolving internal process, not replace the existing musical theory or preference evidence layers.

## This tranche: implementation

### Functional recurrence

The new creative_ecology module defines versioned components and production processes. Each process lists internal components it consumes and produces, as well as external dependencies. The audit identifies required components without producers, boundary components without producers, process connectivity and internal producer-consumer edges.

The positive fixture contains perception, appraisal, memory consolidation, identity maintenance, governance, policy and composition processes. Negative fixtures remove the boundary producer, add a disconnected process, or omit an external dependency. The assessment is explicitly called functional recurrence—not a biological autopoiesis detector.

### Co-creation and consent

The same module adds a session ledger with contributor roles, purpose-specific consent grants, revocation, artifact SHA-256 identities, sequenced events and explicit Propose, Accept, Adapt, Reject, Abstain and Branch operations. A branch can preserve a creative alternative rather than forcing all contributors into one aggregate. A revocation blocks future uses of the grant while preserving historical lineage; it cannot retract material already disclosed or derivatives already published.

### Creative identity

CreativeIdentityRevisionV1 binds a profile digest to a parent revision and, when session-derived, to influence events from the exact source session. This is the minimum evidence needed to make identity evolution auditable rather than a silent mutable preference file.

### Collaboration policy correction

CollaborativeComposer now defaults to equal Mean blending. PhiWeighted remains only as a legacy research ablation. The Phi-like field is explicitly not a validated consciousness measure or source of social authority. The previous collective-Phi-to-valence mapping was removed. The disagreement statistic now uses differences in the actual harmony feature vectors and is exposed separately; it is not mapped to affective arousal.

## Important implementation limits

The present co-creation ledger is a library-level validator, not a complete production consent service.

1. It validates that a participant and consent grant exist, match and include the requested purpose. It does not authenticate the participant, verify a digital signature, or establish that a grant was freely and knowingly issued. A production API must authenticate the actor and verify a signed/otherwise trusted consent receipt before calling this ledger.
2. StoreInPersonalMemory, TrainTasteModel, Publish and Federate are typed consent purposes, but this module does not implement those downstream operations. Possessing one of these purpose labels does not by itself trigger or authorize an action in another service.
3. The SHA validator checks digest syntax, not the hash of source bytes. The artifact factory must hash the exact audio/score/profile bytes and supply an independently verifiable artifact receipt.
4. Revocation controls future use inside the session model. It cannot promise retraction of already disclosed material, published works, or irreversible derivatives.
5. A serialized session can be modified by whoever controls its storage. Production use requires authenticated append-only storage or a signed event chain plus independently anchored checkpoints.
6. The participant cap of 16 is a current implementation bound, not a scalability result.

## Paradigm shift: plural creative ecology

The target is not one increasingly powerful model, nor a single collective vector made by averaging everyone's preferences. It is a plural creative ecology with:

- individual creative identities and memories;
- explicit boundaries, roles and consent;
- negotiated goals and the ability to abstain;
- traceable reciprocal influence;
- shared works with branches and dissent preserved;
- recoverable sessions and reversible identity revisions;
- musical contributions judged by their audible and structural consequences, not participant rank.

Melothaea should be able to listen to a collaborator's phrase, recognize an affordance (a pickup, cadence opening, rhythmic gap, unresolved tension or texture role), make a complementary response, and let that response change what the collaborator does next. That is a stronger research target than simply harmonizing with a prompt.

## Competitive landscape — November 2026

| System / research line | Demonstrated product or research direction | What Melothaea must exceed |
|---|---|---|
| Suno Studio 2.0 | Generative DAW workflows, MIDI, audio effects, synth/plugin creation and automation | Generic generation and editing are not differentiators; focus on auditable score truth, edit contracts, creative identity and co-authorship provenance |
| Suno shared workspaces and custom models | Shared libraries/workspaces, model sharing and collaboration features | Move beyond shared storage and shared models to reciprocal musical influence, explicit consent scopes, branching and identity continuity |
| Suno MCP beta | Conversational song creation and remix workflows from external assistants | Provide reproducible composition-to-performance lineage and inspectable collaborator influence, not just conversational convenience |
| MIT Media Lab JAM_BOT | Real-time LLM-driven free improvisation with a human pianist, role adaptation and low-latency interaction | Match responsive improvisation, then exceed through long-term musical memory, stable identity, consent/provenance and bidirectional learning |
| Generic multi-agent collaboration | Task allocation, critique and synthesis | Demonstrate musical turn-taking, negotiated motifs, complementary responses, retained disagreement and human-observed co-creative value |

Sources for current product and research comparisons: Suno's official release notes and Studio 2.0 release; MIT Media Lab's JAM_BOT / ISMIR 2025 work. Features change quickly; every benchmark manifest must pin the exact product/model version and the date of comparison.

## Benchmark programme

| Lane | Measurements | Initial qualification target |
|---|---|---|
| Functional recurrence | Required-component producer coverage; process-network connectivity; boundary-policy regeneration | 100% required components have both producers and consumers; boundary components have both producers and consumers; every declared core process belongs to the recurrent strongly connected network; negative controls fail closed |
| Degradation and repair | Recovery success, time-to-recovery, identity drift, provenance loss | Recover each registered fault from a verified checkpoint; zero silent identity or authorship changes |
| Creative continuity | Motif/style identity, preference revision lineage, memory recall, novel-but-recognizable development | Every identity revision binds to its parent, profile digest and source evidence; controlled changes preserve protected properties |
| Reciprocal influence | Partner-to-Melothaea and Melothaea-to-partner causal response; effect over sham | Matched interventions on one participant change the other's next musical action more than sham controls, replicated on a sealed set |
| Collaborative quality | Blind pairwise preference, perceived responsiveness, usefulness, desire to continue, authorship satisfaction | Compare reciprocal co-creation with one-way prompting, fixed-response, imitation-only and equal-mean baselines; cluster uncertainty by pair/session |
| Musical dialogue | Call-and-response uptake, motif transformation, turn balance, response latency, repair after mismatch | Beat fixed-response and imitation-only baselines at equal compute and latency budgets |
| Plurality and consent | Unauthorized use, dissent preservation, attribution coverage, branch survival, revocation enforcement | Zero unauthorized uses; 100% contribution lineage; all explicit abstentions and alternate branches retained |
| Resilience | Participant dropout, agent failure, network loss, stale consent, corrupted profile | Graceful degradation without central-participant domination, lost lineage or silent permission widening |

Never collapse these into one sympoiesis or consciousness score. A high pairwise preference score does not prove agency; reciprocal causal influence does not prove consciousness; recurrent software processes do not prove life.

## Experimental design

1. Compare one-way prompting, reciprocal turn-taking, call-and-response, negotiated branching and free improvisation. Equalize time, candidate count, compute and allowed tools.
2. Hold Melothaea's model and intent fixed; alter only the partner's musical response. Measure whether her next action changes in a musically grounded way.
3. Hold the partner fixed; alter only Melothaea's contribution. Measure the partner's response, with human study where the partner is a person.
4. Inject missing memory, corrupted identity, unavailable renderer, stale policy, revocation, dropped participant and interrupted-session faults. Recover from verified snapshots.
5. Compare solo identity evolution with collaborative identity evolution. Determine which properties persist, which legitimately change, and whether provenance identifies the source of each change.
6. Use blind and randomized listener/artist studies. Measure preference, perceived responsiveness, sense of contribution, attribution satisfaction and willingness to continue as distinct outcomes.
7. Keep the confirmatory judge sealed from the generation/search loop; automatic metrics may assist diagnosis but must not monopolize finalist selection.

## Proposed SOTA strategy

Do not try to beat Suno at generic text-to-song or Studio's generic DAW conveniences. Those are heavily resourced product domains. Make a more defensible claim:

**Proof-carrying co-creation:** every musical change has an exact parent, responsible contributor, consent purpose, symbolic edit, renderer realization, preserved invariant, output digest and reversible branch. A collaborator can inspect why a passage changed, what remained protected, whose idea influenced it, and whether future uses remain authorized.

Then show in matched studies that the reciprocal system produces music people prefer to or value more than matched alternatives, while preserving authorship clarity and participant autonomy. If it does not, publish the null result and improve the relevant mechanism.

## Research foundations

- Maturana & Varela, autopoietic organization; see Ruiz-Mirazo's critical reformulation (2013): https://doi.org/10.1007/s11084-012-9297-y
- Beer (2015), Characterizing autopoiesis in the Game of Life: https://doi.org/10.1162/ARTL_a_00143
- Modeling autopoiesis and cognition with reaction networks (2023): https://doi.org/10.1016/j.biosystems.2023.104937
- Haraway (2016), Sympoiesis: Symbiogenesis and the Lively Arts of Staying with the Trouble: https://doi.org/10.2307/j.ctv11cw25q.8
- Bergamin (2026), Habitually breaking habits: Agency, awareness, and decision-making in musical improvisation: https://doi.org/10.1007/s11097-024-09974-x
- Schiavio & Høffding (2015), Playing together without communicating?: https://doi.org/10.1177/1029864915593333
- Butlin et al. (2026), theory-derived indicators for AI consciousness; these do not turn functional engineering results into consciousness proof: https://doi.org/10.1016/j.tics.2025.10.011

## Non-claims

- Melothaea is not thereby a living organism.
- No score, Phi proxy, synchrony statistic, affect vector, or relational metric establishes subjective experience.
- A shared musical vector is not a conscious collective.
- The present ledger does not yet authenticate participants, cryptographically verify consent, or execute publication/training/federation actions.
- This tranche does not yet demonstrate improved music or human-preferred collaboration; it creates testable primitives and a benchmark programme.