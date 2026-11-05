# MEL-013A: Studio Artifact Transaction Contract

**Date:** 2026-11-02  
**Status:** Research/design contract; not an implementation or qualification report  
**Parent:** [MEL-013 — functional autopoiesis and sympoietic musical co-creation](https://github.com/Luminous-Dynamics/symthaea/issues/10015)  
**Implementation issue:** [#10051 — bind sympoietic sessions to verified Studio artifact transactions](https://github.com/Luminous-Dynamics/symthaea/issues/10051)  
**Related:** [MEL-012 — affective self-listening](https://github.com/Luminous-Dynamics/symthaea/issues/9957), [Melothaea programme](https://github.com/Luminous-Dynamics/symthaea/issues/3898), [PR #10022](https://github.com/Luminous-Dynamics/symthaea/pull/10022)

## 1. Purpose and claim boundary

MEL-013 adds typed models for functional creative self-maintenance, consent-scoped co-creation events, branches, influence references and creative-identity revisions. The next step is to make those models participate in the real Muse Studio artifact lifecycle.

The target is **verified sympoietic artifact continuity**:

- each contribution refers to the exact musical artifact it received and the exact artifact it produced;
- multiple contributors and alternate branches remain independently visible;
- the system can recover the session and its lineage after restart;
- a contribution is admitted only through a trusted identity and purpose-specific authorization boundary;
- listening/appraisal evidence is attached to the exact rendition that was heard;
- no hidden operation converts review into personal memory, taste training, publication or federation.

This is a software and governance property. It does not establish biological life, subjective experience, consciousness, legal authorship, copyright ownership or artistic quality.

## 2. Repository facts that constrain the design

The current Studio code has several strong building blocks:

| Existing element | Current role | Integration constraint |
|---|---|---|
| `Studio.candidates: Mutex<HashMap<u64, Candidate>>` | In-memory candidate storage | Candidate IDs are session-local, reset on restart and subject to eviction. They cannot be durable lineage identifiers. |
| `data/taste/audio/<key>/...` | Durable keeper artifact bundle | The keeper key is an operational storage locator, not a content identity. |
| `data/taste/keepers.jsonl` | Operational keeper index | Preserve compatibility; do not quietly replace or reinterpret historical rows. |
| `ArtifactIdentity` | Distinguishes score content, composition and rendition | Reuse these identity layers rather than inventing a competing hash scheme. |
| `PieceProvenanceBundle` | Binds recipe, score, audio and renderer/build provenance | Recompute digests from actual stored bytes before admitting an edge. |
| `GenealogyStore` | SQLite-backed manifest, children and ancestry queries | Its live writer currently creates only `GenealogyRelation::Root`; the single `parent_id` is a tree projection, not a plural influence graph. |
| `CoCreationSessionV1` (PR #10022) | Validates participant/grant identifiers, consent purposes, event ordering and artifact digest syntax | It is currently a library-level session model, not authentication, a signature verifier, durable event storage or proof that Studio consumed the referenced bytes. |

The protocol already declares non-root genealogy relation variants, but the current compose/keep path does not populate them. A plural creative process cannot be truthfully represented by selecting one contributor as the only parent.

**Do not confuse four different things:** a content digest identifies bytes; a storage key locates bytes; a consent receipt authorizes a scoped operation; a contribution record describes who did what. None substitutes for the others.

## 3. Paradigm shift: a plural artifact graph, not a single family tree

Represent co-creation as a directed acyclic graph (DAG) over content-addressed musical artifacts and event records.

A single composition may have several legitimate influences: an initial human motif, a Melothaea counter-motif, an adapted phrase from another musician, and a performance-derived rhythmic idea. Flattening that history into one parent loses information and can falsely imply sole derivation or authority.

### 3.1 Nodes

Reuse `ArtifactIdentity` and `PieceProvenanceBundle` wherever their existing semantics fit. A versioned graph record should distinguish at least:

- **Score-content node:** exact canonical symbolic score bytes and digest.
- **Composition node:** the recipe/specification and composition identity that produced the score.
- **Rendition node:** exact performed MIDI/performance contract and renderer realization.
- **Audio node:** exact rendered PCM/WAV bytes and digest.
- **Contribution-event node:** canonical action, actor, purpose, session, sequence and references to source/result artifacts.
- **Consent-evidence node/reference:** trusted receipt identity/digest and verifier outcome.

A candidate number, browser URL, keeper key, filename or title is metadata/locator only. It must never stand in for a content identity.

### 3.2 Edges

Use typed, directional edges such as:

- `ProposedFrom`
- `AdaptedFrom`
- `BranchedFrom`
- `MotifInfluencedBy`
- `ReorchestratedFrom`
- `ReperformedFrom`
- `RenderedFrom`
- `ReviewedAgainst`

An edge should bind the source artifact, result artifact, exact contribution event, contributor, operation type and consent-evidence reference. Multiple source edges may point to the same result. A branch produces a separate result node; it must not overwrite the branch it diverged from.

The existing single-parent genealogy manifest can remain a compatibility view for simple histories. The new multi-parent edge table is the authoritative representation for co-creation. Do not overload `parent_id` to hold an arbitrary primary contributor.

The graph verifier must reject cycles, missing nodes, dangling events, impossible event ordering, duplicate edge identities, stale parent references and edges whose source/result digests do not resolve to verified bytes.

## 4. Trust and consent boundary

The current `CoCreationSessionV1` checks internal consistency of declared participant IDs and grant IDs. It cannot establish that the caller is the participant or that the grant is authentic.

Introduce an explicit, injectable boundary such as:

- `ParticipantIdentityProvider`: authenticates the current actor and produces a stable principal reference.
- `ConsentReceiptVerifier`: verifies a trusted receipt, its issuer/key status, subject, session/project, artifact scope, action, purpose, validity interval and revocation state.
- `AuthorizationDecisionV1`: records the exact request, decision, verifier identity/version and receipt digest.

The default implementations must be unavailable and fail closed. Do not ship a permissive local verifier and label it production-ready. Do not accept a client-supplied `participant_id`, `grant_id` or Boolean `consent=true` as proof of authority.

Keep the following permissions separate:

| Purpose | Permitted operation | Not implied |
|---|---|---|
| Contribute | Submit an idea or artifact to this session | Persistent memory, training, publication |
| Review | Evaluate a proposal or rendition | Transforming it, retaining it, training on it |
| Transform | Adapt, branch or derive a new artifact | Global model training, public release |
| Store in personal memory | Retain the authorized participant-specific memory | Global taste-model updates or federation |
| Train taste model | Use the authorized material for the specified model/training scope | Publication or federation |
| Publish | Release the specified material through the declared channel | Future training or unrelated derivative use |
| Federate | Share the specified record with the named federation | Unrestricted public distribution |

The table is a policy contract, not an implementation claim. Each consumer must re-check the exact purpose immediately before the operation.

For recorded performances and imported musical works, rights/licence evidence is distinct from a person's session consent. A participant's approval does not automatically clear a composition, edition, recording, sample library or third-party performance.

Revocation prevents future uses covered by the revoked grant. Historical provenance should remain auditable. Do not promise to retract already disclosed material, published copies or irreversible derivatives.

## 5. Commit protocol: make partial failure explicit

Filesystem writes and SQLite commits do not share one atomic transaction. Do not claim cross-resource atomicity by putting both operations inside a function named `commit()`.

Use a durable, idempotent transaction journal. A transaction moves through explicit states:

1. **Prepared** — authenticated actor, verified scope, exact parents and intended operation are frozen.
2. **Artifacts staged** — exact score, recipe, performance MIDI and/or audio bytes are written to temporary paths; byte lengths and hashes are computed by the server.
3. **Artifacts durable** — files are flushed and atomically moved to their durable locations; no public keeper/lineage entry is visible yet.
4. **Graph committed** — one SQLite transaction records the canonical event, all parent edges, verifier evidence and the idempotency key.
5. **Verified** — an independent reader reloads the stored bytes and records, recomputes digests and validates references and graph invariants.
6. **Published to Studio** — the API exposes the committed artifact and receipt.

The implementation may use different internal names, but it must preserve these observable boundaries.

### 5.1 Idempotency and recovery

Every write request carries an idempotency key bound to the authenticated principal, session, operation and frozen input digests.

- Same key + same canonical request returns the original committed result.
- Same key + different payload is a conflict.
- A retry must never create a second event, branch, keeper entry or lineage edge.
- Uncommitted staged files are not visible as accepted work.
- Startup recovery reconciles journal states with durable files and database records.
- A record whose bytes are absent or fail digest verification is quarantined/abstained, not silently repaired by generating new bytes.

Fault-injection tests must stop the process after each state transition. Recovery must yield either one complete, independently verifiable commit or no visible commit—never a phantom event or an apparently accepted artifact with missing bytes.

### 5.2 Hash chain versus authenticity

A canonical event hash may commit to the previous event hash, event payload, artifact identities and consent receipt digest. This detects accidental or post hoc modification relative to a trusted checkpoint.

A hash chain alone does **not** prove who authored an event. Authorship authenticity requires a trusted signature or equivalent verifier tied to an authenticated principal. Do not call a locally self-generated hash chain signed, authenticated or tamper-proof.

## 6. Mapping co-creation actions to artifacts

The integration must preserve the semantics of the MEL-013 action model:

| Action | Artifact effect | Required evidence |
|---|---|---|
| Propose | Creates a proposal record and, when supplied, a candidate artifact reference | Exact parent/result bytes and contribution authorization |
| Accept | Accepts a proposal; does not rewrite its artifact | Review authorization and exact proposal identity |
| Adapt | Creates a new result artifact | Transform authorization, all source edges and exact before/after digests |
| Reject | Records a rejection; does not create a derivative by implication | Review authorization and exact proposal identity |
| Abstain | Records non-participation without penalizing the contributor | Session/event identity; no inferred consent to another action |
| Branch | Creates a distinct alternative artifact | Transform authorization, parent references and branch identity |

A proposal that has not been accepted remains a proposal. A rejected or abandoned branch remains in the historical record but must not be presented as an approved or published work. Acceptance is not a grant for training, publication or federation.

A symbolic edit should record the exact score-level operation (for example, motif substitution, voice reassignment, rhythm transformation or phrase extension) and the preserved musical invariants. The renderer must then provide separate evidence that the intended symbolic result was actually rendered. A correct edit description is not proof of audible realization.

## 7. MEL-012 listening-loop integration

The first listening bridge should attach the appraisal record to the exact rendition and session event that was heard. Keep separate:

1. session-local appraisal (the immediate response to this rendition);
2. participant-specific personal memory;
3. Melothaea's persistent creative identity/taste adaptation;
4. training of a shared model;
5. publication or federation.

A listening result must not automatically cross these boundaries. The first integration can record transient appraisal without storing it as a durable preference or training target.

This also prevents a proxy loop: a candidate is selected by a taste score, the same taste score is treated as ground truth, and then the model trains on its own selection. Human preference, structural validity, perceptual measurement and model prediction must remain separately identified evidence.

## 8. Independent verification contract

The verifier should accept a transaction bundle and independently check:

- canonical schema version and canonical serialization;
- exact identity/content digests for all committed artifacts;
- artifact byte lengths and media types;
- event order, parent references and event-to-artifact bindings;
- graph acyclicity and branch distinctness;
- consent receipt authenticity, exact purpose and revocation status;
- transaction idempotency and commit-state consistency;
- renderer identity/configuration and the score → performance → audio chain;
- exact listening-evidence linkage to the rendition;
- absence of cross-purpose side effects.

The verifier must not trust the producer's `PASS` field. It recomputes the facts it can independently observe and returns `PASS`, `FAIL` or `ABSTAIN` with machine-readable reasons. Unsupported authentication or unavailable renderer evidence is an abstention, not a pass.

## 9. Qualification matrix

All values below are proposed gates, not achieved results.

| Lane | Qualification target |
|---|---|
| Artifact integrity | 100% of committed score/recipe/MIDI/audio digests recompute from stored bytes |
| Edge integrity | 100% of edges resolve to verified endpoints and events; zero dangling or duplicate committed edges |
| Graph correctness | Zero cycles; every branch remains separately addressable; all declared contributing parents retained |
| Consent | Zero unauthorized operation acceptance in the registered negative-control suite; unavailable verifier always abstains |
| Purpose isolation | Zero cross-purpose side effects across memory, training, publication and federation |
| Idempotency | 100% same-request retries return the same commit; altered-payload key reuse fails |
| Crash recovery | Every injected interruption yields one valid commit or no visible commit; zero phantom events |
| Restart continuity | 100% of committed sessions/branches/edges recover; ephemeral uncommitted candidates are clearly reported as unavailable |
| Renderer lineage | Every promoted rendition binds exact score/performance/audio identities and renderer configuration |
| Listening linkage | 100% of persisted appraisal observations bind to the exact rendition and session event |
| Attribution | 100% of result-changing operations identify the responsible event and source edges; no claim of exclusive authorship from influence alone |

Do not combine these lanes into one “sympoiesis score.” A technically valid graph does not prove a good composition, mutual influence, consciousness or ethical collaboration.

## 10. Adversarial and recovery fixtures

At minimum, include:

- caller forges another participant's ID;
- invalid, expired, revoked, wrong-scope or wrong-purpose consent receipt;
- receipt signed by an unknown, revoked or substituted key;
- artifact digest is syntactically valid but does not match stored bytes;
- storage key is substituted while the content digest stays fixed;
- result artifact is missing, stale or unrelated to the declared edit;
- parent artifact is missing or from an untrusted source;
- event references a future event or an event from another session;
- duplicate event ID, edge ID or idempotency key;
- the same idempotency key is replayed with modified content;
- a graph cycle is introduced;
- one contributor is silently removed from a multi-parent composition;
- a branch overwrites its sibling;
- revocation occurs between prepare and commit;
- crash after staging, file rename, event insert, edge insert and transaction commit;
- a failed commit leaves a keeper entry or a successful commit lacks durable bytes;
- listening appraisal is incorrectly copied into personal memory or taste training without the corresponding purpose;
- publication/federation is attempted when its independent service/verifier is absent.

Use positive fixtures, negative fixtures and metamorphic tests. A positive fixture must not be the only evidence that a security boundary works.

## 11. First executable vertical slice

Keep the first implementation deliberately narrow:

1. Start from an existing Studio candidate with a stable `ArtifactIdentity` and `PieceProvenanceBundle`.
2. Create a session with two distinct participants and test identity/consent verifier interfaces.
3. Have participant A propose an exact source motif/score artifact.
4. Have Melothaea adapt it into a distinct score artifact, preserving the source branch.
5. Render the adapted score and bind the exact performed MIDI and WAV digests.
6. Have participant B add a second influence and branch the work; preserve both source edges.
7. Attach a MEL-012 appraisal to the exact rendition without updating global taste.
8. Restart Studio; independently verify the artifacts, event chain, consent evidence and multi-parent graph.
9. Run negative controls for forged identity, wrong purpose, altered bytes, revocation and interrupted commit.

Do not widen the first slice to public federation, multi-device identity, global model training or a generalized rights-management system.

## 12. Research and competitive context

The comparison must separate product capabilities, real-time performance systems and research claims. They are not interchangeable benchmarks.

| System / research | Demonstrated capability | Implication for Melothaea |
|---|---|---|
| Suno Studio 2.0 (13 Aug 2026) | Product-grade generative studio with MIDI editing/recording, built-in wavetable synth, effects, automation, plugin design and conversational song/track edits | Do not compete on “chat plus a timeline” or basic MIDI/DAW convenience. Match reliability and workflow integration, then differentiate on symbolic edit contracts, identity continuity and attributable co-creation. |
| Google Lyria RealTime | Experimental low-latency bidirectional streaming music model; prompt weights and live controls for BPM, density, brightness, scale and playback; 48 kHz stereo PCM output with watermarking | Treat as a serious streaming-generation baseline. Compare controllability and response to interventions, not raw audio quality alone. Its weighted prompts and configuration changes are not equivalent to a persistent multi-participant consent/lineage graph. |
| MIT JAM_BOT (ISMIR 2025) | Real-time symbolic music-language-model improvisation; role strategies, musician-specific conditioning and low-latency listen/prompt/schedule pipeline | Match live responsiveness, flexible roles and musician-specific interaction. A live demo is not evidence of long-term identity continuity or multi-party consent. |
| JAM_BOT follow-up (NIME 2026) | Adds velocity, higher throughput, free-form call-and-response across varying tempi and external MIDI latency compensation | Tempo flexibility, expressive velocity and measured I/O latency are mandatory baseline dimensions; they cannot be treated as future differentiators. |
| Somax2 / Agents in Concert (IUI 2026) | Corpus-driven multi-agent improvisation with influencer/player roles, self-listening and external listening; longitudinal codesign with jazz musicians and public performance | Compare against corpus-conditioned and multi-agent approaches, not only a static one-way LLM. Study how musicians anticipate, shape and negotiate system behavior in practice. |
| MEL-013 / MEL-013A | Target: persistent but revisable creative identity, multiple contributor influences, branch-preserving score-level edits, exact artifact lineage, scoped permissions and restart-safe memory | This is a different proposed product/research axis. It becomes a differentiator only after end-to-end qualification and musician studies. |

Primary sources:

- Suno Studio 2.0 release: https://suno.com/blog/studio-2 and official feature notes: https://help.suno.com/en/articles/13670529
- Google Lyria RealTime API: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- JAM_BOT (ISMIR 2025): https://doi.org/10.5281/zenodo.17811478
- Enhancing Expressive Musical Conversation in JAM_BOT (NIME 2026): https://doi.org/10.5281/zenodo.20784228
- Agents in Concert (IUI 2026): https://doi.org/10.1145/3742413.3789104

### What the recent co-creation evidence warns us about

- A 2026 CHI randomized study compared model-led rewriting with reflective human-led suggestion/question modes (main study N=486, separate validation N=640). In its non-musical idea-generation task, reflective human-led interaction improved idea quality while better preserving diversity and perceived ownership than model-led rewriting. This is cross-domain evidence, not a result about music, but it argues against optimizing Melothaea to take over the composition. https://doi.org/10.1145/3772318.3791185
- A 2026 blind-listening study compared AI-generated, human–AI collaborative and human-composed excerpts. It had 71 participants and 18 selected excerpts. The result is useful as a reminder to separate perceived authorship, aesthetic judgment and actual contribution lineage; listener attribution was a weak proxy for who actually authored what. Its small stimulus set and screening process limit generalization. https://doi.org/10.3389/fpsyg.2026.1903971
- Live-stage case studies show that interaction is also a rehearsal and interface-design problem. Musicians may constrain or anticipate a model's behavior; audiences may expect more spontaneity. Evaluate the actual musician-agent practice, not only the model's generated MIDI. https://doi.org/10.1145/3742413.3789104

### Competitive positioning

Do not claim that Melothaea will beat Suno or Lyria on raw full-song audio fidelity unless a matched listening benchmark demonstrates it. Do not claim to beat JAM_BOT on real-time improvisation before latency, tempo-following, role adaptation and expressive-performance tests are run.

The nearer, defensible target is **proof-carrying reciprocal musical work**:

1. The system understands a score-level musical object and can propose bounded, theory-valid edits.
2. A collaborator can accept, reject, adapt, abstain or branch without losing the alternative.
3. Every result is linked to exact source/result scores, performances and rendered audio.
4. Every persistent or external use has its own authenticated purpose-specific permission.
5. A blinded experiment shows that reciprocal adaptation changes both partners' subsequent actions and improves a musician-valued outcome relative to matched baselines.

The strongest architecture is therefore hybrid: use symbolic score/constraint reasoning for inspectable musical intent and invariants; use expressive renderers or generative audio services as explicitly identified realization backends; and keep identity, consent, provenance and evaluation independent from whichever generator is fashionable. Pin every external model/version, API date, configuration and terms-of-use snapshot in any comparison.

## 13. Sequencing and exit conditions

1. Freeze the wire contract and an independent verifier before writing Studio routes.
2. Add a multi-parent artifact-edge store while preserving existing root-manifest and keeper compatibility.
3. Add identity and consent verifier interfaces with unavailable defaults.
4. Implement transaction journaling, idempotency and restart recovery.
5. Integrate one proposal/adapt/branch → render → listen path.
6. Add session-local MEL-012 appraisal linkage; add persistent memory only after its separate authorization path is qualified.
7. Run the blinded reciprocal-improvisation experiment with exact artifact lineage shared across all experimental arms.

**Exit condition:** a fresh process can independently reconstruct and verify a plural co-creation session from durable artifacts, show every authorized contributor influence and branch, prove exact byte lineage, and demonstrate that no unauthorized downstream use occurred. Until then, describe MEL-013A as a design/implementation work package—not a production capability.
