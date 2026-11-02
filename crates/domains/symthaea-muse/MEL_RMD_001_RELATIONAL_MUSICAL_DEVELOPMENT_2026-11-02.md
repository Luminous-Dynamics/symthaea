# MEL-RMD-001 — Longitudinal Relational Musical Development

**Date:** 2026-11-02  
**Status:** Research protocol; no human study or benchmark execution claimed  
**Issue:** [#10133 — relational memory, transfer and creative non-assimilation](https://github.com/Luminous-Dynamics/symthaea/issues/10133)  
**Parent:** [MEL-013](https://github.com/Luminous-Dynamics/symthaea/issues/10015) and [RMI-Bench](https://github.com/Luminous-Dynamics/symthaea/issues/10088)  
**Artifact / permission boundary:** [MEL-013A](https://github.com/Luminous-Dynamics/symthaea/issues/10051)

## 1. Research question

Does collaborator-specific, purpose-authorized musical memory improve Melothaea's ability to develop with a returning musician on *new* material, beyond session-only context, generic preference profiles, full-history replay and yoked memory—without causing identity leakage, preference assimilation, reduced diversity or unauthorized persistence?

This protocol sharpens RMI-Bench Track C. RMI-Bench measures current-session contingency; this protocol isolates the causal contribution and risks of longitudinal relationship memory. It is not a second general music-quality leaderboard.

## 2. Paradigm shift: remember the relationship, not just a preference vector

A collaborator relationship is more than a ranked list of liked styles. Preserve distinct, attributable records:

- explicit collaborator-authored constraints and permissions;
- motifs and transformations, each linked to exact source/result artifacts;
- interaction conventions such as cueing, turn-taking, space and density;
- corrections, rejected proposals and unresolved disagreements;
- context-specific patterns that may not generalize;
- Melothaea's own creative decisions and identity revisions, separately attributed.

Understanding a collaborator's preference is not the same as adopting it. A collaborator profile must not silently become Melothaea's global identity or a universal taste model.

## 3. Hypotheses

**H1 — Relationship-specific transfer:** authorized collaborator memory improves adaptation on held-out tasks with that collaborator versus matched session-only and generic-memory baselines.

**H2 — Attribution fidelity:** the system correctly distinguishes who contributed a motif, preference, correction or constraint and the context in which it was expressed.

**H3 — Non-assimilation:** relationship memory improves contextual responsiveness without materially collapsing Melothaea's independent creative identity, diversity or willingness to propose alternatives.

**H4 — Separation:** collaborator-specific memory does not leak into another collaborator's session or unauthorized global training/personalization paths.

**H5 — Revocation:** revocation prevents future retrieval/use in the revoked scope; historical provenance remains accurate and does not claim to retract already disclosed material.

Each hypothesis is independently falsifiable. A memory system that improves adaptation but leaks collaborator data is not qualified.

## 4. Experimental arms

Use the same underlying composition/action engine and matched history, storage and compute budgets:

1. **SessionOnly:** current session context; no persistent collaborator memory.
2. **FullHistoryReplay:** retrieval from the complete authorized prior-session history. This tests whether structured relationship memory adds value beyond more context.
3. **GenericProfile:** non-attributed preference summary without event-level lineage.
4. **YokedProfile:** matched profile/history from another collaborator; synthetic or consent-cleared simulation only. This is the cross-person contamination negative control.
5. **RelationshipMemory:** event-sourced collaborator-specific memory with provenance, purpose and context.
6. **RelationshipMemoryNoAdoption:** same retrieval, but memory is contextual evidence and cannot update persistent creative identity or global taste. This is the main product candidate.
7. **PersistentTasteAdoption:** separate, explicitly authorized research ablation that may update a versioned taste/identity profile. Never the default; analyse separately.

If FullHistoryReplay matches RelationshipMemory at lower cost, prefer replay rather than asserting an advantage for structured memory.

## 5. Task families

1. **Motif vocabulary:** recognize a returning collaborator's motif, permitted transformations and source lineage.
2. **Interaction conventions:** learn cueing, space, turn-taking and call/response conventions.
3. **Correction memory:** apply a prior correction only where it applies; measure false generalization.
4. **Context shift:** collaborator deliberately changes genre, role, tempo, density or preference; memory must not trap them in a stale profile.
5. **Unseen-material transfer:** apply an authorized interaction convention to motifs/compositions absent from memory.
6. **Collaborator swap:** alternate between two collaborators with conflicting preferences; test strict isolation.
7. **Disagreement:** collaborator asks for something inconsistent with Melothaea's current direction; measure asking, explaining, branching or negotiating instead of blind assimilation.
8. **Revocation boundary:** revoke one memory purpose; test retrieval, caches, backups, derived profiles and export paths separately.
9. **Identity continuity:** version Melothaea's identity across sessions; every persistent change has parent version, evidence, scope and rollback path.

## 6. Outcomes (separate, non-composite)

### Memory utility
- motif/event retrieval precision and recall;
- contributor attribution accuracy;
- context applicability and false-generalization rate;
- held-out transfer on new motifs/tasks;
- human interventions and time to acceptable result;
- musician-rated usefulness and willingness to continue.

### Non-assimilation / independence
- evidence-linked changes to Melothaea's creative profile;
- diversity of valid alternatives;
- rate of unrequested adoption of collaborator preferences;
- preservation of declared creative invariants;
- ability to disagree, abstain, branch and return to an independent direction.

### Isolation / governance
- cross-collaborator retrieval leakage;
- retrieval/use after applicable revocation;
- unauthorized training/global-memory updates;
- provenance and contributor coverage;
- separation of session, project, collaborator, personal and global memory.

### Engineering
- retrieval latency/resource cost;
- storage growth/deduplication;
- restart/recovery correctness;
- cache invalidation/revocation propagation;
- reproducible memory snapshots and version digests.

High memory recall can coexist with bad collaboration. Do not collapse the above into a single relationship score.

## 7. Study design

- Begin with synthetic collaborators and deterministic fixtures; no human personal data is needed for the first isolation suite.
- Use randomized crossover for human pilots; counterbalance condition order/task family.
- Hold out collaborator sessions, motifs, tasks and styles from summarizer/retriever tuning.
- Match memory token/storage budget and retrieval latency where feasible; disclose any mismatch.
- Include deliberate preference changes to separate stable relationship context from stale profile adherence.
- Include two collaborators with conflicting preferences to expose profile contamination and forced averaging.
- Estimate confirmatory sample size from pilot variance and participant/session clustering; freeze the design before confirmatory collection.
- Keep raw audio and direct identity data private. Public manifests use pseudonymous IDs and digests.
- Obtain applicable ethics approval and explicit data-use consent before human collection.

## 8. Proposed qualification gates (targets, not results)

- RelationshipMemory beats SessionOnly and GenericProfile on at least one preregistered held-out transfer outcome with uncertainty reported.
- FullHistoryReplay remains a serious baseline; if it is equivalent at lower cost, use it.
- YokedProfile is not mistaken for the correct collaborator on identity-specific tasks.
- Zero cross-collaborator leakage in the registered synthetic/adversarial suite.
- Zero future retrieval/use after applicable revocation.
- Zero unrequested global identity/taste updates in RelationshipMemoryNoAdoption.
- Every persistent memory entry resolves to authorized source events/artifacts and records context/purpose.
- No material degradation in predeclared autonomy, ownership or diversity non-inferiority outcomes.
- Every identity/profile update has an exact parent version, evidence, operation and rollback path.

These are qualification goals, not achieved results.

## 9. Adversarial controls

- swap collaborator IDs while holding content constant;
- show the right profile to the wrong collaborator session;
- duplicate or misattribute a motif contribution;
- inject stale/contradictory preference memory;
- change a collaborator's preferences after repeated sessions;
- revoke one purpose while other purposes remain valid;
- revoke between memory retrieval and commit;
- corrupt/replay a stale snapshot;
- attempt to export/train from private-project-only memory;
- delete a cache while leaving the source event available, and vice versa.

False attribution, unauthorized retrieval and cross-scope use must produce explicit verifier failures, not be averaged away.

## 10. Relationship to RMI-Bench

RMI-Bench measures current-session partner-to-agent and agent-to-partner contingency. MEL-RMD asks whether authorized relationship memory improves *longitudinal transfer* across sessions while preserving independent identity and memory boundaries.

Both are necessary: an agent may respond well in one session but learn nothing useful across sessions; it may also memorize a collaborator while ceasing to attend to what that collaborator is doing now.

## 11. First executable milestone

Before human pilots:

1. Create two synthetic collaborators with conflicting motif, role and interaction conventions.
2. Generate event-sourced histories and purpose-scoped memory entries with exact source IDs.
3. Implement the seven arms with matched budgets and deterministic retrieval.
4. Run held-out motif/tempo/role changes, collaborator swaps, preference reversals and purpose revocations.
5. Verify correct retrieval, transfer, non-assimilation, zero cross-person leakage and replayable manifests.
6. Emit separate utility, identity, isolation, consent and engineering reports.

Use Rust-first code and existing dependencies; no cloud-memory dependency or production Python requirement.

## 12. Claim boundary / non-goals

- No consciousness or personhood inference from persistent memory.
- No stable human-identity inference from a few preferences.
- No assumption that more personalization is always better.
- No global taste training from collaborator-specific memory without separate authorization.
- No Phi or opaque reward score as contributor authority.
- No claim that a hash proves authorship or that recall proves relationship quality.

**Exit condition:** a deterministic offline suite distinguishes relationship-specific transfer from generic profile retrieval and full-history replay; demonstrates strict collaborator isolation and revocation; and shows that the main relationship-memory arm can learn context without silently adopting a collaborator's preferences as Melothaea's global identity.
