# PITB proof-impact analysis and proof-maintenance implementation

**Date:** 2026-11-03  
**Status:** executable synthetic analyzer added; no live Lean snapshot, repair run, stale-proof campaign or measured savings claimed.

## Paradigm shift: verified proofs are maintained assets

Proof generation is only the point-in-time act of producing a kernel-checked term. A deployable assurance system must also know whether that proof remains valid after source, dependencies, imports, instances, toolchains, axioms, or semantic review evidence changes.

The unit of operation should be a **versioned assurance graph**:
- theorem/definition/instance/module nodes;
- typed evidence edges with direction and provenance;
- source, statement, body, proof, axiom, repository and environment digests;
- impact sets and action classifications;
- replay, repair, rollback and release receipts.

A repository change becomes a graph transformation. The system should identify the minimal *safe* revalidation frontier, preserve unrelated verified work, repair invalidated obligations, and replay the complete required repository state before release.

The goal is not the fewest rebuilds at any cost. It is the lowest-cost maintenance plan with zero stale-proof acceptance and honest treatment of unknown dependencies.

## Why one dependency edge is not enough

Snapshot edge orientation: `dependent/source -> dependency/target`.

| Relation | Change in target dependency | Required action on source consumer |
|---|---|---|
| `type_reference`, `signature_reference`, `structure_field_reference`, `inheritance_reference` | Type/signature/structure changes | Re-elaborate and re-prove |
| `instance_synthesis` | Instance declaration or synthesis environment changes | Re-elaborate and re-prove |
| `definition_body_reference` | Transparent body changes | Re-prove; downstream reduction may differ |
| `definition_body_reference` | Opaque body/proof change with unchanged type | Revalidate conservatively; do not infer type drift |
| `proof_term_reference` | Referenced proof/axiom evidence changes | Revalidate dependent proof under the updated environment; occurrence is not logical necessity |
| `module_environment_dependency` | Module/import/options/elaborator/build input changes | Rebuild the dependent environment closure |
| `documentation_reference`, Blueprint/informal/cross-formality candidate | Retrieval source changes | Refresh retrieval only; never invalidate formal proofs solely from a candidate edge |

A changed proof body behind an opaque declaration does not change the declaration's type. Type-only consumers should not automatically be reproved, but proof-term users must be revalidated against the updated trusted declaration and axiom footprint. A transparent definition body may be unfolded by consumers and can therefore force reproving. The analyzer keeps these cases separate.

Global Lean environment digest drift is treated as a rebuild of every formal declaration. Incomplete dependency inventories trigger `unknown_conservative` for every formal declaration. Unknown does not mean unaffected.

## Current executable component

The Rust CLI is `tools/pitb-proof-impact/`:
- strict snapshot schema with repository commit and global environment identities;
- typed declaration/module/document nodes and registered relation vocabulary;
- exact node/edge snapshot diff;
- direct and transitive dependency impact propagation;
- action precedence: `retain`, `retrieval_refresh`, `source_revalidate`, `revalidate`, `reprove`, `rebuild`, `unknown_conservative`;
- report with raw snapshot digests, changed nodes/edges, reasons, and action counts;
- synthetic self-tests for no-op, signature change, proof-body-only change, transparent body change, environment change, doc-only change, and incomplete inventory.

It does **not** yet:
- authenticate old/new snapshots or the completeness claim;
- independently extract dependencies from a real Lean environment;
- decide whether a proof term is logically necessary;
- repair proof obligations or re-run Lean;
- validate source-to-statement semantic correspondence;
- issue a release or assurance certificate.

Use it only as a conservative planning tool over already authenticated snapshots. The signed snapshot-attestation workstream currently authenticates the issuer and exact graph bytes; independent Lean re-extraction remains a separate obligation. See [typed graph evidence](https://github.com/Luminous-Dynamics/symthaea/pull/10317) and [graph snapshot attestation](https://github.com/Luminous-Dynamics/symthaea/pull/10336).

## Competitive baseline design

CIPM should compare the same parent repository snapshot, patch bytes, toolchain, verifier and total compute/cost budgets across:

1. **Full rebuild/reproof:** conservative reference policy; highest baseline cost but a useful safety control.
2. **Changed-file heuristic:** only rebuild affected files or modules; tests whether file-level heuristics are sufficient.
3. **Untyped dependency closure:** conservative reverse graph traversal without relation semantics.
4. **Typed relation-aware impact:** this analyzer's policy-driven dependency classification.
5. **Predictive/LLM impact classifier:** optional advisory model that can rank or suggest extra affected nodes but cannot remove hard-rule impacts or promote unknown to unaffected.
6. **Full maintenance controller:** typed impact + selective repair + independent whole-repository replay.

The strongest safe baseline is selected on development patches and frozen before confirmatory mutation families. Model suggestions must never shrink a mechanically required invalidation set.

## Change families and controls

Seed deterministic patches for:
- no-op and irrelevant-file changes;
- source-preserving formatting/refactor;
- theorem statement/signature changes;
- opaque proof-body changes with identical type;
- transparent definition body changes;
- typeclass/instance changes and overlapping-instance edits;
- structure fields and class inheritance;
- helper lemma changes with 1, 2, 4, 8 and 16 downstream consumers;
- import/module/options/notation/macro/elaborator/Lake/Lean toolchain changes;
- axiom-policy and `sorryAx` mutations;
- stale proof and stale receipt replay;
- missing/unknown graph edges and deliberately incomplete inventories;
- documentation/Blueprint/informal extraction-only changes.

Each mutation binds the parent snapshot, patch digest, child snapshot, exact dependency inventory, expected changed nodes, expected invalidation set and independent replay result. Include a hidden project-disjoint set and a later temporal snapshot; never tune against confirmatory patches.

## Metrics and proposed gates

| Metric | Qualification goal |
|---|---:|
| Stale proof acceptance | 0 on the registered mutation suite |
| Stale receipt replay acceptance | 0 |
| Critical-change impact recall | 100% on curated critical controls before hidden evaluation |
| Hidden impact recall | ≥0.99 with cluster-aware interval |
| Impact precision | ≥0.95 on adjudicated changes |
| Unaffected-proof preservation | ≥0.95, conditional on successful independent whole-state replay |
| Full required repository replay | 100% of counted maintained changes |
| Cost per verified maintained change | ≥20% lower than full rebuild at non-inferior safety |
| Change-to-verified-green latency | Report p50 and p95; target ≥20% p95 improvement only if total resources do not increase |
| Invalidation localization | Report direct and transitive impact separately, and all false negatives individually |

Finite-suite 100% controls must include denominators and confidence bounds; they are not universal guarantees. The inference unit is a repository/project or independently sampled patch family, not individual graph edges.

## Research sequence

1. Qualify the isolated synthetic analyzer and exact report schema.
2. Capture a real Lean parent graph with environment, instance and module evidence; authenticate snapshot issuer; independently re-extract the graph to establish completeness.
3. Generate a controlled shared-helper mutation cohort; compare exact invalidation closure against an independently adjudicated oracle.
4. Add instance, transparent-definition, opaque-proof, source and environment change families.
5. Implement typed repair and selective reuse; keep the full-rebuild arm as a conservative comparator.
6. Require exact-head clean checkout and independent kernel replay of the full required repository state.
7. Publish cost/safety frontiers and temporal/project-disjoint replication.

## Relationship to the frontier

CoCo-Prover demonstrates the value of AND/OR proof scheduling and bounded specialist actions, but its headline cost result is on program-verification cohorts and does not itself establish safe proof maintenance under repository evolution. Aegis adds implementation-derived obligations; Vero and VeriSoftBench measure repository-level completion. CIPM asks a different question: what happens to already-verified claims after the software and mathematical environment change?

PITB's opportunity is to unify:
- DAPS/LeanGraphSearch/TheoremGraph retrieval as candidate discovery;
- typed AND/OR support bundles and proof DAG planning;
- source/environment integrity and authenticated receipts;
- exact dependency impact and selective proof maintenance;
- independent full-state replay.

Keep benchmark scorecards separate. A better retrieval recall or initial proof pass rate does not demonstrate safe maintenance.

## References

- Existing design-only CIPM proposal: linked in broad PITB PR #8154 and issue #10272.
- CoCo-Prover: https://arxiv.org/abs/2610.09681
- Aegis / EasyProver: https://arxiv.org/abs/2610.38174
- Vero repository completion: https://rdi.berkeley.edu/blog/vero
- PITB executable analyzer: `tools/pitb-proof-impact/`
