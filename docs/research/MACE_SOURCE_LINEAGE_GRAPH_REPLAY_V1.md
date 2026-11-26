# MACE Source-Lineage Graph Replay V1

**Status:** Consumer-side structural graph replay is implemented as a Rust module on the `mace/source-lineage-graph-replay` branch. The graph manifest schema is published with this change. Tests are authored but have not been run. This is not source-origin authentication or proof that an operator disclosed every real-world dependency.

## 1. What it closes

The source-lineage receipt verifier (#13700) verifies a third authority's signed statement and checks its exact envelope-byte commitment. That establishes which resolver signed the assertion, but not whether the accompanying graph is internally complete and consistent.

This module introduces a canonical graph manifest whose exact UTF-8 bytes are committed by `source_graph_sha256` in each signed source-lineage receipt. The consumer can now replay the declared parent graph and recompute root-family and lineage-closure digests rather than trusting those digest values as opaque labels.

## 2. Graph contract

The `mace_source_lineage_graph_v1` manifest binds:
- the exact qualified evidence-set digest;
- the consumer-owned DecisionNeed digest;
- sorted evidence nodes;
- source identity and source/payload digests;
- evidence-class digest and the DecisionNeed requirement IDs that the node is claimed to satisfy;
- sorted parent evidence IDs and a derivation receipt digest for every non-root node;
- sorted common-cause group digests.

Candidate nodes are compared against the qualified candidate records. Source ID, source digest, payload digest, parent IDs and derivation receipt must match exactly. An omitted or rebound candidate parent is rejected.

Every parent must exist. Kahn's topological algorithm rejects cycles. A non-root node must have a derivation receipt digest; root nodes must have no parent and no derivation receipt. Unknown ancestry is not silently treated as an independent root.

## 3. Deterministic replay

Root identity is computed as:

`SHA256("luminous:mace:source-root-identity:v1\\0" || LP(source_id) || LP(source_sha256))`

where `LP(x)` is an unsigned 64-bit big-endian byte length followed by the UTF-8 bytes of `x`.

Root families are connected components of root identities joined by any shared common-cause group digest found on their ancestry paths. The family commitment is domain-separated with `luminous:mace:source-root-family:v1\\0`, followed by the sorted root identity digests and sorted common-cause digests.

For each DecisionNeed requirement, the replay starts from qualified candidate nodes explicitly tagged as satisfying that requirement, walks all parent paths, and collects the reachable source roots. The lineage closure digest is domain-separated with `luminous:mace:lineage-closure:v1\\0` and commits the requirement ID, root-family digest, and sorted closure nodes, including source identity, payload, evidence class, derivation receipt, parent IDs and common-cause groups.

The replay result must match the source resolver's signed `source_graph_sha256`, `root_family_sha256` and `lineage_closure_sha256` commitments exactly. The graph's raw envelope-independent digest is taken over the exact UTF-8 graph manifest bytes—not a parsed and reserialized representation.

## 4. Consumer use

The entry point is `replay_source_lineage_graph_v1(graph_json, qualified, qualification_binding, current_qualifier_verifier, verified_sufficiency, sufficiency_use_binding, current_evaluator_profile, decision_need, verified_lineage, current_lineage_profile)`. It calls the lineage set's use-time validation itself, so graph replay cannot rely on a stale verified token. All three authorities are checked against one trusted time value before the graph is processed.

Before accepting the graph, the caller must first verify the source-lineage envelopes under #13700, which itself revalidates qualification and sufficiency at the same trusted use-time. Replay then requires:
- matching qualified-set and DecisionNeed digests;
- every qualified candidate's exact source/payload/parent binding;
- valid evidence-class/requirement tags;
- no dangling parents or cycles;
- exact root-family and closure-digest agreement with the signed receipts.

The graph replay step is only meaningful for a sufficient decision; insufficient/abstained assessments cannot become a valid graph-backed sufficient result.

## 5. Remaining trust boundary

Graph replay proves consistency of the disclosed graph. It does **not** prove:
- that an ancestor's source bytes actually hash to the disclosed digest;
- that every source, parent or transformation was disclosed;
- that common-cause identifiers are factually correct;
- that two sources have no hidden upstream dependency;
- that a source publisher or evidence owner is authentic.

Those require source-origin receipts, immutable content retrieval, and independent source metadata acquisition. An independent graph verifier can recompute a graph only after the source records and derivation metadata are authenticated independently of the resolver and evaluator. The root graph is not made true by being self-consistently hashed.

## 6. Tests authored

The module includes tests for:
- domain-separated source-root identity;
- collapsing common-cause roots with disjoint-set union;
- canonical parser rejection of unsorted/duplicate parent edges;
- rejection of missing derivation receipts;
- rejection of dangling parents and cycles.

A full replay integration fixture and executed Rust tests remain outstanding. No test-pass claim is made.
