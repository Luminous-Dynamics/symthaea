# Independent query-ensemble oracle V1

**Tracking:** [#9344 — Query robustness across plausible connectome reconstructions](https://github.com/Luminous-Dynamics/symthaea/issues/9344)  
**Prerequisite implementation:** [PR #8883 — finite query-ensemble evaluator](https://github.com/Luminous-Dynamics/symthaea/pull/8883)  
**Status:** independent reference implementation and exhaustive differential CI added; no CI result or biological performance claim is implied by authoring this change.

## Why this is a separate assurance gate

The V1 Rust evaluator can validate its input, bind exact graph/scenario digests, and deterministically evaluate a finite scenario set. Those checks do not independently establish that its query implementation computes the declared graph semantics correctly.

This extension adds a second implementation written in Python's standard library. It does not import the production source or reuse its graph data structures. In particular:

- production reachability uses adjacency maps and breadth-first search;
- the oracle uses a Boolean reachability matrix and Floyd–Warshall transitive closure;
- graph and scenario-set digests are recomputed independently from the serialized records;
- every reported per-scenario query value and decision is recomputed and compared, in addition to aggregate ranges and disposition counts.

The Python code is test-only. It is not a production dependency and does not participate in query execution or certificate issuance.

## Exhaustive differential domain

The generator builds the exact finite universe of loop-free directed simple graph topologies on four labeled vertices:

    2^(4(4-1)) = 2^12 = 4,096 graph topologies.

Each possible directed edge has a deterministic, distinct positive weight. The generated input evaluates 36 frozen queries per topology:

- 12 ordered-pair edge-weight queries;
- 4 weighted in-degree queries;
- 4 weighted out-degree queries;
- 16 directed reachability queries, including reflexive source/target pairs.

This gives **147,456 production-vs-oracle query/scenario comparisons** in one bounded synthetic run. The 4-node universe is deliberately small enough to enumerate exactly and broad enough to exercise cycles, disconnected graphs, directionality, alternate paths, absent edges, weighted degrees, and empty graphs.

The verifier also checks graph identity, the canonical scenario-set digest, lineage-group cardinality, exact scenario membership, decision counts, minimum/maximum values, range width, and exact-input digest binding.

## Adversarial and metamorphic controls

The oracle's own negative controls deliberately mutate one output value, the scenario-set digest, and scenario membership. Each mutation must be rejected.

The additional oracle checks cover:

- node-label permutation invariance for reachability;
- edge reversal: incoming and outgoing degree exchange;
- exact threshold-boundary semantics for strict, inclusive, and interval decisions;
- cyclic and disconnected directed graphs through the exhaustive graph universe;
- canonical output ordering and exact scenario membership.

The production known-answer fixture is also checked independently. Its reconstruction-sensitive path query must remain SENSITIVE_TO_RECONSTRUCTION; the oracle does not allow agreement on the other queries to compensate for it.

## Run locally

    cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
      --bin connectome_query_ensemble_audit -- \
      --input docs/neuroscience/examples/CONNECTOME_QUERY_ENSEMBLE_AUDIT.example.json \
      --output query-ensemble-report.json

    python3 tools/connectomics-intake-audit/reference/verify_query_ensemble_oracle.py \
      --input docs/neuroscience/examples/CONNECTOME_QUERY_ENSEMBLE_AUDIT.example.json \
      --report query-ensemble-report.json

    python3 tools/connectomics-intake-audit/reference/generate_exhaustive_query_ensemble_input.py \
      /tmp/connectome-query-ensemble-exhaustive.json

    cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
      --bin connectome_query_ensemble_audit -- \
      --input /tmp/connectome-query-ensemble-exhaustive.json \
      --output /tmp/connectome-query-ensemble-exhaustive-report.json

    python3 tools/connectomics-intake-audit/reference/verify_query_ensemble_oracle.py \
      --input /tmp/connectome-query-ensemble-exhaustive.json \
      --report /tmp/connectome-query-ensemble-exhaustive-report.json \
      --exhaustive

The isolated workflow connectome-query-ensemble-oracle.yml performs the same sequence on a pull request or manual dispatch.

## Qualification boundary

A passing differential run would establish agreement between two implementations for the exact supported V1 query language over the exhaustive four-vertex graph domain, plus the checked known-answer fixture. It would not prove:

- that a scientific query's biological construct or threshold is appropriate;
- that the four-vertex synthetic domain covers production-scale graph behavior, numeric ranges, or all parser/runtime failure modes;
- that a declared scenario universe contains every biologically plausible reconstruction;
- that source imagery, segmentation, synapse annotations, or expert adjudications are correct;
- that any model outperforms ConnectomeBench2, NEURD, VLM or other specialist systems.

A later extension should add exhaustive small-graph motif-query checks when those metric semantics are admitted into the finite-ensemble evaluator. Production-scale numerical, performance, and biological qualification remain separate gates.
