# Query-scoped proof-carrying connectome reliability certificate V1

**Status:** design contract; no certificate implementation or biological certification exists.
**Tracking:** [SCI-CONNECTOMICS-012 / #9255](https://github.com/Luminous-Dynamics/symthaea/issues/9255).

## 1. Why global cleanliness is the wrong release contract

A connectome may be stable under global graph invariants while remaining unreliable for a particular scientific question. A dataset-wide “proofread” or “not proofread” label hides local and query-specific uncertainty. Recent graph-invariant stopping work presents global convergence as a useful progress proxy but explicitly notes that it cannot establish progress for every specific scientific question.

The proposed unit of assurance is therefore a scientific query with a declared dependency closure: which volumes, segments, neurons, synapses, edges, morphology subgraphs, source revisions, and corrections can change the answer?

## 2. Certificate schema

Every certificate should bind:

- stable query ID, formal query expression, parameters, query implementation digest, scientific interpretation, and declared applicability scope;
- dataset, volume, imaging, segmentation, annotation, skeleton, mesh, synapse and graph revision digests;
- exact dependency closure across source objects, spatial regions, coordinate transforms, neurons, roots, segments, synapses, edges, morphology subgraphs and derived values;
- correction branch identity, exact applied correction set and order, reviewer/adjudicator records, before/after graph digests, and recomputed query values;
- relevant error families, PSU frame, sampling design, inclusion probabilities, sampled counts, candidate-generation recall bounds and in-pool residual-risk estimates;
- evidence-source ancestry, detector versions, modality missingness, inter-rater disagreement, unresolved and indeterminate cases;
- query-specific null-model version and calibration evidence where statistical significance is asserted;
- MSE/unseen-error sensitivity state, if available, kept separate from observed and design-based estimates;
- exact verifier version, output digest, and reproducible derivation receipt.

## 3. Certificate states

- CERTIFIED_WITHIN_DECLARED_SCOPE: all required data, coverage, uncertainty and scientific-risk gates pass for the specified query and scope.
- CONDITIONAL: a result is usable only under explicitly stated limitations, such as incomplete regional support or a non-critical family with insufficient sample support.
- NOT_CERTIFIED: required evidence, risk bounds, lineage, or calibration is absent or fails.
- INVALIDATED_BY_REVISION: a source, dependency, correction, null-model, or query-code revision changes the dependency closure.

No certificate state is a universal guarantee of biological truth or causality. A digest proves the identity/integrity of recorded bytes, not the correctness of the underlying annotation or biological interpretation.

## 4. Dependency closure and invalidation

Represent provenance as a directed acyclic derivation graph where possible. Source volumes, segmentation snapshots, meshes, skeletons, synapse tables, candidate proposals, review outcomes, applied corrections, query values, and certificates are typed nodes. Edges record the transformation, parameters, software digest, and source coordinates/revisions.

A correction invalidates every certificate whose dependency closure intersects the changed graph objects or source evidence. A certificate must remain valid after edits only when the dependency engine proves those edits cannot affect its query. If dependency analysis is incomplete, invalidate conservatively.

Recomputation is deterministic on the same frozen inputs and tool versions. Preserve historical certificates and supersession links rather than overwriting old evidence.

## 5. Uncertainty composition

Do not simply multiply or add all per-component confidence scores. Identify shared dependence and avoid double-counting common evidence, sampling units, adjudicators, source images, and model lineage.

Separate at least:
- sampling uncertainty from #9204;
- within-candidate residual risk from #9187;
- rater disagreement and reference uncertainty from #9186;
- all-list-missed sensitivity from #9248;
- correction-set interaction uncertainty from #9250;
- morphology / extension uncertainty from #9241 and #9249;
- query/null-model uncertainty from #9156;
- measurement and model uncertainty in the query itself.

Where a defensible joint uncertainty model is unavailable, report components separately and downgrade the certificate state. Do not fabricate an aggregate probability.

## 6. Benchmarks

- Exact dependency closure on enumerable synthetic graphs and query implementations.
- Zero missed dependency edges on the declared exact synthetic suite.
- Zero stale certificates after dependent-object revisions and no invalidation after provably unrelated edits.
- False-certification rate on adversarial synthetic and blinded challenge cases.
- Calibration and empirical coverage of query-error intervals against independently corrected graphs.
- Coverage-risk curves: fraction of queries certified versus independently measured query error.
- Stability under query-preserving transformations and correct invalidation under query-changing transformations.
- Byte-for-byte reproducibility from exact source, verifier and protocol inputs.

## 7. Proposed promotion gates (not results)

- Zero missed dependencies on the finite exact synthetic suite.
- No stale certificates after any dependent source, graph or query-code change.
- At least 95% empirical coverage for nominal 95% query-error intervals in preregistered synthetic finite-population simulations, including cluster dependence and finite-population sampling.
- No critical-query certification when critical-family recall, minimum support, source lineage, or query-specific null-model gates fail.
- Every certificate includes full derivation digests, uncertainty components and scope limitations.

## 8. Relationship to industry systems

ConnectomeBench2 measures task-specific local proofreading decisions. NEURD and RoboEM provide morphology / tracing and correction mechanisms. SynAnno provides structured human workflow. CAVE provides versioned reconstruction and derived annotation update infrastructure. A query-scoped certificate layer composes these capabilities while making the scientific scope, uncertainty, and invalidation semantics explicit.

No certificate implementation or biological certification is claimed by this document.

## Machine-readable contract

- [JSON Schema](CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.schema.json)
- [Synthetic fixture](examples/CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.example.json)

The example is synthetic and intentionally `CONDITIONAL`: a critical coverage gate is not estimable. It is not a biological certificate. V1 forbids a single aggregate-confidence field; uncertainty components remain separate unless a validated joint model exists.


## Semantic verifier

The Rust binary `connectome-query-cert-audit` validates the cross-field invariants that JSON Schema alone does not express:

- every dependency references a source revision digest bound by the certificate;
- all digests are lowercase SHA-256 strings and identities are unique;
- correction order is contiguous and an applied correction requires an `accept` decision;
- a `CERTIFIED_WITHIN_DECLARED_SCOPE` state requires every quality gate to pass and every uncertainty component to be estimated;
- a `CONDITIONAL` certificate cannot conceal a failed critical gate;
- `INVALIDATED_BY_REVISION` requires an explicit event with different previous/current revision digests;
- non-estimable components cannot carry point estimates or intervals.

Run:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-query-cert-audit -- \
  --certificate docs/neuroscience/examples/CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.example.json
```

This is an internal consistency verifier only. It does not fetch source artifacts, independently verify external digest claims, establish the completeness of a dependency closure, or prove biological truth. A successful run means only that the supplied certificate is internally consistent under V1 rules. The example is expected to pass as `CONDITIONAL`, not as a biological certification.


A `CERTIFIED_WITHIN_DECLARED_SCOPE` certificate requires all critical quality gates to be `PASS`; non-critical gates may be `NOT_APPLICABLE` only with a non-empty scope justification. A critical gate marked `NOT_APPLICABLE`, `NOT_ESTIMABLE`, or `FAIL` blocks full certification. A `CONDITIONAL` certificate cannot conceal a failed critical gate. Every uncertainty component must declare at least one explicit assumption; `ESTIMATED` components require a point estimate, while `NOT_ESTIMABLE` components must not carry one or an interval.
