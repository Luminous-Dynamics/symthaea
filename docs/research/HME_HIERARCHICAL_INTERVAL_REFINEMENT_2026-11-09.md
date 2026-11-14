# HME Hierarchical Interval Refinement — 2026-11-09

## Status

Research prototype in \`crates/domains/symthaea-memory/src/hme/\`:

- \`group_envelope.rs\`: group coordinate bounds, centroid/L2 radius, generation and content digests, member payload digests, and exact query-vector digest.
- \`interval_select.rs\`: weighted group selection, opaque in-memory member residual receipts, per-member interval intersection, and two-stage refinement.
- \`mod.rs\`: re-exports the current prototype APIs.
- Synthetic fixture: \`fixtures/hme_hierarchical_interval_selection_v0.json\`.

**No Rust build or tests have run in this environment.** The code and Rust tests are authored, not executed. There is no claim of an HME benchmark win or a production exact-completion receipt.

## 1. The two-level theorem

Let the complete eligible corpus be partitioned into disjoint groups \(G_1,\dots,G_m\). Group \(g\) has \(n_g\) members. Suppose every canonical score of every member in group \(g\) is enclosed by one valid interval:

\[
  s(x) \in [L_g,U_g], \quad x \in G_g.
\]

Construct a weighted multiset containing \(L_g\) exactly \(n_g\) times. Let \(\theta_1\) be its \(k\)-th largest value.

At least \(k\) eligible records have a lower bound at least \(\theta_1\). Consequently, every record in a group with \(U_g < \theta_1\) is below at least \(k\) records and cannot be in exact top-k. Retain every group satisfying \(U_g \ge \theta_1\); equality must remain.

For each retained member \(x\), suppose a valid per-record interval exists:

\[
  s(x) \in [l_x,u_x].
\]

Intersect it with its parent group interval:

\[
  [l'_x,u'_x] =
  [\max(l_x,L_g),\min(u_x,U_g)].
\]

Both source intervals contain the same canonical score, so their intersection also contains that score. Moreover, at least \(k\) records have \(L_g \ge \theta_1\), and therefore their intersected lower endpoints remain at least \(\theta_1\). Let \(\theta_2\) be the \(k\)-th largest \(l'_x\) over all members of retained groups. Then \(\theta_2 \ge \theta_1\). Any member with \(u'_x < \theta_2\) can be excluded. The original payloads of only the final retained member records need exact scoring.

The two levels are compositional: group exclusion proves that the discarded groups cannot contain a winner; per-record exclusion proves that the discarded members of surviving groups cannot contain a winner.

### Preconditions that are not optional

The theorem is deterministic only when all of these hold:

1. Group intervals enclose the canonical score—not an approximate score or a real-cosine surrogate.
2. The groups form an exact, disjoint partition of the eligible records.
3. The group set is complete for the same pinned manifest and eligibility generations.
4. Every group interval and member interval uses the same exact query vector and score contract.
5. Every member receipt binds the exact original payload, compact reconstruction, residual radius, transform/codebook profile, parent group and generation.
6. Authorization, bitemporal validity, forgetting and deletion eligibility are decided before group membership is formed. No mixed-policy or mixed-time group can be treated as eligible wholesale.
7. Only strict \(U < \theta\) comparison excludes. Equality is unresolved.

A group synopsis is not itself proof of group membership, authorization, or a residual bound.

## 2. Implemented prototype path

### Stage A — group synopsis selection

\`GroupScoreIntervalRecord::from_group_envelope\` computes a query-specific interval from the existing component-box and joint-L2 group envelopes. It binds the interval to:

- manifest generation;
- eligibility generation;
- member-set digest;
- payload/content digest;
- exact query-vector digest.

\`select_group_refinement_set\` uses weighted lower endpoints. It rejects malformed intervals, empty groups, duplicate group identities, overlapping member IDs, inconsistent membership metadata, mismatched snapshot generations, and mismatched query digests. It returns snapshot identities, the threshold, and member counts for refined/excluded groups.

### Stage B — member intervals from admission receipts

\`VerifiedMemberResidualCertificate::verify_at_ingest\` reads the original payload at admission/build time and independently checks:

\[
  \|x-\hat{x}\|_2 \le \epsilon.
\]

The in-memory receipt binds the original payload digest, reconstruction digest, residual radius, transform-profile digest, member ID, and group snapshot.

At query time, \`GroupMemberScoreIntervalRecord::from_verified_certificate\` takes the compact reconstruction and opaque receipt. It does not accept the original payload as an argument. It verifies the group identity, original digest in the group envelope, reconstruction digest, and transform-profile digest before propagating the residual bound through the canonical score interval kernel.

The original payload is loaded only for members that survive the second interval selection and need exact canonical scoring.

### Stage C — interval intersection and exact rerank set

\`select_hierarchical_refinement_set\`:
1. computes the weighted group plan;
2. requires one member interval for every member of every retained group;
3. rejects duplicate member intervals, foreign group identities, mixed query digests, incomplete member coverage and disjoint interval intersections;
4. intersects each member interval with its parent group interval;
5. runs the existing global per-record selector over the retained-group population.

The returned plan is not an \`ExactComplete\` receipt. A caller still needs original-payload exact scoring for every final refinement ID, canonical stable-ID tie handling, and a completion receipt bound to the pinned snapshot.

## 3. Important remaining architectural gap: compact partition proofs

The current in-memory \`GroupScoreIntervalRecord\` retains copies of the member-ID and per-member-payload-digest arrays. The group selector also scans those member IDs to reject overlap. Thus:

- it avoids reading original vector payloads for group-excluded records;
- it reduces exact payload scoring when group and member bounds are useful;
- but its current membership validation is still \(O(N)\) in member metadata, not \(O(G)\) in group headers.

Do **not** claim that the current code scans only \(O(G)\) metadata or that it already delivers the I/O profile of a persistent group index.

The next storage-level design should replace per-query materialization of every group member list with a reusable, manifest-bound **Group Membership Partition Certificate**. At segment build/reopen it must verify that the group member sets are pairwise disjoint and their union equals the complete eligible set. It should bind:
- manifest and eligibility generation;
- eligible-member-set root/digest;
- ordered group member-set and content digests;
- group count and eligible member count;
- score-contract and segment format versions.

The query path can then verify the partition certificate once and scan compact group headers in \(O(G)\) memory. Stage B loads membership and residual receipts only for groups that survive Stage A. Reopen must authenticate the certificate and receipt table against the immutable segment manifest; a plain digest without authenticated provenance is not enough.

This belongs with HME segment ownership in #10917. Until this exists, the current selector is a useful in-memory proof prototype, not the intended storage architecture.

## 4. Receipt persistence is a separate proof obligation

The current \`VerifiedMemberResidualCertificate\` is an opaque in-memory Rust object. It is not yet a serialized, signed, checksummed or reopen-authenticated artifact.

A persistent receipt format needs an explicit version and canonical serialization. It must bind:
- segment UUID and generation;
- group identity and member ID;
- original payload digest and canonical encoding/dimension;
- reconstruction digest and quantizer/codebook/transform profile digest;
- residual metric, radius, verifier version and numeric contract;
- admission receipt creation/version and integrity/authentication chain.

On reopen, the receipt must be rejected if any binding differs. A segment digest is an integrity binding, not proof that the verifier actually checked the residual; that verification must be performed at admission and its resulting receipt authenticated by the segment protocol.

The transform-profile digest must come from the trusted codec/segment profile, not a free-form query-time caller label. Residual inequality independently protects interval correctness, but profile identity is needed for reproducibility and proper codec provenance.

## 5. Synthetic selector experiments

All numbers below come from separate deterministic JavaScript simulations of interval selection logic, not from Rust execution, HME payloads, a disk benchmark, or a measured workload.

### Arbitrary group layouts

A 4,096-case simulation with random exact scores, random per-record interval slack, group sizes 1–6 and group bounds equal to the hull of member intervals observed:
- zero disagreements against full exact top-k;
- mean 24.51 records in a corpus of 24.51/query;
- 98.6% of group members in retained groups;
- the per-record selector retained 68.5% on the same fixtures.

This is a deliberately poor partitioning regime. It shows that group weighting alone does not make pruning effective.

### Tight, score-coherent group envelopes

A separate 500-query simulation used 100 groups × 10 members (1,000 eligible records/query), within-group score variation ≤0.015, wider per-member residual uncertainty, and group intervals verified to contain every generated exact score. For top-10:
- zero observed false-prune disagreements;
- 3.74% of records survived Stage A to member-metadata/refinement processing;
- 3.74% survived Stage B for original-payload exact scoring;
- the per-record-only interval baseline retained 12.36% for exact scoring.

This is a useful hypothesis about the value of score-coherent groups, not evidence that current HME data will have those properties. It also does not include the current \(O(N)\) group membership metadata pass, disk overhead, build/compaction cost, CPU time, or memory pressure.

The next benchmark must measure actual group-bound width distributions on real HME data, not just synthetic score clusters.

## 6. Group construction is now an optimization problem

A good group must be compact under the score geometry while remaining homogeneous under eligibility and update policy.

Proposed group-construction objectives:
- minimize expected query-time weighted refinement count and cold payload bytes;
- minimize group interval width and residual interval width jointly;
- avoid grouping across tenant/authorization, valid-time scope, retention/forgetting state, or score-contract boundary;
- constrain update amplification and group rebuild costs;
- choose an adaptive group size based on corpus size, top-k, query mix, local score dispersion and metadata/payload ratio.

Candidate dimensions for clustering are not merely semantic embeddings. They include verified centroid/radius, coordinate extrema, residual-code profile, query workload signatures, and lifecycle/update locality. Query logs may tune grouping, but they may never change the canonical score guarantee or omit eligible groups.

A useful objective to benchmark is:

\[
  C =
  \lambda_p B_{\mathrm{cold\ payload}}
  + \lambda_m B_{\mathrm{metadata}}
  + \lambda_c N_{\mathrm{canonical\ scores}}
  + \lambda_l L_{p99}
  + \lambda_u C_{\mathrm{updates}}
\]

Publish each term separately. Do not hide a regression in update/compaction cost behind QPS.

## 7. Promotion scorecard

Targets below are project aspirations, not industry standards or achieved results.

| Dimension | Gate / target |
|---|---:|
| Exact top-k agreement with full canonical scan | 100% |
| False group/member exclusions | 0 |
| Snapshot/query/policy mismatch | fail closed, 100% |
| Missing, overlapping, or duplicate group membership | fail closed, 100% |
| Receipt/reconstruction/codec mismatch | fail closed, 100% |
| Cold payload bytes | target ≤10% of eligible bytes for top-10 on a qualified compact-cluster workload |
| Member receipt/reconstruction bytes | publish separately; initial target ≤10% of corpus records read for top-10 |
| p99 latency | no more than 1.2× exact/strong ANN baseline at matched quality; aspirational pending workload |
| Update/rebuild cost | report per mutation and after 10%, 50%, 90% corpus churn |
| Memory and storage | include group synopsis, membership proof, receipts, reconstructions, raw payload and all indexes |

The 10% I/O target is not an ANN recall target. It is a payload-byte target under a workload whose group-bound distribution has been independently measured. Exactness remains a hard gate.

## 8. External comparisons

- **Recast** is the closest storage/refinement comparator. It keeps compact codes in memory, retrieves raw vectors from storage only for refinement, and reports 1.56–2.01× QPS over its strongest matched-memory baseline at 0.99 recall, with 1.22–6.75× less external storage in its tested setups. HME must compare at matched quality, data, memory, update/delete rate and payload bytes.
- **PQR-Search** is the per-query probabilistic recall comparator for dynamic ANN. Its \(\Pr[\mathrm{Recall}(q)<\tau]\le\delta\) contract is probabilistic, not deterministic complete top-k. HME should report exact and probabilistic tracks separately.
- **QBAT** is the query-specific budget-allocation comparator. A learned scheduler can predict which group/member to inspect first, but only verified intervals may eliminate records.
- **MEME, DyadMem, LongMemEval, MemoryAgentBench and AML** test agent-memory utility, lifecycle and capture/revision, not vector-kernel speed. Run them with the same frozen reader, evaluator and context budget.

References:
- Recast (2026): https://arxiv.org/abs/2611.02765
- PQR-Search (2026): https://arxiv.org/abs/2611.07579
- MEME (2026): https://arxiv.org/abs/2605.12477
- DyadMem (2026): https://arxiv.org/abs/2610.03020
- Agent Memory Leaderboard: https://agentmemories.ai/evaluation
