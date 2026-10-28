# DSE-012 — Risk-Controlled Context Acquisition for HDC Retrieval

**Status:** Proposed next experiment; no implementation or qualification claimed.  
**Motivation:** DSE-010 and DSE-011 show why a retrieval policy should distinguish a weak score from insufficient information. DSE-010's identical-byte twins cannot be distinguished by a vector-only function. DSE-011's shared-fact groups show that adding more computation to the same shared cues does not recover omitted identity information.

## Hypothesis

The system can improve useful retrieval quality by learning when to stop scoring and instead request or acquire the most informative additional evidence. The objective is not always “answer with a candidate.” Minimize cost across false identity claims, abstention, user clarification, latency and memory.

## Policy interface

Given query cues, candidates and scores, return one of:
- `AcceptedSemantic`: semantic relevance policy passed;
- `ExactIdentityVerified`: requested ID/version and independent evidence checks passed;
- `CandidateOnly`: useful candidate, not confirmed identity;
- `RequestContext`: name the additional cue type needed (for example, another role-bound fact, record ID, version or provenance field) and estimate how it partitions surviving candidates;
- `NoMatch` or `Abstain`: evidence cannot support the request.

The context selector must only select from declared cue channels and must not invent evidence. Its estimated information gain must be learned/evaluated on a separate partition, not the final test set.

## Synthetic workload

Use compositional records with:
- corpus sizes 1,024 / 4,096 / 16,384;
- bundle arities K = 7 / 15;
- cluster widths 8 / 16 / 32;
- shared slots S sweeping from 0 through K;
- query cue budget M sweeping odd values from 1 to K;
- cue corruption 0 / 1 / 2 / 5 / 10%;
- independent unknown queries, stale versions, revoked/deleted records and absent identities;
- multi-seed, entity-disjoint calibration / validation / test partitions.

Ground truth must include whether identity is identifiable from the available cues. If multiple candidates are indistinguishable under the observed facts, the correct policy is to request more context or abstain, not guess.

## Baselines

1. Bundled-prototype exact search plus score threshold.
2. Top-1/top-2 margin threshold.
3. Slot-aware component index with fixed weights.
4. Component index with weights frozen on calibration data.
5. CandidateOnly/Abstain without active context acquisition.
6. Risk-controlled context acquisition with a declared cost per requested cue.
7. FAISS IndexBinaryFlat for exact Hamming-search performance only, not as a semantic decision-policy baseline.

## Metrics

- Identity Hit@1 and semantic Hit@k, separately.
- False identity acceptance and false semantic rejection by negative profile.
- Selective risk versus accepted coverage.
- Abstention, RequestContext rate, mean and p90 additional cues required.
- Cost-weighted correct-resolution rate: correct resolutions divided by query count after accounting for clarification and false-accept costs.
- Target rank/margin before and after each additional cue.
- Latency, throughput, complete memory footprint, index/build costs and added scans.
- Calibration intervals and results stratified by source seed, cluster size and overlap.
- Direct device energy measurement only after the policy is frozen.

## Go/no-go

- In information-limit controls, never emit ExactIdentityVerified using only shared cues; return RequestContext or Abstain.
- With distinguishing cues available, recover held-out targets at a pre-registered rate and quantify added-cue cost.
- Beat fixed-threshold and fixed-margin baselines on risk-versus-coverage or cost-weighted resolution without weakening exact-search guarantees.
- Reuse a published risk-control method only with its assumptions stated and verified. Selective conformal risk control and fixed-sequence confidence-interval calibration are methodological starting points, not assumed drop-in guarantees.
- All policies use identical query/corpus partitions; threshold tuning on the final test partition invalidates the comparison.

## Evidence boundary

DSE-010 and DSE-011 are synthetic NumPy pilots. Their easy 100% Hit@1 cases do not support a production claim; the shared-cue cases illustrate information insufficiency, not an algorithm defect. This DSE proposes a next experiment and is not a result.
