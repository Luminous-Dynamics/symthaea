# WorldFailBench boundary-rate adapter v0.1

**Status:** normalized admission contract and synthetic fixture only.  
**Dataset status:** metadata-only / not admitted.  
**Claim ceiling:** no WorldFailBench run, scorer reproduction or Symthaea-vs-baseline result is claimed.

## Source inspection

The public DAXIAORobotics/WorldFailBench README describes:

- RoboTwin evaluation metadata with policy configuration, task, seed, success label, instruction, split and ordered stages;
- LIBERO metadata with policy variant, task ID, initial-state ID, language, suite, success and ordered stages;
- boundary subsets separated by platform and perturbation dimension (position, orientation, size);
- 60 LIBERO boundary pairs and 60 RoboTwin boundary pairs, for 120 pairs total;
- ten reference rollout outcomes per side, giving 2,400 boundary rollouts.

The sample project site separately states that its original evaluation code is not yet released. The data documentation and metadata are not the original benchmark scorer.

## Important semantics

A WorldFailBench scene-perturbation pair does not assert that the reference policy emitted identical actions in both variants. The two scene variants are executed by the same named reference-policy configuration; its closed-loop action sequence may diverge after the scene perturbation. Therefore:

- bind the same reference-policy configuration/checkpoint across one evaluation;
- bind each side's distinct scene snapshot and the common pair source manifest;
- keep each side's success count and trial denominator;
- keep each policy configuration in a separate evaluation;
- do not claim an identical-action counterfactual unless action traces are independently available and matched.

The normalized record maps to `BoundaryRatePair`:
- `pair.pair_id` → `pair_id`;
- `domain_id`, `environment_family_id`, `environment_cluster_id` → cluster/domain identity;
- `pair.pair_source_manifest_sha256` → pair source manifest digest;
- `scene_sides.success_side.scene_sha256` and failure-side equivalent → side-specific scene digests;
- `policy_configuration_id` and `reference_policy_checkpoint_sha256` → frozen reference-policy identity;
- `candidate_checkpoint_sha256` → frozen candidate checkpoint;
- each side's reference successes/trials → corresponding reference counters;
- each side's candidate-predicted successes/trials → corresponding candidate counters;
- perturbation dimension/values → normalized perturbation identity.

The schema additionally carries dataset ID/revision/archive digest/license. Optional action trace digests may be recorded when available; their absence must not be presented as proof of equal actions.

## Admission blockers

1. **Archive bytes:** the README links `dataset.tar.gz`, but the archive SHA-256 has not been locally verified. Do not use a GitHub repository SHA as an archive digest.
2. **Policy weights:** the README names variants such as `pi05`, `pi05-data-20`, `fastwam`, `lingbot-vla-v2`, and LIBERO `pi05_data20/50/100`, but the public episode schema does not include a policy-checkpoint SHA-256. Resolve the exact checkpoint/commit per policy variant or keep evaluation at metadata-only status.
3. **Candidate identity:** pin candidate model checkpoint, all model assets and action/video prompt/configuration.
4. **Pair source:** compute source manifest digest from the exact pair metadata and both side scene files; compute each scene digest from canonical bytes.
5. **License scope:** the HF dataset card and dataset repository declare Apache-2.0. Verify all linked assets and any policy checkpoints under their own licenses; don't assume the dataset license covers them.
6. **Scorer:** the project's public README states evaluation code is not yet released. The CMAQ-020 rate score is an independently defined diagnostic; it is not the original WorldFailBench scorer.
7. **Statistics:** ten rollouts per side remain repeated observations. Cluster/pair is the unit for inference; #8578 owns cluster-level resampling and confidence intervals.

## Public reference result snapshot

The public RoboTwin table reports its Ref–Ref diagnostic TFE/SFE/TPC/SPC of 0.061/0.094/0.814/0.903. Across the seven model rows, DW05 has the lowest TFE (0.159) and highest TPC (0.402); Ctrl has the best model SPC (0.095). These remain four separate metrics; no overall score is constructed. No Symthaea score is available.

## Files

- Schema: `worldfailbench-boundary-rate-pair-v0.1.schema.json`
- Synthetic mapping fixture: `fixtures/worldfailbench-boundary-rate-pair.example.json`
- Source dataset docs: https://github.com/DAXIAORobotics/WorldFailBench
- HF dataset card: https://huggingface.co/datasets/ACERobotics/WorldFailBench
- Project site / published results / scorer status: https://tangxiaoyang18.github.io/WorldFailBench/
