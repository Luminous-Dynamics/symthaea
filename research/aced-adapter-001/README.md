# ACED-REPAIR-ADAPTER-001

**Status:** pinned upstream inventory and metadata qualifier; no model trials or world payload execution  
**Owner:** [Symthaea #11598](https://github.com/Luminous-Dynamics/symthaea/issues/11598)  
**Parent:** [IG-CAUSAL-PRIMITIVES-001 #11505](https://github.com/Luminous-Dynamics/symthaea/issues/11505)

## Research target

Integrate the official MIT-licensed [ACED-Bench](https://github.com/Rose-STL-Lab/ACED-Bench) as an external active causal investigation comparator.

Pinned upstream revision: `998dc692bfcf5574e5319d67f1cea9c170569d80`  
Paper: [ACED v2, arXiv:2610.14381](https://arxiv.org/abs/2610.14381), revision dated 8 November 2026.

ACED-Repair tests whether an agent can investigate a hidden causal system, select useful interventions under a budget, identify working levers and doses, and recommend a fix that is independently re-simulated against the true objective. Its published protocol uses 15 experiments and 32 turns per world.

The pinned inventory contains:
- **540 test worlds** across nine causal archetypes (60 each).
- **54 evaluation worlds** (six per archetype).
- Ten domains across clinical, fermentation, agronomy, aquaculture, battery, bioprocess, catalysis, datacenter, semiconductor and water treatment.

The evaluation index is a **selected subset of the test index**, not an independent disjoint dataset. The qualifier verifies every eval row is an exact file/domain/archetype/seed match in test. Do not pool these sets as 594 independent worlds or use test/eval overlap as independent statistical evidence.

## What is pinned here

The `upstream/` directory contains only:
- `repair_test_index.json`: 540 world filename/domain/archetype/seed records.
- `repair_eval_index.json`: 54 matching selected records.
- `SHA256SUMS`: the upstream hash ledger for the repository's data files.

The source manifest also pins the raw SHA-256 and Git blob identity of the two index files and checksum ledger, upstream commit, MIT license, expected distributions and split semantics. The upstream license is preserved in `ACED-UPSTREAM-LICENSE`; attribution is in `UPSTREAM_NOTICE.md`.

**No compressed world payloads, structural equations, ground-truth interventions, oracle fixes, scoring batteries, hidden roles or grader state are copied into this adapter.** The checksum ledger exposes file paths and digests, not the payload bytes.

## Reproduction

From the Symthaea repository root:

```bash
cargo fmt --manifest-path research/aced-adapter-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/aced-adapter-001/qualifier/Cargo.toml
cargo run --manifest-path research/aced-adapter-001/qualifier/Cargo.toml -- \
  research/aced-adapter-001/upstream/repair_test_index.json \
  research/aced-adapter-001/upstream/repair_eval_index.json \
  research/aced-adapter-001/ACED_REPAIR_SOURCE_MANIFEST_V0_1.json \
  research/aced-adapter-001/upstream/SHA256SUMS
```

A passing inventory qualifier checks:
1. Exact raw index and checksum-manifest SHA-256.
2. Pinned upstream repository, commit, paper revision and license identity.
3. Exact inventory sizes, unique filenames/seeds and path safety.
4. Domain and archetype distributions.
5. Exact evaluation-subset membership and metadata equality.
6. Presence of all test/evaluation world payload paths in the upstream hash ledger.
7. Checksum-ledger entries for both index files match their pinned raw bytes.

It does **not** download or verify the compressed world payload bytes. For an official upstream reproduction, checkout the pinned upstream commit and run its `prepare_dataset.sh` so the original `data/SHA256SUMS` is checked against the actual payloads. Keep upstream data and source attribution intact.

## Security boundary for future model trials

The upstream repository explicitly warns that its Repair code sandbox is a restricted process safeguard, not a security boundary; it also warns that the world files contain answers and should not be mounted into a candidate process for a public leaderboard. Its Query agents may execute code without restrictions.

Luminous must not simply run the upstream agent code sandbox on the host and call it secure. Before model trials:
- Keep compressed worlds, hidden SCM truth, oracle fixes, objective functions, scoring batteries and grader state in a separate evaluator-only store/process.
- Give candidate agents only the public scenario, permitted measurement/intervention responses, visible evidence ledger, tool schema and remaining budgets.
- Execute candidate code in an independently controlled Firecracker/gVisor or equivalently isolated microVM boundary with no network, cloud metadata, credentials, host mounts or evaluator IPC.
- Separate tool request, authorization decision, dispatch, tool response, independent state observation and outcome grading in their receipts.
- An unknown tool outcome must block subsequent state mutation until independently reconciled.
- Preserve every timeout, invalid result, incomplete world and failed trial in the denominator; never silently drop them.

## Comparative scorecard

Preserve official ACED metrics as their own lane:
- accepted rate;
- utility threshold pass rate (`part_a_utility_ok`);
- mechanism/battery score (`battery_fraction`);
- benefit recovered versus oracle fix;
- recommended utility versus oracle utility.

Add independent Symthaea research measures without altering the official score:
- intervention regret and objective improvement;
- true-lever discovery and unsupported-lever uptake;
- dose/combination selection accuracy;
- experiment efficiency and wasted-experiment rate;
- confounder, collider, instrument-only and surrogate-trap failures;
- subgroup-specific policy quality and worst-subgroup utility;
- evidence-ledger provenance and unsupported mechanism claims;
- experiment/turn/token/tool/compute budget and p50/p95 cost;
- timeout, invalid, incomplete and indeterminate denominators;
- worst-archetype and worst-domain results.

No single aggregate causal score is authorized. Report the complete vector, per-world paired comparisons and uncertainty intervals.

## Baselines and ablations

Reproduce the official scripted/mock controls and official evaluation protocol where licensing and execution contracts allow. Then compare under matched model snapshot, prompts, tools, 15-experiment/32-turn budgets and compute accounting:
- single-agent investigation;
- independent-sample/self-consistency;
- standard multi-agent debate;
- evidence-ledger / revision-aware agent;
- active experiment-selection agent;
- full evidence-gated Symthaea system.

Ablate active experiment selection, causal structure support, evidence ledger, uncertainty calibration, dose/combination search, subgroup policy handling and stopping rules one at a time.

Proposed success condition: demonstrate a preregistered, statistically supported improvement over the strongest matched baseline in official acceptance and/or utility regret **without worsening worst-archetype or worst-domain outcomes**. Do not declare a percentage-point superiority target before establishing a reproduced baseline and power analysis.

## Claim ceiling

A PASS currently qualifies only the pinned source metadata, exact inventory, checksum-ledger coverage and evaluation-subset contract. It does not qualify compressed world payload integrity, model behavior, official grader reproduction, sandbox security, causal discovery or causal repair competence. SOTA or general capability claims require executed trials, matched baselines, independent outcome evaluation, multiple model/harness configurations and uncertainty estimates.
