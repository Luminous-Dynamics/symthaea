# Reward Network III source-version reconciliation: 2026 paper vs older GitHub snapshot

**Status:** reproducible source audit of an older public repository snapshot.  
**Critical boundary:** this is not the final v2.0 Zenodo archive and must not be substituted for it.

## Why this audit exists

The 2026 paper explicitly specifies a 12-node/30-edge task network, hidden node levels 0–3, +400 rewards originating at level 3, and at least three negative-reward edges on every path from a level-0 node to level 3. The older public GitHub repository snapshot inspected below is a different, earlier implementation profile: 10 nodes, 20 directed edges, observed levels 0–4 and eight transitions. The profiles must remain distinct; the older snapshot does not override the published paper specification, and the paper specification does not prove the old snapshot is the final v2.0 archive.

## 2026 paper facts

Brinkmann et al., *Propagation and preservation of AI-discovered problem-solving strategies in human culture*, Nature Communications 17, 8255 (2026), DOI: 10.1038/s41467-026-76113-2.

- Article: https://pmc.ncbi.nlm.nih.gov/articles/PMC13487184/
- Publisher page: https://www.nature.com/articles/s41467-026-76113-2
- The paper states that participants choose **10 moves** through a reward network.
- Rewards range from **−50 to +400**; the paper identifies a three-loss path followed by seven +400 moves as the 2,650-point maximum, versus 2,000 for ten +200 moves.
- The main human experiment had 30 populations, five generations, eight positions per generation, with Human-Only and Human-Machine conditions.
- The final paper identifies the exact code/data reproduction source as Zenodo v2.0: https://zenodo.org/records/18547122 (DOI 10.5281/zenodo.18547122).

The paper explicitly supports the 12-node/30-edge/levels-0–3 profile. This audit's purpose is not to dispute that published task definition; it is to prevent the older public GitHub snapshot's 10-node/20-edge/eight-transition implementation from being mistaken for the final archived v2.0 implementation.

## Older public GitHub snapshot inspected

Repository: https://github.com/LBrinkmann/reward-networks-iii-experiment  
Snapshot commit: `01be138b112122acc0d3898e29d018877b4eb76f`  
This is an older public experiment/deployment snapshot. Its README documents a Docker/FastAPI/React deployment. It is not identified here as the final v2.0 code deposit.

### Exact source files and blob identities

| File | Git blob SHA | Observation |
|---|---|---|
| `backend/app/data/networks.json` | `8ff3950430eb95184e488fe38f3fcf31e189743e` | 5,618,289-character JSON payload |
| `backend/app/data/solutions_loss.json` | `4676fce8fccf9ddf6de2a8b24ee3be4454d1f4c1` | loss-heuristic moves |
| `backend/app/data/solutions_myopic.json` | `55e1c352f1bae10e0333647b3f745a191b567abe` | myopic-heuristic moves |
| `backend/app/models/network.py` | `1476f10144000dad104c9e2913fe8aef18cd5eb0` | node/edge schema |
| `backend/app/utils/utils.py` | `7d4018d95f1d2476ff2b0bb9f86913ca57c64f77` | score helper, default 8 steps |
| `frontend/src/reducers/NetworkReducer.tsx` | `a0f39e874aec37a31ae3b7a3984df715219e5536` | outgoing-edge traversal and maxStep default 8 |
| `backend/app/models/config.py` | `a9a40a579f3c059044127c21eba58c481987552a` | older experiment config defaults |

Direct links:
- [network model](https://github.com/LBrinkmann/reward-networks-iii-experiment/blob/01be138b112122acc0d3898e29d018877b4eb76f/backend/app/models/network.py)
- [network reducer](https://github.com/LBrinkmann/reward-networks-iii-experiment/blob/01be138b112122acc0d3898e29d018877b4eb76f/frontend/src/reducers/NetworkReducer.tsx)
- [score helper](https://github.com/LBrinkmann/reward-networks-iii-experiment/blob/01be138b112122acc0d3898e29d018877b4eb76f/backend/app/utils/utils.py)
- [session generator](https://github.com/LBrinkmann/reward-networks-iii-experiment/blob/01be138b112122acc0d3898e29d018877b4eb76f/backend/app/study_setup/generate_sessions.py)

### Network file inventory and verified invariants

Directly parsed the public `networks.json` Git blob and its two solution files in-memory; no participant records are involved.

| Property | Observed value |
|---|---:|
| Networks | 841 |
| Nodes per network | 10 (all networks) |
| Stored directed edges per network | 20 (all networks) |
| Node levels observed | 0, 1, 2, 3, 4 |
| Maximum level allowed by the old Pydantic validator | 5 (validator condition is `n > 5`, despite error text saying 0–3) |
| Loss-solution records | 841 |
| Myopic-solution records | 841 |
| Network IDs without a matching solution in either file | 0 |
| Move-list length in both solution files | 9 node IDs = 8 traversed edges |
| Starting node at level 0 | all 841 networks |
| Minimum negative edges from each network's start to level 4 | exactly 3 for all 841 networks |
| +400 edges originating at a level other than 4 | 0 |

Reward counts in the full 841-network pool:

| Reward | Stored edge count |
|---:|---:|
| −50 | 5,735 |
| 0 | 2,830 |
| 100 | 3,459 |
| 200 | 3,137 |
| 400 | 1,659 |

The older snapshot therefore appears to encode a **10-node, 20-edge, five-level** task in which the maximum-reward source level is 4. Its level validator is internally inconsistent: the function name says four levels, the conditional admits values up to 5, and the error string says 0–3; the observed data itself uses levels 0–4.

### Directed traversal is confirmed in this older frontend

`NetworkReducer.tsx` computes legal next nodes by filtering edges where `source_num === currentNodeId`. It does not automatically reverse edges. The same reducer looks up exactly one edge with `source_num === currentNode` and `target_num === nextNode`. Thus the older experiment treats stored arcs as directed transitions for legal-move selection, even where some reverse arcs are separately present.

The older UI's `NEXT_NODE` action defaults to `maxSteps = 8`, and the tutorial ends its path demonstration at 9 node entries (start plus eight transitions). The older score helper also defaults to `n_steps = 8`. This is direct evidence for an older **8-transition** protocol, not the paper's final 10-move protocol.

### Older solution scores are not the paper's 2,650/2,000 headline values

The older repository contains 841 move lists per heuristic. Replaying those lists against their own network records produced:
- 0 invalid move sequences for either file;
- the loss heuristic uses exactly three −50 edges in each listed solution;
- its later rewards vary by network, using 0/100/200/400;
- myopic-solution rewards also vary by network.

Do not use those historical scores as the paper's main-task aggregate or claim that these solution lists reproduce the final Zenodo DQN.

### Older configuration defaults also differ

The older `ExperimentSettings` defaults include three generations, 20 sessions per generation, three AI players, four individual trials, one social-learning iteration and zero demonstrations. The 2026 paper reports five generations and a different main-study schedule. These config defaults must not be substituted for the published main experiment.

## CivBench implementation correction

The generic Rust `RewardNetwork` now:
- keeps the published paper profile explicit as 12 nodes, 30 edges, levels 0–3, terminal level 3 and at least three negative-reward edges on every level-0-to-level-3 path;
- exposes source-specific shape validation via `validate_shape(expected_nodes, expected_edges)`;
- separates node-level range validation from graph construction;
- validates a generic terminal-level/negative-edge contract with `validate_terminal_strategy(terminal_level, required_losses)`;
- keeps the paper's 10-move budget separate from the older GitHub snapshot's 8-transition profile;
- retains clearly synthetic fixtures for both shapes, without treating either fixture as imported source data.

The crate defines named `LEGACY_GITHUB_*` constants solely for the older snapshot. They are not the Zenodo v2.0 schema.

## Remaining exact-reproduction gates

Before publishing a v2.0-compatible import or claiming paper reproduction:

1. Run the metadata-only Zenodo v2.0 inventory job and verify the archive digest.
2. Inspect v2.0 network count, node/edge counts, level range, reward distribution, starting-node field and move-sequence lengths.
3. Inspect the v2.0 runtime/reducer for directed traversal, revisit rules, step budget and scoring/penalty semantics.
4. Bind the exact v2.0 train/validation/experimental network splits and solutions by archive paths/digests.
5. Reproduce authors' exclusion rules, DQN training/config and published analyses from the v2.0 archive.

Until those checks are complete, report three separate things: **paper-level published task semantics**, **older GitHub snapshot semantics**, and **our synthetic reference fixtures**. Do not merge them into one supposed canonical schema.
