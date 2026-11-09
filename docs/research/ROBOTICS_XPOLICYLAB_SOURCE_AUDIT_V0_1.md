# XPolicyLab interoperability source audit — v0.1

**Audit date:** 2026-11-09  
**Scope:** source reconnaissance and Luminous integration boundary; no benchmark run  
**Parent:** Symthaea #11285 / HUM-OBS-002 #4781  
**Inspected upstream repository:** https://github.com/XPolicyLab/XPolicyLab  
**Inspected upstream commit:** `6baccd57ff958570d0db1588370289ee36c8bb70` (2026-11-03)

## 1. Finding: reuse a shared policy-serving contract, not 77 custom integrations

XPolicyLab's README describes the policy/environment split as one adapter per policy and one client per environment, targeting O(N+M) rather than O(N×M) bespoke integrations. It provides:
- dependency-isolated policy server and environment client;
- websocket deployment, locally or across machines;
- observation/action/trajectory dictionary conventions;
- single and batched observation/action calls;
- reset semantics and common evaluation scripts;
- a policy zoo with 77 listed adapters at the inspected revision.

This is a strong candidate for the external policy-serving interop layer. It should not replace any Luminous-specific evidence model, controller authority, runtime assurance, or qualification logic.

## 2. Exact API contract observed

The demo adapter at the inspected commit has the following model-facing methods:

| Method | Observed contract |
|---|---|
| `update_obs(obs)` | Accept one observation dictionary |
| `update_obs_batch(obs_list)` | Accept a batch |
| `get_action()` | Return an ordered list of action dictionaries (one action chunk) |
| `get_action_batch(env_idx_list)` | Return action chunks aligned with the active environment indices |
| `reset()` | Clear model-side state between episodes; no arguments |

Relevant source blobs at the inspected commit:
- `README.md`: `73307abad831b96862ea088cd06759d9b943839e`
- `policy/demo_policy/model.py`: `13d0d26a3ee07283b49a2f61b2781dc3d37d5c1c`
- `policy/demo_policy/deploy.yml`: `30de66f6c8326b003d6d287ef78cfaeaf32c6098`

**Important action-semantics rule:** an action is a chunk, not necessarily one atomic vector. The Luminous adapter must preserve its request identity, ordering, chunk length, action-space declaration, batch/environment index mapping and timing. It must not split, reorder, retry or coalesce an action chunk without an explicit upstream contract.

## 3. Transport semantics that affect replay and controller lifecycle

The README documents the default websocket protocol and the following behavior:
- requests carry a `request_id`;
- a reconnect retry reuses the same ID and the server returns a cached response for a duplicate rather than re-running a non-idempotent call;
- a timeout is fatal for that trial because the server may still be processing the request;
- a reconnect to a restarted policy-server process raises `ServerRestartedError`, because model-side state was lost.

This behavior should be mapped explicitly to the Symtropy request/ticket/epoch lifecycle:
- preserve the exact external request ID and XPolicyLab server-process identity;
- distinguish timeout/unknown execution state from definite rejection;
- never retry an ambiguous physical action at the controller boundary just because a policy RPC timed out;
- treat a server restart as an episode/attempt invalidation or explicit recovery event, not as a transparent continuation;
- keep the policy RPC request ID distinct from the authenticated controller request ID unless a deliberate, collision-safe mapping is specified.

The XPolicyLab websocket request cache is not a durable controller ledger and is not physical actuation replay protection.

## 4. Observation and trajectory fields: useful interchange, not evidence-grade metrology

The README's observation convention includes:
- optional `data_format_version`, instruction(s), and environment index;
- RGB images and optional depth, camera intrinsics and extrinsics;
- optional joint, end-effector and mobile-base state fields;
- pose convention `[x, y, z, qw, qx, qy, qz]`.

Trajectory files include time-indexed image, action and state arrays, optional subtask annotations, and a trajectory format version. There is a camera extrinsics field-name difference between runtime observations (`extrinsics_matrix`) and saved trajectory data (`extrinsic_matrix`).

The inspected common observation/trajectory dictionaries do **not** by themselves provide an evidence-grade contract for:
- physical sensor/instrument serial identity;
- sensor source timestamp and clock-domain identity;
- synchronization offset/drift/jitter uncertainty;
- measurement-unit and calibration identity/currentness for every physical channel;
- metrology uncertainty/error model;
- raw hardware sample bytes and decoder lineage;
- physical test-article/as-built configuration identity.

Consequently:

```text
XPolicyLab observation
!= calibrated physical observation
!= qualified experiment evidence
```

Do not retrofit these missing semantics into an incompatible XPolicyLab payload. Attach the canonical Luminous evidence envelope out-of-band, correlating it to observations/actions through immutable run, episode, sample and action identities. Compose #4800 ROB-EXP, #4866 ROB-INSTR, #3695 SE-OBS and shared quantity semantics #4861 when qualified.

## 5. Image/data conversion details worth preserving

The inspected README documents two historical JPEG byte encodings (legacy channel order and the newer `XPL-RGB1` marker). It requires image decoding through XPolicyLab's `decode_image_bit` and encoding through `encode_image_bit`; a hand-written decoder can silently swap red and blue channels.

For an evidence-preserving adapter:
- record the XPolicyLab codec/decoder revision and whether the input was encoded or already decoded;
- preserve the raw trajectory asset digest and the normalized decoded observation digest;
- do not silently change image color, resize, crop, depth projection or camera ordering;
- mark any preprocessing as a derived observation with exact transformer/config identity.

This is not a reason to copy XPolicyLab's Python imaging stack into the Rust core. The upstream policy server can retain its native dependencies; the Luminous evidence adapter should record the transform and verify stable output.

## 6. Benchmark support: distinguish RMBench from RMMBench

The inspected README lists:
- **RoboDojo** — 42 simulation tasks over five dimensions plus 18 physical tasks on three bimanual embodiments;
- **RoboTwin 2.0** — 50 bimanual tasks under clean/randomized conditions;
- **RMBench** — a memory-dependent manipulation benchmark built on RoboTwin 2.0, with nine dual-arm tasks.

**RMBench and RMMBench are different benchmarks.** XPolicyLab's inspected README lists RMBench, not RMMBench. Therefore the compatibility matrix in Symthaea #11285 must keep RMMBench = Unknown until an exact, pinned adapter and evaluator test demonstrates support.

No README claim should mark any integration Qualified. Require an exact upstream revision, task/asset identity, a reset/update/action/episode/abort/malformed-input conformance run and preserved outputs.

## 7. Agent-track task-reference assistance: Q0 Preview audit note

The inspected Q0 Preview sources expose a task-context helper that combines a static task card with a static task-reference record. The returned policy-visible context can include evaluator-derived completion tolerances and assessment notes; examples include numeric position/orientation tolerances, insertion-depth constraints and gripper-opening thresholds. The module docstring states that these are static benchmark definitions, not live object/reward state, and that official assessment occurs outside the policy.

Inspected source blobs:
- `policy/Q0_Preview/runtime/task_context.py`: `7776e51197f824cc754b06a359f4505572c3372d`
- `policy/Q0_Preview/assets/tasks/task_reference.json`: `9e82814b209cb662aef8ca06099127bce2cce2e2`

A separate upstream commit on 2026-11-03 (`6baccd57ff958570d0db1588370289ee36c8bb70`) removed the `source_sha256` comparison in `task_reference(name)` that previously checked the reference-card source hashes against task-card hashes. The current helper still checks that a task reference exists, but that specific cross-file source-consistency guard is absent.

This may be a deliberate maintenance change; this audit does not infer intent or allege benchmark compromise. It is a **qualification and assistance-disclosure issue**:
- pin the exact task card, task reference and evaluator/source revision used;
- establish how task-reference tolerances were derived and reviewed;
- test whether all agent submissions receive the same task-context channel;
- classify Q0 Preview as an agent/tool-assisted track with its task-reference context disclosed;
- do not compare it as if it were a bare policy-only observation/action model;
- preserve source-consistency checks in the Luminous importer, or record the missing upstream source binding as unresolved.

This illustrates why agent-track and policy-only tracks should be separate. A static benchmark definition is not equivalent to live hidden state, but it is still an additional information channel relative to a minimal natural-language instruction.

## 8. Competitive benchmark snapshot — 2026-11-09

Use these as protocol-specific comparison points; do not combine the score scales.

| Benchmark track | Snapshot | Reported result | Interpretation |
|---|---|---|---|
| RoboCasa365 | Official page updated 2026-11-01 | 85.6 overall leader; 64.4 best listed open-source result | 50-task multi-task policy board; atomic-seen, composite-seen and composite-unseen are separate splits |
| RoboDojo simulation, agent track | Official page snapshot 2026-11-09; board data updated 2026-11-03 | Q0-Preview 74.20 composite / 68.87% SR | Agent track, separately displayed from non-agent policy ranking |
| RoboDojo real-world | Official page snapshot 2026-11-09; board data updated 2026-11-03 | SLA-Preview 42.33 composite / 33.89% SR | 18 physical tasks across ARX X5, Piper and Piper X; per-embodiment results differ materially |
| RoboTwin 2.0 | Official RoboTwin page snapshot updated 2026-10-11 | PatchWAM 79.14 average / 66.72% clean-to-random | Its official page remains an older snapshot; RoboDojo's XPolicyLab display shows a separate 2026-10-30 feed with different leaders. Never merge these snapshots |

The RoboDojo real-world board's leading SLA-Preview scores vary by embodiment: ARX X5 50.33/36.67%, Piper 60.33/55.00%, and Piper X 16.33/10.00% (composite score / SR). This is a reminder to retain per-robot results and not hide transfer gaps inside a macro average.

NIST's Performance of Emergency Response Robots programme provides a complementary physical-test philosophy: repeatable elemental test methods across maneuvering, mobility, dexterity, sensing, endurance, communication, durability, reliability, autonomy and safety. Do not reduce these to a single leaderboard score.

## 9. Luminous integration decision

Adopt XPolicyLab only as an **external policy-serving/environment interoperability boundary** after exact source pinning. Keep the architecture:

```
XPolicyLab policy server
        |
        | observation / action chunk (identified)
        v
Luminous interop adapter
        |
        +---- run/episode/action identity
        +---- evidence-envelope correlation
        +---- protocol / model / assistance manifest
        |
        v
Symtropy authenticated admission + epoch/deadline gates
        |
        v
native controller / independent safety authority

Separate, out-of-band:
physical observation -> ROB-INSTR / ROB-EXP -> independent audit
```

The sidecar evaluation should compare the *same frozen base policy* with and without Symthaea cognition, using paired episodes and identical action executors. Start on the official RoboDojo simulation track via an open pinned policy; then consider the physical RealEval process only after equipment, safety, calibration and operational constraints are reviewed. Keep the existing #7969 RoboDojo-RC harness in its own physical partial-progress evidence tier.

## 10. Nonclaims and next steps

This source audit does not establish a qualified Luminous integration, benchmark reproduction, policy performance, physical evidence, actuation safety or SOTA result.

Next:
1. Pin XPolicyLab at the inspected commit and add an upstream policy-server/client smoke adapter.
2. Define correlation IDs and a strict Luminous evidence sidecar for action chunks and observations.
3. Implement protocol conformance fixtures for reset, batch alignment, timeouts, server restart, duplicate request IDs, chunk ordering and malformed actions.
4. Audit Q0 task-reference provenance/assistance before using Q0 as a comparison baseline.
5. Start with one open policy and RoboDojo simulation; compare policy-only vs matched Symthaea sidecar.

## Sources

- XPolicyLab README at inspected commit: https://github.com/XPolicyLab/XPolicyLab/blob/6baccd57ff958570d0db1588370289ee36c8bb70/README.md
- XPolicyLab demo model: https://github.com/XPolicyLab/XPolicyLab/blob/6baccd57ff958570d0db1588370289ee36c8bb70/policy/demo_policy/model.py
- XPolicyLab demo deploy config: https://github.com/XPolicyLab/XPolicyLab/blob/6baccd57ff958570d0db1588370289ee36c8bb70/policy/demo_policy/deploy.yml
- Q0 Preview task context: https://github.com/XPolicyLab/XPolicyLab/blob/6baccd57ff958570d0db1588370289ee36c8bb70/policy/Q0_Preview/runtime/task_context.py
- Q0 Preview task-reference records: https://github.com/XPolicyLab/XPolicyLab/blob/6baccd57ff958570d0db1588370289ee36c8bb70/policy/Q0_Preview/assets/tasks/task_reference.json
- XPolicyLab paper: https://arxiv.org/abs/2608.09892
- RoboDojo real-world board: https://robodojo-benchmark.com/leaderboard?bench=realworld
- RoboDojo simulation board: https://robodojo-benchmark.com/leaderboard?bench=sim
- RoboTwin 2.0 board: https://robotwin-platform.github.io/leaderboard.html
- RoboCasa365 board: https://robocasa.ai/leaderboard
- NIST PER: https://www.nist.gov/programs-projects/performance-emergency-response-robots
