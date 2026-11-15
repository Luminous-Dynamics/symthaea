# MEL-RTI-001A — Streaming musical turn-taking trace scorer V1

**Date:** 2026-11-15  
**Status:** Source/tests authored; exact-head qualification pending  
**Parent:** [#10307 — Streaming musical turn-taking](https://github.com/Luminous-Dynamics/symthaea/issues/10307)  
**Implementation task:** [#12337](https://github.com/Luminous-Dynamics/symthaea/issues/12337)

## Scope

This standalone Rust crate scores supplied chunk annotations and per-chunk action predictions. It does not infer musical invitations from PCM, detect phrase boundaries, or prove the runner actually withheld future input. It qualifies the trace contract and the declared evidence only.

## Frozen session and trace boundary

The request binds:
- exact source-audio digest syntax, sample rate, channels and frame count;
- session, task-manifest and annotation-policy identities;
- immutable policy identity, runner-build identity and initial policy-state digest;
- expected chunk count and seed.

Chunks must be ordered and contiguous, begin at frame zero, finish at the exact source frame count and have unique stable IDs. One truth and one prediction are required for every chunk. Missing records are invalid, never silently treated as WAIT.

Every prediction binds exact chunk identity, input-prefix digest, observed-through frame, monotonic chunk-receipt and decision times, pre/post policy-state digests, immutable policy identity and runner build. State snapshots form an explicit chain. The scorer rejects observed-through frames later than the current chunk end, state-chain discontinuities, model/runner identity changes and clock regression.

## Musical action semantics

- WAIT: preserve listening space, no new floor action.
- RESPOND: start a new contribution; requires a sample-frame effect and monotonic effect-emission time.
- CONTINUE: sustain the current contribution; no new effect timestamp.
- YIELD: relinquish the floor; requires a sample-frame effect and monotonic effect-emission time.
- ParseFailure: invalid parse, not WAIT, with a separate error digest and no action effect.

Truth stores a sorted valid-action set. Multiple valid actions remain ambiguous and are not forced into the single-label confusion matrix. Unambiguous RESPOND/YIELD truth requires a target frame and permitted half-open effect window at or after the input prefix.

The audio sample-frame clock and monotonic runner clock remain separate. The report distinguishes effect-window error in frames from decision/effect latency in nanoseconds.

## Metrics

The report contains:
- 4 × 5 action confusion matrix (four truth actions; five prediction classes including ParseFailure);
- per-action precision/recall/F1 and macro-F1;
- top-1 accuracy over unambiguous labels only;
- ambiguous valid-set acceptance;
- ParseFailure and WAIT false-activation counts;
- CONTINUE premature-YIELD count;
- post-YIELD non-WAIT trace leakage before the next expected RESPOND invitation;
- effect-window eligible/attempt/success counts and rate;
- p50/p95/p99 decision latency, effect latency and absolute sample-frame effect error.

Metrics remain separate. There is no composite turn-taking score.

## Deterministic controls

Fifteen unit tests are authored for:
1. perfect four-action trace and effect windows;
2. ParseFailure distinct from WAIT;
3. ambiguous action sets without forced truth;
4. missing prediction records;
5. chunk gap/overlap detection;
6. one-frame lookahead;
7. decision before chunk receipt;
8. backdated sample effect;
9. backdated monotonic effect;
10. effect timestamps on untimed WAIT/CONTINUE;
11. missing RESPOND effect clock;
12. duplicate effect frame;
13. post-YIELD non-WAIT leakage;
14. policy-state chain and immutable identity binding;
15. deterministic report replay and tamper rejection.

A synthetic five-chunk example covers WAIT → RESPOND → CONTINUE → YIELD → WAIT, with sample-frame effect windows and monotonic runner latency. The workflow runs it in two fresh processes and requires byte-identical JSON reports.

## Qualification

The exact-head workflow pins Rust 1.96.0, checks out and verifies the exact PR head, validates the schema, stages the standalone crate, normalizes/checks formatting, runs tests, executes fresh-process report replay twice, runs Clippy with warnings denied and uploads source/schema/lockfile/report identity artifacts.

No compile/test/Clippy pass is claimed until this workflow succeeds on the exact current head.

## Claim limits and next stage

This V1 scorer does not:
- identify musical invitations from waveforms;
- authenticate that a deployed runner used only the declared input prefix;
- measure audible output energy after YIELD;
- establish that a response was musically relevant or preferred;
- prove human reciprocal influence or ensemble quality;
- qualify an official HEARTBEAT submission.

The next stage is a trusted runner adapter that independently records input-prefix receipts and actual output event receipts, then a musical-cue benchmark with musician annotations. HEARTBEAT-compatible speech action results must be reported separately from music-specific event semantics.