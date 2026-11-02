# Temporal candidate wire protocol v0.1

Tracking: [#9939](https://github.com/Luminous-Dynamics/symthaea/issues/9939)  
Implementation branch: `research/temporal-candidate-wire-v0.3`  
Parent: [HARTH streaming ingestion PR #9917](https://github.com/Luminous-Dynamics/symthaea/pull/9917)

## Purpose

Define a bounded, dependency-light binary boundary between a future isolated candidate process and the temporal benchmark evaluator.

The candidate receives only one public observation frame and returns one bounded action frame. No scenario seed, hidden labels, event schedule, evaluator state, scoring, oracle commitment or evaluator files are part of the protocol.

This protocol is a prerequisite, **not an OS sandbox**. It does not itself launch a subprocess, restrict filesystem access, block network calls, enforce CPU/memory limits or keep oracle data safe from a malicious process. Those are the separate Linux runner requirements in [issue #9939](https://github.com/Luminous-Dynamics/symthaea/issues/9939).

## Frame layout

All integer fields are unsigned little-endian. Every frame has a fixed 48-byte header:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | Magic `SYMEDGE1` |
| 8 | 2 | Protocol version (1) |
| 10 | 1 | Frame kind: 1 observation, 2 action |
| 11 | 1 | Reserved, must be zero |
| 12 | 4 | Exact payload length |
| 16 | 32 | Domain-separated SHA-256 of the payload |

Frame lengths are bounded and exact. Unknown versions/kinds, nonzero reserved fields, truncation, trailing bytes, wrong payload lengths and digest mismatches are rejected.

Observation v1 is exactly 51 payload bytes: current public step; presence flags; last measurement sample-step/sensor/result; measurement age; total sample count; LowPower count; HighFidelity count. Optional fields have a canonical zero encoding when absent, and measurement age is recomputed from sample step and current step.

Action v1 is exactly 30 payload bytes: alarm flag, selected sensor (none/LowPower/HighFidelity), next wake interval, candidate-declared compute cycles, transmitted bytes and memory-write bytes. Wake intervals must be 1–1,024 steps.

Maximum frame size is 99 bytes. No dynamic-length user text, paths, serialized objects or arbitrary metadata are accepted.

## Causal semantics

The simulator emits an observation to the candidate at a wake. The candidate responds with an action. The evaluator applies the alarm decision at the current wake, performs the optional requested sample, then advances to the next wake. The new measurement is visible only in the next observation frame. This preserves the delay between acquisition and policy use.

The host runner must enforce:
- one observation frame -> exactly one action frame;
- bounded frame count and per-frame/episode wall-clock deadline;
- no candidate-controlled evaluator seed, scenario ID or oracle-derived metadata;
- process reset between episodes when the preregistered policy requires it;
- bounded stderr capture and no parsing of stderr as a score.

## Linux isolation requirements

Issue #9939 specifies a Linux-only runner. Mandatory controls must include a restricted user/mount/PID/IPC/UTS/network namespace, minimal read-only runtime, no host home/repository/evaluator mount, no device nodes, bounded private tmpfs, network denial, seccomp policy, Landlock filesystem restrictions when supported, cgroup v2 CPU/memory/pids limits, rlimits and process-tree/cgroup kill on timeout. The runner must record its exact kernel, launcher, policy and resource-limit digests. Unsupported mandatory protections fail closed.

A container or network namespace alone is not enough. A candidate process with arbitrary executable code must be treated as hostile.

## Tests authored

The module includes nine synthetic tests for observation round trips with and without measurements, action round trips with and without sensor acquisition, payload corruption, wrong magic/version/kind, truncation, invalid action bounds, inconsistent measurement age/counters and noncanonical absent-field encoding.

**Execution status:** tests are authored but not compiled or executed. No sandbox or confirmatory execution is claimed.
