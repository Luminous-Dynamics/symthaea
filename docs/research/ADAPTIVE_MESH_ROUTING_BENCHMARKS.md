# Adaptive mesh-routing benchmark protocol

Status: input-validation regression coverage and a repeated-sample CPU microbenchmark are present. No performance improvement is claimed yet.

## Goal

Test whether Symthaea's predictive controller can improve routing decisions over established and simple baselines while preserving the existing transport/security invariants. The controller may select among bounded policies; it must not replace packet authentication, replay protection, TTL checks, or transport correctness.

## Required baselines

1. The current `ConsciousnessAwareRouter` policy at the exact benchmarked commit.
2. A simple signal baseline using per-link delivery ratio, RTT/latency, queue depth, and energy cost, with exponentially weighted smoothing.
3. A fixed-weight multi-objective route score.
4. The strongest applicable deployed protocol for each scenario (e.g. Babel or B.A.T.M.A.N.-adv for the topology / deployment being modeled). Babel's standardized feasibility condition is a particularly useful safety baseline for loop avoidance; do not claim protocol-wide superiority from a synthetic topology alone: https://www.rfc-editor.org/info/rfc8966/.

## Deterministic scenario corpus

Run each scenario with recorded seed, topology, traffic matrix, and configuration:

- Stable sparse and dense graphs.
- Random and clustered node/link failures.
- Mobility and periodic links.
- Bursty congestion and asymmetric links.
- Mixed traffic priorities across high-capacity Wi-Fi, overlay/QUIC, and low-rate / high-latency radio links.
- Intermittent partitions with store-and-forward recovery.
- Byzantine relay behavior: drops, delays, metric inflation, duplicate/replayed reports, and Sybil-like identities.
- Ablations: no HDC retrieval, no CfC forecast, no adaptive policy, no Phi-derived input, and no trust weighting.

Keep workload generation and scoring independent of the adaptive policy. Train/tune only on training seeds and report held-out seeds separately.

## Metrics

End-to-end scenarios must report delivery ratio, p50/p95/p99 latency, convergence / recovery time after a topology change, energy per delivered byte, control bytes per delivered byte, route churn, queue drops, and per-priority deadline misses. Report confidence intervals over independent runs; don't pool away scenario-specific regressions.

The current dependency-free microbenchmark is deliberately narrower:

- Route-policy CPU cost: 200,000 route calls per repeat, 7 repeats, batches of 5,000 calls, with a fixed 16 / 64 / 256-peer state.
- Peer-update CPU cost: 250 rounds per repeat, 7 repeats, updating all peers per round with strictly increasing cycle values.
- It emits mean throughput plus p50/p95/p99 over **batch-mean** latencies. Route percentiles are batch means over 5,000 calls; update percentiles are per-round means over a full peer set. They are not individual-call tail latencies.
- The route workload currently has no live link metrics, network delivery, mobility, failure simulation, or baseline comparison. It does not prove end-to-end routing quality, nor does it measure a speedup without a paired baseline run.

Command:

```sh
cargo bench --bench adaptive_mesh_routing --features mesh
```

The result is a local CPU-cost baseline only. Preserve raw output, CPU, OS, Rust toolchain, commit SHA, feature set, and command line. Compare candidate and baseline commits on the same machine; repeat full process invocations before making claims.

## Hard invariants

- Zero accepted malformed or replayed safety-critical packets in the regression corpus.
- No unauthenticated relaying of safety-critical payloads.
- No routing loops in bounded adversarial and churn scenarios.
- Emergency priority does not bypass authentication, anti-replay checks, TTL enforcement, or hard resource caps.
- Missing, stale, or low-confidence predictions fall back to a known baseline.
- Adaptive routing cannot make the control plane oscillate indefinitely; test hysteresis and cooldown behavior explicitly.
- A peer's reported Phi / consciousness estimate is not proof of forwarding reliability or identity trust. Trust must use independent authenticated observations.
- Preserve negative controls and publish exact commit, scenario seeds, parameters, corpus counts, and raw result artifacts.

## First implementation slice

The router rejects non-finite and out-of-range peer metrics, unknown governance tiers, and obviously stale cycle updates before they alter cached peer state. Regression tests cover those conditions. These checks reduce malformed-input hazards; they do not replace authenticated sequence numbers or establish that a peer is trustworthy.

## Acceptance gates

An adaptive configuration is eligible for further consideration only if it:
- beats the selected baseline on a predeclared primary metric by a practically meaningful margin;
- stays within predeclared latency, control-overhead, and energy budgets;
- has no hard-invariant failures;
- passes ablations demonstrating that the learned/predictive component adds value;
- reports regressions instead of hiding them in an aggregate score.

Queued CI is not a pass. Local results must identify the exact tested commit and command.
