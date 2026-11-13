# ROB-DECISION-RECOVERY-AWARE-CONTEXT-RETENTION-001

Owner: Symthaea issue #11665. Parent fleet-serving protocol: #11627 / PR #11630.

This standalone reference selects resident contexts to evict when a caller requests additional KV tokens. It compares deterministic LRU, phase-affinity and transition-reuse controls against a Luminous decision/recovery-aware lexicographic policy.

## Hard retention constraints

- Hard-pinned local control, RTA, actuator interlock, safe-stop, source-lineage and committed-prefix contexts are never eviction candidates.
- Missing, Unknown, expired or incomplete decision-impact / recovery evidence is protected, not treated as zero regret.
- If protected state plus the requested allocation exceeds cache capacity, the result is DEFER. There is no partial hidden state loss and no permissive eviction default.
- Qualified decision-regret values must share one versioned task-loss scale. Incomparable scales return an explicit error; no universal weighted score is invented.
- Recovery-critical contexts are retained ahead of non-critical alternatives but can be evicted only when the remaining capacity pressure requires it.
- Cache capacity and decision-loss values remain separate units. This reference does not convert task-loss units into tokens, joules or milliseconds.

## Policies

- LRU: oldest last-access tick first.
- Phase affinity: lowest next-phase affinity first (AgentKV-inspired control, not a reproduction of query-subspace scoring).
- Transition reuse: lowest predicted next-use probability first (CacheScout-inspired control, not a reproduction of its learned transition model).
- Luminous: non-recovery-critical first, then lowest qualified eviction-regret upper bound, lowest next-phase affinity, lowest future-reuse probability, cheapest prefill, and least-recent access as a deterministic tie-break.

All policies retain hard-pinned and Unknown-evidence contexts. Their differences are only among currently eligible candidates.

## Evidence contract

Each context carries:
- stable identity and token footprint;
- hard-pin reason, if any;
- recovery-critical flag and recovery evidence digest;
- decision-eviction regret upper bound, task-loss scale, calibration identity/digest and validity tick;
- phase-affinity manifest/phase identity;
- transition-derived future-reuse probability;
- re-prefill cost and last access tick.

The caller supplies these values. This crate does not estimate regret, probability, recovery or prefill cost; it only performs deterministic ordering over qualified evidence.

## Run

    cargo fmt --manifest-path research/rob-decision-recovery-context-retention-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-decision-recovery-context-retention-001/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-decision-recovery-context-retention-001/Cargo.toml --all-targets -- -D warnings

## Claim boundary

The reference qualifies only deterministic context-selection semantics. It does not implement KV-page allocation, model attention/query scoring, real serving, physical recovery validation or actuator authority. Published AgentKV, CacheScout and CONCUR results remain external comparators; no Luminous superiority or hardware result is claimed.
