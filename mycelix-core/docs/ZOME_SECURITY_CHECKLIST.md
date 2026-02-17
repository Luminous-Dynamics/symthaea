# Holochain Zome Security Checklist

**Status**: v0.1 – Based on Agents zome patterns (`zomes/agents/src/lib.rs`)  
**Scope**: Practical hardening guidelines for all Holochain zomes in Mycelix.

This checklist translates the general guidance in `docs/SECURITY_HARDENING.md` into concrete patterns to apply at each zome boundary.

---

## 1. Input Validation (Every `#[hdk_extern]`)

- [ ] Validate all string IDs:
  - Non-empty, trimmed.
  - Length bounded (e.g. ≤256 chars).
  - Restricted character set (e.g. alphanumeric + `- _ :` for DIDs).
  - Example: `validate_agent_id` in `zomes/agents/src/lib.rs`.
- [ ] Validate lists:
  - Bounded length (e.g. max capabilities per agent).
  - No empty elements.
  - Length bounds on each string.
- [ ] Validate numeric ranges:
  - Reputation scores, thresholds, weights in `[0.0, 1.0]`.
  - No NaN / Inf; check with `is_finite()`.
  - Fail fast with clear `Guest` errors.
- [ ] Validate payload size:
  - For vectors/weights, enforce reasonable upper bounds (e.g. `MAX_WEIGHTS_SIZE`).

---

## 2. Rate Limiting & Abuse Prevention

- [ ] Apply per-agent rate limits on write operations:
  - Use path-based anchors + self-links as lightweight counters.
  - Maintain a sliding time window (e.g. last 60 seconds via `sys_time()`).
  - Example: `check_rate_limit` / `record_rate_limit_action` in `zomes/agents/src/lib.rs`.
- [ ] Separate limits per action:
  - Registration vs model updates vs governance actions.
  - Ensure “expensive” operations (aggregation triggers, escalations) have stricter limits.
- [ ] Provide clear error messages for rate limit violations.

---

## 3. Storage & Privacy

- [ ] Store **hashes**, not raw large/secret data, in the DHT:
  - E.g. `weights_hash` + `weights_size`, with actual gradients held off-chain or in a separate backend.
  - Example: `ModelUpdate` in `zomes/agents/src/lib.rs`.
- [ ] Use anchors to structure data:
  - Global anchors like `"agents.all"`, `"training_rounds.{round_id}"`, `"agent_updates.{agent_id}"`.
  - Use link tags for quick filtering (`agent_id`, `round_id` bytes).
- [ ] Avoid unbounded link fan-out:
  - Consider per-round/per-agent anchors instead of a single global list.

---

## 4. Authorization & Ownership

- [ ] Enforce “owner only” updates:
  - For entries with a clear owner (e.g. agent registrations), require that the caller `agent_info()?.agent_initial_pubkey` matches the stored `registrant` or owner field.
  - Example: `deactivate_agent` in `zomes/agents/src/lib.rs`.
- [ ] Restrict privileged operations:
  - Aggregation, threshold changes, escalation decisions should require specific capabilities/roles, not arbitrary callers.
  - Map roles from MATL / governance into explicit checks at the zome boundary.
- [ ] Treat cross-zome calls as untrusted:
  - Validate inputs regardless of which zome called you.

---

## 5. Denial-of-Service Considerations

- [ ] Bound all loops over DHT data:
  - If scanning links, consider maximum limits or pagination.
  - Avoid unbounded recursion or multi-hop searches in a single call.
- [ ] Fail fast on malformed or oversized payloads.
- [ ] Avoid heavyweight computation in externs:
  - Offload expensive ML/crypto work to Rust crates or external processes with well-defined inputs/outputs.

---

## 6. Telemetry & Auditing

- [ ] Log meaningful events at the zome boundary:
  - Security-relevant decisions: rate-limit hits, failed validations, escalation/halt decisions.
  - Do not log sensitive payloads; log IDs/hashes only.
- [ ] For FL/BFT zomes:
  - Track per-round metrics: number of submissions, detected Byzantine fraction, chosen defenses.
  - Emit signals for `SecurityStatus` changes (`Ok` → `Escalate` → `Halt`).

---

## 7. Byzantine-Specific Checks (FL / MATL Zomes)

- [ ] Never trust a single gradient:
  - Always run detection (PoGQ/MATL + `ByzantineDetector`) before aggregation.
- [ ] Make defense choice explicit:
  - Validate `method` in `aggregate_gradients` and reject unknown methods.
  - Ensure `f`, `k`, and trimming parameters are within safe bounds.
- [ ] Document behavior at high adversary ratios:
  - At 40–45% Byzantine/cartel, either:
    - Prove aggregation remains near honest mean, or
    - Explicitly `Escalate`/`Halt` via `evaluate_round_security` and avoid silent success.

---

## 8. How to Apply This Checklist

For every new or modified zome:

1. **Walk each `#[hdk_extern]`** through sections 1–5:
   - Add validation, rate limiting, and ownership checks as needed.
2. **If zome participates in FL/BFT**, also apply section 7.
3. **Update tests**:
   - Add adversarial cases (bad IDs, rates, malformed payloads).
   - Confirm failure modes are explicit and safe.
4. **Cross-reference examples**:
   - Use `zomes/agents/src/lib.rs` as a model for:
     - ID validation,
     - rate limiting,
     - hash-only storage,
     - deactivation/ownership checks.

This file is meant to be a practical complement to `docs/SECURITY_HARDENING.md`. When you review or design zomes, treat this checklist as “definition of done” for security at the zome boundary.

