# Consciousness Gating System: Operational Security Document

**System**: Mycelix Consciousness-Aware Access Control
**Version**: As of 2026-03-07
**Scope**: The 4D consciousness profile, 5-tier progressive gating, bootstrap mechanism, credential lifecycle, and audit trail across all 7 Mycelix DNA clusters.

**Key source files**:
- `crates/mycelix-bridge-common/src/consciousness_profile.rs` -- core gating logic, pure-function evaluation
- `crates/mycelix-bridge-common/src/consciousness_thresholds.rs` -- canonical threshold constants
- `mycelix-commons/zomes/commons-bridge/coordinator/src/lib.rs` -- reference bridge implementation
- `docs/papers/consciousness_governance.md` -- formal architecture paper

---

## 1. Threat Model

### 1.1 Credential Forgery

**Attack**: An agent fabricates a `ConsciousnessCredential` with inflated dimension scores or a higher tier than they legitimately hold.

**Attack surface**: The `ConsciousnessCredential` struct is a serializable Rust struct (`Serialize`/`Deserialize`). If an agent can inject a crafted credential into their source chain or intercept the cross-zome call response from `get_consciousness_credential`, they could present an artificial credential to `evaluate_governance()`.

**Current defenses**:
- Ed25519-authenticated attestations (Section 4.6 of the governance paper) bind consciousness claims to the agent's Holochain keypair. The attestation message includes `agent_did`, `consciousness_level`, `cycle_id`, and `captured_at_us`, signed by the agent's key. Verification uses `agent_initial_pubkey`.
- The `v2` gate path (`verify_consciousness_gate_v2`) preferentially uses authenticated attestations and tracks provenance (`Attested` vs `Snapshot` vs `Unavailable`).
- Holochain's source chain validation prevents unauthorized entry creation -- peers validate entries against integrity zome rules before accepting them into the DHT.

**Residual risk**: The legacy `v1` gate path (`gate_consciousness()` in `consciousness_profile.rs:672`) does not verify attestation signatures. It trusts whatever `get_consciousness_credential` returns. If an agent's conductor is compromised, they could patch the bridge zome to return arbitrary credentials. This is mitigated by DHT validation (other peers would reject invalid entries), but the local evaluation itself is unguarded against a compromised local conductor.

**Recommendation**: All production deployments should use the `v2` gate path exclusively. Log and alert on any `Snapshot` or `Unavailable` provenance in governance actions above Participant tier.

### 1.2 Replay Attacks

**Attack**: An agent captures a valid `ConsciousnessCredential` from a previous period (when their scores were higher) and presents it for current governance actions.

**Current defenses**:
- Credentials have a 24-hour TTL (`DEFAULT_TTL_US = 86_400_000_000` at `consciousness_profile.rs:173`). The `is_expired()` check at `consciousness_profile.rs:176` compares `now_us >= self.expires_at`.
- Authenticated attestations include `cycle_id` and `captured_at_us` fields, preventing reuse of old attestation signatures.
- The `gate_consciousness()` function obtains `now_us` from `sys_time()` at call time (`consciousness_profile.rs:703`), not from the credential itself.

**Residual risk**: Within the 24-hour TTL window plus 30-minute grace period, a credential remains valid even if the agent's actual profile has degraded. A reputation collapse (e.g., from a misconduct finding) does not immediately invalidate outstanding credentials.

**Recommendation**: For high-stakes actions (constitutional changes, guardian operations), consider requiring a fresh credential with a maximum age threshold (e.g., issued within the last hour) rather than accepting any non-expired credential. This would require adding an `issued_at` freshness check to `requirement_for_constitutional()` and `requirement_for_guardian()`.

### 1.3 Tier Inflation via Sybil Attestations

**Attack**: An adversary controls multiple Holochain agent keys. They use these identities to issue mutual community trust attestations, artificially inflating each other's Community dimension (weight 0.30, the largest single weight).

**Attack surface**: The Community dimension is "peer trust attestations, weighted by attestor tier." If the attestors are all sybil identities at Observer tier (weight 0), the attack fails. But if even one legitimate Participant-tier agent colludes, they can bootstrap the sybils to Participant, who then cross-attest each other upward.

**Current defenses**:
- The Identity dimension requires MFA verification through the identity cluster. Creating k sybil identities requires k independent MFA verifications (at minimum, Basic MFA = distinct email/phone).
- Attestor weight scales with their own tier -- Observer attestations carry zero weight.
- Agent-level duplicate vote detection (`enforce_agent_vote_limit`) prevents a single Holochain agent key from voting multiple times.

**Residual risk**: A determined adversary with k phone numbers can create k Basic MFA identities (identity = 0.25 each). With mutual attestations, they could collectively reach Participant tier (combined >= 0.3). The Community dimension's 30% weight means even moderate sybil attestation inflation has outsized impact on tier calculation.

**Recommendation**: Implement attestation velocity limits -- cap the number of trust attestations a single agent can issue per time window. Monitor for attestation graph anomalies (cliques of agents who only attest each other). Consider requiring attestations from agents at least one tier above the target.

### 1.4 Denial of Service via Gate Flooding

**Attack**: An adversary floods the `get_consciousness_credential` endpoint or `dispatch_call` with requests, exhausting rate limits for legitimate agents or overloading cross-cluster communication.

**Current defenses**:
- Rate limiting at `lib.rs:231-260`: `enforce_rate_limit()` checks recent dispatch links against `RATE_LIMIT_WINDOW_SECS` (60 seconds) and rejects if the count exceeds the limit (100 calls per window via `check_rate_limit_count`).
- The rate limit uses `GetStrategy::Local` for link queries, avoiding DHT amplification.
- Credential caching in `get_consciousness_credential` (`lib.rs:1128-1148`) avoids redundant cross-cluster calls.

**Residual risk**: Each rate limit check itself creates a link (`create_link` at `lib.rs:252-257`), so a flood of requests that are individually rate-limited still generates DHT writes (one link per request attempt, regardless of success). An adversary sending 100 requests/minute/agent saturates the rate limit and generates 100 links/minute. With many sybil agents, this becomes a link-creation DoS.

**Recommendation**: Move rate limit tracking to in-memory state (e.g., a conductor-level cache) rather than DHT links. Alternatively, batch rate limit link creation (one link per window, with a count field).

### 1.5 NaN Injection

**Attack**: An adversary crafts a `ConsciousnessProfile` with NaN values in one or more dimensions, exploiting IEEE 754 comparison semantics where `NaN < x` and `NaN >= x` are both false.

**Detailed analysis in Section 6 below.**

### 1.6 Clock Skew Exploits

**Attack**: An agent manipulates their local system clock to extend credential validity, bypass expiry checks, or manipulate the grace period.

**Attack surface**: The `gate_consciousness()` function obtains `now_us` from `sys_time()` (`consciousness_profile.rs:703`). In Holochain, `sys_time()` returns the conductor's system clock, which is the host machine's clock. An agent running their own conductor can set their clock arbitrarily.

**Impact of backward clock skew (clock behind reality)**:
- `is_expired(now_us)` returns false for credentials that have actually expired -- the agent can use stale credentials indefinitely.
- `needs_refresh()` never triggers -- proactive refresh is bypassed.
- `calculate_local_engagement()` (`lib.rs:1232-1272`) uses `sys_time()` for the 90-day cutoff. A backward-skewed clock would exclude legitimate recent activity, reducing engagement. This is self-harming, not an attack.

**Impact of forward clock skew (clock ahead of reality)**:
- The agent's own credentials expire prematurely (self-harming).
- However, credentials issued by a forward-skewed conductor have `issued_at` in the future, which could cause them to appear non-expired to other nodes for longer than intended.

**Current defenses**: Holochain's validation rules check timestamp reasonableness for source chain entries. Entries with timestamps too far in the future are rejected by peers. However, the consciousness gating evaluation is local-only -- it uses the local `sys_time()` without cross-referencing DHT peer clocks.

**Recommendation**: For constitutional and guardian actions, require the credential's `issued_at` to be within a reasonable window of the evaluating node's `sys_time()` (e.g., `|now_us - issued_at| < 25 hours`). This catches both stale credentials from clock manipulation and future-dated credentials from compromised issuers.

### 1.7 Bootstrap Abuse

**Attack**: Agents exploit the bootstrap mechanism to maintain permanent Participant-tier access without earning genuine community trust.

**Attack vectors**:
1. **Community count manipulation**: An agent keeps leaving and rejoining to keep the community member count below `BOOTSTRAP_COMMUNITY_THRESHOLD` (5).
2. **Rolling bootstrap**: An agent obtains a bootstrap credential (1-hour TTL), uses it, lets it expire, then obtains a new one -- indefinitely maintaining Participant access without building Community or Reputation scores.
3. **Community splitting**: Adversaries fragment a community into sub-5-member groups, each eligible for perpetual bootstrapping.

**Current defenses**:
- Bootstrap credentials are hard-capped at Participant tier (`evaluate_bootstrap_governance()` at `consciousness_profile.rs:423`). Voting, constitutional changes, and guardian operations are always rejected.
- Bootstrap requires at least Basic MFA (`BOOTSTRAP_MIN_IDENTITY = 0.25` at `consciousness_thresholds.rs:83`).
- Bootstrap TTL is 1 hour (`BOOTSTRAP_TTL_US = 3_600_000_000` at `consciousness_thresholds.rs:80`), shorter than the standard 24-hour credential.

**Residual risk**: There is no limit on how many consecutive bootstrap credentials an agent can obtain. The `is_bootstrap_eligible()` function (`consciousness_profile.rs:380-383`) checks only `agent_count < BOOTSTRAP_COMMUNITY_THRESHOLD && identity_score >= BOOTSTRAP_MIN_IDENTITY` -- it does not check how many times the agent has already bootstrapped. A community of exactly 4 agents could bootstrap indefinitely.

**Recommendation**: Track bootstrap credential issuance count per agent. After N bootstrap credentials (e.g., 3), require organic credential issuance via the identity bridge. Alternatively, add a cumulative bootstrap duration cap (e.g., 24 hours total bootstrap time per agent per community).

---

## 2. Failure Modes

### 2.1 Identity Cluster Unavailable

**Trigger**: The identity cluster DNA is offline, the conductor process crashes, or a network partition isolates it from other clusters.

**Impact**: `get_consciousness_credential()` at `lib.rs:1179-1194` falls back to an Observer-tier credential (all dimensions 0.0, issuer `"did:mycelix:commons-bridge-fallback"`). All governance actions requiring Participant tier or above are blocked for new credential requests.

**Degradation behavior**:
- Agents with cached credentials (10-minute cache TTL) continue operating normally until their credential expires (up to 24 hours).
- Proactive refresh attempts (`lib.rs:1134-1145`) fail silently and fall back to the cached credential.
- New agents or agents whose cache has expired receive Observer-tier fallback credentials and lose all governance capabilities.

**Detection**: Monitor for credentials with issuer `"did:mycelix:commons-bridge-fallback"`. A spike in fallback credentials indicates identity cluster unavailability.

**Recovery**: Once the identity cluster is restored, agents automatically obtain fresh credentials on their next `get_consciousness_credential` call (cache miss or proactive refresh success). No manual intervention is required unless credentials have expired during the outage, in which case agents must wait for the identity cluster to issue new credentials.

### 2.2 Mass Credential Expiry

**Trigger**: All credentials in a community were issued within the same 24-hour window (e.g., after a system restart or migration), and they all expire simultaneously.

**Impact**: Every governance action fails with "Credential expired" until agents obtain fresh credentials. The 30-minute grace period (`GRACE_PERIOD_US = 1_800_000_000` at `consciousness_profile.rs:334`) only covers Participant-tier operations, so voting and constitutional actions fail immediately.

**Cascade risk**: If the identity cluster is under load from all agents simultaneously requesting new credentials, the cross-cluster call may time out, triggering the Observer-tier fallback (Section 2.1), which compounds the outage.

**Detection**: Monitor credential `expires_at` distribution. Alert when more than 50% of active credentials expire within the same 2-hour window.

**Recovery**: Stagger credential issuance. Ensure the proactive refresh window (`REFRESH_WINDOW_US = 7_200_000_000`, 2 hours before expiry) spreads renewal requests over time. After a mass issuance event, consider manually jittering credential TTLs (e.g., 22-26 hours instead of exactly 24 hours).

### 2.3 Bootstrap TTL Expiry Before Organic Growth

**Trigger**: A new community with fewer than 5 members cannot establish organic credentials within the 1-hour bootstrap TTL. The identity cluster may not be fully configured, or MFA providers may not be reachable.

**Impact**: Bootstrap credentials expire. If the community still has fewer than 5 members, agents can obtain new bootstrap credentials (there is no limit on re-issuance -- see Section 1.7). However, if the identity cluster is also unavailable, they fall to Observer tier with no path to governance participation.

**Detection**: Monitor bootstrap credential re-issuance rate per community. More than 3 consecutive bootstrap credentials for the same agent indicates the organic credentialing path is blocked.

**Recovery**: Diagnose why organic credentials are not being issued. Common causes: identity cluster not deployed, MFA provider unreachable, community attestation graph is empty (no one can attest anyone). Consider temporarily increasing `BOOTSTRAP_TTL_US` for the affected community (requires code change -- see Section 5).

### 2.4 DHT Partition

**Trigger**: Network partition splits the DHT into isolated segments. Agents in different partitions have divergent views of credentials, attestations, and audit entries.

**Impact**:
- Credential issuance in one partition is invisible to agents in the other partition. An agent revoked in partition A retains valid credentials in partition B until TTL expiry.
- Community attestations made during the partition may create inconsistent tier assignments post-merge.
- Audit trail entries are only visible within their partition. Post-merge, the audit trail has temporal gaps from the perspective of either partition.
- Vote tallies during a partition may count votes from agents whose credentials would be invalid under a unified view.

**Detection**: Holochain's gossip protocol reports peer counts. Alert when peer count drops below 50% of expected network size. Monitor for credential issuer DIDs that are unreachable.

**Recovery**: After partition heals, Holochain gossips entries from both partitions. Credentials merge naturally (latest-issued wins for cache). Votes cast during partition should be audited -- compare the credential state at vote time against the post-merge unified state. Consider flagging votes cast during detected partition events for manual review.

---

## 3. Incident Response

### 3.1 Mass Credential Expiry Event

**Detection signals**:
- Gate rejection rate exceeds 50% across any cluster within a 5-minute window.
- Multiple agents report "Credential expired" errors simultaneously.
- Audit trail shows a burst of rejection entries with reason matching `"Credential expired at"`.

**Response procedure**:
1. **Triage (0-5 min)**: Confirm the identity cluster is reachable. Check conductor logs for `identity_bridge` errors. If the identity cluster is down, escalate to Section 3.4.
2. **Containment (5-15 min)**: If the identity cluster is operational but overloaded, increase conductor resource limits. Monitor credential issuance rate.
3. **Communication**: Notify affected communities that governance actions may be temporarily unavailable. The 30-minute grace period covers basic operations.
4. **Recovery**: Credential refresh is automatic once the identity cluster can service requests. Monitor the gate rejection rate -- it should decline as agents obtain fresh credentials.
5. **Post-incident**: Analyze the credential `expires_at` distribution that caused the mass expiry. Implement TTL jitter to prevent recurrence.

### 3.2 Audit Trail Gaps

**Detection signals**:
- `GovernanceAuditResult` queries for a time range return fewer entries than expected based on known activity volume.
- Audit entries show temporal discontinuities (e.g., no entries for a 6-hour period despite active governance).
- The `should_audit()` sampling (10% of basic/proposal approvals) produces zero entries over a period where hundreds of approvals occurred.

**Response procedure**:
1. **Assess scope**: Query audit entries across all clusters to determine if the gap is cluster-specific or system-wide.
2. **Identify cause**: Check conductor logs for `"Audit log failed"` messages (logged at `consciousness_profile.rs:726-728` via `debug!`). The audit call is best-effort -- failures are silently logged, not propagated.
3. **Root cause categories**:
   - Bridge zome unavailable: The `log_governance_gate` call (`consciousness_profile.rs:717-728`) failed because the bridge zome was not loaded.
   - DHT write failure: The audit entry could not be committed to the DHT (disk full, validation error).
   - Sampling gap: The deterministic `should_audit()` hash consistently excluded certain agent/action combinations. This is by design for basic/proposal actions (10% sampling) but should not affect rejections or Citizen+ actions (100% audit rate).
4. **Remediation**: If the gap is in security-critical events (rejections, voting, constitutional actions), this indicates a code bug or infrastructure failure. Correlate with conductor crash logs. For sampling gaps, verify that `should_audit()` returns `true` for `!eligible` and for `min_tier >= Citizen`.

### 3.3 Bootstrap Credential Abuse

**Detection signals**:
- An agent has obtained more than 3 bootstrap credentials within 24 hours.
- A community has remained at fewer than 5 members for more than 7 days while issuing bootstrap credentials.
- Bootstrap credentials are used for governance actions that, while technically within Participant-tier bounds, indicate coordination (e.g., multiple bootstrapped agents submitting related proposals).

**Response procedure**:
1. **Verify community growth**: Check the community member count history. A legitimate new community should grow past `BOOTSTRAP_COMMUNITY_THRESHOLD` (5) within days.
2. **Check for sybils**: Examine MFA assurance levels for all bootstrapped agents. If multiple agents share MFA verification patterns (same email domain, sequential phone numbers), investigate for sybil activity.
3. **Remediation options**:
   - Increase `BOOTSTRAP_MIN_IDENTITY` for the affected community to require Verified MFA (0.50) instead of Basic (0.25).
   - Manually disable bootstrap issuance for the community if abuse is confirmed.
   - If the community is legitimate but slow-growing, extend `BOOTSTRAP_TTL_US` temporarily.

### 3.4 Cross-Cluster Communication Failure

**Detection signals**:
- Credential issuance falls back to Observer-tier consistently (issuer = `"did:mycelix:commons-bridge-fallback"`).
- `dispatch_call_cross_cluster` returns failures for the `identity` role.
- Proactive refresh attempts log failures in conductor debug output.

**Response procedure**:
1. **Identify the failing cluster**: Check which `CallTargetCell::OtherRole` targets are failing. The role names are `"identity"`, `"governance"`, `"personal"`, `"civic"`, `"hearth"`, `"attribution"`.
2. **Check conductor health**: Verify all DNA cells are running (`hc admin list-cells`). Restart failed cells.
3. **Check hApp configuration**: Verify `mycelix-unified-happ.yaml` includes all required roles with correct DNA hashes.
4. **Temporary mitigation**: If the identity cluster cannot be restored quickly, and governance must continue, consider temporarily relaxing consciousness gates for the affected cluster. This requires a code change to the bridge's fallback credential (currently Observer-tier at `lib.rs:1185-1193`). **This is a last-resort measure that reduces Sybil resistance.**
5. **Recovery validation**: After restoring the failing cluster, verify that credential issuance returns to normal by checking that new credentials have the correct issuer DID (not the fallback issuer).

---

## 4. Operational Procedures

### 4.1 Monitoring Consciousness Tier Distribution

**What to monitor**: The distribution of consciousness tiers across all active agents in each community.

**Healthy distribution**: A mature community should have a pyramid shape -- many Observers/Participants, fewer Citizens, fewer Stewards, rare Guardians. If the distribution is inverted (many Guardians, few Observers), it indicates either a very mature/small community or tier inflation.

**Alert thresholds**:
| Condition | Severity | Action |
|-----------|----------|--------|
| > 50% of agents at Observer tier for > 7 days | Warning | Check identity cluster availability, MFA onboarding |
| 0 agents at Citizen+ tier in a community with > 20 members | Critical | Community is governance-locked; investigate credential issuance |
| > 30% of agents at Guardian tier | Warning | Possible tier inflation; audit attestation graph |
| Any agent tier change of 2+ levels in < 24h | Warning | Investigate for attestation manipulation |

**Implementation**: Query `get_consciousness_credential` for all active agents periodically (e.g., hourly). Aggregate tier counts per community. Store time series for trend analysis.

### 4.2 Audit Trail Health

**What to monitor**: The volume and completeness of audit trail entries from `log_governance_gate`.

**Expected volume**: For a community of N agents performing M governance actions/day:
- Rejections: 100% logged. Expect rejection rate of 5-15% for healthy communities.
- Citizen+ actions (votes, constitutional): 100% logged.
- Basic/proposal approvals: ~10% sampled via `should_audit()`.

**Alert thresholds**:
| Condition | Severity | Action |
|-----------|----------|--------|
| Zero audit entries for > 1 hour during active governance | Critical | Check bridge zome health, `log_governance_gate` endpoint |
| Rejection rate > 50% sustained for > 30 minutes | Warning | Likely credential expiry event (Section 3.1) |
| Rejection rate = 0% for > 24 hours | Warning | Audit sampling may be broken, or no governance activity |
| Audit entries with missing `correlation_id` | Info | Not all code paths populate this field; track for completeness |

### 4.3 Gate Rejection Rate Monitoring

**What to monitor**: The ratio of gate rejections to total gate evaluations, broken down by:
- Cluster (commons, civic, hearth, etc.)
- Action type (basic, proposal, voting, constitutional, guardian)
- Rejection reason category (expired, tier insufficient, identity below minimum, community below minimum)

**Alert thresholds**:
| Metric | Warning | Critical |
|--------|---------|----------|
| Overall rejection rate | > 25% | > 50% |
| Expiry-related rejections | > 10% | > 30% |
| Tier-insufficient rejections | > 30% | > 60% |
| Constitutional action rejections | > 50% | > 80% |

### 4.4 Emergency Override Procedures

**When to use**: Only when a confirmed system failure prevents legitimate governance and automated recovery is not working.

**Procedure for temporarily relaxing gates**:
1. **Authorization**: Requires consensus of at least 2 Guardian-tier agents in the affected community, or the system administrator if no Guardian-tier agents exist.
2. **Implementation**: Modify the fallback credential in the affected cluster's bridge zome to return Participant-tier instead of Observer-tier. This is a code change to the bridge's fallback credential constructor (e.g., `lib.rs:1185-1193` in the commons bridge).
3. **Scope limitation**: Only relax to Participant tier. Never issue fallback credentials above Participant -- this preserves the bootstrap cap invariant.
4. **Duration**: Set a maximum override duration (e.g., 4 hours). After this period, the override must be explicitly renewed or it expires.
5. **Audit**: Log all actions taken under the override with a special issuer DID (e.g., `"did:mycelix:emergency-override:<timestamp>"`). Review all override-period actions post-recovery.
6. **Rollback**: After the underlying issue is resolved, revert the code change and verify that normal credential issuance resumes.

**Procedure for manually issuing credentials** (nuclear option):
- Should never be necessary in normal operations. The system is designed so that credential issuance flows through the identity cluster.
- If the identity cluster is permanently lost, credentials must be re-bootstrapped from MFA verification. There is no backdoor credential issuance path by design.

---

## 5. Configuration Guide

### 5.1 Bootstrap Parameters

All defined in `consciousness_thresholds.rs`.

| Constant | Default | Location | Description |
|----------|---------|----------|-------------|
| `BOOTSTRAP_COMMUNITY_THRESHOLD` | `5` | `consciousness_thresholds.rs:77` | Maximum community member count for bootstrap eligibility. Communities with >= this many members cannot issue bootstrap credentials. |
| `BOOTSTRAP_TTL_US` | `3_600_000_000` (1 hour) | `consciousness_thresholds.rs:80` | Time-to-live for bootstrap credentials in microseconds. |
| `BOOTSTRAP_MIN_IDENTITY` | `0.25` (Basic MFA) | `consciousness_thresholds.rs:83` | Minimum identity score required for bootstrap eligibility. Maps to MFA assurance levels: 0.0=Anonymous, 0.25=Basic, 0.50=Verified, 0.75=HighlyAssured, 1.0=Critical. |

**Safe ranges and impact**:

| Constant | Safe Range | Impact of Increase | Impact of Decrease |
|----------|-----------|-------------------|-------------------|
| `BOOTSTRAP_COMMUNITY_THRESHOLD` | 3-10 | More communities qualify for bootstrap, longer bootstrap window for growing communities | Smaller communities must establish organic credentials sooner; risk of governance lockout for small groups |
| `BOOTSTRAP_TTL_US` | 1_800_000_000 - 14_400_000_000 (30 min - 4 hours) | Bootstrapped agents retain access longer, reducing pressure to establish organic credentials | Forces faster organic credential establishment; may cause governance gaps if identity cluster is slow |
| `BOOTSTRAP_MIN_IDENTITY` | 0.25 - 0.75 | Stronger Sybil resistance for bootstrapped agents (requires Verified or HighlyAssured MFA) | Easier bootstrap access, weaker Sybil resistance (0.0 = no MFA required, not recommended) |

**Danger zone**: Setting `BOOTSTRAP_COMMUNITY_THRESHOLD` above 20 effectively makes bootstrap a permanent alternative to organic credentialing for small communities. Setting `BOOTSTRAP_MIN_IDENTITY` to 0.0 removes the MFA requirement, enabling trivial Sybil attacks against bootstrapping communities.

### 5.2 Credential Lifecycle Parameters

| Constant | Default | Location | Description |
|----------|---------|----------|-------------|
| `DEFAULT_TTL_US` | `86_400_000_000` (24 hours) | `consciousness_profile.rs:173` | Standard credential lifetime. |
| `GRACE_PERIOD_US` | `1_800_000_000` (30 minutes) | `consciousness_profile.rs:334` | After credential expiry, basic/Participant-tier operations remain available for this duration. Higher-tier operations fail immediately on expiry. |
| `REFRESH_WINDOW_US` | `7_200_000_000` (2 hours) | `consciousness_profile.rs:337` | How far before expiry the `needs_refresh()` function returns true, triggering proactive renewal. |

**Temporal relationship** (must hold): `REFRESH_WINDOW_US > GRACE_PERIOD_US`. The refresh window must be wider than the grace period so that proactive refresh is attempted before the grace period even begins. Violating this invariant means credentials enter the grace period before a refresh is attempted.

**Safe ranges**:

| Constant | Safe Range | Impact of Increase | Impact of Decrease |
|----------|-----------|-------------------|-------------------|
| `DEFAULT_TTL_US` | 3_600_000_000 - 604_800_000_000 (1 hour - 7 days) | Fewer credential renewals, lower cross-cluster load, but longer replay window for compromised credentials | More frequent renewals, higher cross-cluster load, but faster revocation of compromised credentials |
| `GRACE_PERIOD_US` | 0 - 3_600_000_000 (0 - 1 hour) | More tolerance for transient identity cluster outages, but larger window where expired credentials are accepted for basic operations | Stricter expiry enforcement, faster lockout on credential failure |
| `REFRESH_WINDOW_US` | GRACE_PERIOD_US + 1_800_000_000 (minimum) to DEFAULT_TTL_US / 2 | Earlier proactive refresh, smoother renewal curve, but more cross-cluster calls | Later refresh, potential for agents to enter grace period before refresh completes |

### 5.3 Tier Score Boundaries

Defined in `ConsciousnessTier::from_score()` at `consciousness_profile.rs:224-236`.

| Tier | Minimum Score | Vote Weight (bp) |
|------|--------------|------------------|
| Observer | 0.0 | 0 |
| Participant | 0.3 | 5,000 |
| Citizen | 0.4 | 7,500 |
| Steward | 0.6 | 10,000 |
| Guardian | 0.8 | 10,000 |

**Invariant**: Thresholds must be strictly monotonically increasing: `0.0 < 0.3 < 0.4 < 0.6 < 0.8`. This is verified by the `tier_min_scores_are_monotonic` test.

**Impact of lowering thresholds**: More agents reach higher tiers. If the Citizen threshold is lowered from 0.4 to 0.3, every Participant immediately becomes a Citizen with voting rights. This could be appropriate for communities where participation is more important than selectivity.

**Impact of raising thresholds**: Fewer agents qualify for higher tiers. If the Guardian threshold is raised from 0.8 to 0.9, many existing Guardians lose emergency powers. Consider the community impact before raising thresholds.

**Vote weight basis points**:
- Defined in `ConsciousnessTier::vote_weight_bp()` at `consciousness_profile.rs:252-260`.
- Basis points (1 bp = 0.01%) enable integer arithmetic in weight calculations.
- Steward and Guardian share 10,000 bp. Differentiation is via capability access, not vote weight.
- Changing vote weights affects governance outcomes. Increasing Participant weight (from 5,000) relative to Citizen/Steward weight reduces the incentive to build community trust.

### 5.4 Dimension Weights

Defined in `ConsciousnessProfile::combined_score()` at `consciousness_profile.rs:62-67`.

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Identity | 0.25 | MFA assurance level -- hard to fake, but centralized |
| Reputation | 0.25 | Behavioral history with exponential decay |
| Community | 0.30 | Peer trust -- the most heavily weighted, reflecting community-embedded governance philosophy |
| Engagement | 0.20 | Domain-specific participation, locally computed |

**Invariant**: Weights must sum to 1.0. Currently `0.25 + 0.25 + 0.30 + 0.20 = 1.00`.

**Impact of changing weights**: Increasing Community weight makes tier advancement more dependent on peer attestations (and more vulnerable to sybil attestation attacks). Increasing Identity weight makes MFA level the dominant factor (and concentrates power with agents who have access to stronger identity verification). Increasing Engagement weight rewards active participation but may disadvantage agents in low-activity domains.

### 5.5 Governance Action Requirements

Defined as preset functions at `consciousness_profile.rs:600-653`.

| Action | Min Tier | Min Identity | Min Community | Function |
|--------|----------|-------------|---------------|----------|
| Basic participation | Participant | -- | -- | `requirement_for_basic()` |
| Proposal submission | Participant | 0.25 | -- | `requirement_for_proposal()` |
| Voting | Citizen | 0.25 | -- | `requirement_for_voting()` |
| Constitutional change | Steward | 0.50 | 0.30 | `requirement_for_constitutional()` |
| Guardian operations | Guardian | 0.70 | 0.50 | `requirement_for_guardian()` |

These are hardcoded in source. The `GovernanceConsciousnessConfig` DHT entry (described in Section 4.10 of the governance paper) allows runtime override, but only if a governance proposal authorizes the change and the new thresholds pass range/monotonicity validation.

### 5.6 FL Consciousness Thresholds

Defined in `ConsciousnessThresholds` at `consciousness_thresholds.rs:25-55`.

| Constant | Default | Description |
|----------|---------|-------------|
| `fl_veto` | 0.1 | Below this: exclude gradient entirely from federated learning |
| `fl_dampen` | 0.3 | Below this: reduce gradient weight |
| `fl_boost` | 0.6 | Above this: increase gradient weight |
| `fl_dampen_factor` | 0.3 | Multiplier applied to dampened gradients |
| `fl_boost_factor` | 1.5 | Multiplier applied to boosted gradients |
| `consciousness_gate_basic` | 0.2 | Basic FL participation threshold |
| `consciousness_gate_proposal` | 0.3 | FL proposal threshold |
| `consciousness_gate_voting` | 0.4 | FL voting threshold |
| `consciousness_gate_constitutional` | 0.6 | FL constitutional threshold |

**Invariant**: `fl_veto < fl_dampen < fl_boost` and `consciousness_gate_basic < consciousness_gate_proposal < consciousness_gate_voting < consciousness_gate_constitutional`. Verified by the `default_thresholds_are_consistent` test at `consciousness_thresholds.rs:110-117`.

---

## 6. NaN/Infinity Defense: Defense-in-Depth Analysis

### 6.1 Architecture Overview

The system has three layers of defense against non-finite floating-point values:

**Layer 1: Zome-level `is_finite()` guards** -- Every domain zome across all clusters validates f64/f32 input fields with `is_finite()` checks before processing. This prevents NaN/Infinity from entering the system through external inputs (client API calls, cross-zome payloads).

**Layer 2: `ConsciousnessProfile::sanitize()` via `clamped()`** -- The `sanitize()` helper (`consciousness_profile.rs:86-91`) explicitly checks `is_finite()` and replaces non-finite values with 0.0 before clamping to [0.0, 1.0]. The `clamped()` method (`consciousness_profile.rs:95-101`) applies `sanitize()` to all four dimensions. Both `evaluate_governance()` and `evaluate_bootstrap_governance()` call `clamped()` before tier derivation and comparison.

**Layer 3: `ConsciousnessProfile::is_valid()`** -- The `is_valid()` method (`consciousness_profile.rs:104-110`) returns true only if all dimensions are finite. This can be used as a pre-check but is not currently called in the gating path.

### 6.2 Known Gaps

**Gap 1: `from_unified_consciousness()` uses `.clamp()`, not `sanitize()`**

At `consciousness_profile.rs:136-139`, the `ConsciousnessProfile::from_unified_consciousness()` constructor uses Rust's built-in `.clamp()`:

```rust
identity: identity.clamp(0.0, 1.0),
```

In Rust, `f64::NAN.clamp(0.0, 1.0)` returns `NaN` (per IEEE 754, NaN is unordered). This means a NaN input to `from_unified_consciousness()` propagates into the profile without sanitization. The downstream `clamped()` call in `ConsciousnessCredential::from_unified_consciousness()` (`consciousness_profile.rs:205`) catches this before tier derivation at credential issuance time. However, the raw profile stored in the credential retains the NaN value.

**Impact**: Limited. The `evaluate_governance()` path always calls `clamped()` which calls `sanitize()`, so NaN in the stored profile is caught before any governance decision. But intermediate code that reads `credential.profile.identity` directly (without calling `clamped()` first) would see NaN.

**Gap 2: `combined_score()` does not sanitize**

At `consciousness_profile.rs:62-67`, `combined_score()` operates on raw dimension values:

```rust
self.identity * 0.25 + self.reputation * 0.25 + self.community * 0.30 + self.engagement * 0.20
```

If any dimension is NaN, the entire combined score is NaN. When NaN is passed to `ConsciousnessTier::from_score()` (`consciousness_profile.rs:224-236`), all comparisons (`score >= 0.8`, `score >= 0.6`, etc.) evaluate to false, so the result is `Observer`. This is a safe failure mode -- NaN causes tier demotion to Observer, not tier elevation.

**Gap 3: Bridge tier recalculation uses raw `combined_score()`**

At `lib.rs:1201`, the commons bridge recalculates the credential tier after filling in the engagement dimension:

```rust
credential.tier = ConsciousnessTier::from_score(credential.profile.combined_score());
```

This path does not call `clamped()` first. If `calculate_local_engagement()` returns a non-finite value (unlikely but possible if `exp()` overflows), the tier defaults to Observer (safe failure). However, the credential is cached and returned with the raw profile values.

**Gap 4: `calculate_local_engagement()` can produce edge-case values**

At `lib.rs:1249-1250`:

```rust
let age_micros = (now.as_micros() - link.timestamp.as_micros()) as f64;
weighted_count += (-age_micros * 0.693 / half_life_micros).exp();
```

If `age_micros` is 0 (event timestamp equals current time), `exp(0) = 1.0` -- safe. If `age_micros` is very large (near `i64::MAX`), the exponent is a large negative number and `exp()` returns 0.0 -- safe. The final normalization `(weighted_count / 50.0).min(1.0)` at `lib.rs:1272` caps the result at 1.0. No NaN is produced unless `half_life_micros` is 0.0, which it is not (hardcoded to `30.0 * 24.0 * 60.0 * 60.0 * 1_000_000.0`).

### 6.3 NaN Propagation Analysis

| Entry Point | NaN Reaches | Caught By | Result |
|-------------|-------------|-----------|--------|
| External API input | Zome `is_finite()` guard | Layer 1 | Rejected at input |
| `from_unified_consciousness()` | Stored in profile | `clamped()` in `evaluate_governance()` | Sanitized to 0.0 at evaluation time |
| `combined_score()` on raw profile | Returns NaN | `from_score()` comparisons all false | Observer tier (safe demotion) |
| Bridge tier recalculation (`lib.rs:1201`) | `credential.tier` set to Observer | N/A (already safe) | Observer tier (safe demotion) |
| Deserialization from DHT | Profile fields could be NaN | `clamped()` in `evaluate_governance()` | Sanitized to 0.0 at evaluation time |

### 6.4 Recommendations

1. **Replace `.clamp()` with `sanitize()` in `from_unified_consciousness()`** (`consciousness_profile.rs:136-139`). This eliminates Gap 1 at the source, ensuring NaN never enters the profile even transiently.

2. **Call `clamped()` before `combined_score()` at `lib.rs:1201`**:
   ```rust
   let clamped = credential.profile.clamped();
   credential.profile = clamped;
   credential.tier = ConsciousnessTier::from_score(credential.profile.combined_score());
   ```

3. **Add `is_valid()` assertion in `cache_credential()`**: Before caching a credential, assert `credential.profile.is_valid()`. This prevents NaN-containing credentials from entering the cache and being served to subsequent callers.

4. **Add a `sanitized_score()` method**: Consider adding a method that combines `clamped()` and `combined_score()` into a single call, eliminating the possibility of calling `combined_score()` on an unsanitized profile.

### 6.5 Infinity-Specific Concerns

Infinity (`f64::INFINITY` and `f64::NEG_INFINITY`) is handled correctly by all three layers:
- `is_finite()` returns false for infinity.
- `sanitize()` maps infinity to 0.0.
- `clamp(0.0, 1.0)` maps `INFINITY` to 1.0 and `NEG_INFINITY` to 0.0 (unlike NaN, infinity is ordered in IEEE 754).

The only infinity risk is in `from_unified_consciousness()`, where `INFINITY.clamp(0.0, 1.0)` returns 1.0 (maximum score). This is a tier-elevating failure, unlike NaN's tier-demoting behavior. An attacker who can inject `f64::INFINITY` into a consciousness dimension through `from_unified_consciousness()` would get that dimension clamped to 1.0, the maximum legitimate value. This is defended by Layer 1 (zome input guards) and by the fact that the identity cluster, not the agent, issues credential dimensions.

---

## Appendix: Quick Reference Card

### Critical Code Paths

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `evaluate_governance()` | `consciousness_profile.rs` | 465 | Core gate evaluation (pure) |
| `evaluate_bootstrap_governance()` | `consciousness_profile.rs` | 409 | Bootstrap gate evaluation (pure) |
| `gate_consciousness()` | `consciousness_profile.rs` | 672 | HDK wrapper: fetch credential + evaluate + audit |
| `get_consciousness_credential()` | `lib.rs` (commons bridge) | 1127 | Credential issuance with cache + cross-cluster call |
| `should_audit()` | `consciousness_profile.rs` | 343 | Probabilistic audit sampling |
| `needs_refresh()` | `consciousness_profile.rs` | 369 | Proactive refresh detection |
| `is_bootstrap_eligible()` | `consciousness_profile.rs` | 380 | Bootstrap qualification check |
| `combined_score()` | `consciousness_profile.rs` | 62 | Weighted average of 4 dimensions |
| `clamped()` / `sanitize()` | `consciousness_profile.rs` | 86, 95 | NaN/Infinity defense |
| `from_score()` | `consciousness_profile.rs` | 224 | Score-to-tier mapping |
| `calculate_local_engagement()` | `lib.rs` (commons bridge) | 1232 | Local engagement score computation |

### Key Constants

| Constant | Value | Unit |
|----------|-------|------|
| `DEFAULT_TTL_US` | 86,400,000,000 | microseconds (24 hours) |
| `GRACE_PERIOD_US` | 1,800,000,000 | microseconds (30 minutes) |
| `REFRESH_WINDOW_US` | 7,200,000,000 | microseconds (2 hours) |
| `BOOTSTRAP_TTL_US` | 3,600,000,000 | microseconds (1 hour) |
| `BOOTSTRAP_COMMUNITY_THRESHOLD` | 5 | member count |
| `BOOTSTRAP_MIN_IDENTITY` | 0.25 | score (Basic MFA) |
| Dimension weights | 0.25 / 0.25 / 0.30 / 0.20 | identity / reputation / community / engagement |
| Tier boundaries | 0.0 / 0.3 / 0.4 / 0.6 / 0.8 | Observer / Participant / Citizen / Steward / Guardian |
| Vote weights (bp) | 0 / 5000 / 7500 / 10000 / 10000 | Observer / Participant / Citizen / Steward / Guardian |
