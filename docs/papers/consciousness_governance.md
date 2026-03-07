# Consciousness-Aware Access Control for Distributed Governance: A Multi-Dimensional Profile Approach

**Authors:** Tristan Stoltz, Luminous Dynamics

**Target Venue:** IEEE International Conference on Blockchain and Cryptocurrency (ICBC) / Distributed Ledger Technology Workshop

---

## Abstract

Decentralized governance systems face persistent challenges: Sybil attacks undermine one-person-one-vote integrity, token-weighted voting concentrates power among wealthy participants, and low participation rates erode democratic legitimacy. We present Mycelix, a distributed governance framework built on Holochain that introduces *consciousness-aware access control* -- a novel governance primitive that gates participation and weights votes through a four-dimensional agent profile comprising Identity verification, Reputation history, Community trust attestations, and domain-specific Engagement. These dimensions are combined into a weighted score that maps agents to five progressive tiers (Observer through Guardian), each granting incrementally greater governance capabilities and vote weights. The system supports cold-start communities through a bootstrap credential mechanism with bounded privileges, handles credential expiry through a grace period and proactive refresh protocol, and maintains an audit trail through probabilistic sampling that reduces DHT writes by approximately 90%. We implement this architecture across 7 Holochain DNA clusters comprising 85 zomes with over 9,800 tests. The gating kernel is a pure function with O(1) evaluation cost, requiring no network calls at governance decision time. We analyze the system's security properties including Sybil resistance through Ed25519-authenticated consciousness attestations and anti-plutocracy constraints that cap financial influence at 5% of vote weight. To our knowledge, this is the first distributed governance system that integrates multi-dimensional identity and behavioral profiling into a formally structured access control layer with progressive capability escalation.

---

## 1. Introduction

Governance in decentralized systems remains an unsolved problem. The design space is constrained by a trilemma: systems must simultaneously resist Sybil attacks (where a single adversary creates multiple identities to gain disproportionate influence), prevent plutocratic capture (where wealth determines political power), and maintain sufficient participation to legitimize collective decisions. Existing approaches typically optimize for one or two of these properties at the expense of the third.

Token-weighted governance, as deployed in most Decentralized Autonomous Organizations (DAOs), addresses Sybil resistance by requiring economic commitment but introduces severe plutocratic tendencies [Buterin 2018]. Quadratic voting mechanisms [Buterin and Weyl 2019] reduce the marginal influence of large token holders but remain vulnerable to collusion through identity splitting. Reputation-based systems [Kamvar et al. 2003] build on behavioral history but face bootstrapping problems and offer limited resistance to long-term strategic manipulation.

A deeper issue underlies these technical challenges: existing governance systems treat all participants as interchangeable units differentiated only by token holdings or accumulated reputation points. They lack a mechanism to assess the *quality* of participation -- the degree to which an agent has established verified identity, built community trust, demonstrated domain engagement, and maintained a consistent behavioral record across multiple dimensions simultaneously.

We propose *consciousness-aware governance*, a framework that constructs a multi-dimensional profile for each agent and uses it to gate governance actions with progressive capability escalation. The term "consciousness" here refers not to phenomenal experience but to a composite measure of an agent's verified presence, demonstrated trustworthiness, community embeddedness, and active engagement -- an operational analog to the degree of awareness and intentionality an agent brings to governance participation.

Our system, Mycelix, implements this framework on Holochain [Harris-Braun et al. 2018], an agent-centric distributed computing platform. Each agent carries a *consciousness credential* -- a time-limited, cryptographically signed attestation of their four-dimensional profile -- that is evaluated locally by pure functions requiring no network calls at governance decision time. This design yields O(1) gate evaluation, graceful degradation when subsystems are unavailable, and clean separation between credential issuance (which may involve cross-cluster communication) and credential evaluation (which is purely local).

The contributions of this paper are:

1. **A formal 4D consciousness profile** with weighted combination, tier derivation, and progressive vote weighting in basis points (Section 4.1).
2. **A five-tier progressive access control model** that maps combined scores to governance capabilities ranging from read-only observation to emergency powers (Section 4.2).
3. **A bootstrap mechanism** for cold-start communities that provides bounded temporary credentials (Section 4.4).
4. **A pure-function gating kernel** that evaluates credentials without network calls, with probabilistic audit sampling (Section 4.5).
5. **An authenticated attestation protocol** using Ed25519 signatures over structured messages that binds consciousness claims to agent identity (Section 4.6).
6. **A production implementation** across 85 Holochain zomes in 7 DNA clusters with over 9,800 tests (Section 6).

---

## 2. Background

### 2.1 Holochain

Holochain [Harris-Braun et al. 2018] is an agent-centric distributed computing framework in which each participant maintains a local hash chain (source chain) and shares data through a distributed hash table (DHT) validated by a subset of peers. Unlike blockchain, there is no global consensus; validation is performed by the nodes responsible for storing each piece of data. Applications are structured as *DNAs* containing *zomes* (WebAssembly modules) that define entry types, link types, and validation rules. Multiple DNAs can be bundled into a *hApp* and assigned roles, enabling cross-DNA communication via `CallTargetCell::OtherRole`.

### 2.2 The Sybil Problem in Decentralized Governance

The Sybil attack [Douceur 2002] is the fundamental threat to any system where influence scales with the number of identities. In governance, a single adversary who controls multiple identities can manufacture artificial majorities. Proof-of-work and proof-of-stake blockchains address this by making identity creation expensive, but this conflates economic resources with political voice -- the very plutocratic tendency that motivates decentralized governance in the first place.

### 2.3 Consciousness as a Governance Primitive

We use "consciousness" in an operational sense: the degree to which an agent is verifiably present, historically trustworthy, community-embedded, and actively engaged. This aligns with the philosophical notion of *consciousness as integration* [Tononi 2004] -- not at the level of neural correlates, but at the level of social and informational integration within a governance community. An agent with high consciousness, in our usage, is one whose multiple dimensions of participation are simultaneously strong and mutually reinforcing.

---

## 3. Related Work

**Quadratic Voting.** Buterin and Weyl [2019] propose quadratic voting (QV) as a mechanism to reduce the marginal influence of large token holders. Under QV, the cost of votes grows quadratically: casting *n* votes on a proposal costs *n^2* voice credits. This achieves optimal preference expression under a social welfare framework but requires a trusted identity layer to prevent one agent from splitting into multiple identities, each casting fewer votes at lower cost. Our system provides this identity layer through the Identity dimension of the consciousness profile.

**Token-Weighted Governance.** Most DAOs (Compound, MakerDAO, Uniswap) use token-weighted voting where governance power is proportional to token holdings [Barbereau et al. 2022]. This creates plutocratic dynamics: in practice, a small number of "whale" addresses control most governance outcomes. Mycelix explicitly constrains financial influence: the `STAKE_MAX_BONUS` constant caps stake contribution at 5% of total vote weight, and the vote weight formula squares reputation to make behavioral history the dominant factor.

**Reputation Systems.** EigenTrust [Kamvar et al. 2003] computes global trust values through iterative aggregation of local trust assessments. PageRank-style approaches [Page et al. 1999] rank participants by the structure of endorsement graphs. These systems capture a single dimension (trust/reputation) and are vulnerable to whitewashing attacks where misbehaving agents abandon compromised identities and re-enter with fresh ones. Our system's Identity dimension, anchored in multi-factor authentication with assurance levels (Anonymous through Critical), makes whitewashing costly.

**Liquid Democracy.** Liquid democracy [Blum and Zuber 2016] allows voters to delegate their votes to domain experts, who may further re-delegate. This addresses the competence problem (voters may lack expertise) but introduces centralization risk through delegation chains. Mycelix supports delegation with decay -- the `resolve_delegation_chain` function reduces delegated vote weight at each hop -- limiting the influence of delegation hubs.

**Proof of Humanity.** Proof of Humanity [Clement 2020] and similar protocols (BrightID, Worldcoin) attempt to establish unique human identity through social vouching, biometric verification, or in-person ceremonies. These systems solve the Sybil problem at the identity layer but do not integrate identity verification into governance weighting. Our Identity dimension maps directly to MFA assurance levels (0.0 for anonymous, 0.25 for basic, 0.50 for verified, 0.75 for highly assured, 1.0 for critical), making identity strength a continuous variable rather than a binary gate.

**Gap in Existing Work.** No existing system combines multi-dimensional profiling (identity, reputation, community trust, engagement) with progressive access control tiers, pure-function evaluation, cold-start bootstrapping, and anti-plutocracy constraints in a single coherent framework. Our contribution fills this gap.

---

## 4. Architecture

### 4.1 The Four-Dimensional Consciousness Profile

Each agent in the Mycelix network carries a `ConsciousnessProfile` -- a four-dimensional vector where each dimension is normalized to [0.0, 1.0]:

1. **Identity** (*w* = 0.25): The strength of the agent's identity verification, derived from the MFA assurance level in the identity cluster. Values map to discrete levels: Anonymous (0.0), Basic (0.25), Verified (0.50), Highly Assured (0.75), and Critical (1.0).

2. **Reputation** (*w* = 0.25): Cross-hApp aggregated reputation computed from behavioral history. Reputation decays exponentially with a 30-day half-life to ensure that historical contributions do not confer permanent privilege. Multiple signal sources are aggregated with domain-specific weights; per the Commons Charter, financial domain signals (stake weight, payment reliability, escrow completion rate) contribute at most 5% to the aggregated score.

3. **Community** (*w* = 0.30): Peer trust attestations, weighted by the attestor's own consciousness tier. Higher-tier attestors contribute more to a target's Community score, creating a virtuous cycle where trusted agents' endorsements carry greater weight. This is the most heavily weighted dimension, reflecting the design philosophy that community-embedded agents should have the strongest governance voice.

4. **Engagement** (*w* = 0.20): Domain-specific participation computed locally by each cluster's bridge zome. Based on event and query participation counts with temporal decay. In the Minimal Viable Bridge to external consciousness systems (such as the Symthaea cognitive engine), the unified consciousness score maps 1:1 to this dimension.

The **combined score** is the weighted average:

```
S = 0.25 * Identity + 0.25 * Reputation + 0.30 * Community + 0.20 * Engagement
```

All dimension values are clamped to [0.0, 1.0] before combination, preventing adversarial injection of negative or supranormal values.

### 4.2 Five-Tier Progressive Access

The combined score maps to one of five tiers through a monotonically increasing threshold function:

| Tier | Min Score | Vote Weight (bp) | Capabilities |
|------|-----------|-------------------|-------------|
| Observer | 0.0 | 0 | Read-only access |
| Participant | 0.3 | 5,000 | Basic proposals, commenting |
| Citizen | 0.4 | 7,500 | Voting rights |
| Steward | 0.6 | 10,000 | Constitutional amendments |
| Guardian | 0.8 | 10,000 | Emergency powers, system administration |

Vote weights are expressed in basis points (1 bp = 0.01%) to enable integer arithmetic in weight calculations. The tier ordering is total and monotonic: `Observer < Participant < Citizen < Steward < Guardian`, enforced by the derived `Ord` implementation.

Each governance action specifies a `GovernanceRequirement` consisting of:
- A minimum tier (always checked)
- An optional minimum Identity dimension (e.g., voting requires Identity >= 0.25)
- An optional minimum Community dimension (e.g., constitutional changes require Community >= 0.30)

Standard requirement presets are provided as pure functions:

| Action | Min Tier | Min Identity | Min Community |
|--------|----------|-------------|---------------|
| Basic participation | Participant | -- | -- |
| Proposal submission | Participant | 0.25 | -- |
| Voting | Citizen | 0.25 | -- |
| Constitutional change | Steward | 0.50 | 0.30 |
| Guardian operations | Guardian | 0.70 | 0.50 |

Constitutional changes require both strong identity verification (at least Verified MFA) and meaningful community trust, ensuring that the most consequential governance actions are restricted to well-established, multiply-verified agents.

### 4.3 Consciousness Credentials

The `ConsciousnessCredential` is a time-limited container for the consciousness profile, issued by bridge zomes and cached on the agent's source chain:

- **TTL**: 24 hours by default (`DEFAULT_TTL_US = 86,400,000,000` microseconds)
- **Grace period**: 30 minutes after expiry (`GRACE_PERIOD_US = 1,800,000,000` microseconds), during which basic-tier (Participant or below) operations remain available
- **Proactive refresh**: Credentials within 2 hours of expiry (`REFRESH_WINDOW_US = 7,200,000,000` microseconds) are flagged for refresh via the `needs_refresh()` function
- **Issuer tracking**: Each credential records the DID of the issuing bridge zome for provenance

This three-layer temporal design (proactive refresh window, nominal expiry, grace period) ensures that governance is never abruptly interrupted by clock skew or transient unavailability of the credential issuance infrastructure, while still guaranteeing that stale credentials are eventually rejected.

### 4.4 Bootstrap Mechanism

Cold-start communities face a chicken-and-egg problem: agents need community trust attestations to earn governance privileges, but they cannot participate in the community without governance privileges. The bootstrap mechanism resolves this:

**Eligibility**: An agent qualifies for bootstrap if:
- The community has fewer than `BOOTSTRAP_COMMUNITY_THRESHOLD` (5) members
- The agent's Identity score meets `BOOTSTRAP_MIN_IDENTITY` (0.25, i.e., at least Basic MFA)

**Credential**: Bootstrap credentials grant Participant tier with a 1-hour TTL (`BOOTSTRAP_TTL_US = 3,600,000,000` microseconds) and vote weight of 5,000 bp. They are issued by a synthetic issuer (`did:mycelix:bootstrap`).

**Cap**: Bootstrap credentials are hard-capped at Participant tier. Any governance action requiring Citizen tier or above (voting, constitutional changes, guardian operations) is rejected even with a valid bootstrap credential. This ensures that bootstrapped communities cannot perform high-stakes governance actions until agents have earned genuine community trust.

**Evaluation**: The `evaluate_bootstrap_governance()` function checks expiry, enforces the Participant cap, and validates per-dimension minimums if required by the governance action. It is separate from the standard `evaluate_governance()` path to maintain clear separation of security properties.

### 4.5 Pure-Function Gating Kernel

The core gate evaluation (`evaluate_governance`) is a pure function: it takes a `ConsciousnessCredential`, a `GovernanceRequirement`, and the current timestamp, and returns a `GovernanceEligibility` result. This design has several critical properties:

1. **No HDK dependency**: The function can be unit-tested without a Holochain conductor, enabling comprehensive testing (the implementation includes 212 tests in the bridge-common crate alone).

2. **O(1) evaluation**: Gate checking involves a fixed sequence of comparisons -- credential expiry check, tier comparison, and at most two dimension comparisons. There are no loops, no network calls, and no DHT lookups.

3. **Deterministic**: Given identical inputs, the function always produces identical outputs, enabling reproducible auditing.

The shared `gate_consciousness()` function wraps the pure evaluation with HDK calls:
1. Fetch the agent's DID from `agent_info()`
2. Cross-zome call to the cluster's bridge for `get_consciousness_credential`
3. Call the pure `evaluate_governance()`
4. Probabilistic audit logging via `should_audit()`
5. Return eligibility or reject with `WasmError`

Each domain coordinator keeps a thin 3-line wrapper that passes its cluster's bridge zome name (e.g., `"commons_bridge"`, `"civic_bridge"`, `"hearth_bridge"`), maintaining the DRY principle across 85 zomes.

### 4.6 Authenticated Consciousness Attestation

To prevent agents from fabricating consciousness scores, the system supports Ed25519-authenticated attestations:

```
message = "symthaea-consciousness-attestation:v1:{agent_did}:{consciousness_level:.6}:{cycle_id}:{captured_at_us}"
```

The agent signs this structured message with their Holochain agent key. The governance bridge verifies the signature against the agent's public key before committing the attestation entry. This binds the consciousness claim to the agent's cryptographic identity, preventing:

- **Score fabrication**: An agent cannot claim a consciousness level they did not actually achieve, as the signature must originate from the same key that controls the agent's source chain.
- **Replay attacks**: The `cycle_id` and `captured_at_us` fields ensure that old attestations cannot be resubmitted as current.
- **Impersonation**: The signature verification uses the caller's `agent_initial_pubkey`, preventing one agent from attesting on behalf of another.

The v2 gate verification (`verify_consciousness_gate_v2`) preferentially uses authenticated attestations when available, falling back to legacy snapshots, and tracking provenance (`Attested`, `Snapshot`, or `Unavailable`) throughout.

### 4.7 Phi-Weighted Voting

The voting system integrates consciousness into vote weight through a multi-factor formula:

```
weight = Reputation^2 * consciousness_multiplier * participation_bonus * stake_modifier * domain_modifier
```

Where:
- `consciousness_multiplier = 0.7 + 0.3 * Phi` when consciousness data is available, or `1.0` when unavailable (neutral, never fabricated)
- `participation_bonus = 1.0 + 0.1 * participation_score`
- `stake_modifier = 1.0 + 0.05 * stake_weight` (capped at 5% per anti-plutocracy charter)
- `domain_modifier = 1.0 + 0.1 * domain_reputation`

The final weight is clamped to [`MIN_VOTING_WEIGHT` (0.1), `MAX_VOTING_WEIGHT` (1.5)], ensuring that no single voter can have more than 15x the influence of the least-weighted voter. Reputation is squared to amplify the difference between high and low reputation agents -- an agent with reputation 0.5 gets only 25% of the weight contribution of an agent with reputation 1.0, creating strong incentives for consistent trustworthy behavior.

When Phi provenance is `Unavailable`, the consciousness multiplier defaults to 1.0 rather than penalizing or rewarding the voter. This is a conscious design decision: agents who have not integrated consciousness measurement should receive reputation-only weighting, not a fabricated consciousness bonus.

### 4.8 Cross-Cluster Architecture

Mycelix is organized as a unified hApp with seven DNA clusters, each assigned a role:

| Cluster | Role | Domains | Zomes |
|---------|------|---------|-------|
| Commons | `commons` | Property, housing, care, mutual aid, water, food, transport | 35 |
| Civic | `civic` | Justice, emergency, media | 16 |
| Hearth | `hearth` | Kinship, gratitude, care, autonomy, decisions | 12 |
| Identity | `identity` | DID registry, MFA, trust credentials | 9 |
| Governance | `governance` | Proposals, voting, DKG threshold signing, councils | 7 |
| Personal | `personal` | Identity vault, health vault, credential wallet | 4 |
| Attribution | `attribution` | Dependency registry, usage receipts, reciprocity | 3 |

Cross-cluster calls use `CallTargetCell::OtherRole`, which routes through Holochain's conductor to the appropriate DNA. Governance queries consciousness data through two paths:

1. **Personal cluster path**: Governance calls `personal_bridge::present_phi_credential` to request a signed credential presentation from the agent's personal vault.
2. **Local bridge path**: Governance calls `governance_bridge::verify_consciousness_gate_v2` for attestation-based verification.

Both paths are attempted in sequence with graceful fallback: if the personal cluster is unavailable, the system falls back to local bridge data; if no consciousness data exists at all, the voter receives reputation-only weighting (consciousness multiplier = 1.0).

The cross-cluster dispatch enforces allowlists: each dispatch function (e.g., `dispatch_personal_call`, `dispatch_identity_call`) validates that the target zome name is in a hardcoded `ALLOWED_*_ZOMES` list, preventing arbitrary cross-cluster calls.

### 4.9 Audit Sampling

Full audit logging of every gate decision would impose unsustainable DHT write load. The `should_audit()` function implements a deterministic sampling strategy:

- **All rejections** are logged (security-critical events)
- **All Citizen-tier-and-above actions** are logged (voting, constitutional changes, guardian operations)
- **~10% of basic/proposal approvals** are sampled using an action-name-salted hash: `agent_hash.last() + salt(action_name) < 26` (26/256 ~ 10.2%)

The salt is the byte-sum of the action name, ensuring that different actions produce different sampling patterns even for the same agent. This prevents systematic gaps where certain agents are never audited for certain actions.

### 4.10 Configurable Consciousness Parameters

Consciousness gate thresholds are not hardcoded in governance zomes. Instead, a `GovernanceConsciousnessConfig` entry on the DHT serves as the runtime source of truth. This configuration includes:

- Per-action-type gate thresholds (basic, proposal, voting, constitutional)
- Per-proposal-type minimum voter consciousness
- Maximum voting weight cap
- Optional per-dimension gates (minimum true Phi for constitutional actions, minimum coherence for voting)

Configuration updates require a governance proposal ID, creating an audit trail. The `update_consciousness_config` function validates range constraints and monotonicity (basic < proposal < voting < constitutional) before committing. If no configuration has been bootstrapped, hardcoded defaults are used.

---

## 5. Formal Properties

We identify five key properties of the consciousness-aware governance system and provide arguments for their satisfaction.

### 5.1 Tier Monotonicity

**Property**: For any two profiles *P1* and *P2*, if *S(P1) < S(P2)* then *tier(P1) <= tier(P2)*.

**Argument**: The `from_score()` function implements a simple threshold ladder with strictly increasing cutpoints (0.0, 0.3, 0.4, 0.6, 0.8). Since the combined score is a linear combination of clamped [0,1] values with positive weights summing to 1.0, the score is itself in [0,1]. The threshold function is monotonically non-decreasing. This property is verified by the `tier_min_scores_are_monotonic` test, which checks that `tiers[i].min_score() > tiers[i-1].min_score()` for all consecutive tiers.

### 5.2 Vote Weight Progressivity

**Property**: For any two tiers *T1 < T2*, `vote_weight_bp(T1) <= vote_weight_bp(T2)`.

**Argument**: The `vote_weight_bp()` function returns 0, 5000, 7500, 10000, 10000 for the five tiers respectively. The sequence is monotonically non-decreasing. This is verified by the `tier_vote_weights_are_progressive` test.

### 5.3 Gate Soundness

**Property**: An agent is granted governance eligibility only if (a) their credential is non-expired (or within grace period for basic actions), (b) their tier meets or exceeds the requirement, and (c) all per-dimension minimums are satisfied.

**Argument**: The `evaluate_governance()` function checks these three conditions in sequence, accumulating rejection reasons. Eligibility is granted only when the reasons list is empty (`reasons.is_empty()`). The grace period relaxation applies only when `requirement.min_tier <= ConsciousnessTier::Participant`, ensuring that high-stakes actions cannot bypass expiry checks. Tests `evaluate_observer_rejected_for_basic`, `evaluate_participant_passes_basic`, `evaluate_participant_rejected_for_voting`, and `evaluate_citizen_passes_voting` verify boundary cases.

### 5.4 Bootstrap Capability Cap

**Property**: Bootstrap credentials never grant capabilities beyond Participant tier.

**Argument**: The `evaluate_bootstrap_governance()` function explicitly checks `requirement.min_tier > ConsciousnessTier::Participant` and rejects with a diagnostic message if true. The bootstrap credential's tier field is always set to `ConsciousnessTier::Participant` by `bootstrap_credential()`. The capability cap is independent of the agent's identity score -- even an agent with Critical-level MFA (identity = 1.0) receives at most Participant tier through bootstrapping.

### 5.5 Audit Completeness for Security-Critical Actions

**Property**: All gate rejections and all Citizen-tier-and-above actions are logged.

**Argument**: The `should_audit()` function returns `true` unconditionally for `!eligible` (rejections) and for `min_tier` of `Citizen`, `Steward`, or `Guardian`. Only `Observer` and `Participant` tier approvals are subject to the 10% sampling. This ensures that the audit trail is complete for all security-relevant events while reducing DHT writes for routine low-privilege operations.

---

## 6. Implementation

### 6.1 Technology Stack

Mycelix is implemented on Holochain 0.6 using the HDK (Holochain Development Kit) for Rust. All zomes compile to WebAssembly (`wasm32-unknown-unknown` target). The codebase consists of:

- **7 DNA clusters** with a total of **85 zomes** (domain zomes plus bridge zomes per cluster)
- **Shared types crate** (`mycelix-bridge-entry-types`): DHT entry type definitions shared across clusters
- **Shared logic crate** (`mycelix-bridge-common`): Pure-function gating kernel, consciousness profile types, threshold constants, cross-cluster dispatch utilities
- **Unified hApp** (`mycelix-unified-happ.yaml`): All 7 clusters deployed as roles in a single hApp, enabling `CallTargetCell::OtherRole` cross-cluster communication

### 6.2 Test Coverage

The system is validated by over 9,800 tests:

| Component | Test Count |
|-----------|-----------|
| Commons cluster | 5,276 |
| Civic cluster | 2,273 |
| Hearth cluster | 1,023 |
| Bridge-common (incl. consciousness profile) | 212 |
| Identity cluster | 123+ |
| Governance cluster | 174+ |
| Personal cluster | 20 |
| Attribution cluster | 17 |
| SDK (Rust + TypeScript) | 7,646 |

The consciousness profile module alone has 50+ unit tests covering: combined score weight correctness, clamping behavior, tier boundary conditions, tier monotonicity, vote weight progressivity, credential expiry with grace periods, governance evaluation for each tier/requirement combination, bootstrap credential cap enforcement, audit sampling determinism, and serialization roundtrips.

### 6.3 Pure-Function Kernel

The gating kernel is deliberately separated from HDK-dependent code. The following functions are pure (no side effects, no Holochain calls) and independently testable:

- `ConsciousnessProfile::combined_score()` -- weighted average
- `ConsciousnessProfile::clamped()` -- dimension normalization
- `ConsciousnessTier::from_score()` -- tier derivation
- `ConsciousnessTier::vote_weight_bp()` -- weight lookup
- `evaluate_governance()` -- credential evaluation
- `evaluate_bootstrap_governance()` -- bootstrap evaluation
- `should_audit()` -- audit sampling
- `needs_refresh()` -- proactive refresh detection
- `is_bootstrap_eligible()` -- bootstrap qualification
- `compute_vote_weight()` -- voting weight computation
- `compute_tally_result()` -- tally aggregation

This separation enables testing without a Holochain conductor and makes the security-critical logic amenable to formal verification.

### 6.4 Security Hardening

All floating-point inputs are validated with `is_finite()` guards across every zome. The `ConsciousnessProfile::clamped()` function normalizes all dimension values to [0.0, 1.0] before any governance decision. Input validation functions (`check_snapshot_input`, `check_alignment_input`, `check_weighted_vote_input`, etc.) enforce length limits on string fields (1-256 characters for IDs, up to 4096 for content) and range constraints on numeric fields.

The Sybil resistance layer operates at two levels:
1. **Agent-level**: `enforce_agent_vote_limit()` prevents a single Holochain agent key from voting multiple times on the same proposal, regardless of which DID they present.
2. **DID-level**: Duplicate vote detection checks whether the presented DID has already voted on the proposal.

---

## 7. Evaluation

### 7.1 Security Analysis

**Sybil Resistance.** An attacker attempting to create multiple identities faces three barriers: (1) each Holochain agent key must be backed by a unique cryptographic keypair, (2) the Identity dimension requires MFA verification through the identity cluster, and (3) the Community dimension requires attestations from existing high-tier members. Creating *k* fake identities requires either compromising *k* independent MFA verification processes or colluding with existing high-tier community members -- both significantly more expensive than simply creating pseudonymous accounts.

**Plutocratic Resistance.** The `STAKE_MAX_BONUS` constant (0.05) caps the financial contribution to vote weight at 5%. Even an agent with maximum stake (1.0) receives at most a 5% weight bonus. By contrast, reputation is squared and contributes a factor of up to 1.0 to the weight, and consciousness contributes up to 30% (0.7 base + 0.3 * Phi). This makes behavioral history and consciousness measurement approximately 20x more influential than financial stake.

**Consciousness Fabrication.** Without Ed25519-authenticated attestations, an agent could self-report an inflated consciousness level. The authenticated attestation protocol binds the reported value to a specific cognitive cycle ID and timestamp, signed by the agent's key. While this does not prevent a compromised consciousness measurement system from producing inflated values, it does ensure that the reported value is the one actually produced by the measurement system -- moving the trust boundary from the governance layer to the consciousness measurement layer.

**Credential Staleness.** The 24-hour TTL, 2-hour proactive refresh window, and 30-minute grace period create a layered defense against stale credentials. Under normal operation, credentials are refreshed proactively before expiry. If the refresh infrastructure fails, agents retain basic capabilities for 30 minutes while high-stakes operations are immediately gated. After the grace period, all operations are blocked until a fresh credential is obtained.

### 7.2 Performance

**Gate Evaluation.** The `evaluate_governance()` function performs: one integer comparison (expiry check), one enum comparison (tier check), and at most two floating-point comparisons (per-dimension minimums). This is O(1) with a small constant factor. No heap allocation occurs during evaluation. The function is called inline at the entry point of every gated coordinator function.

**Credential Issuance.** Issuance involves one cross-zome call (to the identity bridge for MFA level), one DHT read (for reputation history), aggregation of community attestation links, and one DHT write (the credential entry). This is O(k) where k is the number of community attestations, dominated by link traversal.

**DHT Efficiency.** The audit sampling strategy reduces DHT writes for routine operations by approximately 90% (only ~10% of basic/proposal approvals are logged). For a community of 1,000 agents performing 10 basic governance actions per day, this reduces audit writes from 10,000/day to approximately 1,000/day, with all security-critical events (rejections, votes, constitutional actions) still fully logged.

**Vote Weight Computation.** The `compute_vote_weight()` function involves 5 floating-point multiplications, 5 clamp operations, and 1 min/max clamp. All inputs are pre-fetched before the pure computation. The function is independently benchmarkable.

### 7.3 Comparison with Existing Systems

| Property | Token-weighted DAOs | Quadratic Voting | EigenTrust | Mycelix |
|----------|-------------------|-----------------|-----------|---------|
| Sybil resistance | Economic (buy tokens) | Requires external identity | Iterative trust | Multi-factor (MFA + community + reputation) |
| Plutocratic resistance | None | Partial (sqrt cost) | N/A | Strong (5% stake cap, rep-squared) |
| Progressive access | Binary (hold token or not) | Binary (has identity or not) | Continuous score | 5-tier with per-dimension gates |
| Cold-start | Token sale | N/A | Bootstrapping problem | Explicit bootstrap with Participant cap |
| Gate evaluation cost | O(1) balance check | O(1) credit check | O(n) iterative | O(1) pure function |
| Audit trail | On-chain (all txns) | On-chain (all txns) | None | Probabilistic sampling (~10% routine, 100% critical) |

---

## 8. Discussion

### 8.1 Limitations

**Eventual Consistency.** Holochain's DHT provides eventual consistency, not strong consistency. A consciousness credential issued on one node may not be immediately visible to all nodes. This means that in rare cases, an agent could have their credential revoked on one node while still using a cached version on another. The 24-hour TTL bounds the window of this inconsistency, and the grace period provides a safety margin. For constitutional-level actions, the system could be extended to require confirmation from multiple DHT peers before accepting a credential.

**Deterministic Audit Sampling.** The `should_audit()` function uses a deterministic hash of the agent's public key and the action name. A sophisticated adversary who knows their key's hash could predict which actions will be audited and behave differently for unaudited actions. This could be mitigated by incorporating a random beacon or using the Holochain `sys_time()` as additional entropy, at the cost of reproducibility.

**Identity Dimension Centralization.** The Identity dimension relies on MFA verification, which typically involves centralized identity providers (email, phone, government ID). This introduces a single point of failure and a potential censorship vector. Future work could integrate decentralized identity verification (e.g., web-of-trust-based key ceremonies) to reduce this dependency.

**Consciousness Measurement Validity.** When integrated with external consciousness measurement systems (e.g., Symthaea), the Engagement dimension's validity depends on the measurement system's accuracy. The Ed25519 attestation protocol ensures provenance but not correctness -- a systematically biased measurement system would produce systematically biased governance outcomes. The system's defense is the multi-dimensionality of the profile: even with an unreliable Engagement dimension, the Identity, Reputation, and Community dimensions provide independent constraints.

### 8.2 Future Work

**Formal Verification.** The pure-function gating kernel is amenable to formal verification using tools such as Kani or MIRI for Rust. Key properties to verify include: tier monotonicity (proven by test but not formally), gate soundness (all paths through `evaluate_governance` either populate reasons or leave it empty), and bootstrap cap enforcement.

**Consciousness Dashboard.** A real-time visualization showing each agent's four-dimensional profile, current tier, credential status, and vote weight history would improve transparency and enable agents to understand why they are eligible or ineligible for specific governance actions.

**Credential Refresh Automation.** Currently, credential refresh is triggered when `needs_refresh()` returns true. This could be automated through a background task that monitors credential TTL and proactively re-issues credentials before the refresh window opens, eliminating the possibility of expiry-related disruption.

**Adaptive Tier Thresholds.** The current tier thresholds (0.3, 0.4, 0.6, 0.8) are fixed. Community-specific calibration -- where thresholds adjust based on the distribution of combined scores in the population -- could prevent communities from being entirely locked at low tiers (if no member reaches 0.4, no one can vote) or from having all members at Guardian tier (eliminating differentiation).

**Cross-Community Credential Portability.** Currently, consciousness credentials are scoped to a single Mycelix network. A federation protocol that allows agents to present credentials from one community to another -- with appropriate attestation and trust discounting -- would enable governance participation in multiple communities without rebuilding reputation from scratch.

---

## 9. Conclusion

We have presented Mycelix, a consciousness-aware distributed governance system that addresses the Sybil-plutocracy-participation trilemma through a multi-dimensional agent profiling approach. The four-dimensional consciousness profile (Identity, Reputation, Community, Engagement) provides a richer basis for governance decisions than any single metric, while the five-tier progressive access model ensures that governance capabilities scale with demonstrated trustworthiness.

The system's architecture separates credential issuance (which involves cross-cluster communication) from credential evaluation (which is a pure O(1) function), enabling efficient gate checking at every governance action without network overhead. The bootstrap mechanism solves the cold-start problem with bounded temporary credentials, and the authenticated attestation protocol prevents consciousness score fabrication through Ed25519 signature verification.

Our implementation across 85 Holochain zomes in 7 DNA clusters demonstrates the framework's practicality at scale, with over 9,800 tests validating correctness. The anti-plutocracy constraints (5% stake cap, reputation squaring) and the multi-factor Sybil resistance (MFA + community attestations + behavioral history) address governance challenges that no existing system handles in combination.

The fundamental insight of this work is that governance participation quality can be meaningfully assessed through the integration of multiple independent dimensions -- identity verification strength, behavioral reputation, community embeddedness, and active engagement -- and that this assessment can be performed efficiently, securely, and transparently in a fully decentralized setting. By treating consciousness as a governance primitive rather than an afterthought, Mycelix provides a foundation for distributed governance systems that are simultaneously resistant to manipulation, accessible to genuine participants, and responsive to the quality of collective engagement.

---

## References

Barbereau, T., Smethurst, R., Papez, O., Liebenau, J., and Sedlmeir, J. (2022). Decentralised Finance's Unregulated Governance: Minority Rule in the Digital Wild West. *SSRN Electronic Journal*.

Blum, C. and Zuber, C. I. (2016). Liquid Democracy: Potentials, Problems, and Perspectives. *Journal of Political Philosophy*, 24(2):162--182.

Buterin, V. (2018). Notes on Blockchain Governance. *Personal Blog*, https://vitalik.ca/general/2017/12/17/voting.html.

Buterin, V. and Weyl, E. G. (2019). Liberal Radicalism: A Flexible Design for Philanthropic Matching Funds. *Management Science*, 65(11):5171--5187.

Clement, N. (2020). Proof of Humanity: A Social Identity Verification System. *White Paper*, https://proofofhumanity.id.

Douceur, J. R. (2002). The Sybil Attack. In *Proceedings of the 1st International Workshop on Peer-to-Peer Systems (IPTPS)*, pages 251--260. Springer.

Harris-Braun, E., Luck, N., and Brock, A. (2018). Holochain: A Framework for Distributed Applications. *White Paper*, https://holochain.org.

Kamvar, S. D., Schlosser, M. T., and Garcia-Molina, H. (2003). The EigenTrust Algorithm for Reputation Management in P2P Networks. In *Proceedings of the 12th International Conference on World Wide Web (WWW)*, pages 640--651. ACM.

Page, L., Brin, S., Motwani, R., and Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. *Stanford InfoLab Technical Report*.

Tononi, G. (2004). An Information Integration Theory of Consciousness. *BMC Neuroscience*, 5(1):42.
