# Security Architecture

*Protecting Truth-Seeking Infrastructure*

---

> "Security is not a feature; it is a foundation."
> — Us

> "The best security makes attacks uneconomical, not impossible."
> — Cryptographic wisdom

---

## Introduction

Epistemic Markets handles valuable stakes—monetary, reputational, and social. Security is not optional. This document describes our threat model, security architecture, and the mechanisms that protect participants and the protocol.

We approach security with epistemic humility: we don't assume we've thought of everything. This document is a living artifact that evolves as new threats emerge.

---

## Part I: Threat Model

### 1.1 Adversary Categories

#### Category A: Individual Attackers

**A1: Profit-Seeking Attacker**
- **Motivation**: Financial gain
- **Capabilities**: Moderate technical skill, limited resources
- **Attack vectors**: Market manipulation, oracle gaming, stake front-running
- **Risk level**: High (most common)

**A2: Reputation Farmer**
- **Motivation**: Inflated reputation without genuine expertise
- **Capabilities**: Multiple accounts, social engineering
- **Attack vectors**: Sybil attacks, calibration gaming, sock puppet networks
- **Risk level**: Medium

**A3: Vandal/Troll**
- **Motivation**: Disruption, entertainment
- **Capabilities**: Low to moderate
- **Attack vectors**: Spam, harassment, noise injection
- **Risk level**: Low (annoying but manageable)

#### Category B: Organized Groups

**B1: Coordinated Manipulation Ring**
- **Motivation**: Profit or ideology
- **Capabilities**: Multiple coordinated actors, shared resources
- **Attack vectors**: Cartel formation, coordinated voting, information warfare
- **Risk level**: High

**B2: State Actor**
- **Motivation**: Political influence, censorship
- **Capabilities**: Extensive resources, legal pressure
- **Attack vectors**: Infrastructure attacks, targeted takedowns, propaganda
- **Risk level**: Medium (decentralization is defense)

**B3: Competitive Platform**
- **Motivation**: Market share, discrediting rival
- **Capabilities**: Technical expertise, financial resources
- **Attack vectors**: Spam attacks, negative campaigns, poaching users
- **Risk level**: Low to Medium

#### Category C: Insider Threats

**C1: Malicious Oracle**
- **Motivation**: Profit from position of trust
- **Capabilities**: Oracle privileges, MATL score
- **Attack vectors**: False resolution votes, collusion with traders
- **Risk level**: Medium (MATL weighting mitigates)

**C2: Rogue Developer**
- **Motivation**: Various (profit, ideology, coercion)
- **Capabilities**: Code access, system knowledge
- **Attack vectors**: Backdoors, selective bugs, key theft
- **Risk level**: Medium (code review, multi-sig mitigate)

**C3: Compromised Governance**
- **Motivation**: Capture for external interests
- **Capabilities**: Voting power, proposal rights
- **Attack vectors**: Rule changes favoring attackers
- **Risk level**: Medium (checks and balances mitigate)

#### Category D: Technical Threats

**D1: Smart Contract Bugs**
- **Cause**: Coding errors, edge cases
- **Impact**: Fund loss, incorrect resolution
- **Mitigation**: Formal verification, audits, bug bounties

**D2: Cryptographic Breaks**
- **Cause**: Algorithm weakness, quantum computing
- **Impact**: Signature forgery, privacy breach
- **Mitigation**: Algorithm agility, PQC readiness

**D3: Network Attacks**
- **Cause**: DDoS, partition attacks
- **Impact**: Availability loss, consistency issues
- **Mitigation**: Decentralization, partition tolerance

---

### 1.2 Attack Surface Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                       ATTACK SURFACE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   CLIENT    │────▶│   NETWORK   │────▶│   HOLOCHAIN │       │
│  │  INTERFACE  │     │   LAYER     │     │    NODE     │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│       │                    │                    │               │
│       ▼                    ▼                    ▼               │
│  • Phishing          • MITM attacks       • Zome vulnerabilities│
│  • Session theft     • DDoS              • State manipulation   │
│  • XSS/CSRF          • Eclipse attacks    • Entry corruption    │
│  • Malicious input   • Partition          • Validation bypass   │
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   ORACLE    │────▶│  GOVERNANCE │────▶│  EXTERNAL   │       │
│  │   SYSTEM    │     │   LAYER     │     │  SERVICES   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│       │                    │                    │               │
│       ▼                    ▼                    ▼               │
│  • Collusion         • Vote buying        • API compromise     │
│  • Bribery           • Proposal spam      • Data feed attacks  │
│  • Wrong resolution  • Gridlock           • Key management     │
│  • Selective voting  • Emergency abuse    • Third-party risk   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part II: Security Principles

### Principle 1: Defense in Depth

No single security measure is sufficient. We layer defenses so that breaching one doesn't compromise the system.

```
┌────────────────────────────────────────────┐
│           DEFENSE IN DEPTH                 │
├────────────────────────────────────────────┤
│                                            │
│  Layer 1: Cryptographic Foundation         │
│    └── Signatures, encryption, hashing     │
│                                            │
│  Layer 2: Protocol Rules                   │
│    └── Validation, rate limits, locks      │
│                                            │
│  Layer 3: Economic Incentives              │
│    └── Stakes, penalties, rewards          │
│                                            │
│  Layer 4: Social Mechanisms                │
│    └── Reputation, community vigilance     │
│                                            │
│  Layer 5: Governance Response              │
│    └── Emergency powers, dispute process   │
│                                            │
└────────────────────────────────────────────┘
```

### Principle 2: Fail Secure

When something goes wrong, the system should fail in the most secure state, not the most convenient.

```rust
// Fail secure example
pub fn resolve_market(market_id: EntryHash) -> ExternResult<Resolution> {
    let market = get_market(market_id.clone())?;

    // Fail secure: if anything is uncertain, don't resolve
    if !all_oracles_verified(&market)? {
        return Err(WasmError::Guest(
            "Cannot resolve: oracle verification incomplete".into()
        ));
    }

    if has_pending_disputes(&market)? {
        return Err(WasmError::Guest(
            "Cannot resolve: disputes pending".into()
        ));
    }

    if consensus_uncertain(&market)? {
        // Don't guess—wait for clarity
        return Err(WasmError::Guest(
            "Cannot resolve: consensus not reached".into()
        ));
    }

    // Only resolve when everything is certain
    execute_resolution(market)
}
```

### Principle 3: Least Privilege

Every component has only the permissions it needs, nothing more.

```rust
// Capability-based permissions
pub enum Capability {
    // Market capabilities
    CreateMarket,
    CloseMarket,
    ModifyMarketParameters,

    // Prediction capabilities
    SubmitPrediction,
    WithdrawPrediction,

    // Oracle capabilities
    VoteOnResolution { markets: Vec<MarketType> },
    DisputeResolution,

    // Governance capabilities
    ProposeParameterChange,
    VoteOnProposal,
    ExecuteProposal,
    EmergencyHalt,
}

pub struct CapabilityGrant {
    pub holder: AgentPubKey,
    pub capabilities: Vec<Capability>,
    pub constraints: CapabilityConstraints,
    pub expiration: Option<Timestamp>,
    pub issuer: AgentPubKey,
    pub revocable: bool,
}

pub struct CapabilityConstraints {
    pub rate_limits: Option<RateLimits>,
    pub stake_requirements: Option<StakeRequirements>,
    pub temporal_windows: Option<Vec<TimeWindow>>,
    pub geographic_restrictions: Option<Vec<Region>>,
}
```

### Principle 4: Transparency Over Obscurity

Security through obscurity is not security. Our mechanisms are public and auditable.

**What we keep secret:**
- Private keys (obviously)
- Pending predictions before reveal (for independence)
- Vote weights during active voting (to prevent cascade)

**What we make public:**
- All code (open source)
- All validation rules
- All economic parameters
- Historical data (after resolution)

### Principle 5: Economic Security

Make attacks expensive, not impossible. The cost of an attack should exceed the expected benefit.

```rust
pub fn calculate_attack_cost(attack: &AttackVector) -> AttackEconomics {
    let stake_required = estimate_stake_for_attack(attack);
    let reputation_at_risk = estimate_reputation_burn(attack);
    let detection_probability = estimate_detection_rate(attack);
    let penalty_if_detected = calculate_penalties(attack);

    let expected_cost = stake_required
        + reputation_at_risk * REPUTATION_VALUE
        + detection_probability * penalty_if_detected;

    let expected_benefit = estimate_attack_profit(attack);

    AttackEconomics {
        expected_cost,
        expected_benefit,
        profitability_ratio: expected_benefit / expected_cost,
        is_economically_viable: expected_benefit > expected_cost,
    }
}

// Design goal: profitability_ratio < 0.1 for all known attacks
```

---

## Part III: Cryptographic Foundation

### 3.1 Key Management

```rust
pub struct KeyHierarchy {
    // Master key (cold storage)
    pub master_key: MasterKey,

    // Derived keys for different purposes
    pub signing_key: SigningKey,      // For transactions
    pub encryption_key: EncryptionKey, // For private data
    pub session_key: SessionKey,       // For temporary sessions
}

pub struct KeyRotation {
    pub rotation_schedule: Duration,
    pub grace_period: Duration,
    pub old_key_retention: Duration,
    pub emergency_rotation: bool,
}

impl KeyManagement {
    pub fn rotate_keys(&mut self) -> Result<(), KeyError> {
        // Generate new key pair
        let new_key = self.generate_new_key()?;

        // Create rotation proof (signed by old key)
        let rotation_proof = RotationProof {
            old_key: self.current_key.public(),
            new_key: new_key.public(),
            timestamp: current_time(),
            signature: self.current_key.sign(&rotation_message),
        };

        // Publish rotation
        self.publish_rotation(rotation_proof)?;

        // Enter grace period (both keys valid)
        self.enter_grace_period(new_key)?;

        // After grace period, old key only for verification
        Ok(())
    }
}
```

### 3.2 Signature Schemes

```rust
// Multi-signature for high-value operations
pub struct MultiSig {
    pub required_signatures: u8,
    pub total_signers: u8,
    pub signers: Vec<AgentPubKey>,
    pub signatures: Vec<Signature>,
}

impl MultiSig {
    pub fn verify(&self, message: &[u8]) -> bool {
        let valid_count = self.signatures.iter()
            .filter(|sig| sig.verify(message).is_ok())
            .count();

        valid_count >= self.required_signatures as usize
    }
}

// Threshold signatures for distributed oracle
pub struct ThresholdSignature {
    pub threshold: u8,
    pub shares: Vec<SignatureShare>,
}

impl ThresholdSignature {
    pub fn combine(&self) -> Result<Signature, CryptoError> {
        if self.shares.len() < self.threshold as usize {
            return Err(CryptoError::InsufficientShares);
        }

        // Lagrange interpolation to recover signature
        self.lagrange_combine()
    }
}
```

### 3.3 Commit-Reveal Scheme

Used for prediction independence:

```rust
pub struct CommitRevealPrediction {
    // Phase 1: Commit
    pub commitment: Hash,  // H(prediction || nonce || secret)
    pub commit_time: Timestamp,

    // Phase 2: Reveal (after commit phase ends)
    pub revealed_prediction: Option<Prediction>,
    pub reveal_nonce: Option<Nonce>,
    pub reveal_time: Option<Timestamp>,
}

impl CommitRevealPrediction {
    pub fn commit(prediction: &Prediction, secret: &[u8]) -> Self {
        let nonce = generate_random_nonce();
        let commitment = hash(&[
            prediction.to_bytes(),
            nonce.as_ref(),
            secret,
        ].concat());

        CommitRevealPrediction {
            commitment,
            commit_time: current_time(),
            revealed_prediction: None,
            reveal_nonce: None,
            reveal_time: None,
        }
    }

    pub fn reveal(
        &mut self,
        prediction: Prediction,
        nonce: Nonce,
        secret: &[u8]
    ) -> Result<(), RevealError> {
        // Verify commitment matches
        let expected = hash(&[
            prediction.to_bytes(),
            nonce.as_ref(),
            secret,
        ].concat());

        if expected != self.commitment {
            return Err(RevealError::CommitmentMismatch);
        }

        self.revealed_prediction = Some(prediction);
        self.reveal_nonce = Some(nonce);
        self.reveal_time = Some(current_time());

        Ok(())
    }
}
```

### 3.4 Zero-Knowledge Proofs

For private aggregation without revealing individual predictions:

```rust
pub struct ZKPredictionProof {
    // Proves: "I predicted outcome X with confidence Y"
    // Without revealing: X, Y, or my identity (until resolution)
    pub proof: ZKProof,
    pub commitment: Commitment,
    pub nullifier: Nullifier,  // Prevents double-voting
}

pub struct PrivateAggregation {
    pub proofs: Vec<ZKPredictionProof>,
    pub aggregated_result: EncryptedAggregate,
}

impl PrivateAggregation {
    pub fn aggregate(proofs: Vec<ZKPredictionProof>) -> Self {
        // Homomorphic aggregation
        let aggregated = proofs.iter()
            .fold(EncryptedAggregate::zero(), |acc, proof| {
                acc.add_encrypted(&proof.commitment)
            });

        PrivateAggregation {
            proofs,
            aggregated_result: aggregated,
        }
    }

    pub fn reveal_aggregate(&self, threshold_key: &ThresholdKey) -> Aggregate {
        // Only revealed after threshold of keyholders agree
        threshold_key.decrypt(&self.aggregated_result)
    }
}
```

---

## Part IV: Protocol Security

### 4.1 Entry Validation

Every entry is validated before acceptance:

```rust
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened()? {
        FlatOp::StoreEntry(store_entry) => {
            match store_entry.entry {
                Entry::App(entry) => validate_entry(entry, &op),
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_entry(entry: AppEntry, op: &Op) -> ExternResult<ValidateCallbackResult> {
    // Deserialize and dispatch to type-specific validation
    let entry_type = entry.entry_type();

    match entry_type.as_str() {
        "Market" => validate_market(entry),
        "Prediction" => validate_prediction(entry, op),
        "OracleVote" => validate_oracle_vote(entry, op),
        "Resolution" => validate_resolution(entry, op),
        _ => Ok(ValidateCallbackResult::Invalid(
            "Unknown entry type".into()
        )),
    }
}

fn validate_prediction(entry: AppEntry, op: &Op) -> ExternResult<ValidateCallbackResult> {
    let prediction: Prediction = entry.try_into()?;

    // 1. Check market exists and is open
    let market = get_market(prediction.market_id.clone())?;
    if market.status != MarketStatus::Open {
        return Ok(ValidateCallbackResult::Invalid(
            "Market is not open for predictions".into()
        ));
    }

    // 2. Check prediction deadline not passed
    if current_time()? > market.closes_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Prediction deadline passed".into()
        ));
    }

    // 3. Check stake is within allowed range
    if !market.allowed_stake_range.contains(&prediction.stake.total_value()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Stake outside allowed range".into()
        ));
    }

    // 4. Verify signature
    let author = op.author()?;
    if !verify_prediction_signature(&prediction, &author)? {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid signature".into()
        ));
    }

    // 5. Check rate limits
    if exceeds_rate_limit(&author, "prediction")? {
        return Ok(ValidateCallbackResult::Invalid(
            "Rate limit exceeded".into()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
```

### 4.2 Rate Limiting

Protection against spam and DoS:

```rust
pub struct RateLimiter {
    pub limits: HashMap<ActionType, RateLimit>,
    pub windows: HashMap<(AgentPubKey, ActionType), SlidingWindow>,
}

pub struct RateLimit {
    pub max_actions: u32,
    pub window_duration: Duration,
    pub burst_allowance: u32,
    pub cooldown: Duration,
}

pub struct SlidingWindow {
    pub actions: VecDeque<Timestamp>,
    pub burst_used: u32,
    pub cooldown_until: Option<Timestamp>,
}

impl RateLimiter {
    pub fn check(&mut self, agent: &AgentPubKey, action: ActionType) -> RateLimitResult {
        let limit = self.limits.get(&action)
            .ok_or(RateLimitError::UnknownAction)?;

        let window = self.windows
            .entry((agent.clone(), action))
            .or_insert_with(|| SlidingWindow::new());

        // Check cooldown
        if let Some(cooldown_until) = window.cooldown_until {
            if current_time() < cooldown_until {
                return RateLimitResult::Blocked {
                    reason: "In cooldown period",
                    retry_after: cooldown_until,
                };
            }
        }

        // Slide window
        let cutoff = current_time() - limit.window_duration;
        while let Some(oldest) = window.actions.front() {
            if *oldest < cutoff {
                window.actions.pop_front();
            } else {
                break;
            }
        }

        // Check limit
        if window.actions.len() >= limit.max_actions as usize {
            if window.burst_used < limit.burst_allowance {
                window.burst_used += 1;
                RateLimitResult::BurstAllowed { remaining_burst: limit.burst_allowance - window.burst_used }
            } else {
                window.cooldown_until = Some(current_time() + limit.cooldown);
                RateLimitResult::Blocked {
                    reason: "Rate limit exceeded",
                    retry_after: window.cooldown_until.unwrap(),
                }
            }
        } else {
            window.actions.push_back(current_time());
            RateLimitResult::Allowed {
                remaining: limit.max_actions - window.actions.len() as u32
            }
        }
    }
}
```

### 4.3 Economic Locks

Preventing double-spend and ensuring accountability:

```rust
pub struct StakeLock {
    pub locked_amount: u64,
    pub locked_reputation: u64,
    pub lock_reason: LockReason,
    pub lock_until: Timestamp,
    pub unlock_conditions: Vec<UnlockCondition>,
}

pub enum LockReason {
    PendingPrediction { market_id: EntryHash },
    OracleVote { resolution_id: EntryHash },
    GovernanceVote { proposal_id: EntryHash },
    DisputeStake { dispute_id: EntryHash },
    CooldownPeriod { reason: String },
}

pub enum UnlockCondition {
    TimeElapsed { duration: Duration },
    MarketResolved { market_id: EntryHash },
    DisputeSettled { dispute_id: EntryHash },
    ProposalCompleted { proposal_id: EntryHash },
    ManualRelease { authority: AgentPubKey },
}

impl StakeLock {
    pub fn can_unlock(&self) -> Result<bool, LockError> {
        for condition in &self.unlock_conditions {
            match condition {
                UnlockCondition::TimeElapsed { duration } => {
                    if current_time() < self.lock_until + *duration {
                        return Ok(false);
                    }
                }
                UnlockCondition::MarketResolved { market_id } => {
                    if !is_market_resolved(market_id)? {
                        return Ok(false);
                    }
                }
                // ... other conditions
            }
        }
        Ok(true)
    }
}
```

---

## Part V: Oracle Security

### 5.1 Oracle Selection

```rust
pub fn select_oracles(market: &Market) -> Result<Vec<Oracle>, OracleError> {
    let eligible = get_eligible_oracles(&market.domains)?;

    // Filter by minimum MATL score
    let qualified: Vec<_> = eligible.iter()
        .filter(|o| o.matl_score >= market.min_oracle_matl)
        .collect();

    if qualified.len() < market.min_oracles as usize {
        return Err(OracleError::InsufficientQualifiedOracles);
    }

    // Weighted random selection
    let selected = weighted_sample(
        &qualified,
        |o| o.matl_score.powi(2),  // Square for stronger preference
        market.target_oracle_count,
    );

    // Verify diversity
    if !verify_oracle_diversity(&selected)? {
        return Err(OracleError::InsufficientDiversity);
    }

    Ok(selected)
}

fn verify_oracle_diversity(oracles: &[Oracle]) -> Result<bool, OracleError> {
    // Check geographic distribution
    let regions: HashSet<_> = oracles.iter()
        .filter_map(|o| o.region.as_ref())
        .collect();

    if regions.len() < MIN_REGION_DIVERSITY {
        return Ok(false);
    }

    // Check no single entity controls majority
    let entities: HashMap<_, u32> = oracles.iter()
        .fold(HashMap::new(), |mut acc, o| {
            *acc.entry(&o.entity_id).or_insert(0) += 1;
            acc
        });

    let max_from_single = entities.values().max().unwrap_or(&0);
    if *max_from_single as f64 / oracles.len() as f64 > MAX_SINGLE_ENTITY_SHARE {
        return Ok(false);
    }

    Ok(true)
}
```

### 5.2 Collusion Detection

```rust
pub struct CollusionDetector {
    pub detection_window: Duration,
    pub correlation_threshold: f64,
    pub timing_variance_threshold: Duration,
}

impl CollusionDetector {
    pub fn analyze_votes(&self, votes: &[OracleVote]) -> CollusionAnalysis {
        let mut indicators = Vec::new();

        // 1. Timing analysis
        let timing = self.analyze_timing(votes);
        if timing.variance < self.timing_variance_threshold {
            indicators.push(CollusionIndicator::SuspiciousTiming {
                variance: timing.variance,
                expected_min: self.timing_variance_threshold,
            });
        }

        // 2. Voting pattern correlation
        let correlation = self.analyze_correlation(votes);
        if correlation > self.correlation_threshold {
            indicators.push(CollusionIndicator::HighCorrelation {
                correlation,
                threshold: self.correlation_threshold,
            });
        }

        // 3. Network analysis
        let clusters = self.detect_clusters(votes);
        for cluster in clusters {
            if cluster.size > 2 && cluster.internal_correlation > 0.95 {
                indicators.push(CollusionIndicator::SuspiciousCluster {
                    members: cluster.members.clone(),
                    correlation: cluster.internal_correlation,
                });
            }
        }

        // 4. Historical pattern matching
        let known_patterns = self.match_known_collusion_patterns(votes);
        for pattern in known_patterns {
            indicators.push(CollusionIndicator::KnownPattern {
                pattern_id: pattern.id,
                confidence: pattern.match_confidence,
            });
        }

        CollusionAnalysis {
            indicators,
            overall_risk: self.calculate_overall_risk(&indicators),
            recommended_action: self.recommend_action(&indicators),
        }
    }

    fn recommend_action(&self, indicators: &[CollusionIndicator]) -> CollusionAction {
        let risk_score: f64 = indicators.iter()
            .map(|i| i.risk_weight())
            .sum();

        if risk_score > 0.8 {
            CollusionAction::HaltAndEscalate
        } else if risk_score > 0.5 {
            CollusionAction::ExpandOraclePool
        } else if risk_score > 0.2 {
            CollusionAction::FlagForReview
        } else {
            CollusionAction::Continue
        }
    }
}
```

### 5.3 Byzantine Fault Tolerance

```rust
pub struct ByzantineConsensus {
    pub threshold: f64,  // 0.45 = 45% Byzantine tolerance
    pub quorum: f64,     // Minimum participation
}

impl ByzantineConsensus {
    pub fn check_consensus(&self, votes: &[WeightedVote]) -> ConsensusResult {
        let total_weight: f64 = votes.iter().map(|v| v.weight).sum();

        // Check quorum
        if total_weight < self.quorum {
            return ConsensusResult::InsufficientQuorum {
                current: total_weight,
                required: self.quorum,
            };
        }

        // Group by outcome
        let mut outcome_weights: HashMap<Outcome, f64> = HashMap::new();
        for vote in votes {
            *outcome_weights.entry(vote.outcome.clone()).or_insert(0.0) += vote.weight;
        }

        // Find winning outcome
        let (winner, winner_weight) = outcome_weights.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // Check if winner has sufficient margin
        let required_margin = total_weight * (1.0 - self.threshold);

        if *winner_weight >= required_margin {
            ConsensusResult::Achieved {
                outcome: winner.clone(),
                weight: *winner_weight,
                margin: *winner_weight - required_margin,
            }
        } else {
            ConsensusResult::NoConsensus {
                leading_outcome: winner.clone(),
                leading_weight: *winner_weight,
                required_weight: required_margin,
            }
        }
    }
}
```

---

## Part VI: Sybil Resistance

### 6.1 Identity Verification Layers

```rust
pub struct IdentityVerification {
    pub layers: Vec<VerificationLayer>,
    pub minimum_score: f64,
}

pub enum VerificationLayer {
    // Layer 1: Basic (anyone can do)
    EmailVerification { weight: f64 },
    SocialAccountLink { platform: String, weight: f64 },

    // Layer 2: Investment (costs money/time)
    StakeDeposit { min_amount: u64, weight: f64 },
    TimeInSystem { min_duration: Duration, weight: f64 },
    ActivityHistory { min_actions: u32, weight: f64 },

    // Layer 3: Social (requires network)
    VouchingFromVerified { min_vouchers: u32, weight: f64 },
    CommunityEndorsement { weight: f64 },

    // Layer 4: External (third-party)
    KYCProvider { provider: String, weight: f64 },
    CredentialVerification { credential_type: String, weight: f64 },
    BiometricVerification { weight: f64 },
}

impl IdentityVerification {
    pub fn calculate_score(&self, user: &User) -> f64 {
        let mut total_weight = 0.0;
        let mut earned_weight = 0.0;

        for layer in &self.layers {
            let (weight, verified) = match layer {
                VerificationLayer::EmailVerification { weight } => {
                    (*weight, user.email_verified)
                }
                VerificationLayer::StakeDeposit { min_amount, weight } => {
                    (*weight, user.total_stake >= *min_amount)
                }
                VerificationLayer::TimeInSystem { min_duration, weight } => {
                    let in_system = current_time() - user.joined_at;
                    (*weight, in_system >= *min_duration)
                }
                VerificationLayer::VouchingFromVerified { min_vouchers, weight } => {
                    let vouches = count_verified_vouches(&user.id);
                    (*weight, vouches >= *min_vouchers as usize)
                }
                // ... other layers
            };

            total_weight += weight;
            if verified {
                earned_weight += weight;
            }
        }

        earned_weight / total_weight
    }

    pub fn is_verified(&self, user: &User) -> bool {
        self.calculate_score(user) >= self.minimum_score
    }
}
```

### 6.2 Sybil Detection

```rust
pub struct SybilDetector {
    pub behavioral_analyzer: BehavioralAnalyzer,
    pub network_analyzer: NetworkAnalyzer,
    pub temporal_analyzer: TemporalAnalyzer,
}

impl SybilDetector {
    pub fn analyze_account(&self, account: &Account) -> SybilAnalysis {
        let mut risk_factors = Vec::new();

        // Behavioral analysis
        let behavioral = self.behavioral_analyzer.analyze(account);
        if behavioral.uniformity_score > 0.9 {
            risk_factors.push(SybilRiskFactor::HighBehavioralUniformity {
                score: behavioral.uniformity_score,
            });
        }

        // Network analysis
        let network = self.network_analyzer.analyze(account);
        if network.cluster_coefficient > 0.95 {
            risk_factors.push(SybilRiskFactor::TightClusterMembership {
                cluster_id: network.primary_cluster,
                coefficient: network.cluster_coefficient,
            });
        }

        // Temporal analysis
        let temporal = self.temporal_analyzer.analyze(account);
        if temporal.activity_correlation > 0.9 {
            risk_factors.push(SybilRiskFactor::SynchronizedActivity {
                correlated_accounts: temporal.correlated_with,
                correlation: temporal.activity_correlation,
            });
        }

        // IP/device fingerprint analysis (if available)
        if let Some(fingerprint) = &account.device_fingerprint {
            let similar = self.find_similar_fingerprints(fingerprint);
            if similar.len() > SUSPICIOUS_FINGERPRINT_THRESHOLD {
                risk_factors.push(SybilRiskFactor::SharedDeviceFingerprint {
                    similar_accounts: similar,
                });
            }
        }

        SybilAnalysis {
            account_id: account.id.clone(),
            risk_factors,
            overall_sybil_probability: self.calculate_probability(&risk_factors),
            recommended_action: self.recommend_action(&risk_factors),
        }
    }

    pub fn detect_sybil_clusters(&self, accounts: &[Account]) -> Vec<SybilCluster> {
        // Graph-based community detection
        let graph = self.build_similarity_graph(accounts);
        let communities = graph.detect_communities();

        communities.into_iter()
            .filter(|c| c.internal_similarity > SYBIL_CLUSTER_THRESHOLD)
            .filter(|c| c.size > 2)
            .map(|c| SybilCluster {
                accounts: c.members,
                confidence: c.internal_similarity,
                detected_at: current_time(),
            })
            .collect()
    }
}
```

---

## Part VII: Emergency Response

### 7.1 Circuit Breakers

```rust
pub struct CircuitBreaker {
    pub name: String,
    pub triggers: Vec<CircuitTrigger>,
    pub state: CircuitState,
    pub cooldown: Duration,
}

pub enum CircuitState {
    Closed,  // Normal operation
    Open { opened_at: Timestamp, reason: String },  // Halted
    HalfOpen { test_until: Timestamp },  // Testing recovery
}

pub enum CircuitTrigger {
    ErrorRate { threshold: f64, window: Duration },
    VolumeSpike { multiplier: f64, window: Duration },
    SuspiciousActivity { score_threshold: f64 },
    ExternalSignal { source: String },
    ManualTrigger { authority: AgentPubKey },
}

impl CircuitBreaker {
    pub fn check(&mut self) -> CircuitAction {
        match &self.state {
            CircuitState::Closed => {
                for trigger in &self.triggers {
                    if self.is_triggered(trigger) {
                        self.state = CircuitState::Open {
                            opened_at: current_time(),
                            reason: format!("Triggered by {:?}", trigger),
                        };
                        return CircuitAction::Halt {
                            reason: format!("Circuit breaker {} opened", self.name),
                        };
                    }
                }
                CircuitAction::Continue
            }
            CircuitState::Open { opened_at, reason } => {
                if current_time() - *opened_at > self.cooldown {
                    self.state = CircuitState::HalfOpen {
                        test_until: current_time() + Duration::from_secs(300),
                    };
                    CircuitAction::TestRecovery
                } else {
                    CircuitAction::Halt { reason: reason.clone() }
                }
            }
            CircuitState::HalfOpen { test_until } => {
                if current_time() > *test_until {
                    // If we made it through the test period, close
                    self.state = CircuitState::Closed;
                    CircuitAction::Continue
                } else {
                    // Still testing - allow limited traffic
                    CircuitAction::LimitedOperation
                }
            }
        }
    }
}
```

### 7.2 Emergency Procedures

```rust
pub struct EmergencyProcedure {
    pub id: EmergencyId,
    pub severity: EmergencySeverity,
    pub triggers: Vec<EmergencyTrigger>,
    pub actions: Vec<EmergencyAction>,
    pub authorization_required: AuthorizationLevel,
}

pub enum EmergencySeverity {
    Low,     // Monitoring only
    Medium,  // Partial restrictions
    High,    // Major restrictions
    Critical, // Full halt
}

pub enum EmergencyAction {
    // Operational
    PauseMarketCreation,
    PausePredictions { markets: MarketFilter },
    PauseResolutions,
    PauseWithdrawals,
    FullHalt,

    // Protective
    LockSuspiciousAccounts { criteria: AccountCriteria },
    FreezeMarkets { markets: MarketFilter },
    RevertTransactions { since: Timestamp, filter: TxFilter },

    // Communication
    NotifyUsers { message: String, severity: AlertSeverity },
    NotifyGovernance { urgency: GovernanceUrgency },
    PublishIncidentReport { report: IncidentReport },

    // Recovery
    ActivateBackupOracles,
    EnableSafeMode,
    InitiateGracefulDegradation,
}

impl EmergencyProcedure {
    pub fn execute(&self, context: &EmergencyContext) -> Result<EmergencyResponse, EmergencyError> {
        // Verify authorization
        if !self.verify_authorization(context)? {
            return Err(EmergencyError::InsufficientAuthorization);
        }

        // Log initiation
        log_emergency_initiation(self, context)?;

        // Execute actions in order
        let mut results = Vec::new();
        for action in &self.actions {
            match self.execute_action(action, context) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Log failure but continue critical actions
                    log_action_failure(action, &e);
                    if action.is_critical() {
                        return Err(EmergencyError::CriticalActionFailed(e));
                    }
                }
            }
        }

        // Schedule follow-up
        schedule_emergency_review(self)?;

        Ok(EmergencyResponse {
            procedure_id: self.id.clone(),
            actions_taken: results,
            timestamp: current_time(),
        })
    }
}
```

### 7.3 Recovery Protocols

```rust
pub struct RecoveryProtocol {
    pub incident_id: IncidentId,
    pub phases: Vec<RecoveryPhase>,
    pub current_phase: usize,
    pub verification_requirements: Vec<VerificationRequirement>,
}

pub struct RecoveryPhase {
    pub name: String,
    pub actions: Vec<RecoveryAction>,
    pub success_criteria: Vec<SuccessCriterion>,
    pub rollback_procedure: RollbackProcedure,
}

pub enum RecoveryAction {
    // Data recovery
    RestoreFromSnapshot { snapshot_id: SnapshotId },
    ReprocessEntries { from: Timestamp, to: Timestamp },
    ReconcileStates { nodes: Vec<NodeId> },

    // System recovery
    RestartServices { services: Vec<ServiceId> },
    ClearCaches,
    ReloadConfiguration,
    ReestablishConnections,

    // Business recovery
    ResumeOperations { gradually: bool },
    ProcessPendingTransactions,
    NotifyAffectedUsers { users: UserFilter },
    IssueCompensation { affected: Vec<Compensation> },
}

impl RecoveryProtocol {
    pub fn execute_next_phase(&mut self) -> Result<PhaseResult, RecoveryError> {
        let phase = &self.phases[self.current_phase];

        // Execute phase actions
        for action in &phase.actions {
            self.execute_action(action)?;
        }

        // Verify success criteria
        for criterion in &phase.success_criteria {
            if !self.verify_criterion(criterion)? {
                // Rollback this phase
                self.execute_rollback(&phase.rollback_procedure)?;
                return Err(RecoveryError::PhaseFailed {
                    phase: phase.name.clone(),
                    criterion: criterion.clone(),
                });
            }
        }

        // Advance to next phase
        self.current_phase += 1;

        if self.current_phase >= self.phases.len() {
            // All phases complete
            self.finalize_recovery()?;
            Ok(PhaseResult::RecoveryComplete)
        } else {
            Ok(PhaseResult::PhaseComplete {
                next_phase: self.phases[self.current_phase].name.clone(),
            })
        }
    }
}
```

---

## Part VIII: Privacy Protection

### 8.1 Data Classification

```rust
pub enum DataClassification {
    Public,      // Can be shared openly
    Internal,    // Visible within the system
    Confidential,// Limited access
    Secret,      // Highly restricted
}

pub struct DataClassificationPolicy {
    pub classifications: HashMap<DataType, DataClassification>,
}

impl DataClassificationPolicy {
    pub fn default() -> Self {
        let mut classifications = HashMap::new();

        // Public data
        classifications.insert(DataType::MarketQuestion, DataClassification::Public);
        classifications.insert(DataType::ResolvedOutcome, DataClassification::Public);
        classifications.insert(DataType::AggregatedPrediction, DataClassification::Public);

        // Internal data
        classifications.insert(DataType::ReputationScore, DataClassification::Internal);
        classifications.insert(DataType::ActivityHistory, DataClassification::Internal);

        // Confidential data
        classifications.insert(DataType::IndividualPrediction, DataClassification::Confidential);
        classifications.insert(DataType::StakeAmount, DataClassification::Confidential);
        classifications.insert(DataType::WalletBalance, DataClassification::Confidential);

        // Secret data
        classifications.insert(DataType::PrivateKey, DataClassification::Secret);
        classifications.insert(DataType::IdentityDocument, DataClassification::Secret);
        classifications.insert(DataType::LocationData, DataClassification::Secret);

        Self { classifications }
    }
}
```

### 8.2 Privacy-Preserving Aggregation

```rust
pub struct PrivacyPreservingAggregator {
    pub min_anonymity_set: usize,
    pub differential_privacy: DifferentialPrivacyConfig,
}

pub struct DifferentialPrivacyConfig {
    pub epsilon: f64,  // Privacy budget
    pub delta: f64,    // Failure probability
}

impl PrivacyPreservingAggregator {
    pub fn aggregate_predictions(
        &self,
        predictions: &[EncryptedPrediction]
    ) -> Result<AggregatedPrediction, PrivacyError> {
        // Check anonymity set size
        if predictions.len() < self.min_anonymity_set {
            return Err(PrivacyError::InsufficientAnonymitySet {
                current: predictions.len(),
                required: self.min_anonymity_set,
            });
        }

        // Homomorphic aggregation (no decryption needed)
        let raw_aggregate = predictions.iter()
            .fold(EncryptedValue::zero(), |acc, p| {
                acc.add(&p.encrypted_value)
            });

        // Add differential privacy noise
        let noise = self.generate_noise();
        let noised_aggregate = raw_aggregate.add_noise(&noise);

        // Decrypt only the aggregate (not individuals)
        let aggregate = threshold_decrypt(&noised_aggregate)?;

        Ok(AggregatedPrediction {
            value: aggregate,
            count: predictions.len(),
            noise_magnitude: noise.magnitude(),
            privacy_guarantee: self.calculate_guarantee(),
        })
    }

    fn generate_noise(&self) -> Noise {
        // Laplace mechanism for differential privacy
        let scale = 1.0 / self.differential_privacy.epsilon;
        Noise::laplace(scale)
    }
}
```

---

## Part IX: Audit and Compliance

### 9.1 Audit Trail

```rust
pub struct AuditTrail {
    pub retention_period: Duration,
    pub immutable: bool,
}

pub struct AuditEntry {
    pub id: AuditId,
    pub timestamp: Timestamp,
    pub actor: ActorId,
    pub action: AuditableAction,
    pub target: Option<TargetId>,
    pub details: AuditDetails,
    pub outcome: ActionOutcome,
    pub previous_hash: Hash,
    pub hash: Hash,
}

pub enum AuditableAction {
    // User actions
    AccountCreated,
    PredictionSubmitted { market_id: EntryHash },
    StakePlaced { amount: u64 },
    WithdrawalRequested { amount: u64 },

    // Oracle actions
    OracleVoteSubmitted { resolution_id: EntryHash },
    DisputeRaised { resolution_id: EntryHash },

    // Admin actions
    MarketPaused { market_id: EntryHash, reason: String },
    AccountSuspended { account_id: AgentPubKey, reason: String },
    EmergencyActivated { procedure_id: EmergencyId },
    ParameterChanged { parameter: String, old_value: String, new_value: String },

    // System actions
    SystemStartup,
    SystemShutdown,
    BackupCreated { backup_id: BackupId },
    RecoveryInitiated { incident_id: IncidentId },
}

impl AuditTrail {
    pub fn log(&mut self, action: AuditableAction, context: &AuditContext) -> AuditId {
        let previous = self.get_latest_hash();

        let entry = AuditEntry {
            id: generate_audit_id(),
            timestamp: current_time(),
            actor: context.actor.clone(),
            action,
            target: context.target.clone(),
            details: context.details.clone(),
            outcome: context.outcome.clone(),
            previous_hash: previous,
            hash: Hash::default(), // Will be set
        };

        let hash = self.calculate_hash(&entry);
        let entry = AuditEntry { hash, ..entry };

        self.append(entry.clone());

        entry.id
    }

    pub fn verify_integrity(&self) -> IntegrityResult {
        let mut expected_prev = Hash::default();

        for entry in self.entries() {
            // Verify chain
            if entry.previous_hash != expected_prev {
                return IntegrityResult::ChainBroken {
                    at_entry: entry.id.clone(),
                    expected: expected_prev,
                    found: entry.previous_hash.clone(),
                };
            }

            // Verify hash
            let calculated = self.calculate_hash(&entry);
            if entry.hash != calculated {
                return IntegrityResult::HashMismatch {
                    at_entry: entry.id.clone(),
                    expected: calculated,
                    found: entry.hash.clone(),
                };
            }

            expected_prev = entry.hash.clone();
        }

        IntegrityResult::Valid {
            entries_verified: self.entries().count(),
        }
    }
}
```

### 9.2 Security Monitoring

```rust
pub struct SecurityMonitor {
    pub detectors: Vec<Box<dyn ThreatDetector>>,
    pub alert_handlers: Vec<Box<dyn AlertHandler>>,
}

pub trait ThreatDetector {
    fn analyze(&self, event: &SecurityEvent) -> Option<ThreatDetection>;
}

pub struct ThreatDetection {
    pub threat_type: ThreatType,
    pub severity: ThreatSeverity,
    pub confidence: f64,
    pub indicators: Vec<ThreatIndicator>,
    pub recommended_actions: Vec<ThreatResponse>,
}

impl SecurityMonitor {
    pub fn process_event(&self, event: SecurityEvent) -> Vec<ThreatDetection> {
        let detections: Vec<_> = self.detectors.iter()
            .filter_map(|d| d.analyze(&event))
            .collect();

        // Handle alerts
        for detection in &detections {
            if detection.severity >= ThreatSeverity::High {
                self.send_alert(detection);
            }
        }

        // Auto-respond to critical threats
        for detection in &detections {
            if detection.severity == ThreatSeverity::Critical
               && detection.confidence > 0.9 {
                self.auto_respond(detection);
            }
        }

        detections
    }

    fn auto_respond(&self, detection: &ThreatDetection) {
        for action in &detection.recommended_actions {
            if action.can_auto_execute() {
                if let Err(e) = action.execute() {
                    log_auto_response_failure(detection, action, &e);
                }
            }
        }
    }
}
```

---

## Part X: Security Governance

### 10.1 Responsible Disclosure

```markdown
## Responsible Disclosure Policy

### Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:

1. **Email**: security@epistemic-markets.org
2. **PGP Key**: [Link to public key]
3. **Bug Bounty Portal**: [Link if applicable]

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested mitigation (if any)

### Our Commitment

- Acknowledge receipt within 24 hours
- Provide initial assessment within 72 hours
- Keep you informed of progress
- Credit you in advisory (unless you prefer anonymity)
- Not pursue legal action for good-faith research

### Bug Bounty Rewards

| Severity | Reward Range |
|----------|--------------|
| Critical | $10,000 - $50,000 |
| High     | $5,000 - $10,000 |
| Medium   | $1,000 - $5,000 |
| Low      | $100 - $1,000 |

Severity is determined by impact and exploitability.

### Scope

In scope:
- Core protocol smart contracts/zomes
- Oracle system
- Cryptographic implementations
- Authentication/authorization
- Economic mechanisms

Out of scope:
- Third-party integrations
- Social engineering attacks
- Physical attacks
- DoS attacks (report, but no bounty)
```

### 10.2 Security Reviews

```rust
pub struct SecurityReviewProcess {
    pub review_types: Vec<ReviewType>,
    pub schedule: ReviewSchedule,
    pub required_approvals: u8,
}

pub enum ReviewType {
    // Internal reviews
    CodeReview { frequency: Duration },
    ArchitectureReview { frequency: Duration },
    ThreatModelUpdate { frequency: Duration },

    // External reviews
    ExternalAudit { provider: String, frequency: Duration },
    PenetrationTest { provider: String, frequency: Duration },
    FormalVerification { scope: String },

    // Continuous
    DependencyAudit,
    ConfigurationReview,
}

pub struct ReviewSchedule {
    pub regular: HashMap<ReviewType, Duration>,
    pub after_changes: Vec<ChangeType>,
    pub ad_hoc_triggers: Vec<ReviewTrigger>,
}

impl SecurityReviewProcess {
    pub fn get_pending_reviews(&self) -> Vec<PendingReview> {
        let mut pending = Vec::new();

        for (review_type, frequency) in &self.schedule.regular {
            let last_review = self.get_last_review(review_type);
            if current_time() - last_review > *frequency {
                pending.push(PendingReview {
                    review_type: review_type.clone(),
                    due_date: last_review + *frequency,
                    overdue_by: current_time() - (last_review + *frequency),
                });
            }
        }

        pending
    }
}
```

---

## Conclusion

Security in Epistemic Markets is not a feature—it's a foundation. Every design decision considers security implications. Every new feature undergoes security review.

We don't claim to be perfect. We claim to be vigilant.

Our security model is:
- **Layered**: No single point of failure
- **Economic**: Attacks are unprofitable
- **Transparent**: Open source, audited
- **Responsive**: Active monitoring and incident response
- **Evolving**: Continuous improvement

If you find a vulnerability, please report it. If you have suggestions, we welcome them. Security is a community effort.

---

*"Perfect security is impossible. Adequate security is essential. The difference between the two is wisdom."*

*We secure what matters. We acknowledge what we cannot.*

---

## Appendix: Security Checklist

### For Users

- [ ] Use a strong, unique password
- [ ] Enable two-factor authentication
- [ ] Verify wallet addresses before transactions
- [ ] Keep private keys secure and backed up
- [ ] Report suspicious activity immediately

### For Operators

- [ ] Regular security audits completed
- [ ] Incident response plan tested
- [ ] Backups verified and tested
- [ ] Access controls reviewed
- [ ] Monitoring alerts configured

### For Developers

- [ ] Code reviewed by multiple people
- [ ] Static analysis tools run
- [ ] Dependency vulnerabilities checked
- [ ] Tests include security cases
- [ ] Documentation updated

