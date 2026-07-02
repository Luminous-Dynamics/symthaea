# Mycelix-Oracle: Decentralized Data Feeds

## Vision & Design Document

**Version**: 1.0.0
**Created**: December 30, 2025
**Status**: Design Phase
**Priority**: Tier 1 - High Synergy

---

## Executive Summary

Mycelix-Oracle is a Byzantine-resistant decentralized oracle network that brings real-world data into the Mycelix ecosystem. By leveraging the proven 45% Byzantine fault tolerance of 0TML and MATL trust scoring for data provider selection, Oracle provides reliable external data feeds for SupplyChain, Marketplace, and other hApps.

### The Oracle Problem

Decentralized systems need external data:
- **SupplyChain**: Temperature sensors, GPS locations, customs data
- **Marketplace**: Exchange rates, commodity prices, shipping status
- **Arbiter**: Legal records, verified documents
- **Insurance (Mutual)**: Weather data, event verification

But external data can be manipulated. Traditional oracles solve this with economic incentives or trusted committees. Mycelix-Oracle uses Byzantine consensus and reputation-weighted aggregation.

---

## Core Principles

### 1. Byzantine Resilience
Up to 45% of data providers can submit false data without corrupting the output.

### 2. Reputation-Weighted Aggregation
Data from high-trust providers counts more than data from new or low-trust providers.

### 3. Epistemic Transparency
Every data feed carries explicit epistemic classification—consumers know how verified the data is.

### 4. Source Diversity
Critical feeds require data from multiple independent sources to prevent single-point failures.

### 5. Economic Incentives
Providers stake reputation and may stake tokens. Accurate data earns rewards; false data loses stake.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        External Data Sources                         │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │
│  │ API  │ │ IoT  │ │ Gov  │ │Market│ │Weather│ │Custom│            │
│  │ Feed │ │Sensor│ │ Data │ │ Data │ │ Data │ │Source│            │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘            │
│     └────────┴────────┴────────┴────────┴────────┘                  │
│                              │                                       │
├──────────────────────────────┼───────────────────────────────────────┤
│                    Data Provider Layer                               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        ││
│  │  │Provider A│  │Provider B│  │Provider C│  │Provider N│        ││
│  │  │Trust: 0.9│  │Trust: 0.7│  │Trust: 0.8│  │Trust: 0.6│        ││
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        ││
│  └───────┴─────────────┴─────────────┴─────────────┴────────────────┘│
│                              │                                       │
├──────────────────────────────┼───────────────────────────────────────┤
│                        Oracle hApp                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Coordinator Zomes                             ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │   Feed   │ │ Provider │ │ Consensus│ │ Consumer │           ││
│  │  │ Manager  │ │ Registry │ │  Engine  │ │ Service  │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                        ││
│  │  │ Staking  │ │ Dispute  │ │ Analytics│                        ││
│  │  │ Manager  │ │ Handler  │ │ & Health │                        ││
│  │  └──────────┘ └──────────┘ └──────────┘                        ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                        Consuming hApps                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ SupplyChain  │  │ Marketplace  │  │   Mutual     │              │
│  │ (IoT, GPS)   │  │ (Prices)     │  │ (Weather)    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// Definition of a data feed
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FeedDefinition {
    /// Unique feed identifier
    pub feed_id: String,
    /// Human-readable name
    pub name: String,
    /// Description
    pub description: String,
    /// Data type schema
    pub data_schema: DataSchema,
    /// Update frequency
    pub update_frequency: UpdateFrequency,
    /// Minimum providers required
    pub min_providers: u32,
    /// Consensus mechanism
    pub consensus_mechanism: ConsensusMechanism,
    /// Epistemic classification of this feed
    pub epistemic_level: EmpiricalLevel,
    /// Who created this feed
    pub creator: AgentPubKey,
    /// Is this feed active?
    pub active: bool,
    /// Staking requirements for providers
    pub provider_stake_required: Option<StakeRequirement>,
    /// Consumer fee per request
    pub consumer_fee: Option<Fee>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DataSchema {
    /// Simple numeric value
    Numeric { unit: String, precision: u8 },
    /// String value with validation
    String { max_length: usize, pattern: Option<String> },
    /// Boolean value
    Boolean,
    /// Geographic coordinates
    GeoLocation,
    /// Timestamp
    Timestamp,
    /// Complex JSON schema
    Json { schema: String },
    /// Binary data with type
    Binary { content_type: String, max_size: usize },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum UpdateFrequency {
    /// Update on every request
    OnDemand,
    /// Regular interval updates
    Interval { seconds: u64 },
    /// Update when value changes by threshold
    OnChange { threshold: f64 },
    /// One-time event verification
    OneTime,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsensusMechanism {
    /// Simple median (for numeric data)
    Median,
    /// Weighted median by trust score
    WeightedMedian,
    /// Mode (most common value)
    Mode,
    /// Byzantine fault tolerant consensus
    ByzantineFT { tolerance: f64 },
    /// All providers must agree
    Unanimous,
    /// Custom aggregation function
    Custom { function_hash: String },
}

/// A data provider's registration
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProviderProfile {
    /// Provider's agent key
    pub agent: AgentPubKey,
    /// Feeds this provider supplies
    pub feeds: Vec<String>,
    /// Data sources used
    pub data_sources: Vec<DataSourceDeclaration>,
    /// MATL trust score
    pub trust_score: f64,
    /// Historical accuracy
    pub accuracy_rating: f64,
    /// Uptime percentage
    pub uptime: f64,
    /// Total submissions
    pub total_submissions: u64,
    /// Successful submissions (within consensus)
    pub successful_submissions: u64,
    /// Staked amount (if applicable)
    pub staked: Option<StakedAmount>,
    /// Registered at
    pub registered_at: Timestamp,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct DataSourceDeclaration {
    /// Source name
    pub name: String,
    /// Source type
    pub source_type: DataSourceType,
    /// How data is obtained
    pub acquisition_method: String,
    /// Last verified date
    pub last_verified: Option<Timestamp>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DataSourceType {
    /// Direct API access
    DirectAPI { provider: String },
    /// IoT device
    IoTDevice { device_type: String, attestation: Option<String> },
    /// Public record
    PublicRecord { jurisdiction: String },
    /// Aggregated from multiple sources
    Aggregated { sources: Vec<String> },
    /// Human reporter
    HumanReport,
}

/// A single data submission from a provider
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DataSubmission {
    /// Feed this submission is for
    pub feed_id: String,
    /// Provider who submitted
    pub provider: AgentPubKey,
    /// The actual data value
    pub value: DataValue,
    /// When the data was observed
    pub observed_at: Timestamp,
    /// When submitted to Oracle
    pub submitted_at: Timestamp,
    /// Source used for this submission
    pub source: String,
    /// Confidence level (self-reported)
    pub confidence: f64,
    /// Signature for verification
    pub signature: Signature,
    /// Round this submission is for (if interval-based)
    pub round: Option<u64>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DataValue {
    Numeric(f64),
    String(String),
    Boolean(bool),
    GeoLocation { lat: f64, lon: f64, accuracy_m: f64 },
    Timestamp(u64),
    Json(String),
    Binary { hash: String, size: usize },
}

/// Aggregated result after consensus
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OracleResult {
    /// Feed ID
    pub feed_id: String,
    /// Round number (for interval feeds)
    pub round: u64,
    /// The consensus value
    pub value: DataValue,
    /// Confidence in this value (0-1)
    pub confidence: f64,
    /// Epistemic classification
    pub epistemic: EpistemicClaim,
    /// Submissions considered
    pub submissions: Vec<ActionHash>,
    /// Submissions excluded (outliers/byzantine)
    pub excluded: Vec<ExcludedSubmission>,
    /// Provider weights used
    pub provider_weights: Vec<(AgentPubKey, f64)>,
    /// Aggregated at
    pub aggregated_at: Timestamp,
    /// Valid until
    pub valid_until: Timestamp,
    /// Consensus method used
    pub consensus_used: ConsensusMechanism,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ExcludedSubmission {
    pub submission: ActionHash,
    pub provider: AgentPubKey,
    pub reason: ExclusionReason,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ExclusionReason {
    OutlierValue { deviation: f64 },
    ByzantineDetected,
    LateSubmission,
    InvalidFormat,
    ProviderSuspended,
}

/// Dispute about a data value
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DataDispute {
    /// Result being disputed
    pub oracle_result: ActionHash,
    /// Who is disputing
    pub disputer: AgentPubKey,
    /// Claimed correct value
    pub claimed_value: DataValue,
    /// Evidence supporting claim
    pub evidence: Vec<ActionHash>,
    /// Stake deposited for dispute
    pub stake: StakedAmount,
    /// Status
    pub status: DisputeStatus,
}
```

---

## Consensus Mechanisms

### 1. Weighted Median (Default for Numeric)

```rust
fn weighted_median(submissions: &[DataSubmission], weights: &[(AgentPubKey, f64)]) -> f64 {
    // Sort by value
    let mut sorted: Vec<_> = submissions
        .iter()
        .filter_map(|s| {
            if let DataValue::Numeric(v) = s.value {
                let weight = weights
                    .iter()
                    .find(|(a, _)| *a == s.provider)
                    .map(|(_, w)| *w)
                    .unwrap_or(0.1);
                Some((v, weight))
            } else {
                None
            }
        })
        .collect();

    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Find weighted median
    let total_weight: f64 = sorted.iter().map(|(_, w)| w).sum();
    let half_weight = total_weight / 2.0;

    let mut cumulative = 0.0;
    for (value, weight) in sorted {
        cumulative += weight;
        if cumulative >= half_weight {
            return value;
        }
    }

    sorted.last().unwrap().0
}
```

### 2. Byzantine Fault Tolerant Consensus

```rust
fn byzantine_ft_consensus(
    submissions: &[DataSubmission],
    weights: &[(AgentPubKey, f64)],
    tolerance: f64,
) -> Option<DataValue> {
    // Group similar values
    let clusters = cluster_submissions(submissions, 0.05); // 5% tolerance

    // Find cluster with most weight
    let cluster_weights: Vec<_> = clusters
        .iter()
        .map(|cluster| {
            let weight: f64 = cluster
                .iter()
                .map(|s| {
                    weights
                        .iter()
                        .find(|(a, _)| *a == s.provider)
                        .map(|(_, w)| *w)
                        .unwrap_or(0.1)
                })
                .sum();
            (cluster, weight)
        })
        .collect();

    let total_weight: f64 = cluster_weights.iter().map(|(_, w)| w).sum();

    // Need more than (1 - tolerance) weight to agree
    let required_weight = (1.0 - tolerance) * total_weight;

    for (cluster, weight) in cluster_weights {
        if weight >= required_weight {
            // Return centroid of winning cluster
            return Some(compute_centroid(cluster));
        }
    }

    // No consensus reached
    None
}
```

### 3. Mode (Most Common Value)

```rust
fn mode_consensus(submissions: &[DataSubmission]) -> Option<DataValue> {
    let mut counts: HashMap<String, (DataValue, usize, f64)> = HashMap::new();

    for submission in submissions {
        let key = serialize_value(&submission.value);
        let entry = counts.entry(key).or_insert((submission.value.clone(), 0, 0.0));
        entry.1 += 1;
        entry.2 += submission.confidence;
    }

    counts
        .into_values()
        .max_by_key(|(_, count, conf)| (*count, *conf as u64))
        .map(|(value, _, _)| value)
}
```

---

## Zome Specifications

### 1. Feed Manager Zome

```rust
// feed_manager/src/lib.rs

/// Create a new data feed
#[hdk_extern]
pub fn create_feed(input: CreateFeedInput) -> ExternResult<ActionHash> {
    let creator = agent_info()?.agent_latest_pubkey;

    // Validate schema
    validate_data_schema(&input.data_schema)?;

    // Determine epistemic level based on feed characteristics
    let epistemic_level = determine_epistemic_level(&input)?;

    let feed = FeedDefinition {
        feed_id: generate_feed_id(&input.name),
        name: input.name,
        description: input.description,
        data_schema: input.data_schema,
        update_frequency: input.update_frequency,
        min_providers: input.min_providers.unwrap_or(3),
        consensus_mechanism: input.consensus_mechanism.unwrap_or(ConsensusMechanism::WeightedMedian),
        epistemic_level,
        creator,
        active: true,
        provider_stake_required: input.provider_stake,
        consumer_fee: input.consumer_fee,
    };

    let hash = create_entry(&EntryTypes::FeedDefinition(feed.clone()))?;

    // Index feed
    create_link(
        all_feeds_path().path_entry_hash()?,
        hash.clone(),
        LinkTypes::AllFeeds,
        feed.feed_id.as_bytes(),
    )?;

    Ok(hash)
}

/// Get current value for a feed
#[hdk_extern]
pub fn get_feed_value(feed_id: String) -> ExternResult<OracleResult> {
    // Get latest aggregated result
    let result = get_latest_result(&feed_id)?;

    // Check if still valid
    if sys_time()? > result.valid_until {
        // Trigger new aggregation if needed
        trigger_aggregation(&feed_id)?;
        return get_latest_result(&feed_id);
    }

    Ok(result)
}

/// Subscribe to feed updates
#[hdk_extern]
pub fn subscribe_to_feed(input: SubscribeInput) -> ExternResult<ActionHash> {
    let subscriber = agent_info()?.agent_latest_pubkey;

    // Verify feed exists
    let feed = get_feed(&input.feed_id)?;

    // Handle consumer fee if applicable
    if let Some(fee) = &feed.consumer_fee {
        process_subscription_fee(&subscriber, fee, &input)?;
    }

    // Create subscription
    let subscription = FeedSubscription {
        feed_id: input.feed_id,
        subscriber,
        callback_type: input.callback_type,
        filters: input.filters,
        created_at: sys_time()?,
        expires_at: input.expires_at,
    };

    create_entry(&EntryTypes::FeedSubscription(subscription))
}

/// Trigger manual feed update
#[hdk_extern]
pub fn request_update(feed_id: String) -> ExternResult<()> {
    let requester = agent_info()?.agent_latest_pubkey;
    let feed = get_feed(&feed_id)?;

    // For on-demand feeds, trigger collection
    if matches!(feed.update_frequency, UpdateFrequency::OnDemand) {
        // Signal providers to submit
        signal_providers_to_submit(&feed_id)?;

        // Set timeout for aggregation
        schedule_aggregation(&feed_id, 30)?; // 30 seconds
    }

    Ok(())
}

fn determine_epistemic_level(input: &CreateFeedInput) -> EmpiricalLevel {
    // IoT with attestation = E3
    // Multiple independent APIs = E2-E3
    // Single API = E2
    // Human report = E1

    match input.min_providers {
        n if n >= 5 => EmpiricalLevel::E3Cryptographic,
        n if n >= 3 => EmpiricalLevel::E2PrivateVerify,
        _ => EmpiricalLevel::E1Testimonial,
    }
}
```

### 2. Provider Registry Zome

```rust
// provider_registry/src/lib.rs

/// Register as a data provider
#[hdk_extern]
pub fn register_provider(input: ProviderRegistration) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Get MATL trust score
    let trust_score = bridge_call::<f64>(
        "bridge",
        "get_cross_happ_reputation",
        agent.clone(),
    )?;

    // Minimum trust requirement
    if trust_score < 0.5 {
        return Err(WasmError::Guest("Trust score too low (min 0.5)".into()));
    }

    // Handle staking if required
    for feed_id in &input.feeds {
        let feed = get_feed(feed_id)?;
        if let Some(stake_req) = &feed.provider_stake_required {
            verify_stake_deposited(&agent, stake_req)?;
        }
    }

    let profile = ProviderProfile {
        agent,
        feeds: input.feeds,
        data_sources: input.data_sources,
        trust_score,
        accuracy_rating: 0.0, // Will be computed over time
        uptime: 0.0,
        total_submissions: 0,
        successful_submissions: 0,
        staked: input.staked,
        registered_at: sys_time()?,
    };

    let hash = create_entry(&EntryTypes::ProviderProfile(profile))?;

    // Index by feed
    for feed_id in &input.feeds {
        create_link(
            feed_providers_path(feed_id).path_entry_hash()?,
            hash.clone(),
            LinkTypes::FeedToProvider,
            (),
        )?;
    }

    Ok(hash)
}

/// Update provider's data sources
#[hdk_extern]
pub fn update_data_sources(sources: Vec<DataSourceDeclaration>) -> ExternResult<()> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Get current profile
    let profile = get_provider_profile(&agent)?;

    // Update sources
    let updated = ProviderProfile {
        data_sources: sources,
        ..profile
    };

    update_entry(get_provider_profile_hash(&agent)?, &updated)?;

    Ok(())
}

/// Suspend a provider (governance action)
#[hdk_extern]
pub fn suspend_provider(input: SuspendProviderInput) -> ExternResult<()> {
    // Verify caller has authority (governance or dispute resolution)
    verify_suspension_authority()?;

    let profile = get_provider_profile(&input.provider)?;

    // Create suspension record
    let suspension = ProviderSuspension {
        provider: input.provider,
        reason: input.reason,
        suspended_at: sys_time()?,
        suspended_until: input.until,
        suspended_by: agent_info()?.agent_latest_pubkey,
    };

    create_entry(&EntryTypes::ProviderSuspension(suspension))?;

    // Slash stake if applicable
    if let Some(slash_amount) = input.slash_amount {
        slash_provider_stake(&input.provider, slash_amount)?;
    }

    Ok(())
}

/// Get providers for a feed with weights
pub fn get_weighted_providers(feed_id: &str) -> ExternResult<Vec<(AgentPubKey, f64)>> {
    let providers = get_feed_providers(feed_id)?;

    let weighted: Vec<_> = providers
        .into_iter()
        .filter(|p| !is_suspended(&p.agent))
        .map(|p| {
            let weight = compute_provider_weight(&p);
            (p.agent, weight)
        })
        .collect();

    Ok(weighted)
}

fn compute_provider_weight(provider: &ProviderProfile) -> f64 {
    // Weight based on:
    // - MATL trust score (40%)
    // - Historical accuracy (30%)
    // - Uptime (20%)
    // - Stake amount (10%)

    let trust_component = provider.trust_score * 0.4;
    let accuracy_component = provider.accuracy_rating * 0.3;
    let uptime_component = provider.uptime * 0.2;

    let stake_component = if let Some(stake) = &provider.staked {
        (stake.amount as f64 / 10000.0).min(1.0) * 0.1
    } else {
        0.0
    };

    trust_component + accuracy_component + uptime_component + stake_component
}
```

### 3. Consensus Engine Zome

```rust
// consensus_engine/src/lib.rs

/// Submit data for a feed
#[hdk_extern]
pub fn submit_data(input: DataSubmissionInput) -> ExternResult<ActionHash> {
    let provider = agent_info()?.agent_latest_pubkey;

    // Verify provider is registered for this feed
    let profile = get_provider_profile(&provider)?;
    if !profile.feeds.contains(&input.feed_id) {
        return Err(WasmError::Guest("Not registered for this feed".into()));
    }

    // Check provider is not suspended
    if is_suspended(&provider) {
        return Err(WasmError::Guest("Provider is suspended".into()));
    }

    // Validate data format
    let feed = get_feed(&input.feed_id)?;
    validate_data_format(&input.value, &feed.data_schema)?;

    // Create submission
    let submission = DataSubmission {
        feed_id: input.feed_id,
        provider: provider.clone(),
        value: input.value,
        observed_at: input.observed_at,
        submitted_at: sys_time()?,
        source: input.source,
        confidence: input.confidence,
        signature: sign_submission(&input)?,
        round: get_current_round(&input.feed_id)?,
    };

    let hash = create_entry(&EntryTypes::DataSubmission(submission.clone()))?;

    // Check if we have enough submissions for aggregation
    let round_submissions = get_round_submissions(&input.feed_id, submission.round)?;
    if round_submissions.len() >= feed.min_providers as usize {
        // Trigger aggregation
        aggregate_round(&input.feed_id, submission.round.unwrap_or(0))?;
    }

    Ok(hash)
}

/// Aggregate submissions into result
pub fn aggregate_round(feed_id: &str, round: u64) -> ExternResult<ActionHash> {
    let feed = get_feed(feed_id)?;
    let submissions = get_round_submissions(feed_id, Some(round))?;

    if submissions.len() < feed.min_providers as usize {
        return Err(WasmError::Guest("Insufficient submissions".into()));
    }

    // Get provider weights
    let weights = get_weighted_providers(feed_id)?;

    // Run consensus
    let (value, excluded) = match &feed.consensus_mechanism {
        ConsensusMechanism::Median => {
            let value = median_consensus(&submissions);
            let excluded = identify_outliers(&submissions, &value);
            (Some(value), excluded)
        }
        ConsensusMechanism::WeightedMedian => {
            let value = weighted_median(&submissions, &weights);
            let excluded = identify_outliers(&submissions, &value);
            (Some(value), excluded)
        }
        ConsensusMechanism::ByzantineFT { tolerance } => {
            let value = byzantine_ft_consensus(&submissions, &weights, *tolerance);
            let excluded = if value.is_some() {
                identify_byzantine(&submissions, value.as_ref().unwrap(), &weights)
            } else {
                vec![]
            };
            (value, excluded)
        }
        ConsensusMechanism::Mode => {
            let value = mode_consensus(&submissions);
            (value, vec![])
        }
        ConsensusMechanism::Unanimous => {
            let value = unanimous_consensus(&submissions);
            (value, vec![])
        }
        ConsensusMechanism::Custom { function_hash } => {
            run_custom_consensus(function_hash, &submissions, &weights)?
        }
    };

    let consensus_value = value.ok_or(WasmError::Guest("Consensus not reached".into()))?;

    // Calculate confidence
    let confidence = calculate_confidence(&submissions, &consensus_value, &weights);

    // Create epistemic claim
    let epistemic = EpistemicClaim::new(
        format!("Oracle feed {} round {}", feed_id, round),
        feed.epistemic_level,
        NormativeLevel::N2Network, // Network-wide consensus
        MaterialityLevel::M1Temporal, // Valid until next update
    );

    // Calculate validity period
    let valid_until = match &feed.update_frequency {
        UpdateFrequency::Interval { seconds } => sys_time()?.add_seconds(*seconds),
        _ => sys_time()?.add_seconds(3600), // 1 hour default
    };

    // Create result
    let result = OracleResult {
        feed_id: feed_id.to_string(),
        round,
        value: consensus_value,
        confidence,
        epistemic,
        submissions: submissions.iter().map(|s| s.hash()).collect(),
        excluded,
        provider_weights: weights,
        aggregated_at: sys_time()?,
        valid_until,
        consensus_used: feed.consensus_mechanism.clone(),
    };

    let hash = create_entry(&EntryTypes::OracleResult(result.clone()))?;

    // Update provider statistics
    update_provider_stats(&submissions, &result)?;

    // Notify subscribers
    notify_subscribers(feed_id, &result)?;

    // Link as latest result
    update_latest_result_link(feed_id, &hash)?;

    Ok(hash)
}

/// Identify outliers/byzantine submissions
fn identify_byzantine(
    submissions: &[DataSubmission],
    consensus: &DataValue,
    weights: &[(AgentPubKey, f64)],
) -> Vec<ExcludedSubmission> {
    let mut excluded = vec![];

    for submission in submissions {
        let deviation = calculate_deviation(&submission.value, consensus);

        // Significant deviation from consensus
        if deviation > 0.2 { // 20% threshold
            excluded.push(ExcludedSubmission {
                submission: submission.hash(),
                provider: submission.provider.clone(),
                reason: if deviation > 0.5 {
                    ExclusionReason::ByzantineDetected
                } else {
                    ExclusionReason::OutlierValue { deviation }
                },
            });
        }
    }

    excluded
}

/// Update provider accuracy stats
fn update_provider_stats(
    submissions: &[DataSubmission],
    result: &OracleResult,
) -> ExternResult<()> {
    for submission in submissions {
        let mut profile = get_provider_profile(&submission.provider)?;

        profile.total_submissions += 1;

        // Check if submission was excluded
        let excluded = result.excluded.iter().any(|e| e.provider == submission.provider);

        if !excluded {
            profile.successful_submissions += 1;
        }

        // Update accuracy rating
        profile.accuracy_rating =
            profile.successful_submissions as f64 / profile.total_submissions as f64;

        // Reputation impact
        let impact = if excluded { -0.01 } else { 0.001 };
        bridge_call::<()>(
            "bridge",
            "apply_reputation_impact",
            ReputationImpact {
                agent: submission.provider.clone(),
                happ: "oracle".to_string(),
                impact,
                reason: if excluded {
                    "Excluded from consensus".to_string()
                } else {
                    "Contributed to consensus".to_string()
                },
            },
        )?;

        update_provider_profile(&profile)?;
    }

    Ok(())
}
```

### 4. Consumer Service Zome

```rust
// consumer_service/src/lib.rs

/// Request data from Oracle (for other hApps)
#[hdk_extern]
pub fn request_data(input: DataRequest) -> ExternResult<OracleResponse> {
    let requester = agent_info()?.agent_latest_pubkey;

    // Get feed
    let feed = get_feed(&input.feed_id)?;

    // Handle consumer fee
    if let Some(fee) = &feed.consumer_fee {
        process_request_fee(&requester, fee)?;
    }

    // Get latest result
    let result = get_feed_value(input.feed_id.clone())?;

    // Check if result meets requester's requirements
    if let Some(min_confidence) = input.min_confidence {
        if result.confidence < min_confidence {
            return Err(WasmError::Guest("Confidence below threshold".into()));
        }
    }

    if let Some(max_age) = input.max_age_seconds {
        let age = sys_time()?.as_secs() - result.aggregated_at.as_secs();
        if age > max_age {
            // Trigger fresh aggregation if on-demand
            if matches!(feed.update_frequency, UpdateFrequency::OnDemand) {
                request_update(input.feed_id)?;
                // Wait for new result (with timeout)
                return wait_for_fresh_result(&input.feed_id, 60);
            }
            return Err(WasmError::Guest("Data too old".into()));
        }
    }

    Ok(OracleResponse {
        feed_id: input.feed_id,
        value: result.value,
        confidence: result.confidence,
        epistemic: result.epistemic,
        aggregated_at: result.aggregated_at,
        valid_until: result.valid_until,
        providers_count: result.provider_weights.len(),
    })
}

/// Batch request for multiple feeds
#[hdk_extern]
pub fn batch_request(feed_ids: Vec<String>) -> ExternResult<Vec<OracleResponse>> {
    let mut responses = vec![];

    for feed_id in feed_ids {
        match request_data(DataRequest {
            feed_id,
            min_confidence: None,
            max_age_seconds: None,
        }) {
            Ok(response) => responses.push(response),
            Err(_) => continue, // Skip failed feeds
        }
    }

    Ok(responses)
}

/// Verify a historical oracle result
#[hdk_extern]
pub fn verify_result(result_hash: ActionHash) -> ExternResult<VerificationResult> {
    let result = get_oracle_result(&result_hash)?;

    // Verify all submissions
    let mut valid_submissions = 0;
    let mut invalid_submissions = 0;

    for submission_hash in &result.submissions {
        let submission = get_data_submission(submission_hash)?;

        // Verify signature
        if verify_submission_signature(&submission)? {
            valid_submissions += 1;
        } else {
            invalid_submissions += 1;
        }
    }

    // Re-run consensus to verify result
    let recalculated = recalculate_consensus(&result)?;

    let matches = recalculated.value == result.value;

    Ok(VerificationResult {
        result: result_hash,
        valid: matches && invalid_submissions == 0,
        valid_submissions,
        invalid_submissions,
        consensus_matches: matches,
        details: if matches {
            "Result verified successfully".to_string()
        } else {
            "Consensus mismatch detected".to_string()
        },
    })
}
```

---

## Feed Categories

### Price Feeds
```rust
// Example: ETH/USD price feed
FeedDefinition {
    feed_id: "price/eth-usd",
    name: "ETH/USD Price",
    data_schema: DataSchema::Numeric { unit: "USD", precision: 2 },
    update_frequency: UpdateFrequency::Interval { seconds: 60 },
    min_providers: 5,
    consensus_mechanism: ConsensusMechanism::WeightedMedian,
    epistemic_level: EmpiricalLevel::E3Cryptographic,
    ...
}
```

### IoT Sensor Feeds
```rust
// Example: Temperature sensor
FeedDefinition {
    feed_id: "iot/temp/warehouse-a",
    name: "Warehouse A Temperature",
    data_schema: DataSchema::Numeric { unit: "Celsius", precision: 1 },
    update_frequency: UpdateFrequency::Interval { seconds: 300 },
    min_providers: 3, // Redundant sensors
    consensus_mechanism: ConsensusMechanism::ByzantineFT { tolerance: 0.33 },
    epistemic_level: EmpiricalLevel::E2PrivateVerify,
    ...
}
```

### Event Verification
```rust
// Example: Flight arrival
FeedDefinition {
    feed_id: "event/flight-arrival",
    name: "Flight Arrival Verification",
    data_schema: DataSchema::Json { schema: FLIGHT_SCHEMA },
    update_frequency: UpdateFrequency::OnDemand,
    min_providers: 3,
    consensus_mechanism: ConsensusMechanism::Mode,
    epistemic_level: EmpiricalLevel::E2PrivateVerify,
    ...
}
```

### Geographic Data
```rust
// Example: Asset location tracking
FeedDefinition {
    feed_id: "geo/asset/container-123",
    name: "Container 123 Location",
    data_schema: DataSchema::GeoLocation,
    update_frequency: UpdateFrequency::Interval { seconds: 3600 },
    min_providers: 2,
    consensus_mechanism: ConsensusMechanism::WeightedMedian, // For lat/lon
    epistemic_level: EmpiricalLevel::E2PrivateVerify,
    ...
}
```

---

## Cross-hApp Integration

### SupplyChain Integration
```rust
// SupplyChain requests temperature data
let temp = bridge_call::<OracleResponse>(
    "oracle",
    "request_data",
    DataRequest {
        feed_id: "iot/temp/container-123".to_string(),
        min_confidence: Some(0.8),
        max_age_seconds: Some(3600),
    },
)?;

// Record in provenance
create_provenance_event(ProvenanceEvent {
    product_id,
    event_type: ProvenanceEventType::EnvironmentReading,
    data: temp.value,
    oracle_proof: temp.result_hash,
    ..
})?;
```

### Marketplace Integration
```rust
// Marketplace uses price feeds
let price = bridge_call::<OracleResponse>(
    "oracle",
    "request_data",
    DataRequest {
        feed_id: format!("price/{}-{}", base_currency, quote_currency),
        min_confidence: Some(0.9),
        max_age_seconds: Some(300),
    },
)?;
```

### Mutual (Insurance) Integration
```rust
// Weather-indexed insurance
let weather = bridge_call::<OracleResponse>(
    "oracle",
    "request_data",
    DataRequest {
        feed_id: format!("weather/precip/{}", location_id),
        min_confidence: Some(0.7),
        max_age_seconds: Some(86400),
    },
)?;

// Trigger payout if conditions met
if weather.value.precipitation_mm > threshold {
    trigger_payout(policy_id)?;
}
```

---

## Economic Model

### Provider Incentives

| Activity | Reward/Penalty |
|----------|---------------|
| Accurate submission | +0.001 MATL score |
| Outlier submission | -0.01 MATL score |
| Byzantine detection | -0.05 MATL score + stake slash |
| High uptime (>99%) | Bonus multiplier |

### Consumer Fees (Optional)

Feed creators can set fees:
- Per-request fee
- Subscription fee
- Premium for low-latency

Fee distribution:
- 60% to providers
- 30% to feed creator
- 10% to Oracle development

---

## Success Metrics

| Metric | 1 Year | 3 Years |
|--------|--------|---------|
| Active feeds | 100 | 1,000 |
| Data providers | 50 | 500 |
| Daily requests | 10,000 | 1M |
| Consensus accuracy | 99% | 99.9% |
| Byzantine detection rate | 95% | 99% |
| Median latency | <10s | <5s |

---

*"Truth is not decreed from above. It emerges from the consensus of many honest observers."*
