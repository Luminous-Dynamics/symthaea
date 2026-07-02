# Mycelix-Sentinel: Network Security Monitor

## Vision & Design Document

**Version**: 1.0.0
**Created**: December 30, 2025
**Status**: Design Phase
**Priority**: Tier 1 - Research/Experimental

---

## Executive Summary

Mycelix-Sentinel is the immune system of the Mycelix ecosystem. It provides real-time Byzantine behavior detection, security anomaly monitoring, threat intelligence sharing, and coordinated response to attacks. Built on the 0TML Byzantine detection algorithms, Sentinel protects the network while preserving privacy and decentralization.

### Why Sentinel?

Decentralized networks face unique security challenges:
- **Byzantine actors**: Up to 45% may be adversarial
- **Sybil attacks**: Fake identities at scale
- **Collusion**: Coordinated manipulation
- **Eclipse attacks**: Network isolation
- **Slow-roll attacks**: Gradual reputation manipulation

Sentinel makes these attacks visible, attributable, and defensible.

---

## Core Principles

### 1. Proactive Defense
Detect threats before they cause harm, not after.

### 2. Privacy-Preserving Monitoring
Monitor behavior patterns, not personal data.

### 3. Distributed Sensing
Every node is a sensor; intelligence is collective.

### 4. Proportional Response
Responses match threat severity; no overreaction.

### 5. Transparent Accountability
Security actions are auditable and challengeable.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Threat Landscape                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │Byzantine │ │  Sybil   │ │ Collusion│ │  Eclipse │              │
│  │ Actors   │ │ Attacks  │ │  Rings   │ │ Attacks  │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│                       Sentinel hApp                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Coordinator Zomes                             ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │  Threat  │ │ Anomaly  │ │ Response │ │   Intel  │           ││
│  │  │ Detector │ │ Analyzer │ │Coordinator│ │  Sharing │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Network  │ │ Forensics│ │ Bounty   │ │ Alert    │           ││
│  │  │ Health   │ │ Engine   │ │ Manager  │ │ System   │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                      Data Sources                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ 0TML/MATL    │  │ All hApps    │  │ Network Layer            │   │
│  │ Trust Scores │  │ Activity Logs│  │ (Kitsune2 metrics)       │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// A security threat report
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ThreatReport {
    /// Unique report ID
    pub report_id: String,
    /// Threat type
    pub threat_type: ThreatType,
    /// Severity level
    pub severity: Severity,
    /// Confidence in detection
    pub confidence: f64,
    /// Suspected agents
    pub suspects: Vec<AgentPubKey>,
    /// Evidence
    pub evidence: Vec<ThreatEvidence>,
    /// Detection method
    pub detection_method: DetectionMethod,
    /// Affected hApps
    pub affected_happs: Vec<String>,
    /// Reported at
    pub reported_at: Timestamp,
    /// Reporter (can be system or agent)
    pub reporter: Reporter,
    /// Status
    pub status: ThreatStatus,
    /// Response actions taken
    pub responses: Vec<ResponseAction>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ThreatType {
    /// Byzantine behavior in consensus
    Byzantine {
        context: String,
        deviation: f64,
    },
    /// Sybil attack (fake identities)
    Sybil {
        estimated_fake_count: u32,
        pattern: String,
    },
    /// Collusion ring
    Collusion {
        ring_size: u32,
        coordination_pattern: String,
    },
    /// Eclipse attack
    Eclipse {
        target: AgentPubKey,
        isolation_degree: f64,
    },
    /// Spam/DoS
    Spam {
        rate: f64,
        pattern: String,
    },
    /// Reputation manipulation
    ReputationManipulation {
        direction: String, // inflation or deflation
        target: Option<AgentPubKey>,
    },
    /// Data poisoning (in FL)
    DataPoisoning {
        model_affected: String,
        impact_estimate: f64,
    },
    /// Credential fraud
    CredentialFraud {
        credential_type: String,
    },
    /// Smart contract exploit
    ContractExploit {
        contract: String,
        exploit_type: String,
    },
    /// Network layer attack
    NetworkAttack {
        attack_type: String,
    },
    /// Other
    Other(String),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Severity {
    /// Informational only
    Info,
    /// Low severity, monitor
    Low,
    /// Medium severity, investigate
    Medium,
    /// High severity, respond
    High,
    /// Critical, immediate action needed
    Critical,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreatEvidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Data or reference
    pub data: String,
    /// Collection timestamp
    pub collected_at: Timestamp,
    /// Epistemic classification
    pub epistemic: EpistemicClaim,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Transaction pattern
    TransactionPattern,
    /// Network behavior
    NetworkBehavior,
    /// Timing analysis
    TimingAnalysis,
    /// Reputation trajectory
    ReputationTrajectory,
    /// Cross-correlation
    CrossCorrelation,
    /// FL gradient analysis
    GradientAnalysis,
    /// Witness report
    WitnessReport,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// 0TML Byzantine detection
    ZeroTML { algorithm: String },
    /// MATL anomaly detection
    MATL { threshold: f64 },
    /// Machine learning model
    MLModel { model_id: String },
    /// Heuristic rules
    Heuristic { rule_id: String },
    /// Human report
    HumanReport,
    /// Cross-hApp correlation
    CrossHappCorrelation,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Reporter {
    System { detector_id: String },
    Agent(AgentPubKey),
    Anonymous { proof: String }, // ZKP that reporter is valid
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ThreatStatus {
    Reported,
    UnderInvestigation,
    Confirmed,
    FalsePositive,
    Mitigated,
    Escalated,
    Closed,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseAction {
    pub action_type: ResponseType,
    pub taken_at: Timestamp,
    pub taken_by: AgentPubKey,
    pub result: ActionResult,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseType {
    /// Flag agent for monitoring
    Flag { duration: u64 },
    /// Reduce agent's voting weight
    ReduceWeight { factor: f64 },
    /// Suspend from specific hApp
    SuspendFromHapp { happ: String },
    /// Network-wide suspension
    NetworkSuspend,
    /// Quarantine data/transactions
    Quarantine { items: Vec<ActionHash> },
    /// Alert other nodes
    AlertNetwork,
    /// Trigger Arbiter dispute
    DisputeReferral { dispute: ActionHash },
    /// Request governance action
    GovernanceReferral { proposal: ActionHash },
}

/// Anomaly detection pattern
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AnomalyPattern {
    /// Pattern ID
    pub pattern_id: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Detection rules
    pub rules: Vec<DetectionRule>,
    /// Threshold for alert
    pub alert_threshold: f64,
    /// Active
    pub active: bool,
    /// Created by
    pub created_by: AgentPubKey,
    /// Effectiveness score
    pub effectiveness: f64,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionRule {
    pub metric: String,
    pub operator: Operator,
    pub threshold: f64,
    pub time_window: Option<u64>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Operator {
    GreaterThan,
    LessThan,
    Equals,
    DeviatesFromMean { std_devs: f64 },
    RateOfChange { threshold: f64 },
    Correlation { with: String, min_correlation: f64 },
}

/// Security alert
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SecurityAlert {
    /// Alert ID
    pub alert_id: String,
    /// Related threat report
    pub threat_report: Option<ActionHash>,
    /// Alert level
    pub level: AlertLevel,
    /// Message
    pub message: String,
    /// Affected entities
    pub affected: Vec<AffectedEntity>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Created at
    pub created_at: Timestamp,
    /// Acknowledged by
    pub acknowledged_by: Vec<AgentPubKey>,
    /// Resolved
    pub resolved: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertLevel {
    Watch,
    Advisory,
    Warning,
    Emergency,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AffectedEntity {
    pub entity_type: EntityType,
    pub identifier: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum EntityType {
    Agent,
    HApp,
    Network,
    Transaction,
    Credential,
}

/// Bug bounty program
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BugBounty {
    /// Bounty ID
    pub bounty_id: String,
    /// Scope
    pub scope: BountyScope,
    /// Rewards by severity
    pub rewards: HashMap<String, u64>,
    /// Rules
    pub rules: Vec<String>,
    /// Active period
    pub active_from: Timestamp,
    pub active_until: Option<Timestamp>,
    /// Total pool
    pub reward_pool: u64,
    /// Claimed so far
    pub claimed: u64,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct BountyScope {
    pub happs: Vec<String>,
    pub in_scope: Vec<String>,
    pub out_of_scope: Vec<String>,
}

/// Bug bounty submission
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BountySubmission {
    /// Bounty program
    pub bounty: ActionHash,
    /// Submitter
    pub submitter: AgentPubKey,
    /// Vulnerability details (encrypted for reviewers)
    pub details_encrypted: String,
    /// Severity claimed
    pub severity_claimed: Severity,
    /// Proof of concept
    pub proof_of_concept: Option<String>,
    /// Submitted at
    pub submitted_at: Timestamp,
    /// Status
    pub status: BountyStatus,
    /// Reward paid
    pub reward_paid: Option<u64>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum BountyStatus {
    Submitted,
    UnderReview,
    Confirmed { severity: Severity },
    Duplicate,
    Invalid,
    Paid,
}
```

---

## Detection Algorithms

### 1. Byzantine Behavior Detection (0TML Integration)

```rust
/// Monitor for Byzantine behavior in FL rounds
pub fn monitor_fl_round(round_id: &str) -> ExternResult<Vec<ThreatReport>> {
    // Get round data from 0TML
    let round_data = bridge_call::<FLRoundData>(
        "zerotrustml",
        "get_round_data",
        round_id,
    )?;

    let mut threats = vec![];

    // Check each participant
    for contribution in &round_data.contributions {
        let pogq = bridge_call::<ProofOfGradientQuality>(
            "zerotrustml",
            "get_pogq",
            contribution.contributor.clone(),
        )?;

        // Check against adaptive threshold
        let threshold = get_adaptive_threshold(&contribution.contributor)?;

        if pogq.quality < threshold {
            let severity = if pogq.quality < threshold * 0.5 {
                Severity::High
            } else {
                Severity::Medium
            };

            threats.push(ThreatReport {
                report_id: generate_report_id(),
                threat_type: ThreatType::Byzantine {
                    context: format!("FL round {}", round_id),
                    deviation: threshold - pogq.quality,
                },
                severity,
                confidence: 1.0 - pogq.quality,
                suspects: vec![contribution.contributor.clone()],
                evidence: vec![ThreatEvidence {
                    evidence_type: EvidenceType::GradientAnalysis,
                    data: serde_json::to_string(&pogq)?,
                    collected_at: sys_time()?,
                    epistemic: EpistemicClaim::new(
                        "PoGQ analysis",
                        EmpiricalLevel::E3Cryptographic,
                        NormativeLevel::N2Network,
                        MaterialityLevel::M1Temporal,
                    ),
                }],
                detection_method: DetectionMethod::ZeroTML {
                    algorithm: "adaptive_pogq".to_string(),
                },
                affected_happs: vec!["zerotrustml".to_string()],
                reported_at: sys_time()?,
                reporter: Reporter::System {
                    detector_id: "byzantine_detector_v1".to_string(),
                },
                status: ThreatStatus::Reported,
                responses: vec![],
            });
        }
    }

    // Check for cartel behavior
    let cartel_detection = bridge_call::<CartelAnalysis>(
        "zerotrustml",
        "analyze_cartel",
        round_id,
    )?;

    if cartel_detection.detected {
        threats.push(ThreatReport {
            report_id: generate_report_id(),
            threat_type: ThreatType::Collusion {
                ring_size: cartel_detection.suspected_members.len() as u32,
                coordination_pattern: cartel_detection.pattern.clone(),
            },
            severity: Severity::High,
            confidence: cartel_detection.confidence,
            suspects: cartel_detection.suspected_members.clone(),
            evidence: vec![ThreatEvidence {
                evidence_type: EvidenceType::CrossCorrelation,
                data: serde_json::to_string(&cartel_detection)?,
                collected_at: sys_time()?,
                epistemic: EpistemicClaim::new(
                    "Cartel analysis",
                    EmpiricalLevel::E2PrivateVerify,
                    NormativeLevel::N2Network,
                    MaterialityLevel::M2Persistent,
                ),
            }],
            detection_method: DetectionMethod::ZeroTML {
                algorithm: "cartel_detection".to_string(),
            },
            affected_happs: vec!["zerotrustml".to_string()],
            reported_at: sys_time()?,
            reporter: Reporter::System {
                detector_id: "cartel_detector_v1".to_string(),
            },
            status: ThreatStatus::Reported,
            responses: vec![],
        });
    }

    Ok(threats)
}
```

### 2. Sybil Attack Detection

```rust
/// Detect potential Sybil attacks
pub fn detect_sybil_patterns() -> ExternResult<Vec<ThreatReport>> {
    let mut threats = vec![];

    // Get recent new identities
    let new_identities = bridge_call::<Vec<IdentityAnchor>>(
        "attest",
        "get_recent_identities",
        RecentQuery { hours: 24 },
    )?;

    // Analyze patterns
    let analysis = analyze_identity_patterns(&new_identities)?;

    // Check for suspicious patterns
    if analysis.creation_rate > analysis.historical_avg * 3.0 {
        threats.push(ThreatReport {
            threat_type: ThreatType::Sybil {
                estimated_fake_count: (new_identities.len() as f64 * analysis.suspicion_rate) as u32,
                pattern: "Abnormal creation rate".to_string(),
            },
            severity: Severity::High,
            confidence: analysis.confidence,
            suspects: analysis.suspicious_identities.clone(),
            ..Default::default()
        });
    }

    // Check for correlated behavior
    let behavior_clusters = cluster_by_behavior(&new_identities)?;

    for cluster in behavior_clusters {
        if cluster.correlation > 0.9 && cluster.members.len() > 3 {
            threats.push(ThreatReport {
                threat_type: ThreatType::Sybil {
                    estimated_fake_count: cluster.members.len() as u32,
                    pattern: "Highly correlated behavior".to_string(),
                },
                severity: Severity::Medium,
                confidence: cluster.correlation,
                suspects: cluster.members.clone(),
                ..Default::default()
            });
        }
    }

    // Check for shared infrastructure indicators
    // (same timing patterns, similar network signatures)
    let infra_analysis = analyze_infrastructure_sharing(&new_identities)?;

    if infra_analysis.likely_shared_infra {
        threats.push(ThreatReport {
            threat_type: ThreatType::Sybil {
                estimated_fake_count: infra_analysis.estimated_count,
                pattern: "Shared infrastructure indicators".to_string(),
            },
            severity: Severity::Medium,
            confidence: infra_analysis.confidence,
            suspects: infra_analysis.suspects,
            ..Default::default()
        });
    }

    Ok(threats)
}

fn analyze_identity_patterns(identities: &[IdentityAnchor]) -> ExternResult<PatternAnalysis> {
    // Analyze timing of creation
    let timestamps: Vec<u64> = identities.iter()
        .map(|i| i.created_at.as_secs())
        .collect();

    // Check for burst patterns
    let burst_analysis = detect_bursts(&timestamps);

    // Check for sequential patterns (programmatic creation)
    let sequential_score = detect_sequential_patterns(&timestamps);

    // Get historical baseline
    let historical = get_historical_creation_rate()?;

    Ok(PatternAnalysis {
        creation_rate: identities.len() as f64 / 24.0,
        historical_avg: historical.avg_per_hour,
        suspicion_rate: (burst_analysis.score + sequential_score) / 2.0,
        confidence: 0.7,
        suspicious_identities: identities
            .iter()
            .filter(|i| burst_analysis.suspicious_times.contains(&i.created_at.as_secs()))
            .map(|i| i.agent.clone())
            .collect(),
    })
}
```

### 3. Reputation Manipulation Detection

```rust
/// Detect reputation manipulation attempts
pub fn detect_reputation_manipulation() -> ExternResult<Vec<ThreatReport>> {
    let mut threats = vec![];

    // Get recent reputation changes
    let changes = bridge_call::<Vec<ReputationChange>>(
        "bridge",
        "get_recent_reputation_changes",
        RecentQuery { hours: 24 },
    )?;

    // Build graph of reputation flows
    let graph = build_reputation_graph(&changes);

    // Detect wash trading (circular reputation boosting)
    let cycles = detect_cycles(&graph);

    for cycle in cycles {
        if cycle.len() >= 3 && is_suspicious_cycle(&cycle, &changes) {
            threats.push(ThreatReport {
                threat_type: ThreatType::ReputationManipulation {
                    direction: "inflation".to_string(),
                    target: None,
                },
                severity: Severity::High,
                confidence: 0.8,
                suspects: cycle.clone(),
                evidence: vec![ThreatEvidence {
                    evidence_type: EvidenceType::TransactionPattern,
                    data: format!("Circular reputation cycle detected: {:?}", cycle),
                    collected_at: sys_time()?,
                    epistemic: EpistemicClaim::new(
                        "Reputation graph analysis",
                        EmpiricalLevel::E3Cryptographic,
                        NormativeLevel::N2Network,
                        MaterialityLevel::M2Persistent,
                    ),
                }],
                ..Default::default()
            });
        }
    }

    // Detect coordinated reputation attacks (brigading)
    let brigading = detect_brigading(&changes);

    for attack in brigading {
        threats.push(ThreatReport {
            threat_type: ThreatType::ReputationManipulation {
                direction: "deflation".to_string(),
                target: Some(attack.target.clone()),
            },
            severity: if attack.participants.len() > 10 {
                Severity::High
            } else {
                Severity::Medium
            },
            confidence: attack.confidence,
            suspects: attack.participants.clone(),
            ..Default::default()
        });
    }

    Ok(threats)
}
```

---

## Zome Specifications

### 1. Threat Detector Zome

```rust
// threat_detector/src/lib.rs

/// Report a security threat
#[hdk_extern]
pub fn report_threat(input: ReportThreatInput) -> ExternResult<ActionHash> {
    let reporter = agent_info()?.agent_latest_pubkey;

    // Validate evidence
    for evidence in &input.evidence {
        validate_threat_evidence(evidence)?;
    }

    // Calculate initial severity
    let severity = calculate_severity(&input)?;

    let report = ThreatReport {
        report_id: generate_report_id(),
        threat_type: input.threat_type,
        severity,
        confidence: input.confidence,
        suspects: input.suspects,
        evidence: input.evidence,
        detection_method: DetectionMethod::HumanReport,
        affected_happs: input.affected_happs,
        reported_at: sys_time()?,
        reporter: Reporter::Agent(reporter.clone()),
        status: ThreatStatus::Reported,
        responses: vec![],
    };

    let hash = create_entry(&EntryTypes::ThreatReport(report.clone()))?;

    // Create alert if high severity
    if matches!(severity, Severity::High | Severity::Critical) {
        create_alert_from_threat(&hash, &report)?;
    }

    // Notify security team
    notify_security_team(&hash, &report)?;

    // Reputation reward for valid report (later, after confirmation)

    Ok(hash)
}

/// Run automated threat detection cycle
#[hdk_extern]
pub fn run_detection_cycle() -> ExternResult<DetectionResult> {
    let mut threats_found = vec![];

    // Run all detection algorithms
    let byzantine_threats = monitor_all_fl_rounds()?;
    threats_found.extend(byzantine_threats);

    let sybil_threats = detect_sybil_patterns()?;
    threats_found.extend(sybil_threats);

    let reputation_threats = detect_reputation_manipulation()?;
    threats_found.extend(reputation_threats);

    let network_threats = analyze_network_health()?;
    threats_found.extend(network_threats);

    // Store all threats
    let mut threat_hashes = vec![];
    for threat in &threats_found {
        let hash = create_entry(&EntryTypes::ThreatReport(threat.clone()))?;
        threat_hashes.push(hash.clone());

        if matches!(threat.severity, Severity::High | Severity::Critical) {
            create_alert_from_threat(&hash, threat)?;
        }
    }

    // Update detection metrics
    update_detection_metrics(&threats_found)?;

    Ok(DetectionResult {
        threats_found: threats_found.len(),
        threat_hashes,
        cycle_timestamp: sys_time()?,
    })
}

/// Investigate a reported threat
#[hdk_extern]
pub fn investigate_threat(threat_hash: ActionHash) -> ExternResult<InvestigationResult> {
    let investigator = agent_info()?.agent_latest_pubkey;

    // Verify investigator has authority
    verify_security_authority(&investigator)?;

    let mut threat = get_threat_report(&threat_hash)?;

    // Update status
    threat.status = ThreatStatus::UnderInvestigation;
    update_threat_report(&threat_hash, &threat)?;

    // Gather additional evidence
    let additional_evidence = gather_evidence(&threat)?;

    // Cross-reference with other reports
    let related = find_related_threats(&threat)?;

    // Analyze suspects
    let suspect_analysis: Vec<_> = threat.suspects
        .iter()
        .map(|s| analyze_suspect(s))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(InvestigationResult {
        threat: threat_hash,
        additional_evidence,
        related_threats: related,
        suspect_analysis,
        recommendation: generate_recommendation(&threat, &suspect_analysis),
    })
}
```

### 2. Response Coordinator Zome

```rust
// response_coordinator/src/lib.rs

/// Initiate response to confirmed threat
#[hdk_extern]
pub fn initiate_response(input: InitiateResponseInput) -> ExternResult<ActionHash> {
    let responder = agent_info()?.agent_latest_pubkey;

    // Verify authority
    verify_response_authority(&responder, &input.response_type)?;

    let mut threat = get_threat_report(&input.threat)?;

    // Verify threat is confirmed
    if threat.status != ThreatStatus::Confirmed {
        return Err(WasmError::Guest("Threat must be confirmed before response".into()));
    }

    // Execute response
    let result = execute_response(&input.response_type, &threat)?;

    // Record response
    threat.responses.push(ResponseAction {
        action_type: input.response_type.clone(),
        taken_at: sys_time()?,
        taken_by: responder.clone(),
        result,
    });

    update_threat_report(&input.threat, &threat)?;

    // Create audit record
    let audit = ResponseAudit {
        threat: input.threat.clone(),
        response: input.response_type.clone(),
        responder,
        timestamp: sys_time()?,
        evidence: input.justification,
    };

    create_entry(&EntryTypes::ResponseAudit(audit))
}

/// Execute a response action
fn execute_response(response: &ResponseType, threat: &ThreatReport) -> ExternResult<ActionResult> {
    match response {
        ResponseType::Flag { duration } => {
            for suspect in &threat.suspects {
                bridge_call::<()>(
                    "bridge",
                    "flag_agent",
                    FlagInput {
                        agent: suspect.clone(),
                        reason: format!("Security threat: {}", threat.report_id),
                        duration: *duration,
                    },
                )?;
            }
            Ok(ActionResult::Success)
        }

        ResponseType::ReduceWeight { factor } => {
            for suspect in &threat.suspects {
                bridge_call::<()>(
                    "bridge",
                    "apply_weight_modifier",
                    WeightModifier {
                        agent: suspect.clone(),
                        factor: *factor,
                        reason: format!("Security response: {}", threat.report_id),
                    },
                )?;
            }
            Ok(ActionResult::Success)
        }

        ResponseType::SuspendFromHapp { happ } => {
            for suspect in &threat.suspects {
                bridge_call::<()>(
                    happ,
                    "suspend_agent",
                    SuspendInput {
                        agent: suspect.clone(),
                        reason: format!("Security suspension: {}", threat.report_id),
                    },
                )?;
            }
            Ok(ActionResult::Success)
        }

        ResponseType::AlertNetwork => {
            create_network_alert(&threat)?;
            Ok(ActionResult::Success)
        }

        ResponseType::DisputeReferral { dispute: _ } => {
            // Create Arbiter dispute
            for suspect in &threat.suspects {
                bridge_call::<ActionHash>(
                    "arbiter",
                    "file_dispute",
                    DisputeInput {
                        source_happ: "sentinel".to_string(),
                        source_reference: threat.report_id.parse()?,
                        respondent: suspect.clone(),
                        dispute_type: DisputeType::SecurityViolation,
                        ..Default::default()
                    },
                )?;
            }
            Ok(ActionResult::Success)
        }

        ResponseType::GovernanceReferral { proposal: _ } => {
            // Create Agora proposal for governance action
            let proposal = bridge_call::<ActionHash>(
                "agora",
                "create_proposal",
                CreateProposalInput {
                    proposal_type: ProposalType::Membership {
                        action: MembershipAction::Remove(threat.suspects[0].clone()),
                    },
                    title: format!("Security response: {}", threat.report_id),
                    description: format!("Remove agent due to security threat: {:?}", threat.threat_type),
                    ..Default::default()
                },
            )?;
            Ok(ActionResult::Success)
        }

        _ => Ok(ActionResult::NotImplemented),
    }
}

/// Escalate threat to higher authority
#[hdk_extern]
pub fn escalate_threat(input: EscalateInput) -> ExternResult<()> {
    let mut threat = get_threat_report(&input.threat)?;

    // Increase severity
    threat.severity = escalate_severity(&threat.severity);
    threat.status = ThreatStatus::Escalated;

    update_threat_report(&input.threat, &threat)?;

    // Notify governance
    bridge_call::<()>(
        "agora",
        "create_emergency_notification",
        EmergencyNotification {
            source: "sentinel".to_string(),
            threat: input.threat,
            severity: threat.severity.clone(),
        },
    )?;

    Ok(())
}
```

### 3. Intel Sharing Zome

```rust
// intel_sharing/src/lib.rs

/// Share threat intelligence
#[hdk_extern]
pub fn share_intel(input: ShareIntelInput) -> ExternResult<ActionHash> {
    let sharer = agent_info()?.agent_latest_pubkey;

    // Create intel report
    let intel = ThreatIntel {
        intel_id: generate_intel_id(),
        threat_type: input.threat_type,
        indicators: input.indicators,
        severity: input.severity,
        context: input.context,
        mitigation: input.mitigation,
        shared_by: sharer.clone(),
        shared_at: sys_time()?,
        visibility: input.visibility,
        verifications: vec![],
    };

    let hash = create_entry(&EntryTypes::ThreatIntel(intel))?;

    // Share with network based on visibility
    match input.visibility {
        IntelVisibility::Public => {
            broadcast_intel(&hash)?;
        }
        IntelVisibility::Restricted { recipients } => {
            for recipient in recipients {
                send_intel_to_agent(&recipient, &hash)?;
            }
        }
        IntelVisibility::TrustedOnly { min_trust } => {
            let trusted = get_trusted_agents(min_trust)?;
            for agent in trusted {
                send_intel_to_agent(&agent, &hash)?;
            }
        }
    }

    Ok(hash)
}

/// Verify shared intelligence
#[hdk_extern]
pub fn verify_intel(intel_hash: ActionHash) -> ExternResult<()> {
    let verifier = agent_info()?.agent_latest_pubkey;

    let mut intel = get_threat_intel(&intel_hash)?;

    // Add verification
    intel.verifications.push(IntelVerification {
        verifier: verifier.clone(),
        verified_at: sys_time()?,
        confirmed: true,
        notes: None,
    });

    update_threat_intel(&intel_hash, &intel)?;

    // Reputation boost for original sharer if verified
    if intel.verifications.len() >= 3 {
        bridge_call::<()>(
            "bridge",
            "apply_reputation_impact",
            ReputationImpact {
                agent: intel.shared_by.clone(),
                happ: "sentinel".to_string(),
                impact: 0.02,
                reason: "Verified threat intel".to_string(),
            },
        )?;
    }

    Ok(())
}

/// Get relevant threat intelligence
#[hdk_extern]
pub fn get_relevant_intel(query: IntelQuery) -> ExternResult<Vec<ThreatIntel>> {
    let requestor = agent_info()?.agent_latest_pubkey;

    // Get requestor's trust level
    let trust = get_reputation(&requestor)?;

    // Filter by visibility
    let all_intel = get_recent_intel(&query)?;

    let accessible: Vec<_> = all_intel
        .into_iter()
        .filter(|intel| {
            match &intel.visibility {
                IntelVisibility::Public => true,
                IntelVisibility::Restricted { recipients } => {
                    recipients.contains(&requestor)
                }
                IntelVisibility::TrustedOnly { min_trust } => {
                    trust >= *min_trust
                }
            }
        })
        .collect();

    Ok(accessible)
}
```

### 4. Bounty Manager Zome

```rust
// bounty_manager/src/lib.rs

/// Create a bug bounty program
#[hdk_extern]
pub fn create_bounty_program(input: CreateBountyInput) -> ExternResult<ActionHash> {
    let creator = agent_info()?.agent_latest_pubkey;

    // Verify authority (governance)
    verify_bounty_authority(&creator)?;

    let bounty = BugBounty {
        bounty_id: generate_bounty_id(),
        scope: input.scope,
        rewards: input.rewards,
        rules: input.rules,
        active_from: input.active_from.unwrap_or(sys_time()?),
        active_until: input.active_until,
        reward_pool: input.reward_pool,
        claimed: 0,
    };

    create_entry(&EntryTypes::BugBounty(bounty))
}

/// Submit a vulnerability
#[hdk_extern]
pub fn submit_vulnerability(input: SubmitVulnerabilityInput) -> ExternResult<ActionHash> {
    let submitter = agent_info()?.agent_latest_pubkey;

    // Verify bounty is active
    let bounty = get_bounty(&input.bounty)?;
    verify_bounty_active(&bounty)?;

    // Encrypt details for security team
    let encrypted = encrypt_for_security_team(&input.details)?;

    let submission = BountySubmission {
        bounty: input.bounty,
        submitter: submitter.clone(),
        details_encrypted: encrypted,
        severity_claimed: input.severity,
        proof_of_concept: input.poc,
        submitted_at: sys_time()?,
        status: BountyStatus::Submitted,
        reward_paid: None,
    };

    let hash = create_entry(&EntryTypes::BountySubmission(submission))?;

    // Notify security team
    notify_security_team_bounty(&hash)?;

    Ok(hash)
}

/// Review and pay bounty
#[hdk_extern]
pub fn process_bounty(input: ProcessBountyInput) -> ExternResult<()> {
    let reviewer = agent_info()?.agent_latest_pubkey;

    // Verify reviewer authority
    verify_security_authority(&reviewer)?;

    let mut submission = get_bounty_submission(&input.submission)?;

    if input.valid {
        // Confirm and pay
        let confirmed_severity = input.confirmed_severity.unwrap_or(submission.severity_claimed.clone());

        let bounty = get_bounty(&submission.bounty)?;
        let reward = bounty.rewards
            .get(&severity_to_string(&confirmed_severity))
            .copied()
            .unwrap_or(0);

        submission.status = BountyStatus::Confirmed { severity: confirmed_severity };
        submission.reward_paid = Some(reward);

        // Process payment
        process_bounty_payment(&submission.submitter, reward)?;

        // Update bounty pool
        update_bounty_claimed(&submission.bounty, reward)?;

        // Reputation boost
        bridge_call::<()>(
            "bridge",
            "apply_reputation_impact",
            ReputationImpact {
                agent: submission.submitter.clone(),
                happ: "sentinel".to_string(),
                impact: 0.05,
                reason: "Valid bug bounty submission".to_string(),
            },
        )?;
    } else {
        submission.status = if input.duplicate {
            BountyStatus::Duplicate
        } else {
            BountyStatus::Invalid
        };
    }

    update_bounty_submission(&input.submission, &submission)?;

    Ok(())
}
```

---

## Success Metrics

| Metric | 1 Year | 3 Years |
|--------|--------|---------|
| Threats detected | 100 | 5,000 |
| True positive rate | 90% | 95% |
| False positive rate | <10% | <5% |
| Mean time to detect | <1 hour | <15 min |
| Mean time to respond | <24 hours | <4 hours |
| Bounties paid | $10K | $100K |
| Intel reports shared | 500 | 10,000 |

---

*"The best defense is not a wall, but a thousand watchful eyes."*
