# Edge Cases

*The Weird Situations Where Design Gets Tested*

---

> "The devil is in the details."
> — Proverb

> "The wisdom is in the edges."
> — Us

---

## Introduction

Edge cases are where systems reveal their true nature. They are the boundary conditions, the unexpected situations, the "what if" scenarios that stress-test design.

This document catalogs the edge cases we've considered and how we handle them. It is both a design reference and an invitation: if you find an edge case we haven't considered, please tell us.

---

## Part I: Prediction Edge Cases

### E1: The Unknowable Question

**Scenario:** A market is created for a question that turns out to be fundamentally unknowable.

**Examples:**
- "What is the true value of pi?" (infinite, never fully known)
- "Did consciousness exist before the Big Bang?"
- "What would have happened if Hitler had won WWII?"

**Handling:**
```rust
pub fn handle_unknowable(market: &Market) -> UnknowableResolution {
    // Detection: oracle votes are highly dispersed with "unknowable" votes
    if market.oracle_votes.unknowable_percentage() > 0.5 {
        // Refund stakes
        refund_all_stakes(market);

        // Record as unknowable for learning
        mark_as_unknowable(market);

        // Creator may lose creation fee (depends on preventability)
        if was_foreseeable_unknowable(market) {
            penalize_creator(market);
        }

        UnknowableResolution::Voided {
            reason: "Question determined to be fundamentally unknowable",
            refunds: true,
        }
    }
}
```

**Design principle:** Some questions should not become markets. The system should learn to recognize them.

---

### E2: The Self-Fulfilling Prophecy

**Scenario:** The prediction itself causes the outcome.

**Examples:**
- "Will this company's stock price drop?" → Market reveals pessimism → Stock drops
- "Will the candidate drop out?" → Market shows low chances → Donors flee → Candidate drops out

**Handling:**
```typescript
interface SelfFulfillingAnalysis {
  // Detect potential self-influence
  detectSelfInfluence(market: Market): SelfInfluenceRisk {
    const factors = {
      marketVisibility: market.participantCount > 1000,
      stakeholderParticipation: market.hasStakeholderPredictions,
      outcomeControllability: market.subjectCanAffectOutcome,
      informationAsymmetry: market.predictorsHaveInsiderAccess,
    };

    const risk = calculateRisk(factors);

    if (risk > HIGH_THRESHOLD) {
      return SelfInfluenceRisk.High;
    }
    // ...
  }

  // Mitigation options
  mitigate(risk: SelfInfluenceRisk): Mitigation {
    switch (risk) {
      case SelfInfluenceRisk.High:
        return {
          delayVisibility: true,       // Don't show predictions until resolved
          excludeStakeholders: true,   // Those who can affect outcome can't participate
          adjustScoring: true,         // Different scoring for self-influenced markets
        };
      // ...
    }
  }
}
```

**Design principle:** Acknowledge that prediction changes reality. Design for it.

---

### E3: The Moving Target

**Scenario:** The question's meaning changes over time.

**Examples:**
- "Will Twitter reach 1 billion users?" → Twitter becomes X
- "Will the EU adopt a digital currency?" → Definition of "digital currency" evolves
- "Will AI pass the Turing test?" → The Turing test itself is questioned

**Handling:**
```rust
pub struct MovingTargetProtocol {
    // Original definition
    pub original_definition: Definition,

    // Amendment process
    pub amendment_history: Vec<Amendment>,

    // Stake-weighted approval for amendments
    pub amendment_threshold: f64,

    // Final interpretation authority
    pub interpretation_authority: InterpretationAuthority,
}

pub fn handle_moving_target(market: &Market, change: &DefinitionChange) -> MovingTargetResolution {
    // Assess materiality of change
    let materiality = assess_materiality(change, market);

    match materiality {
        Materiality::Trivial => {
            // Minor change: proceed with updated understanding
            update_definition(market, change);
            continue_market(market)
        }
        Materiality::Significant => {
            // Significant change: stake-weighted vote on interpretation
            let vote = conduct_interpretation_vote(market, change);
            apply_interpretation(market, vote)
        }
        Materiality::Fundamental => {
            // Fundamental change: void and refund
            void_market(market, "Fundamental redefinition of question");
            refund_stakes(market)
        }
    }
}
```

**Design principle:** Questions exist in time. Build in interpretation mechanisms.

---

### E4: The Confidence Paradox

**Scenario:** Extreme confidence that seems paradoxically wrong.

**Examples:**
- Someone predicts 99.99% on a complex question → Likely miscalibrated
- Everyone predicts 50% → No information being aggregated
- Prediction exactly matches base rate → Just guessing?

**Handling:**
```typescript
interface ConfidenceAnalysis {
  // Detect suspicious confidence patterns
  analyzeConfidence(predictions: Prediction[]): ConfidenceReport {
    const extremeHigh = predictions.filter(p => p.confidence > 0.99);
    const extremeLow = predictions.filter(p => p.confidence < 0.01);
    const exactMiddle = predictions.filter(p => p.confidence === 0.5);

    // Flag patterns for review
    const flags: ConfidenceFlag[] = [];

    if (extremeHigh.length > predictions.length * 0.1) {
      flags.push({
        type: "extreme_overconfidence",
        description: ">10% of predictions at 99%+",
        recommendation: "Review for miscalibration or manipulation",
      });
    }

    if (exactMiddle.length > predictions.length * 0.3) {
      flags.push({
        type: "excessive_uncertainty",
        description: ">30% of predictions exactly 50%",
        recommendation: "Question may be too hard or poorly defined",
      });
    }

    return { predictions, flags };
  }

  // Adjust scoring for extreme confidence
  adjustScoring(prediction: Prediction): ScoringAdjustment {
    // Extreme confidence carries extreme risk
    if (prediction.confidence > 0.99 || prediction.confidence < 0.01) {
      return {
        rewardMultiplier: 3.0,  // Big reward if right
        penaltyMultiplier: 3.0, // Big penalty if wrong
        warning: "Extreme confidence detected. Scoring adjusted.",
      };
    }
    return ScoringAdjustment.default();
  }
}
```

**Design principle:** Confidence is information. Analyze it.

---

### E5: The Quantum Prediction

**Scenario:** The question has a superposition-like quality where observation affects outcome.

**Examples:**
- "Will this secret project succeed?" → Revealing the prediction reveals the project
- "Will the negotiations succeed if I tell them my expectations?"
- Schrödinger's startup: alive and dead until investors look

**Handling:**
```rust
pub struct QuantumMarket {
    // Prediction visible only to predictor until resolution
    pub visibility: Visibility::Private,

    // Aggregate visible only after threshold
    pub aggregate_threshold: f64,

    // Resolution can be delayed until observation-safe
    pub observation_safe_delay: Option<Duration>,
}

pub fn handle_quantum_market(market: &QuantumMarket) -> QuantumResolution {
    // Keep predictions private until resolution
    if market.visibility == Visibility::Private {
        // Predictions are committed but not revealed
        let commitments = get_commitments(market);

        // Resolve privately, then reveal
        let resolution = resolve_privately(market);

        // Reveal predictions and resolution together
        reveal_all(commitments, resolution)
    }
}
```

**Design principle:** Some predictions are destroyed by observation. Protect them.

---

## Part II: Resolution Edge Cases

### E6: The Split Outcome

**Scenario:** The outcome is genuinely ambiguous or partial.

**Examples:**
- "Will the bill pass?" → Passes but with amendments that gut it
- "Will the project ship on time?" → Ships 80% of features on time
- "Will the company succeed?" → Survives but as a shadow of its vision

**Handling:**
```rust
pub enum OutcomeType {
    Binary(bool),
    Partial(f64),           // 0.0 to 1.0
    Multi(Vec<(String, f64)>), // Multiple outcomes with weights
    Narrative(String),      // Too complex for numbers
}

pub fn handle_split_outcome(market: &Market, outcome: OutcomeType) -> SplitResolution {
    match outcome {
        OutcomeType::Partial(fraction) => {
            // Proportional payout based on how "right" each side was
            let yes_payout = calculate_partial_payout(&market.yes_predictions, fraction);
            let no_payout = calculate_partial_payout(&market.no_predictions, 1.0 - fraction);

            SplitResolution::Partial { yes_payout, no_payout }
        }
        OutcomeType::Narrative(description) => {
            // Qualitative resolution requires deliberation
            initiate_deliberative_resolution(market, description)
        }
        // ...
    }
}
```

**Design principle:** Reality is often fuzzy. Allow for fuzzy resolutions.

---

### E7: The Oracle Paradox

**Scenario:** All oracles are compromised, conflicted, or unavailable.

**Examples:**
- The question is about the oracles themselves
- All qualified oracles have bets in the market
- The resolution requires expertise that no longer exists

**Handling:**
```typescript
interface OracleFailsafe {
  // Fallback mechanisms
  fallbacks: [
    {
      trigger: "insufficient_eligible_oracles",
      mechanism: "community_deliberation",
      threshold: 0.67,
    },
    {
      trigger: "all_oracles_conflicted",
      mechanism: "external_arbitration",
      arbitrator: "pre-designated neutral party",
    },
    {
      trigger: "expertise_unavailable",
      mechanism: "best_effort_with_disclosure",
      discount: 0.5, // Stakes returned at 50%
    },
    {
      trigger: "complete_failure",
      mechanism: "void_and_refund",
      compensation: "creation_fee_returned",
    },
  ];
}
```

**Design principle:** Oracle systems fail. Plan for it.

---

### E8: The Time Paradox

**Scenario:** The resolution deadline passes without resolution.

**Examples:**
- External events delay verification
- Required information is never released
- The thing that was supposed to happen... just doesn't

**Handling:**
```rust
pub struct TimeParadoxProtocol {
    // Grace period after deadline
    pub grace_period: Duration,

    // Extension mechanisms
    pub extension_process: ExtensionProcess,

    // Ultimate timeout
    pub ultimate_timeout: Duration,

    // Timeout resolution
    pub timeout_resolution: TimeoutResolution,
}

pub enum TimeoutResolution {
    // Void: refund all stakes
    Void { refund_percentage: f64 },

    // Defer: extend until resolvable
    Defer { max_extension: Duration },

    // Default: resolve to default outcome
    Default { outcome: Outcome, reasoning: String },

    // Proportional: split based on current probabilities
    Proportional,
}
```

**Design principle:** Time runs out. Have a plan.

---

### E9: The Resurrection

**Scenario:** A resolved market turns out to have been resolved incorrectly.

**Examples:**
- New evidence emerges after resolution
- The oracle lied and was later caught
- A "final" result is overturned on appeal

**Handling:**
```rust
pub struct ResurrectionProtocol {
    // Window for challenges
    pub challenge_window: Duration,

    // Evidence requirements
    pub evidence_threshold: EvidenceThreshold,

    // Reversal process
    pub reversal_process: ReversalProcess,

    // Finality after challenge window
    pub finality: Finality,
}

pub fn handle_resurrection(
    market: &Market,
    new_evidence: Evidence,
) -> ResurrectionResult {
    // Assess evidence quality
    let evidence_quality = assess_evidence(&new_evidence);

    if evidence_quality > REVERSAL_THRESHOLD {
        // Initiate reversal process
        let reversal = initiate_reversal(market, new_evidence);

        // Notify affected parties
        notify_affected(market, reversal);

        // Execute reversal if approved
        if reversal.approved {
            execute_reversal(market, reversal);
            compensate_wronged_parties(market, reversal);
        }

        ResurrectionResult::Reversed(reversal)
    } else {
        ResurrectionResult::Finality("Evidence insufficient for reversal")
    }
}
```

**Design principle:** Finality must be balanced with correctability.

---

## Part III: Participation Edge Cases

### E10: The Ghost Predictor

**Scenario:** A predictor disappears before resolution.

**Examples:**
- Account abandoned
- User deceased
- Private keys lost

**Handling:**
```rust
pub struct GhostPredictorProtocol {
    // Inactivity threshold
    pub inactivity_threshold: Duration,

    // Warning process
    pub warning_process: WarningProcess,

    // Stakes handling
    pub stakes_handling: StakesHandling,

    // Reputation handling
    pub reputation_handling: ReputationHandling,
}

pub enum StakesHandling {
    // Keep in escrow for recovery period
    Escrow { period: Duration },

    // Donate to treasury after period
    TreasuryAfter { period: Duration },

    // Distribute to designated beneficiary
    Beneficiary { agent: Option<AgentPubKey> },
}

pub fn handle_ghost_predictor(agent: &AgentPubKey) -> GhostResolution {
    // Check for designated successor
    if let Some(successor) = get_successor(agent) {
        transfer_to_successor(agent, successor)
    } else {
        // Escrow and wait
        escrow_assets(agent, ESCROW_PERIOD);

        // After period, handle according to protocol
        schedule_final_disposition(agent, ESCROW_PERIOD)
    }
}
```

**Design principle:** People disappear. Handle their legacy gracefully.

---

### E11: The Whale

**Scenario:** A single participant has outsized influence.

**Examples:**
- One predictor has 50% of the stake
- A wealthy participant manipulates prices
- An early participant has accumulated vast reputation

**Handling:**
```typescript
interface WhaleProtection {
  // Detection
  detectWhale(market: Market): WhaleAnalysis {
    const concentrationMetrics = {
      topParticipantShare: calculateTopShare(market),
      herfindahlIndex: calculateHHI(market),
      minParticipantsFor50Percent: calculateMinFor50(market),
    };

    return {
      isConcentrated: concentrationMetrics.topParticipantShare > 0.3,
      concentrationLevel: categorize(concentrationMetrics),
      recommendations: generateRecommendations(concentrationMetrics),
    };
  }

  // Mitigation
  mitigateWhale(market: Market, whale: Agent): Mitigation {
    return {
      // Cap influence
      maxInfluence: market.totalStake * 0.2, // Max 20% influence

      // Diminishing returns on large stakes
      stakeWeighting: (stake) => Math.sqrt(stake),

      // Public visibility of concentration
      transparencyRequirements: true,

      // Dynamic fees for very large positions
      dynamicFees: (stake) => baseFee * (1 + stake / threshold),
    };
  }
}
```

**Design principle:** Wealth should not determine truth. Limit plutocratic influence.

---

### E12: The Bot

**Scenario:** Automated participants dominate the market.

**Examples:**
- High-frequency prediction algorithms
- AI agents making predictions
- Bot networks gaming the system

**Handling:**
```rust
pub struct BotPolicy {
    // Disclosure requirements
    pub disclosure: DisclosureRequirement,

    // Participation limits
    pub limits: BotLimits,

    // Differentiated treatment
    pub treatment: BotTreatment,
}

pub enum BotTreatment {
    // Full participation with disclosure
    FullParticipation { disclosure_required: true },

    // Limited participation
    Limited {
        max_markets: u32,
        max_stake_percentage: f64,
        cooldown_between_predictions: Duration,
    },

    // Separate bot leagues
    Separate {
        bot_markets: Vec<MarketHash>,
        human_markets: Vec<MarketHash>,
    },

    // Prohibited
    Prohibited,
}

pub fn handle_bot_detection(agent: &AgentPubKey) -> BotResolution {
    let bot_probability = detect_bot_behavior(agent);

    if bot_probability > BOT_THRESHOLD {
        if is_disclosed_bot(agent) {
            // Known bot: apply limits
            apply_bot_limits(agent)
        } else {
            // Undisclosed bot: investigate
            initiate_bot_investigation(agent)
        }
    }
}
```

**Design principle:** Bots are inevitable. Integrate them thoughtfully.

---

### E13: The Insider

**Scenario:** Someone with privileged information participates.

**Examples:**
- Company employee betting on earnings
- Politician betting on policy
- Researcher betting on their own results

**Handling:**
```typescript
interface InsiderProtocol {
  // Detection mechanisms
  detection: {
    unusualAccuracy: "track accuracy by relationship to subject",
    timingAnalysis: "compare prediction timing to information release",
    declarativeDisclosure: "require conflict of interest disclosure",
  };

  // Handling options
  handling: {
    // Option 1: Exclude
    exclude: {
      scope: "any market where participant has material non-public info",
      enforcement: "self-declaration + detection",
    },

    // Option 2: Delayed participation
    delayed: {
      windowBeforeResolution: "24 hours",
      preventLastMinuteInsider: true,
    },

    // Option 3: Separate track
    separateTrack: {
      insiderMarkets: "markets specifically for those with inside info",
      aggregateSeparately: true,
      discloseInsiderAggregate: true,
    },

    // Option 4: Embrace with rules
    embraceWithRules: {
      disclosure: "mandatory disclosure of relationship",
      limitedStake: "can participate but with capped stake",
      informationContribution: "insider info is valuable - use it",
    };
  };
}
```

**Design principle:** Insider information is valuable but creates unfairness. Balance carefully.

---

## Part IV: Market Edge Cases

### E14: The Empty Market

**Scenario:** A market has no participants.

**Examples:**
- Obscure question nobody cares about
- Question too hard to understand
- Market created but never promoted

**Handling:**
```rust
pub struct EmptyMarketProtocol {
    // Grace period for participation
    pub grace_period: Duration,

    // Minimum viability threshold
    pub minimum_participants: u32,
    pub minimum_stake: u64,

    // Handling if empty
    pub empty_handling: EmptyHandling,
}

pub enum EmptyHandling {
    // Extend: give more time
    Extend { max_extensions: u32, extension_duration: Duration },

    // Void: cancel market, refund creator
    Void { refund_creation_fee: bool },

    // Subsidize: add liquidity from treasury
    Subsidize { max_subsidy: u64 },

    // Archive: keep for future reference
    Archive { reason: String },
}
```

**Design principle:** Not all markets attract interest. Fail gracefully.

---

### E15: The Duplicate Market

**Scenario:** Multiple markets exist for the same question.

**Examples:**
- Same question asked with different wording
- Overlapping resolution criteria
- Intentional fragmentation to confuse

**Handling:**
```typescript
interface DuplicateProtocol {
  // Detection
  detectDuplicate(newMarket: Market, existingMarkets: Market[]): DuplicateAnalysis {
    return existingMarkets
      .map(existing => ({
        market: existing,
        similarity: calculateSimilarity(newMarket, existing),
        overlap: calculateOverlap(newMarket, existing),
      }))
      .filter(d => d.similarity > DUPLICATE_THRESHOLD);
  }

  // Resolution
  resolveDuplicate(duplicate: DuplicateAnalysis): DuplicateResolution {
    if (duplicate.similarity > 0.95) {
      // Near-identical: merge or reject
      return DuplicateResolution.Merge(duplicate.market);
    } else if (duplicate.similarity > 0.7) {
      // Similar: link and warn
      return DuplicateResolution.LinkWithWarning(duplicate.market);
    } else {
      // Sufficiently different: allow
      return DuplicateResolution.Allow;
    }
  }
}
```

**Design principle:** Fragmentation destroys liquidity. Consolidate wisely.

---

### E16: The Paradox Market

**Scenario:** The market contains a logical paradox.

**Examples:**
- "Will this prediction be wrong?"
- "Will nobody predict Yes on this market?"
- Markets that reference their own outcome

**Handling:**
```rust
pub fn detect_paradox(market: &Market) -> Option<ParadoxType> {
    // Self-reference detection
    if market.resolution_criteria.references(&market.id) {
        return Some(ParadoxType::SelfReference);
    }

    // Logical contradiction detection
    if is_logically_contradictory(&market.question) {
        return Some(ParadoxType::Contradiction);
    }

    // Game-theoretic paradox detection
    if creates_unstable_equilibrium(&market) {
        return Some(ParadoxType::GameTheoretic);
    }

    None
}

pub fn handle_paradox(market: &Market, paradox: ParadoxType) -> ParadoxResolution {
    // Paradoxes cannot be created
    // If detected post-creation, void
    void_market(market, format!("Paradox detected: {:?}", paradox));
    refund_stakes(market);

    ParadoxResolution::Voided
}
```

**Design principle:** Paradoxes break systems. Detect and prevent them.

---

## Part V: System Edge Cases

### E17: The Fork

**Scenario:** The community splits into irreconcilable factions.

**Examples:**
- Fundamental disagreement on resolution
- Protocol changes that some reject
- Values divergence

**Handling:**
```rust
pub struct ForkProtocol {
    // Fork triggers
    pub triggers: Vec<ForkTrigger>,

    // Fork process
    pub process: ForkProcess,

    // Asset division
    pub asset_division: AssetDivision,

    // History handling
    pub history_handling: HistoryHandling,
}

pub struct ForkProcess {
    // Notification period
    pub notification_period: Duration,

    // Stake commitment
    pub stake_commitment: StakeCommitment,

    // Technical execution
    pub execution: ForkExecution,
}

pub fn execute_fork(fork: &Fork) -> ForkResult {
    // Snapshot state
    let snapshot = snapshot_state();

    // Divide assets proportionally
    let division = divide_assets(snapshot, &fork.participants);

    // Create new chains
    let chain_a = create_chain(snapshot, fork.faction_a);
    let chain_b = create_chain(snapshot, fork.faction_b);

    // Handle ongoing markets
    for market in snapshot.open_markets {
        // Each chain gets a copy; they resolve independently
        clone_market_to_chain(&market, &chain_a);
        clone_market_to_chain(&market, &chain_b);
    }

    ForkResult {
        chains: vec![chain_a, chain_b],
        division,
        history: archive_fork_history(fork),
    }
}
```

**Design principle:** Forks are the ultimate escape valve. Make them orderly.

---

### E18: The Apocalypse

**Scenario:** External events make the entire system irrelevant.

**Examples:**
- Civilizational collapse
- Technology obsolescence
- Legal prohibition in all jurisdictions

**Handling:**
```rust
pub struct ApocalypseProtocol {
    // Detection triggers
    pub triggers: Vec<ApocalypseTrigger>,

    // Graceful shutdown process
    pub shutdown_process: ShutdownProcess,

    // Data preservation
    pub data_preservation: DataPreservation,

    // Asset distribution
    pub final_distribution: FinalDistribution,
}

pub enum ApocalypseTrigger {
    // No activity for extended period
    Inactivity { threshold: Duration },

    // Legal shutdown in all jurisdictions
    GlobalProhibition,

    // Infrastructure collapse
    InfrastructureFailure,

    // Unanimous community decision
    CommunityDecision { threshold: f64 },
}

pub fn handle_apocalypse(trigger: ApocalypseTrigger) -> ApocalypseResolution {
    // Notify all participants
    broadcast_apocalypse_warning(trigger);

    // Void all open markets
    void_all_markets();

    // Distribute remaining assets
    distribute_treasury_to_participants();

    // Archive everything
    archive_to_permanent_storage();

    // Publish lessons learned
    publish_postmortem();

    ApocalypseResolution::GracefulDeath {
        archive_location: get_archive_url(),
        postmortem: get_postmortem_url(),
    }
}
```

**Design principle:** Everything ends. Die well.

---

### E19: The Genesis Edge

**Scenario:** The very first moments of the system have special challenges.

**Examples:**
- No reputation to weight anything
- No historical data for calibration
- No oracles to resolve anything

**Handling:**
```typescript
interface GenesisProtocol {
  // Bootstrap mechanisms
  bootstrap: {
    // Initial reputation seeding
    initialReputation: {
      method: "equal_starting_score",
      score: 50,
      source: "founding_participation",
    };

    // First oracles
    firstOracles: {
      selection: "founding_team",
      duration: "until_matl_scores_meaningful",
      transition: "gradual_handoff_to_matl_selected",
    };

    // Calibration bootstrap
    calibrationBootstrap: {
      source: "external_prediction_history",
      weight: "decreasing_over_time",
    };
  };

  // Special rules for genesis period
  genesisRules: {
    duration: "6 months or 1000 resolved markets",
    reducedStakes: true,
    enhancedMonitoring: true,
    frequentReview: true,
  };
}
```

**Design principle:** Beginnings require special care. Bootstrap deliberately.

---

## Conclusion

Edge cases are not exceptions—they are the places where our design is truly tested. A system that handles edge cases gracefully is a system that can survive.

This document is incomplete by definition. New edge cases will emerge. When they do:

1. **Document** them here
2. **Design** handling mechanisms
3. **Implement** with care
4. **Monitor** for effectiveness
5. **Learn** and update

The goal is not to prevent all edge cases—that is impossible—but to handle them with grace, wisdom, and fairness.

---

*"The edge is the location of transformation."*
*— Terence McKenna*

*May we transform well at every edge.*

---

## How to Report New Edge Cases

If you discover an edge case not documented here:

1. Open an issue with title "Edge Case: [Brief Description]"
2. Describe the scenario
3. Explain why it's problematic
4. Suggest potential handling (optional)
5. Tag with "edge-case"

Your edge case might save the system. Please share.
