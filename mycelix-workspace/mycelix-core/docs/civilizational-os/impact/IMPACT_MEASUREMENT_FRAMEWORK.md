# Mycelix Impact Measurement Framework

## Philosophy: Measuring What Matters

> "Not everything that counts can be counted, and not everything that can be counted counts." — William Bruce Cameron

This framework measures the actual impact of Mycelix on communities and the planet—not vanity metrics, but genuine indicators of flourishing.

---

## Part 1: Multi-Dimensional Impact Model

### 1.1 The Four Domains of Impact

```
                    INDIVIDUAL                          COLLECTIVE
                  ┌─────────────────────────────────────────────────────┐
                  │                                                     │
    INTERIOR      │   WELLBEING          │        CULTURE              │
    (Subjective)  │   • Life satisfaction│        • Trust levels       │
                  │   • Sense of agency  │        • Social cohesion    │
                  │   • Belonging        │        • Shared meaning     │
                  │   • Purpose          │        • Collective wisdom  │
                  │                      │                             │
                  ├──────────────────────┼─────────────────────────────┤
                  │                      │                             │
    EXTERIOR      │   BEHAVIOR           │        SYSTEMS              │
    (Objective)   │   • Participation    │        • Economic flows     │
                  │   • Skill growth     │        • Resource efficiency│
                  │   • Health outcomes  │        • Environmental regen│
                  │   • Time use         │        • Governance quality │
                  │                                                     │
                  └─────────────────────────────────────────────────────┘
```

### 1.2 Core Impact Areas

```typescript
interface ImpactDomains {
  // Domain 1: Individual Wellbeing (UL Quadrant)
  wellbeing: {
    psychological: PsychologicalWellbeing;
    developmental: DevelopmentalGrowth;
    meaning: MeaningAndPurpose;
    autonomy: SenseOfAgency;
  };

  // Domain 2: Cultural Health (LL Quadrant)
  culture: {
    trust: TrustMetrics;
    cohesion: SocialCohesion;
    wisdom: CollectiveWisdom;
    values: SharedValues;
  };

  // Domain 3: Individual Behavior (UR Quadrant)
  behavior: {
    participation: EngagementMetrics;
    growth: SkillDevelopment;
    health: HealthOutcomes;
    timeUse: TimeAllocation;
  };

  // Domain 4: Systems & Structures (LR Quadrant)
  systems: {
    economic: EconomicHealth;
    ecological: EnvironmentalImpact;
    governance: GovernanceQuality;
    infrastructure: InfrastructureResilience;
  };
}
```

---

## Part 2: Wellbeing Indicators (Interior Individual)

### 2.1 Psychological Wellbeing

```typescript
interface PsychologicalWellbeing {
  // Life Satisfaction (WHO-5 adapted)
  lifeSatisfaction: {
    question: "How satisfied are you with life in your community?";
    scale: 1-10;
    frequency: 'monthly';
    method: 'optional-survey';
  };

  // Eudaimonic Wellbeing
  eudaimonia: {
    purpose: number;        // "My actions contribute to something meaningful"
    growth: number;         // "I am learning and growing"
    autonomy: number;       // "I make choices that matter"
    connection: number;     // "I have meaningful relationships"
    competence: number;     // "I can handle challenges"
  };

  // Measurement approach
  measurement: {
    // Privacy-preserving aggregate data
    privacyLevel: 'aggregate-only';
    minGroupSize: 10;       // No individual tracking

    // Voluntary participation
    voluntary: true;
    incentivized: false;    // No rewards for responses (avoid bias)

    // Temporal patterns
    trends: 'quarterly-rolling';
  };
}

// Behavioral proxies (no surveys needed)
interface BehavioralWellbeingProxies {
  // Engagement patterns suggest wellbeing
  engagementQuality: {
    voluntaryParticipation: number;   // % unprompted engagement
    diverseActivities: number;         // Breadth of participation
    reciprocalInteractions: number;    // Give and take balance
  };

  // Communication patterns
  communicationHealth: {
    positivityRatio: number;           // Positive/negative content
    supportGiving: number;             // Offers of help
    gratitudeExpression: number;       // Thanks and appreciation
  };
}
```

### 2.2 Developmental Growth

```typescript
interface DevelopmentalGrowth {
  // Stage progression (from Spiral hApp)
  stageMovement: {
    upwardMovement: number;     // Growth to higher stages
    healthyExpression: number;  // Healthy vs. unhealthy at each stage
    integrationDepth: number;   // Integration of previous stages
  };

  // Shadow integration
  shadowWork: {
    triggersIdentified: number;
    patternsRecognized: number;
    transformationsReported: number;
  };

  // Perspective-taking
  perspectiveGrowth: {
    viewpointsConsidered: number;   // In governance discussions
    empathyExpressions: number;      // Understanding others
    dialecticalThinking: number;     // Holding paradox
  };
}
```

### 2.3 Sense of Agency

```typescript
interface SenseOfAgency {
  // Decision-making power
  decisionInfluence: {
    proposalsCreated: number;
    proposalsSupported: number;
    outcomeInfluence: number;     // Votes on winning side
    feedbackIncorporated: number; // Suggestions adopted
  };

  // Resource control
  resourceAgency: {
    resourcesControlled: number;  // Assets under personal stewardship
    allocationDecisions: number;  // Spending/investing choices
    earningsFromContributions: number;
  };

  // Self-determination
  autonomyIndicators: {
    goalSetting: number;          // Personal goals created
    boundariesSet: number;        // Declined requests
    initiativeTaken: number;      // Proactive actions
  };
}
```

---

## Part 3: Cultural Health Indicators (Interior Collective)

### 3.1 Trust Metrics

```typescript
interface TrustMetrics {
  // Interpersonal trust
  interpersonalTrust: {
    // Behavioral measures (no surveys)
    unsecuredLoans: number;        // Mutual credit extended
    reputationWaived: number;      // Actions without checking reputation
    conflictsResolvedPeacefully: number;
    promisesKept: number;          // Commitments honored
  };

  // Institutional trust
  institutionalTrust: {
    governanceParticipation: number;  // % who vote
    proposalSubmission: number;        // % who propose
    voluntaryCompliance: number;       // Following norms without enforcement
    transparencyViews: number;         // People checking records
  };

  // Trust network analysis
  networkAnalysis: {
    trustDensity: number;           // Connections/possible connections
    bridgingTrust: number;          // Trust across subgroups
    trustReciprocity: number;       // Mutual trust relationships
  };
}
```

### 3.2 Social Cohesion

```typescript
interface SocialCohesion {
  // Connection density
  connectionMetrics: {
    activeRelationships: number;    // Regular interactions
    crossGroupConnections: number;  // Bridging social capital
    supportNetworkSize: number;     // People who help in crisis
    communityEvents: number;        // Shared gatherings
  };

  // Inclusion measures
  inclusionMetrics: {
    diversityInConversations: number; // Varied voices in discussions
    peripheryToCoreMoves: number;     // Newcomers becoming central
    isolatedMembers: number;          // People with few connections
    welcomingBehaviors: number;       // Outreach to newcomers
  };

  // Collective action capacity
  collectiveAction: {
    sharedProjects: number;         // Multi-person initiatives
    emergencyResponse: number;      // Speed/scale of crisis help
    resourcePooling: number;        // Shared resources
    conflictResolution: number;     // Disputes resolved
  };
}
```

### 3.3 Collective Wisdom

```typescript
interface CollectiveWisdom {
  // Decision quality
  decisionQuality: {
    informedVoting: number;         // % who read proposals before voting
    deliberationDepth: number;      // Quality of discussion
    minorityConsideration: number;  // Attention to dissenting views
    longTermThinking: number;       // Decisions considering future
  };

  // Knowledge integration
  knowledgeIntegration: {
    expertiseRecognition: number;   // Right people consulted
    localKnowledgeUse: number;      // Indigenous/place-based wisdom
    learningFromFailure: number;    // Adaptation after mistakes
    knowledgeSharing: number;       // Teaching and documentation
  };

  // Emergent intelligence
  emergentIntelligence: {
    novelSolutions: number;         // Creative problem-solving
    patternRecognition: number;     // Early warning detection
    systemsThinking: number;        // Understanding interconnections
    adaptiveCapacity: number;       // Response to change
  };
}
```

---

## Part 4: Behavioral Indicators (Exterior Individual)

### 4.1 Participation Metrics

```typescript
interface ParticipationMetrics {
  // Engagement breadth
  breadth: {
    activeUsers: number;            // Monthly active
    hAppsUsed: number;              // Average per user
    rolesHeld: number;              // Governance positions
    projectsInvolved: number;
  };

  // Engagement depth
  depth: {
    timeInPlatform: number;         // Hours per week
    contributionQuality: number;    // Peer-rated value
    leadershipActions: number;      // Initiative-taking
    mentorshipProvided: number;
  };

  // Engagement equity
  equity: {
    participationGini: number;      // Distribution of participation
    voiceDistribution: number;      // Who speaks in discussions
    leadershipDiversity: number;    // Demographics of leaders
    barrierReductions: number;      // Accessibility improvements
  };
}
```

### 4.2 Skill Development

```typescript
interface SkillDevelopment {
  // Technical skills
  technicalSkills: {
    digitalLiteracy: number;        // Platform competency
    dataLiteracy: number;           // Understanding analytics
    toolMastery: number;            // Advanced feature use
  };

  // Social skills
  socialSkills: {
    conflictResolution: number;     // Peaceful dispute resolution
    facilitation: number;           // Meeting/process leadership
    collaboration: number;          // Team effectiveness
    communication: number;          // Clear expression
  };

  // Civic skills
  civicSkills: {
    deliberation: number;           // Quality of discourse
    proposalWriting: number;        // Effective proposals
    consensus: number;              // Building agreement
    systemsThinking: number;        // Understanding complexity
  };

  // Tracking method
  tracking: {
    selfAssessment: boolean;
    peerAssessment: boolean;
    behavioralProxies: boolean;    // Actions demonstrate skills
    credentialEarned: boolean;      // Formal recognition
  };
}
```

### 4.3 Health Outcomes

```typescript
interface HealthOutcomes {
  // Self-reported health (voluntary)
  selfReported: {
    overallHealth: number;          // 1-10 scale
    stressLevels: number;
    socialConnection: number;
    purposefulness: number;
  };

  // Behavioral health indicators
  behavioralIndicators: {
    regularActivity: number;        // Consistent engagement
    socialInteractions: number;     // Connection frequency
    learningActivities: number;     // Growth behaviors
    helpGiving: number;             // Prosocial behavior
  };

  // Community health correlation
  communityHealth: {
    mutualAidRequests: number;      // Health-related help
    wellnessResources: number;      // Health info shared
    preventiveActivities: number;   // Wellness programs
  };
}
```

---

## Part 5: Systems Indicators (Exterior Collective)

### 5.1 Economic Health

```typescript
interface EconomicHealth {
  // Circulation metrics
  circulation: {
    velocityOfMoney: number;        // Transactions per credit per period
    localCirculation: number;       // % staying in community
    mutualCreditBalance: number;    // Total system credit
    creditUtilization: number;      // Used vs. available credit
  };

  // Distribution metrics
  distribution: {
    giniCoefficient: number;        // Wealth inequality
    medianWealth: number;           // Middle point
    bottomQuintileWealth: number;   // Poorest 20%
    wealthMobility: number;         // Movement between quintiles
  };

  // Regenerative metrics
  regenerative: {
    valueCreated: number;           // New resources generated
    wasteReduced: number;           // Efficiency gains
    commonsGrowth: number;          // Shared resources increase
    externalDependence: number;     // Reliance on outside economy
  };

  // Resilience metrics
  resilience: {
    diversification: number;        // Economic diversity
    localSufficiency: number;       // Needs met locally
    shockAbsorption: number;        // Response to crises
    redundancy: number;             // Backup systems
  };
}
```

### 5.2 Environmental Impact

```typescript
interface EnvironmentalImpact {
  // Ecological footprint
  footprint: {
    carbonFootprint: number;        // CO2 equivalent per capita
    waterUsage: number;             // Liters per capita
    landUse: number;                // Hectares per capita
    materialConsumption: number;    // Kg per capita
  };

  // Regeneration metrics
  regeneration: {
    carbonSequestration: number;    // CO2 captured
    biodiversityIndex: number;      // Species diversity
    soilHealth: number;             // Soil organic matter
    waterQuality: number;           // Local water health
  };

  // Behavioral change
  behaviorChange: {
    sustainableChoices: number;     // % of eco-friendly decisions
    resourceSharing: number;        // Commons utilization
    circularEconomy: number;        // Reuse/recycle rate
    localSourcing: number;          // % local procurement
  };

  // Planetary boundary alignment
  planetaryBoundaries: {
    climateChange: 'safe' | 'caution' | 'danger';
    biodiversity: 'safe' | 'caution' | 'danger';
    landUse: 'safe' | 'caution' | 'danger';
    freshwater: 'safe' | 'caution' | 'danger';
    nitrogen: 'safe' | 'caution' | 'danger';
    phosphorus: 'safe' | 'caution' | 'danger';
  };
}
```

### 5.3 Governance Quality

```typescript
interface GovernanceQuality {
  // Participation metrics
  participation: {
    voterTurnout: number;           // % who vote
    proposalSubmitters: number;     // % who propose
    discussionParticipants: number; // % who deliberate
    roleHolders: number;            // % in governance roles
  };

  // Process quality
  processQuality: {
    deliberationTime: number;       // Time spent discussing
    amendmentRate: number;          // Proposals improved
    consensusLevel: number;         // Agreement strength
    minorityVoice: number;          // Dissent considered
  };

  // Outcome quality
  outcomeQuality: {
    implementationRate: number;     // Decisions enacted
    goalAchievement: number;        // Objectives met
    satisfactionRate: number;       // Post-decision satisfaction
    reversalRate: number;           // Decisions overturned
  };

  // Legitimacy
  legitimacy: {
    trustInProcess: number;         // Belief in fairness
    voluntaryCompliance: number;    // Following without enforcement
    conflictEscalation: number;     // Disputes reaching formal process
    exitRate: number;               // People leaving community
  };
}
```

### 5.4 Infrastructure Resilience

```typescript
interface InfrastructureResilience {
  // Technical resilience
  technical: {
    uptime: number;                 // System availability
    dataIntegrity: number;          // No data loss
    recoveryTime: number;           // Time to restore
    redundancy: number;             // Backup systems
  };

  // Social resilience
  social: {
    skillDistribution: number;      // Knowledge across members
    successionPlanning: number;     // Leadership backup
    documentationQuality: number;   // Institutional memory
    crossTraining: number;          // Multiple people can do roles
  };

  // Economic resilience
  economic: {
    reserveFunds: number;           // Emergency resources
    diversifiedIncome: number;      // Multiple revenue streams
    debtLevels: number;             // Financial obligations
    liquidityRatio: number;         // Available vs. committed
  };
}
```

---

## Part 6: Measurement Methodology

### 6.1 Data Collection Principles

```typescript
interface DataCollectionPrinciples {
  // Privacy by design
  privacy: {
    minimization: true;             // Collect minimum needed
    aggregation: true;              // Only aggregate reports
    anonymization: true;            // No individual identification
    consent: 'explicit';            // Opt-in only
    retention: 'limited';           // Delete after use
  };

  // Avoid Goodhart's Law
  antiGaming: {
    multipleIndicators: true;       // Triangulate metrics
    qualitativeBalance: true;       // Mix quantitative + qualitative
    behavioralFocus: true;          // Actions over self-report
    rotatingMetrics: true;          // Prevent metric fixation
  };

  // Participatory measurement
  participatory: {
    communityDefined: true;         // Community chooses metrics
    transparentMethods: true;       // Open methodology
    sharedInterpretation: true;     // Collective sensemaking
    adaptiveFramework: true;        // Evolving measurement
  };
}
```

### 6.2 Data Sources

```typescript
interface DataSources {
  // Automatic collection (from platform use)
  automatic: {
    transactions: 'all-economic-activity';
    governance: 'voting-proposal-discussion';
    social: 'connections-messages-reactions';
    temporal: 'timing-patterns-frequency';

    // Privacy preservation
    processing: 'aggregate-only';
    granularity: 'community-level';
  };

  // Voluntary surveys
  surveys: {
    frequency: 'quarterly';
    length: 'max-5-minutes';
    incentive: 'none';              // Avoid response bias
    reminder: 'single';             // No nagging
    required: false;
  };

  // Qualitative methods
  qualitative: {
    stories: 'most-significant-change';
    interviews: 'semi-structured-sample';
    observations: 'participatory-notes';
    artifacts: 'community-created-content';
  };

  // External data
  external: {
    environmental: 'public-datasets';
    economic: 'regional-statistics';
    health: 'public-health-data';
    // Linked at community level, not individual
  };
}
```

### 6.3 Analysis Methods

```typescript
interface AnalysisMethods {
  // Quantitative analysis
  quantitative: {
    descriptive: ['means', 'medians', 'distributions'];
    trends: ['time-series', 'moving-averages', 'seasonality'];
    comparative: ['benchmarking', 'cohort-analysis'];
    predictive: ['regression', 'forecasting'];
  };

  // Qualitative analysis
  qualitative: {
    thematic: 'coding-and-themes';
    narrative: 'story-analysis';
    participatory: 'community-sensemaking';
  };

  // Mixed methods
  mixed: {
    triangulation: 'multiple-sources-same-phenomenon';
    complementarity: 'quant-breadth-qual-depth';
    expansion: 'different-questions-different-methods';
  };

  // AI-assisted (with human oversight)
  aiAssisted: {
    patternDetection: 'anomaly-and-trend';
    textAnalysis: 'sentiment-and-theme';
    prediction: 'early-warning-systems';
    humanOversight: 'required';
  };
}
```

---

## Part 7: Impact Dashboards

### 7.1 Community Dashboard

```typescript
interface CommunityDashboard {
  // Overview metrics
  overview: {
    overallHealth: HealthScore;     // Composite score
    trendDirection: 'improving' | 'stable' | 'declining';
    alertsCount: number;
    celebrationsCount: number;
  };

  // Domain scores
  domains: {
    wellbeing: DomainScore;
    culture: DomainScore;
    behavior: DomainScore;
    systems: DomainScore;
  };

  // Visualizations
  visualizations: {
    radarChart: QuadrantRadar;      // Balance across domains
    trendLines: TimeSeriesChart;    // Change over time
    comparisons: BenchmarkChart;    // Vs. similar communities
    stories: StoryCarousel;         // Qualitative highlights
  };

  // Drill-down capability
  drillDown: {
    byMetric: MetricDetail;
    byTimeframe: TemporalAnalysis;
    bySubgroup: SegmentAnalysis;    // If group is large enough
  };
}
```

### 7.2 Individual Dashboard (Privacy-Preserving)

```typescript
interface IndividualDashboard {
  // Personal growth (private to individual)
  personalGrowth: {
    skillsGained: Skill[];
    contributionImpact: number;
    connectionsMade: number;
    goalsAchieved: number;
  };

  // Community contribution (shareable)
  contribution: {
    proposalsCreated: number;
    votesParticipated: number;
    resourcesShared: number;
    helpProvided: number;
  };

  // Privacy controls
  privacy: {
    viewableBy: 'self' | 'connections' | 'community';
    dataExport: 'full-personal-data';
    dataDelete: 'right-to-erasure';
  };
}
```

### 7.3 Network Dashboard (Multi-Community)

```typescript
interface NetworkDashboard {
  // Aggregate metrics
  aggregate: {
    totalCommunities: number;
    totalMembers: number;
    totalTransactions: number;
    totalProposals: number;
  };

  // Cross-community patterns
  patterns: {
    successPatterns: Pattern[];     // What works
    challengePatterns: Pattern[];   // Common struggles
    innovationSpread: Innovation[]; // Ideas traveling
  };

  // Benchmarking
  benchmarks: {
    bySize: SizeBenchmark;
    byType: TypeBenchmark;
    byAge: MaturityBenchmark;
    byRegion: RegionalBenchmark;
  };

  // Learning network
  learning: {
    bestPractices: Practice[];
    lessonsLearned: Lesson[];
    peerConnections: Connection[];
  };
}
```

---

## Part 8: Reporting & Learning

### 8.1 Report Types

```typescript
interface ReportTypes {
  // Regular reports
  regular: {
    weekly: 'key-metrics-snapshot';
    monthly: 'trend-analysis';
    quarterly: 'comprehensive-review';
    annual: 'impact-report';
  };

  // Event-driven reports
  eventDriven: {
    milestone: 'achievement-celebration';
    alert: 'concern-requiring-attention';
    experiment: 'initiative-evaluation';
    crisis: 'emergency-assessment';
  };

  // Audience-specific
  audienceSpecific: {
    members: 'accessible-summary';
    leadership: 'detailed-analysis';
    funders: 'impact-evidence';
    researchers: 'full-dataset';
  };
}
```

### 8.2 Learning Loops

```typescript
interface LearningLoops {
  // Single-loop learning (are we doing things right?)
  singleLoop: {
    monitor: 'track-metrics';
    compare: 'vs-targets';
    adjust: 'tweak-implementation';
    frequency: 'weekly';
  };

  // Double-loop learning (are we doing the right things?)
  doubleLoop: {
    question: 'are-goals-appropriate';
    reflect: 'why-these-outcomes';
    redesign: 'change-approach';
    frequency: 'quarterly';
  };

  // Triple-loop learning (how do we learn?)
  tripleLoop: {
    examine: 'learning-process-itself';
    evolve: 'measurement-framework';
    transform: 'underlying-assumptions';
    frequency: 'annual';
  };
}
```

### 8.3 Celebration & Course Correction

```typescript
interface ResponseMechanisms {
  // Celebration (reinforce positives)
  celebration: {
    triggers: {
      milestoneReached: true,
      significantImprovement: true,
      innovationSucceeded: true,
      challengeOvercome: true
    };

    responses: {
      publicRecognition: true,
      storySharing: true,
      gratitudeRituals: true,
      learningDocumentation: true
    };
  };

  // Course correction (address concerns)
  courseCorrection: {
    triggers: {
      metricDecline: { threshold: 10, duration: 'weeks' },
      inequityDetected: true,
      participationDrop: { threshold: 20 },
      conflictEscalation: true
    };

    responses: {
      rootCauseAnalysis: true,
      stakeholderDialogue: true,
      experimentProposal: true,
      resourceReallocation: true
    };
  };

  // Early warning system
  earlyWarning: {
    leadingIndicators: [
      'engagement-velocity',
      'sentiment-trends',
      'connection-density',
      'governance-participation'
    ];

    alertThresholds: {
      watch: 'one-deviation';
      concern: 'two-deviations';
      crisis: 'three-deviations';
    };
  };
}
```

---

## Part 9: Implementation Guide

### 9.1 Starting Simple

```typescript
// Phase 1: Essential Metrics (First 6 months)
const essentialMetrics = {
  // Just 5 core metrics
  activeMembers: 'monthly-active-users',
  participation: 'governance-participation-rate',
  economicFlow: 'transaction-volume',
  socialConnection: 'average-connections-per-member',
  satisfaction: 'quarterly-pulse-survey'
};

// Phase 2: Expanded Metrics (6-12 months)
const expandedMetrics = {
  // Add domain-specific metrics
  wellbeing: ['agency-score', 'belonging-score'],
  culture: ['trust-index', 'cohesion-score'],
  systems: ['economic-velocity', 'governance-quality']
};

// Phase 3: Full Framework (12+ months)
const fullFramework = {
  // Complete measurement system
  allDomains: true,
  qualitativeMethods: true,
  learningLoops: true,
  networkComparisons: true
};
```

### 9.2 Community Configuration

```typescript
interface CommunityMeasurementConfig {
  // Community chooses priorities
  priorityDomains: Domain[];

  // Custom indicators
  customIndicators: Indicator[];

  // Measurement preferences
  preferences: {
    surveyFrequency: 'never' | 'quarterly' | 'monthly';
    publicMetrics: Metric[];
    privateMetrics: Metric[];
    benchmarkComparisons: boolean;
  };

  // Reporting preferences
  reporting: {
    dashboardAccess: 'all' | 'members' | 'leaders';
    reportFrequency: 'weekly' | 'monthly' | 'quarterly';
    externalSharing: boolean;
  };
}
```

---

## Part 10: Ethical Considerations

### 10.1 Measurement Ethics

```typescript
interface MeasurementEthics {
  // Do no harm
  doNoHarm: {
    noSurveillance: true,          // Measurement != monitoring
    noControl: true,               // Data for learning, not control
    noPunishment: true,            // No negative consequences from data
    noExploitation: true           // Data not sold or misused
  };

  // Data rights
  dataRights: {
    ownership: 'community';        // Community owns its data
    consent: 'explicit';           // Opt-in only
    access: 'full';                // See all data about you
    correction: 'allowed';         // Fix errors
    deletion: 'supported';         // Right to be forgotten
    portability: 'full';           // Export your data
  };

  // Power dynamics
  powerDynamics: {
    whoMeasures: 'community-controlled';
    whoInterprets: 'participatory';
    whoBenefits: 'community-first';
    whoDecides: 'democratic';
  };
}
```

### 10.2 Avoiding Pitfalls

```typescript
interface MeasurementPitfalls {
  // Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure"
  antiGoodhart: {
    multipleIndicators: true,      // Triangulate
    rotateMetrics: true,           // Change focal metrics
    qualitativeBalance: true,      // Stories + numbers
    processOverOutcome: true       // How we act matters
  };

  // McNamara Fallacy: Measure what's easy, ignore what's important
  antiMcNamara: {
    difficultMetrics: true,        // Measure hard things
    qualitativeData: true,         // Include stories
    localKnowledge: true,          // What community knows
    longTermView: true             // Slow variables
  };

  // Surveillance risk
  antiSurveillance: {
    aggregateOnly: true,           // No individual tracking
    powerDistribution: true,       // Data doesn't concentrate power
    optionalParticipation: true,   // No measurement required
    communityControl: true         // Community decides what's measured
  };
}
```

---

## Appendix: Metric Catalog

### Quick Reference: 50 Core Metrics

| Domain | Metric | Source | Frequency |
|--------|--------|--------|-----------|
| **Wellbeing** | | | |
| | Life satisfaction | Survey | Quarterly |
| | Sense of agency | Behavioral | Monthly |
| | Purpose score | Survey | Quarterly |
| | Social support | Behavioral | Monthly |
| | Growth rate | Spiral hApp | Monthly |
| **Culture** | | | |
| | Trust index | Behavioral | Monthly |
| | Social cohesion | Network analysis | Monthly |
| | Collective efficacy | Behavioral | Monthly |
| | Diversity index | Demographic | Quarterly |
| | Conflict resolution | Process data | Monthly |
| **Behavior** | | | |
| | Active users | Platform | Weekly |
| | Participation rate | Platform | Weekly |
| | Contribution diversity | Platform | Monthly |
| | Skill acquisition | Credentials | Quarterly |
| | Help provided | Platform | Monthly |
| **Systems** | | | |
| | Economic velocity | Transactions | Weekly |
| | Wealth distribution | Ledger | Monthly |
| | Carbon footprint | Calculated | Quarterly |
| | Governance quality | Process data | Monthly |
| | Infrastructure uptime | Technical | Daily |

### Calculation Formulas

```typescript
// Trust Index
const trustIndex = (
  (unsecuredLoansExtended / totalLoans) * 0.3 +
  (promisesKept / promisesMade) * 0.3 +
  (conflictsResolvedPeacefully / totalConflicts) * 0.2 +
  (governanceParticipation / eligibleMembers) * 0.2
);

// Social Cohesion Score
const socialCohesion = (
  (actualConnections / possibleConnections) * 0.25 +    // Density
  (crossGroupConnections / totalConnections) * 0.25 +   // Bridging
  (mutualConnections / totalConnections) * 0.25 +       // Reciprocity
  (1 - giniOfConnections) * 0.25                        // Equality
);

// Economic Health Index
const economicHealth = (
  velocityOfMoney * 0.2 +
  localCirculationRate * 0.2 +
  (1 - giniCoefficient) * 0.2 +
  commonsGrowthRate * 0.2 +
  resilienceScore * 0.2
);

// Overall Community Health (composite)
const communityHealth = (
  wellbeingScore * 0.25 +
  cultureScore * 0.25 +
  behaviorScore * 0.25 +
  systemsScore * 0.25
);
```

---

*"What we measure reflects what we value. Measure what truly matters—the flourishing of all beings and the regeneration of our living world."*
