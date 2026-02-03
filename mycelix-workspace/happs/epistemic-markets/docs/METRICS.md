# Metrics

*What We Measure and Why*

---

> "What gets measured gets managed."
> — Peter Drucker

> "What gets measured wrongly gets managed wrongly."
> — Our addendum

---

## Introduction

Metrics shape behavior. The metrics we choose reveal our values and influence what participants optimize for. This document defines the metrics we track, why we chose them, and how we use them.

We distinguish between:
- **North Star Metrics**: The ultimate measures of success
- **Health Metrics**: Indicators of system health
- **Operational Metrics**: Day-to-day performance
- **Research Metrics**: Understanding and improvement
- **Anti-Metrics**: What we explicitly don't optimize

---

## Part I: North Star Metrics

These are the ultimate measures of whether we're succeeding.

### 1. Collective Calibration

**Definition**: How well the community's aggregate predictions match outcomes.

**Formula**:
```
Collective Brier Score = (1/N) × Σ(aggregate_prediction - outcome)²
```

**Target**: Brier score < 0.20 (better than chance, approaching expert)

**Why this matters**: This directly measures our core purpose—collective truth-seeking accuracy.

**Measurement**:
- Calculated across all resolved markets
- Weighted by market significance
- Tracked over time windows (30d, 90d, 365d, all-time)

```typescript
interface CollectiveCalibrationMetric {
  brierScore: number;
  marketCount: number;
  predictionCount: number;
  timeWindow: string;
  trend: "improving" | "stable" | "declining";
  comparisonToExpert: number;  // Ratio vs. known expert benchmarks
}
```

---

### 2. Wisdom Accumulation Rate

**Definition**: The rate at which lasting, valuable knowledge is being created.

**Components**:
- Wisdom seeds planted
- Wisdom seeds verified as useful (by future predictors)
- Citations of past wisdom in current predictions
- Intergenerational knowledge transfer

**Target**: Positive year-over-year growth in verified wisdom

**Why this matters**: Short-term accuracy isn't enough—we want cumulative understanding.

**Measurement**:
```typescript
interface WisdomAccumulationMetric {
  seedsPlanted: number;
  seedsVerified: number;
  verificationRate: number;
  averageSeedRating: number;
  citationsPerSeed: number;
  knowledgeGraphConnections: number;
  intergenerationalTransfers: number;  // Seeds cited >1 year after planting
}
```

---

### 3. Epistemic Diversity Index

**Definition**: The diversity of perspectives and expertise contributing to predictions.

**Components**:
- Geographic distribution
- Domain expertise distribution
- Viewpoint diversity (measured by prediction variance)
- Socioeconomic diversity (proxy measures)

**Target**: Diversity index > 0.7 (on 0-1 scale)

**Why this matters**: Collective intelligence requires diverse perspectives.

**Measurement**:
```typescript
interface DiversityMetric {
  geographicEntropy: number;    // Shannon entropy of participant locations
  domainCoverage: number;       // Percentage of domains with experts
  viewpointVariance: number;    // Prediction variance before consensus
  newcomerIntegration: number;  // Rate of newcomer retention
  expertDistribution: number;   // Gini coefficient of expertise (lower = more equal)
}
```

---

### 4. Participant Growth and Retention

**Definition**: Sustainable growth with high-quality retention.

**Components**:
- Net new participants (quality-filtered)
- Retention rates (7d, 30d, 90d, 365d)
- Engagement depth (not just breadth)
- Graduation through stages

**Target**:
- 30-day retention > 50%
- 365-day retention > 25%
- Positive net growth

**Why this matters**: A thriving community requires sustainable participation.

**Measurement**:
```typescript
interface GrowthMetric {
  totalParticipants: number;
  activeParticipants: {
    daily: number;
    weekly: number;
    monthly: number;
  };
  retention: {
    day7: number;
    day30: number;
    day90: number;
    day365: number;
  };
  churn: {
    rate: number;
    reasons: Record<string, number>;
  };
  stageDistribution: Record<string, number>;  // How many at each stage
  netPromoterScore: number;
}
```

---

### 5. Truth-Seeking Culture Index

**Definition**: Qualitative measure of whether the culture embodies our values.

**Components** (survey and behavioral):
- Willingness to update beliefs
- Reasoning quality trends
- Constructive disagreement rate
- Commitment fulfillment rate
- Community helpfulness

**Target**: Culture index > 0.75 (on 0-1 scale)

**Why this matters**: Metrics alone can be gamed; culture is the true measure.

**Measurement**:
```typescript
interface CultureMetric {
  beliefUpdateRate: number;      // How often predictions are updated
  reasoningQualityTrend: number; // Change in average reasoning scores
  constructiveDisagreement: number; // Disagreements that led to synthesis
  commitmentFulfillment: number; // Percentage of commitments honored
  mentoringActivity: number;     // Mentorship sessions per month
  communityHelpfulness: number;  // Survey-based
  toxicityIncidents: number;     // Should be very low
}
```

---

## Part II: Health Metrics

These indicate whether the system is functioning properly.

### 6. Market Quality

**Definition**: The quality of markets being created and participated in.

**Components**:
- Questions clarity score
- Outcome definitiveness
- Appropriate E-N-M classification
- Participation depth

```typescript
interface MarketQualityMetric {
  averageClarityScore: number;     // From community ratings
  classificationAccuracy: number;   // Post-hoc review of E-N-M
  averageParticipants: number;
  averagePredictions: number;
  averageReasoningLength: number;
  averageEvidenceLinks: number;
  resolutionSuccessRate: number;   // Resolved without dispute
  marketCompletionRate: number;    // Markets that resolve vs. void
}
```

---

### 7. Resolution Health

**Definition**: How well the resolution process is working.

**Components**:
- Time to resolution
- Oracle participation rate
- Consensus achievement rate
- Dispute rate
- Dispute resolution success

```typescript
interface ResolutionHealthMetric {
  averageTimeToResolution: number;  // In hours
  oracleParticipationRate: number;  // Percentage of invited oracles who vote
  consensusAchievementRate: number; // Percentage reaching consensus
  disputeRate: number;              // Percentage of resolutions disputed
  disputeOverturnRate: number;      // Disputes that changed outcome
  byzantineIncidents: number;       // Detected manipulation attempts
}
```

---

### 8. Economic Health

**Definition**: The sustainability of the economic model.

**Components**:
- Stake volume
- Fee revenue
- Treasury balance
- Economic attack resistance

```typescript
interface EconomicHealthMetric {
  totalStakeVolume: number;
  averageStakeSize: number;
  feeRevenue: number;
  treasuryBalance: number;
  protocolRunwayDays: number;      // How long treasury lasts at current burn
  stakeProfitability: number;      // Are skilled predictors rewarded?
  economicAttacks: number;         // Detected/prevented
  wealthConcentration: number;     // Gini coefficient of holdings
}
```

---

### 9. Technical Health

**Definition**: System reliability and performance.

**Components**:
- Uptime
- Latency
- Error rates
- Network health

```typescript
interface TechnicalHealthMetric {
  uptime: number;                  // Percentage (target: 99.9%)
  averageLatency: number;          // Milliseconds
  p99Latency: number;
  errorRate: number;               // Percentage of failed operations
  nodeCount: number;               // Network participants
  networkPartitions: number;       // Should be 0
  dataIntegrity: number;           // Validation success rate
}
```

---

### 10. Security Health

**Definition**: Protection against attacks and failures.

**Components**:
- Attack attempts detected
- Sybil detection rate
- Collusion detection rate
- Recovery time from incidents

```typescript
interface SecurityHealthMetric {
  attacksDetected: number;
  attacksPrevented: number;
  sybilAccountsBlocked: number;
  collusionIncidents: number;
  securityIncidents: number;       // Total incidents
  meanTimeToDetection: number;     // In minutes
  meanTimeToRecovery: number;      // In minutes
  lastSecurityAuditDate: number;
  vulnerabilitiesOpen: number;
}
```

---

## Part III: Operational Metrics

Day-to-day performance indicators.

### 11. Engagement Metrics

```typescript
interface EngagementMetric {
  dailyActiveUsers: number;
  weeklyActiveUsers: number;
  monthlyActiveUsers: number;
  predictionsPerDay: number;
  marketsCreatedPerDay: number;
  questionsProposedPerDay: number;
  wisdomSeedsPerDay: number;
  averageSessionDuration: number;
  averageActionsPerSession: number;
}
```

---

### 12. Funnel Metrics

```typescript
interface FunnelMetric {
  visitorToSignup: number;
  signupToFirstPrediction: number;
  firstPredictionToSecond: number;
  casualToRegular: number;         // Making 5+ predictions
  regularToExpert: number;         // Becoming an oracle
  expertToElder: number;           // Long-term contribution
}
```

---

### 13. Support Metrics

```typescript
interface SupportMetric {
  supportTickets: number;
  averageResponseTime: number;
  resolutionRate: number;
  satisfactionScore: number;
  documentationHelpfulness: number;
  selfServiceSuccessRate: number;
}
```

---

## Part IV: Research Metrics

For understanding and improvement.

### 14. Calibration Research

```typescript
interface CalibrationResearchMetric {
  averageOverconfidence: number;
  overconfidenceByDomain: Record<string, number>;
  calibrationImprovementRate: number;  // How much users improve
  trainingEffectiveness: number;       // Improvement from training
  expertVsCrowdGap: number;            // Domain expert vs. aggregate
}
```

---

### 15. Aggregation Research

```typescript
interface AggregationResearchMetric {
  aggregationMethods: {
    method: string;
    brierScore: number;
    sampleSize: number;
  }[];
  optimalAggregation: string;          // Best performing method
  independenceViolations: number;      // Cascade detection
  extremizationEffect: number;         // Effect of confidence weighting
}
```

---

### 16. Question Market Research

```typescript
interface QuestionResearchMetric {
  questionSpawnRate: number;
  spawnedMarketSuccessRate: number;
  questionValuePredictiveness: number; // Does value predict good markets?
  timeToSpawn: number;                 // Average days
  questionQualityCorrelation: number;  // Value vs. eventual market quality
}
```

---

## Part V: Anti-Metrics

What we explicitly **don't** optimize for.

### 1. Raw Engagement Time

**Why not**: We don't want addictive engagement; we want meaningful participation.

**Instead**: Measure quality of engagement, not quantity.

---

### 2. Prediction Volume (Alone)

**Why not**: More predictions isn't better if they're low-quality.

**Instead**: Track volume × quality.

---

### 3. Market Count (Alone)

**Why not**: Spam markets don't add value.

**Instead**: Track meaningful, well-participated markets.

---

### 4. Growth Rate (Without Quality)

**Why not**: Fast growth with poor retention is worse than slow sustainable growth.

**Instead**: Track retention and quality alongside growth.

---

### 5. Profit/Revenue (As Primary)

**Why not**: We're not a profit-maximizing entity; we're epistemic infrastructure.

**Instead**: Track sustainability, not profit.

---

## Part VI: Metric Dashboards

### North Star Dashboard

| Metric | Current | Target | Trend |
|--------|---------|--------|-------|
| Collective Brier Score | 0.23 | < 0.20 | ↓ (improving) |
| Wisdom Accumulation | +142/month | +100/month | → (stable) |
| Diversity Index | 0.68 | > 0.70 | ↑ (improving) |
| 30-day Retention | 52% | > 50% | → (stable) |
| Culture Index | 0.77 | > 0.75 | → (stable) |

### Health Dashboard

| Category | Status | Details |
|----------|--------|---------|
| Market Quality | 🟢 Good | Clarity 4.2/5, 94% resolve |
| Resolution | 🟢 Good | 6h avg, 98% consensus |
| Economic | 🟡 Watch | Treasury at 85% target |
| Technical | 🟢 Good | 99.95% uptime |
| Security | 🟢 Good | 0 incidents this month |

### Operational Dashboard

| Metric | Today | 7d Avg | 30d Avg |
|--------|-------|--------|---------|
| DAU | 1,247 | 1,189 | 1,102 |
| Predictions | 423 | 398 | 356 |
| Markets Created | 12 | 11 | 9 |
| Questions Proposed | 34 | 29 | 24 |
| Wisdom Seeds | 8 | 7 | 6 |

---

## Part VII: Metric Collection

### Data Collection Principles

1. **Privacy-preserving**: Aggregate when possible, anonymize always
2. **Transparent**: Users can see what we collect
3. **Purposeful**: No data collection without clear use
4. **Minimal**: Collect only what's needed
5. **Secure**: Encryption and access controls

### Collection Methods

```typescript
interface MetricCollectionConfig {
  // What to collect
  metrics: MetricDefinition[];

  // How often
  collectionInterval: Duration;

  // Retention
  retentionPeriod: Duration;

  // Aggregation
  aggregationLevel: "individual" | "cohort" | "global";

  // Privacy
  privacyLevel: "public" | "internal" | "restricted";
}
```

### Metric Events

```typescript
// Events automatically collected
interface MetricEvent {
  timestamp: number;
  eventType: string;
  anonymizedUserId?: string;  // Hashed, not raw
  properties: Record<string, unknown>;
}

// Example events
"prediction.submitted" -> { marketId, confidence, stakeLevel }
"market.created" -> { epistemicPosition, outcomeCount }
"resolution.completed" -> { timeToResolve, oracleCount }
"wisdom.planted" -> { marketId, wordCount }
```

---

## Part VIII: Using Metrics

### Decision Framework

When making decisions:

1. **Check North Stars first**: Does this improve collective calibration, wisdom, diversity, retention, or culture?

2. **Check Health metrics**: Will this maintain system health?

3. **Consider Anti-metrics**: Are we accidentally optimizing for what we shouldn't?

4. **Run experiments**: A/B test when uncertain, measure results.

### Metric Reviews

**Daily**: Operational dashboard review
**Weekly**: Health metrics review, anomaly investigation
**Monthly**: North Star progress review, goal setting
**Quarterly**: Deep research metrics analysis, strategy adjustment
**Annually**: Comprehensive review, target recalibration

### Alert Thresholds

```typescript
interface MetricAlert {
  metric: string;
  condition: "above" | "below" | "change";
  threshold: number;
  severity: "info" | "warning" | "critical";
  action: string;
}

const alerts: MetricAlert[] = [
  { metric: "collective_brier", condition: "above", threshold: 0.30, severity: "warning", action: "Review aggregation" },
  { metric: "uptime", condition: "below", threshold: 0.99, severity: "critical", action: "Page on-call" },
  { metric: "dispute_rate", condition: "above", threshold: 0.10, severity: "warning", action: "Review resolution" },
  { metric: "security_incidents", condition: "above", threshold: 0, severity: "critical", action: "Security review" },
];
```

---

## Part IX: Metric Evolution

### How We Update Metrics

Metrics aren't fixed. As we learn:

1. **Propose change**: Explain why current metric is insufficient
2. **Design new metric**: Define clearly with collection method
3. **Run in parallel**: Collect new metric alongside old
4. **Evaluate**: Compare and verify new metric is better
5. **Migrate**: Switch to new metric, archive old

### Metric Versioning

```typescript
interface MetricVersion {
  name: string;
  version: string;
  definition: string;
  formula?: string;
  introducedAt: number;
  deprecatedAt?: number;
  replacedBy?: string;
}

// Example
{
  name: "collective_calibration",
  version: "2.0",
  definition: "MATL-weighted average Brier score",
  formula: "Σ(matl_weight × brier_score) / Σ(matl_weight)",
  introducedAt: 1704067200,
  deprecatedAt: null,
  replacedBy: null
}
```

---

## Part X: Honest Assessment

### What Our Metrics Don't Capture

1. **Genuine understanding**: Calibration measures accuracy, not comprehension
2. **Wisdom quality**: Verification is proxy, not direct measure
3. **Cultural nuance**: Survey-based culture metrics are limited
4. **Long-term impact**: Some effects take years to manifest
5. **Counterfactuals**: We can't measure what would have happened otherwise

### Known Limitations

- **Goodhart's Law risk**: Metrics can be gamed if incentivized wrong
- **Lag indicators**: Many metrics are trailing, not leading
- **Aggregation hides variance**: Averages mask important differences
- **Selection bias**: Metrics reflect active users, not potential users

### Mitigation

- **Multiple metrics**: No single metric dominates decisions
- **Qualitative balance**: Regular user research alongside quantitative
- **Anti-gaming design**: Metrics designed to resist manipulation
- **Regular review**: Continuous assessment of metric validity

---

## Conclusion

Metrics are tools, not goals. We track what matters while remembering that not everything that matters can be measured, and not everything that's measured matters.

Our commitment:
- Measure what reflects our values
- Be transparent about what we measure and why
- Evolve our metrics as we learn
- Never let metrics override wisdom

---

*"Not everything that counts can be counted, and not everything that can be counted counts."*
*— William Bruce Cameron*

*We count carefully. We also look up from the numbers.*

---

## Appendix: Metric Definitions

For programmatic access to all metric definitions:

```typescript
import { metricDefinitions } from "@mycelix/epistemic-markets-metrics";

// Get all metrics
const allMetrics = metricDefinitions.list();

// Get specific metric
const brierDef = metricDefinitions.get("collective_brier");

// Calculate metric
const value = await metricDefinitions.calculate("collective_brier", { timeWindow: "30d" });
```

See the metrics package for complete definitions and calculation methods.

