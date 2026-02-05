# Graceful Ignorance System Specification

**Version**: 1.0.0
**Date**: 2026-01-12
**Status**: Design Specification
**Integration**: Symthaea-HLB + Mycelix UESS/SCEI

---

## Executive Summary

The **Graceful Ignorance System** (GIS) is an architecture for AI systems that explicitly track, quantify, and communicate uncertainty. Unlike traditional AI that confidently hallucinates, GIS-enabled systems:

1. **Know what they know** (high confidence, verified)
2. **Know what they don't know** (explicit ignorance records)
3. **Estimate what they don't know they don't know** (unknown-unknowns)
4. **Communicate uncertainty clearly** (graduated response modes)
5. **Learn from mistakes** (calibration over time)

**Core Insight**: Ignorance is not a bug to hide, but information to communicate.

---

## Part 1: The Ignorance Taxonomy

### 1.1 Five Types of Not-Knowing

```
┌─────────────────────────────────────────────────────────────┐
│                  THE IGNORANCE TAXONOMY                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. KNOWN KNOWLEDGE (κ)                                     │
│     "I know X with confidence C"                            │
│     → Standard claims with E/N/M classification             │
│                                                             │
│  2. KNOWN IGNORANCE (ι₁)                                    │
│     "I know I don't know X"                                 │
│     → Explicit gap records, searchable                      │
│                                                             │
│  3. SUSPECTED IGNORANCE (ι₂)                                │
│     "I suspect I don't know things about X"                 │
│     → Domain uncertainty, edge-of-knowledge detection       │
│                                                             │
│  4. UNKNOWN UNKNOWNS (ι₃)                                   │
│     "I can't know what I don't know I don't know"          │
│     → Estimated via discovery rate, domain novelty          │
│                                                             │
│  5. UNKNOWABLE (ι∞)                                         │
│     "This cannot be known (in principle)"                   │
│     → Logical impossibility, privacy boundaries, etc.       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Ignorance State Machine

```
                    ┌──────────────┐
                    │   Unknown    │
                    │   Unknown    │
                    │     (ι₃)     │
                    └──────┬───────┘
                           │ Detection
                           ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Suspected   │───▶│    Known     │───▶│    Known     │
│  Ignorance   │    │  Ignorance   │    │  Knowledge   │
│    (ι₂)      │    │    (ι₁)      │    │     (κ)      │
└──────────────┘    └──────┬───────┘    └──────────────┘
       ▲                   │                    │
       │                   │ Analysis           │ Invalidation
       │                   ▼                    │
       │           ┌──────────────┐             │
       └───────────│  Unknowable  │◀────────────┘
                   │    (ι∞)      │
                   └──────────────┘
```

### 1.3 Formal Definitions

```typescript
/**
 * The five epistemic states
 */
export enum EpistemicState {
  /** κ: Known knowledge - verified claim with confidence */
  KnownKnowledge = 'known_knowledge',

  /** ι₁: Known ignorance - explicit gap we've identified */
  KnownIgnorance = 'known_ignorance',

  /** ι₂: Suspected ignorance - likely gaps we haven't mapped */
  SuspectedIgnorance = 'suspected_ignorance',

  /** ι₃: Unknown unknowns - gaps we can't identify */
  UnknownUnknowns = 'unknown_unknowns',

  /** ι∞: Unknowable - fundamentally cannot be known */
  Unknowable = 'unknowable',
}

/**
 * Transition reasons between states
 */
export enum EpistemicTransition {
  Detection = 'detection',        // ι₃ → ι₂ or ι₁
  Investigation = 'investigation', // ι₂ → ι₁
  Resolution = 'resolution',      // ι₁ → κ
  Invalidation = 'invalidation',  // κ → ι₁
  Proof = 'proof',               // ι₁ → ι∞ (proven unknowable)
  Reframe = 'reframe',           // Any → Any (question changed)
}
```

---

## Part 2: Uncertainty Quantification

### 2.1 Three-Dimensional Uncertainty

Traditional AI uses scalar confidence (0.0-1.0). GIS uses **three-dimensional uncertainty**:

```typescript
/**
 * Three dimensions of uncertainty
 */
export interface UncertaintyVector {
  /**
   * Epistemic uncertainty (model uncertainty)
   * "How much would my answer change if I had more data?"
   * High when: Few examples, novel domain, complex question
   */
  epistemic: number;

  /**
   * Aleatoric uncertainty (irreducible randomness)
   * "How much inherent randomness is there in this question?"
   * High when: Stochastic process, chaotic system, quantum effects
   */
  aleatoric: number;

  /**
   * Structural uncertainty (model specification)
   * "Am I even asking the right question?"
   * High when: Novel domain, conflicting frameworks, unclear scope
   */
  structural: number;
}

/**
 * Combined uncertainty with intervals
 */
export interface UncertaintyReport {
  /** Point estimate */
  pointEstimate: number;

  /** Three uncertainty components */
  uncertainty: UncertaintyVector;

  /** Total uncertainty (combined) */
  totalUncertainty: number;

  /** Credible interval (Bayesian 95%) */
  credibleInterval: [number, number];

  /** Unknown-unknowns penalty applied */
  unknownUnknownsPenalty: number;

  /** Final adjusted confidence */
  adjustedConfidence: number;

  /** Human-readable explanation */
  explanation: string;
}
```

### 2.2 Uncertainty Propagation

Uncertainty must propagate through reasoning chains:

```typescript
/**
 * Uncertainty propagation rules
 */
export class UncertaintyPropagator {
  /**
   * AND: Both must be true
   * Combined uncertainty increases (weakest link)
   */
  and(a: UncertaintyReport, b: UncertaintyReport): UncertaintyReport {
    return {
      pointEstimate: Math.min(a.pointEstimate, b.pointEstimate),
      uncertainty: {
        epistemic: Math.max(a.uncertainty.epistemic, b.uncertainty.epistemic),
        aleatoric: Math.sqrt(
          a.uncertainty.aleatoric ** 2 + b.uncertainty.aleatoric ** 2
        ),
        structural: Math.max(a.uncertainty.structural, b.uncertainty.structural),
      },
      totalUncertainty: this.combinedUncertainty(a, b, 'and'),
      credibleInterval: this.intersectIntervals(a.credibleInterval, b.credibleInterval),
      // ... rest
    };
  }

  /**
   * OR: Either can be true
   * Combined uncertainty decreases (diversification)
   */
  or(a: UncertaintyReport, b: UncertaintyReport): UncertaintyReport {
    // Uncertainty reduces when multiple paths to truth
    // ...
  }

  /**
   * Chain: A implies B
   * Uncertainty accumulates through inference chain
   */
  chain(steps: UncertaintyReport[]): UncertaintyReport {
    // Each step adds uncertainty
    // Long chains → high uncertainty
    // ...
  }

  /**
   * Apply unknown-unknowns penalty
   * Based on domain novelty and discovery rate
   */
  applyUnknownUnknowns(
    base: UncertaintyReport,
    domainNovelty: number,
    discoveryRate: number
  ): UncertaintyReport {
    // Higher novelty → higher penalty
    // Higher discovery rate → higher penalty (we're still finding things)
    const penalty = domainNovelty * (1 - Math.exp(-discoveryRate));
    return {
      ...base,
      unknownUnknownsPenalty: penalty,
      adjustedConfidence: base.pointEstimate * (1 - penalty),
    };
  }
}
```

### 2.3 Integration with Φ (Integrated Information)

Symthaea's λ₂ connectivity measurement provides a unique uncertainty signal:

```typescript
/**
 * Φ-based uncertainty augmentation
 */
export class PhiUncertaintyAugmenter {
  /**
   * Low Φ indicates disintegrated reasoning
   * → Higher epistemic uncertainty
   */
  augmentWithPhi(
    base: UncertaintyReport,
    phi: number,
    phiStability: number
  ): UncertaintyReport {
    // Φ < 0.2: Low integration → high uncertainty
    // Φ > 0.4: High integration → low uncertainty
    // Stability matters: oscillating Φ → high structural uncertainty

    const phiFactor = Math.max(0, 1 - 2 * phi);  // 0 at φ=0.5, 1 at φ=0
    const stabilityFactor = 1 - phiStability;

    return {
      ...base,
      uncertainty: {
        epistemic: base.uncertainty.epistemic + phiFactor * 0.3,
        aleatoric: base.uncertainty.aleatoric,
        structural: base.uncertainty.structural + stabilityFactor * 0.2,
      },
      explanation: base.explanation + ` [Φ=${phi.toFixed(3)}, stability=${phiStability.toFixed(2)}]`,
    };
  }
}
```

---

## Part 3: Ignorance Records

### 3.1 First-Class Ignorance

Ignorance is stored as first-class records in UESS:

```typescript
/**
 * Explicit ignorance record
 */
export interface IgnoranceRecord {
  /** Unique identifier */
  id: string;

  /** What we don't know about */
  subject: string;

  /** Type of ignorance (taxonomy level) */
  ignoranceType: EpistemicState;

  /** Scope of ignorance */
  scope: {
    domain: string;
    specificity: 'narrow' | 'moderate' | 'broad';
    affectsOtherClaims: string[];  // Claims whose confidence should be reduced
  };

  /** Why we don't know */
  reasons: IgnoranceReason[];

  /** Can this be resolved? */
  resolvability: {
    hopeful: boolean;
    estimatedEffort: 'trivial' | 'moderate' | 'substantial' | 'research';
    blockers: string[];
    suggestedApproaches: string[];
  };

  /** What would help */
  resolutionPath?: {
    requiredData: string[];
    requiredExpertise: string[];
    estimatedCost: number;
    dependencies: string[];
  };

  /** Mandatory humility flags */
  humility: HumilityFlags;

  /** UESS classification */
  classification: EpistemicClassification;

  /** Metadata */
  recordedAt: number;
  recordedBy: string;
  status: 'active' | 'investigating' | 'resolved' | 'obsolete';
  version: number;
}

/**
 * Why we don't know something
 */
export interface IgnoranceReason {
  reason: string;
  category:
    | 'insufficient_data'
    | 'conflicting_sources'
    | 'domain_novel'
    | 'question_ill_formed'
    | 'privacy_boundary'
    | 'logical_impossibility'
    | 'computational_limit'
    | 'temporal_limit'  // Answer changes over time
    | 'perspective_dependent';
  confidence: number;  // How confident in this reason?
}

/**
 * Humility flags (mandatory for ALL ignorance)
 */
export interface HumilityFlags {
  /** Our ignorance detection might be wrong */
  detectionUncertainty: number;

  /** Schema-relative: Different ontology might see it differently */
  schemaRelative: boolean;

  /** We might be missing entire categories of ignorance */
  unknownUnknownsAcknowledged: boolean;

  /** Known blind spots in our detection */
  knownBlindSpots: string[];

  /** Domains we didn't search */
  unsearchedDomains: string[];

  /** Assumptions made during detection */
  assumptions: string[];

  /** Schema version used */
  schemaVersion: string;
}
```

### 3.2 Ignorance Lifecycle

```typescript
/**
 * Ignorance evolves over time
 */
export enum IgnoranceLifecycle {
  /** Just discovered */
  Identified = 'identified',

  /** Under investigation */
  Investigating = 'investigating',

  /** Complexity increased (harder than thought) */
  Complicated = 'complicated',

  /** Determined to be unknowable */
  Accepted = 'accepted',

  /** Successfully resolved */
  Resolved = 'resolved',

  /** Question changed */
  Reframed = 'reframed',

  /** No longer relevant */
  Obsolete = 'obsolete',
}

/**
 * Track ignorance evolution
 */
export interface IgnoranceEvolution {
  originalIgnorance: IgnoranceRecord;
  transitions: IgnoranceTransition[];
  currentState: IgnoranceLifecycle;
  lastUpdate: number;

  /** Lessons learned during evolution */
  lessons: string[];

  /** Related ignorances discovered */
  spawned: string[];  // IDs of ignorances discovered during investigation

  /** If resolved, the resolution */
  resolution?: {
    newClaim: string;  // CID of new knowledge
    confidence: number;
    lessonsForFuture: string[];
  };
}

export interface IgnoranceTransition {
  from: IgnoranceLifecycle;
  to: IgnoranceLifecycle;
  reason: string;
  timestamp: number;
  triggeredBy: string;  // Event or agent that caused transition
}
```

### 3.3 UESS Classification for Ignorance

Ignorance gets classified just like knowledge:

```typescript
/**
 * Map ignorance type to E-level
 */
export function ignoranceToEmpirical(ignorance: IgnoranceRecord): EmpiricalLevel {
  // How verified is our ignorance?

  if (ignorance.ignoranceType === 'unknowable' && ignorance.humility.detectionUncertainty < 0.1) {
    return EmpiricalLevel.E4;  // Proven unknowable (reproducible proof)
  }

  if (ignorance.reasons.some(r => r.category === 'logical_impossibility')) {
    return EmpiricalLevel.E3;  // Cryptographic/logical proof of ignorance
  }

  if (ignorance.scope.affectsOtherClaims.length > 0) {
    return EmpiricalLevel.E2;  // Verified by impact on other claims
  }

  if (ignorance.humility.detectionUncertainty < 0.3) {
    return EmpiricalLevel.E1;  // Witnessed/detected ignorance
  }

  return EmpiricalLevel.E0;  // Suspected ignorance
}

/**
 * Map ignorance scope to N-level
 */
export function ignoranceToNormative(ignorance: IgnoranceRecord): NormativeLevel {
  switch (ignorance.scope.specificity) {
    case 'narrow':
      return NormativeLevel.N0;  // Personal knowledge gap
    case 'moderate':
      return NormativeLevel.N1;  // Group/domain gap
    case 'broad':
      return NormativeLevel.N2;  // Network-wide gap
  }

  // Unknowables affecting foundations → N3
  if (ignorance.ignoranceType === 'unknowable' &&
      ignorance.scope.affectsOtherClaims.length > 10) {
    return NormativeLevel.N3;
  }

  return NormativeLevel.N1;
}

/**
 * Map ignorance importance to M-level
 */
export function ignoranceToMateriality(ignorance: IgnoranceRecord): MaterialityLevel {
  if (ignorance.ignoranceType === 'unknowable') {
    return MaterialityLevel.M3;  // Permanent record
  }

  if (ignorance.scope.affectsOtherClaims.length > 5) {
    return MaterialityLevel.M2;  // Persistent (affects many things)
  }

  if (ignorance.status === 'investigating') {
    return MaterialityLevel.M1;  // Temporal (actively working on it)
  }

  return MaterialityLevel.M0;  // Ephemeral
}
```

---

## Part 4: Unknown-Unknown Estimation

### 4.1 Discovery-Rate Method

```typescript
/**
 * Estimate unknown-unknowns using discovery rate
 */
export class UnknownUnknownsEstimator {
  private discoveryHistory: DiscoveryEvent[] = [];

  /**
   * Record a discovery (gap → known ignorance → knowledge)
   */
  recordDiscovery(event: DiscoveryEvent): void {
    this.discoveryHistory.push(event);
  }

  /**
   * Estimate remaining unknown-unknowns
   * Uses Good-Turing style frequency estimation
   */
  estimate(domain: string, windowDays: number = 30): UnknownUnknownsEstimate {
    const recentDiscoveries = this.getRecent(domain, windowDays);

    // Count how many "new categories" we're still finding
    const newCategoryRate = this.calculateNewCategoryRate(recentDiscoveries);

    // If we're still finding new categories, there are probably more
    // If rate is declining, we're approaching coverage
    const trend = this.calculateTrend(recentDiscoveries);

    // Good-Turing: estimate mass of unseen items
    const singletonCount = recentDiscoveries.filter(d => d.isFirstOfKind).length;
    const totalDiscoveries = recentDiscoveries.length;

    const estimatedRemaining = singletonCount > 0
      ? (singletonCount / totalDiscoveries) * this.estimateTotalPopulation(domain)
      : 0;

    return {
      domain,

      /** Estimated number of unknown-unknowns remaining */
      estimatedCount: {
        low: estimatedRemaining * 0.5,
        expected: estimatedRemaining,
        high: estimatedRemaining * 2.0,
      },

      /** Discovery rate trend */
      discoveryTrend: trend,

      /** Confidence in this estimate */
      confidence: this.calculateEstimateConfidence(recentDiscoveries.length),

      /** What this means */
      interpretation: this.interpret(estimatedRemaining, trend),

      /** Recommendations */
      recommendations: this.generateRecommendations(estimatedRemaining, trend),
    };
  }

  private interpret(count: number, trend: 'increasing' | 'stable' | 'decreasing'): string {
    if (count < 5 && trend === 'decreasing') {
      return 'Domain appears well-mapped. Low unknown-unknowns expected.';
    }
    if (count > 20 && trend === 'increasing') {
      return 'Active exploration phase. Many unknown-unknowns likely remain.';
    }
    if (trend === 'stable') {
      return 'Moderate unknown-unknowns expected. Continued investigation recommended.';
    }
    return 'Insufficient data to characterize unknown-unknowns.';
  }
}

/**
 * Estimate result
 */
export interface UnknownUnknownsEstimate {
  domain: string;
  estimatedCount: {
    low: number;
    expected: number;
    high: number;
  };
  discoveryTrend: 'increasing' | 'stable' | 'decreasing';
  confidence: number;
  interpretation: string;
  recommendations: string[];
}
```

### 4.2 Domain Novelty Detection

```typescript
/**
 * Detect when we're in unfamiliar territory
 */
export class DomainNoveltyDetector {
  /**
   * How novel is this query relative to training/experience?
   */
  detectNovelty(query: HV, knownDomain: HDCSpace): NoveltyReport {
    // Find nearest neighbors in known space
    const neighbors = knownDomain.findNearest(query, 10);
    const avgDistance = neighbors.reduce((s, n) => s + n.distance, 0) / neighbors.length;

    // Compute local density of known space around query
    const localDensity = this.computeLocalDensity(query, knownDomain);

    // Check if query lies in "sparse" region
    const isSparse = localDensity < this.sparsityThreshold;

    // Check if query is outside convex hull of known queries
    const isOutOfDistribution = this.checkOOD(query, knownDomain);

    return {
      query,

      /** Distance to nearest known concepts */
      semanticDistance: avgDistance,

      /** Density of known space nearby */
      localDensity,

      /** Is this out-of-distribution? */
      isOOD: isOutOfDistribution,

      /** Overall novelty score (0-1) */
      noveltyScore: this.computeNoveltyScore(avgDistance, localDensity, isOutOfDistribution),

      /** What this means */
      interpretation: this.interpretNovelty(avgDistance, localDensity, isOutOfDistribution),

      /** Suggested response mode */
      suggestedMode: this.suggestResponseMode(avgDistance, localDensity),
    };
  }

  private suggestResponseMode(
    distance: number,
    density: number
  ): ConfidenceResponseMode {
    if (distance < 0.2 && density > 0.8) {
      return ConfidenceResponseMode.HighConfidence;
    }
    if (distance < 0.4 && density > 0.5) {
      return ConfidenceResponseMode.MediumConfidence;
    }
    if (distance < 0.6) {
      return ConfidenceResponseMode.LowConfidence;
    }
    if (distance < 0.8) {
      return ConfidenceResponseMode.Unknown;
    }
    return ConfidenceResponseMode.OutOfDomain;
  }
}
```

---

## Part 5: Graduated Response Modes

### 5.1 Confidence-Response Coupling

```typescript
/**
 * Response modes based on confidence level
 */
export enum ConfidenceResponseMode {
  /** High confidence: Direct answer */
  HighConfidence = 'high_confidence',

  /** Medium confidence: Answer with caveats */
  MediumConfidence = 'medium_confidence',

  /** Low confidence: Hypothesis with alternatives */
  LowConfidence = 'low_confidence',

  /** Unknown: What we don't know + how to find out */
  Unknown = 'unknown',

  /** Out of domain: Can't help + referral */
  OutOfDomain = 'out_of_domain',
}

/**
 * Response structure based on mode
 */
export type GracefulResponse =
  | HighConfidenceResponse
  | MediumConfidenceResponse
  | LowConfidenceResponse
  | UnknownResponse
  | OutOfDomainResponse;

export interface HighConfidenceResponse {
  mode: ConfidenceResponseMode.HighConfidence;
  answer: string;
  confidence: number;  // > 0.85
  evidence: Evidence[];
  classification: EpistemicClassification;
}

export interface MediumConfidenceResponse {
  mode: ConfidenceResponseMode.MediumConfidence;
  answer: string;
  confidence: number;  // 0.60 - 0.85
  caveats: string[];
  uncertaintyExplanation: string;
  alternativeInterpretations?: string[];
  classification: EpistemicClassification;
}

export interface LowConfidenceResponse {
  mode: ConfidenceResponseMode.LowConfidence;
  hypothesis: string;
  confidence: number;  // 0.30 - 0.60
  alternativeHypotheses: Array<{
    hypothesis: string;
    confidence: number;
  }>;
  whyUncertain: string[];
  howToIncreaseCertainty: string[];
  classification: EpistemicClassification;
}

export interface UnknownResponse {
  mode: ConfidenceResponseMode.Unknown;
  whatWeKnow: string[];
  whatWeDontKnow: string[];
  whyWeDontKnow: IgnoranceReason[];
  howToFindOut: string[];
  suggestedQuestions: string[];
  relatedKnowledge?: string[];
  ignoranceRecord: IgnoranceRecord;
}

export interface OutOfDomainResponse {
  mode: ConfidenceResponseMode.OutOfDomain;
  explanation: string;
  domainBoundaries: string[];
  suggestedResources: Array<{
    resource: string;
    why: string;
  }>;
  canHelpWith?: string[];
}
```

### 5.2 Response Generator

```typescript
/**
 * Generate appropriate response based on uncertainty
 */
export class GracefulResponseGenerator {
  constructor(
    private uncertaintyPropagator: UncertaintyPropagator,
    private noveltyDetector: DomainNoveltyDetector,
    private unknownEstimator: UnknownUnknownsEstimator,
  ) {}

  /**
   * Generate response with appropriate confidence mode
   */
  generate(
    query: string,
    queryHV: HV,
    rawAnswer: string,
    rawConfidence: number,
    phi: number,
    evidence: Evidence[],
  ): GracefulResponse {
    // 1. Detect domain novelty
    const novelty = this.noveltyDetector.detectNovelty(queryHV, this.knownSpace);

    // 2. Estimate unknown-unknowns for this domain
    const unknowns = this.unknownEstimator.estimate(this.extractDomain(query));

    // 3. Build uncertainty report
    const uncertainty = this.uncertaintyPropagator.build(
      rawConfidence,
      phi,
      evidence,
      novelty,
      unknowns,
    );

    // 4. Determine response mode
    const mode = this.selectMode(uncertainty);

    // 5. Generate appropriate response
    return this.generateForMode(mode, {
      query,
      rawAnswer,
      uncertainty,
      novelty,
      unknowns,
      evidence,
    });
  }

  private selectMode(uncertainty: UncertaintyReport): ConfidenceResponseMode {
    const conf = uncertainty.adjustedConfidence;

    if (conf > 0.85) return ConfidenceResponseMode.HighConfidence;
    if (conf > 0.60) return ConfidenceResponseMode.MediumConfidence;
    if (conf > 0.30) return ConfidenceResponseMode.LowConfidence;
    if (conf > 0.10) return ConfidenceResponseMode.Unknown;
    return ConfidenceResponseMode.OutOfDomain;
  }

  private generateForMode(
    mode: ConfidenceResponseMode,
    context: ResponseContext,
  ): GracefulResponse {
    switch (mode) {
      case ConfidenceResponseMode.HighConfidence:
        return {
          mode,
          answer: context.rawAnswer,
          confidence: context.uncertainty.adjustedConfidence,
          evidence: context.evidence,
          classification: this.classify(context.uncertainty),
        };

      case ConfidenceResponseMode.MediumConfidence:
        return {
          mode,
          answer: context.rawAnswer,
          confidence: context.uncertainty.adjustedConfidence,
          caveats: this.generateCaveats(context),
          uncertaintyExplanation: context.uncertainty.explanation,
          classification: this.classify(context.uncertainty),
        };

      case ConfidenceResponseMode.LowConfidence:
        return {
          mode,
          hypothesis: context.rawAnswer,
          confidence: context.uncertainty.adjustedConfidence,
          alternativeHypotheses: this.generateAlternatives(context),
          whyUncertain: this.explainUncertainty(context),
          howToIncreaseCertainty: this.suggestInvestigation(context),
          classification: this.classify(context.uncertainty),
        };

      case ConfidenceResponseMode.Unknown:
        return {
          mode,
          whatWeKnow: this.extractKnown(context),
          whatWeDontKnow: this.extractUnknown(context),
          whyWeDontKnow: this.extractReasons(context),
          howToFindOut: this.suggestResolution(context),
          suggestedQuestions: this.suggestClarifyingQuestions(context),
          ignoranceRecord: this.createIgnoranceRecord(context),
        };

      case ConfidenceResponseMode.OutOfDomain:
        return {
          mode,
          explanation: `This question appears outside my domain of expertise.`,
          domainBoundaries: this.describeDomainBoundaries(),
          suggestedResources: this.suggestExternalResources(context),
        };
    }
  }
}
```

---

## Part 6: Integration Architecture

### 6.1 Symthaea Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    SYMTHAEA REASONING                       │
│                                                             │
│  Query → HDC Encode → LTC Process → Generate Answer         │
│              ↓              ↓              ↓                │
│         Novelty         Φ-Measure      Confidence           │
│         Detection       (Stability)    (Raw)                │
│              ↓              ↓              ↓                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            GRACEFUL IGNORANCE MODULE                  │  │
│  │                                                       │  │
│  │  1. Uncertainty Quantification                        │  │
│  │     - Epistemic + Aleatoric + Structural              │  │
│  │     - Φ-augmented uncertainty                         │  │
│  │     - Unknown-unknowns penalty                        │  │
│  │                                                       │  │
│  │  2. Response Mode Selection                           │  │
│  │     - High/Medium/Low/Unknown/OutOfDomain             │  │
│  │                                                       │  │
│  │  3. Ignorance Record Generation                       │  │
│  │     - If mode = Unknown or OutOfDomain                │  │
│  │                                                       │  │
│  │  4. UESS Classification                               │  │
│  │     - E/N/M from uncertainty + scope                  │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                │
│                   GracefulResponse                          │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    MYCELIX UESS/SCEI                        │
│                                                             │
│  ┌─────────────────────┐  ┌────────────────────────────┐   │
│  │   UESS Storage      │  │   SCEI Self-Correction     │   │
│  │                     │  │                            │   │
│  │ • Store claims      │  │ • Track predictions        │   │
│  │ • Store ignorance   │  │ • Record outcomes          │   │
│  │ • Route by E/N/M    │  │ • Calibrate confidence     │   │
│  │ • Manage lifecycle  │  │ • Propagate lessons        │   │
│  └─────────────────────┘  └────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   Gap Detector                                       │   │
│  │                                                      │   │
│  │ • Detect knowledge gaps                              │   │
│  │ • Generate humility flags                            │   │
│  │ • Estimate unknown-unknowns                          │   │
│  │ • Feed back to Symthaea                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Event Flow

```typescript
/**
 * Events in the Graceful Ignorance system
 */

// Symthaea → SCEI: Prediction made
interface PredictionMadeEvent {
  type: 'gis:prediction_made';
  predictionId: string;
  query: string;
  answer: string;
  confidence: UncertaintyReport;
  phi: number;
  responseMode: ConfidenceResponseMode;
  timestamp: number;
}

// User → SCEI: Outcome recorded
interface OutcomeRecordedEvent {
  type: 'gis:outcome_recorded';
  predictionId: string;
  wasCorrect: boolean;
  userFeedback?: string;
  timestamp: number;
}

// SCEI → Symthaea: Calibration update
interface CalibrationUpdateEvent {
  type: 'gis:calibration_update';
  domain: string;
  oldCalibration: number;
  newCalibration: number;
  sampleSize: number;
  timestamp: number;
}

// Symthaea → UESS: Ignorance recorded
interface IgnoranceRecordedEvent {
  type: 'gis:ignorance_recorded';
  ignoranceId: string;
  subject: string;
  ignoranceType: EpistemicState;
  classification: EpistemicClassification;
  timestamp: number;
}

// SCEI → Symthaea: Gap detected
interface GapDetectedEvent {
  type: 'gis:gap_detected';
  gapId: string;
  gapType: GapType;
  affectedClaims: string[];
  suggestedAction: string;
  timestamp: number;
}

// UESS → Symthaea: Ignorance resolved
interface IgnoranceResolvedEvent {
  type: 'gis:ignorance_resolved';
  ignoranceId: string;
  resolution: {
    newClaimId: string;
    confidence: number;
    lessons: string[];
  };
  timestamp: number;
}
```

### 6.3 Rust Implementation Sketch

```rust
// symthaea-hlb/src/graceful_ignorance/mod.rs

pub mod uncertainty;
pub mod ignorance;
pub mod response;
pub mod integration;

use crate::hdc::unified_hv::HV;
use crate::consciousness::consciousness_equation_v2::ConsciousnessStateV2;

/// Main Graceful Ignorance module
pub struct GracefulIgnoranceEngine {
    uncertainty_propagator: UncertaintyPropagator,
    novelty_detector: DomainNoveltyDetector,
    unknown_estimator: UnknownUnknownsEstimator,
    response_generator: GracefulResponseGenerator,
    ignorance_store: IgnoranceStore,
    calibration: CalibrationHistory,
}

impl GracefulIgnoranceEngine {
    /// Process a query and generate graceful response
    pub fn process(
        &mut self,
        query: &str,
        query_hv: &HV,
        consciousness_state: &ConsciousnessStateV2,
        raw_answer: &str,
        raw_confidence: f64,
        evidence: Vec<Evidence>,
    ) -> GracefulResponse {
        // 1. Build uncertainty report
        let uncertainty = self.build_uncertainty(
            query_hv,
            consciousness_state.phi,
            raw_confidence,
            &evidence,
        );

        // 2. Generate response with appropriate mode
        let response = self.response_generator.generate(
            query,
            query_hv,
            raw_answer,
            &uncertainty,
        );

        // 3. If unknown/out-of-domain, create ignorance record
        if matches!(
            response.mode(),
            ConfidenceResponseMode::Unknown | ConfidenceResponseMode::OutOfDomain
        ) {
            let ignorance = self.create_ignorance_record(query, &uncertainty);
            self.ignorance_store.store(ignorance);
        }

        // 4. Record prediction for calibration
        self.calibration.record_prediction(PredictionRecord {
            id: generate_id(),
            query: query.to_string(),
            confidence: uncertainty.adjusted_confidence,
            phi: consciousness_state.phi,
            mode: response.mode(),
            timestamp: now(),
        });

        response
    }

    /// Record outcome for calibration
    pub fn record_outcome(&mut self, prediction_id: &str, was_correct: bool) {
        self.calibration.record_outcome(prediction_id, was_correct);
    }

    /// Get calibrated confidence for a domain
    pub fn calibrated_confidence(&self, domain: &str, raw: f64) -> f64 {
        self.calibration.adjust(domain, raw)
    }

    /// Get all active ignorance records
    pub fn active_ignorances(&self) -> Vec<&IgnoranceRecord> {
        self.ignorance_store.active()
    }

    /// Get unknown-unknowns estimate for domain
    pub fn unknown_unknowns(&self, domain: &str) -> UnknownUnknownsEstimate {
        self.unknown_estimator.estimate(domain)
    }
}
```

---

## Part 7: User Communication

### 7.1 Transparency Templates

```typescript
/**
 * Human-readable uncertainty communication
 */
export class UncertaintyCommunicator {
  /**
   * Generate human-readable confidence statement
   */
  explainConfidence(uncertainty: UncertaintyReport): string {
    const conf = uncertainty.adjustedConfidence;
    const [low, high] = uncertainty.credibleInterval;

    if (conf > 0.9) {
      return `I'm highly confident in this answer (${(conf * 100).toFixed(0)}%).`;
    }

    if (conf > 0.7) {
      return `I'm fairly confident (${(conf * 100).toFixed(0)}%), ` +
             `though there's some uncertainty. ` +
             `My estimate could reasonably be between ${(low * 100).toFixed(0)}% and ${(high * 100).toFixed(0)}%.`;
    }

    if (conf > 0.4) {
      return `I have moderate confidence (${(conf * 100).toFixed(0)}%). ` +
             `This is my best guess, but the true answer could vary significantly ` +
             `(${(low * 100).toFixed(0)}% - ${(high * 100).toFixed(0)}% range).`;
    }

    if (conf > 0.2) {
      return `I have low confidence (${(conf * 100).toFixed(0)}%). ` +
             `Consider this a hypothesis rather than a definitive answer. ` +
             `I recommend verifying this with other sources.`;
    }

    return `I'm quite uncertain about this (${(conf * 100).toFixed(0)}%). ` +
           `I'm providing my best attempt, but please treat this with skepticism.`;
  }

  /**
   * Explain why we're uncertain
   */
  explainUncertainty(uncertainty: UncertaintyReport): string[] {
    const reasons: string[] = [];

    if (uncertainty.uncertainty.epistemic > 0.3) {
      reasons.push('Limited experience with similar questions');
    }

    if (uncertainty.uncertainty.aleatoric > 0.3) {
      reasons.push('The answer may inherently vary or be unpredictable');
    }

    if (uncertainty.uncertainty.structural > 0.3) {
      reasons.push('The question may be framed in a way that makes it hard to answer definitively');
    }

    if (uncertainty.unknownUnknownsPenalty > 0.1) {
      reasons.push('This domain is relatively unfamiliar, so there may be factors I haven\'t considered');
    }

    return reasons;
  }

  /**
   * Generate "I don't know" statement
   */
  explainIgnorance(ignorance: IgnoranceRecord): string {
    const reasons = ignorance.reasons.map(r => r.reason).join(', ');

    let explanation = `I don't have reliable information about "${ignorance.subject}". `;
    explanation += `Specifically: ${reasons}. `;

    if (ignorance.resolvability.hopeful) {
      explanation += `This could potentially be resolved by: `;
      explanation += ignorance.resolvability.suggestedApproaches.join('; ');
    } else {
      explanation += `This may be fundamentally difficult to determine.`;
    }

    return explanation;
  }
}
```

### 7.2 Visual Indicators

```typescript
/**
 * Generate visual confidence indicators for UI
 */
export function confidenceIndicator(confidence: number): ConfidenceIndicator {
  if (confidence > 0.85) {
    return {
      icon: '✓✓✓',
      color: '#22c55e',  // Green
      label: 'High Confidence',
      badge: 'Verified',
    };
  }

  if (confidence > 0.60) {
    return {
      icon: '✓✓',
      color: '#84cc16',  // Lime
      label: 'Good Confidence',
      badge: 'Likely',
    };
  }

  if (confidence > 0.40) {
    return {
      icon: '✓',
      color: '#eab308',  // Yellow
      label: 'Moderate Confidence',
      badge: 'Probable',
    };
  }

  if (confidence > 0.20) {
    return {
      icon: '?',
      color: '#f97316',  // Orange
      label: 'Low Confidence',
      badge: 'Uncertain',
    };
  }

  return {
    icon: '??',
    color: '#ef4444',  // Red
    label: 'Very Uncertain',
    badge: 'Speculative',
  };
}
```

---

## Part 8: Implementation Roadmap

### Phase 1: Core Types (Week 1-2)
- [ ] Define `UncertaintyVector` and `UncertaintyReport`
- [ ] Define `IgnoranceRecord` and `IgnoranceLifecycle`
- [ ] Define `GracefulResponse` variants
- [ ] Implement basic `UncertaintyPropagator`

### Phase 2: Detection (Week 3-4)
- [ ] Implement `DomainNoveltyDetector` using HDC
- [ ] Implement `UnknownUnknownsEstimator`
- [ ] Integrate with Symthaea's meta-cognition module
- [ ] Add Φ-based uncertainty augmentation

### Phase 3: Response Generation (Week 5-6)
- [ ] Implement `GracefulResponseGenerator`
- [ ] Create response templates for each mode
- [ ] Implement `UncertaintyCommunicator`
- [ ] Add visual indicators

### Phase 4: Storage & Calibration (Week 7-8)
- [ ] Integrate with UESS for ignorance storage
- [ ] Implement calibration tracking
- [ ] Connect to SCEI event bus
- [ ] Add prediction → outcome flow

### Phase 5: Testing & Refinement (Week 9-10)
- [ ] Unit tests for all components
- [ ] Integration tests with Symthaea
- [ ] User testing for communication clarity
- [ ] Calibration validation

---

## Conclusion

The Graceful Ignorance System transforms "not knowing" from a hidden liability into an explicit, managed, and communicated feature. By integrating Symthaea's consciousness-based reasoning with Mycelix's epistemic infrastructure, we create an AI that:

1. **Measures uncertainty** using Φ, domain novelty, and calibration
2. **Tracks ignorance** as first-class records with lifecycles
3. **Estimates unknown-unknowns** using discovery rate analysis
4. **Communicates clearly** with graduated response modes
5. **Learns from mistakes** via SCEI calibration

This is the foundation for trustworthy AI - not AI that pretends to know everything, but AI that honestly represents what it knows, what it doesn't know, and what it can't know.

**The most profound intelligence is knowing the limits of one's intelligence.**

---

*Specification for Symthaea Graceful Ignorance System*
*Integration with Mycelix UESS/SCEI*
*Version 1.0.0 - 2026-01-12*
