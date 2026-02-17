/**
 * Trust Market Protocol
 *
 * Revolutionary prediction markets for trust claims themselves.
 * Creates self-correcting belief revelation through market mechanisms.
 *
 * Key Innovation: Markets for "Will Alice's reputation stay > 0.7 for 30 days?"
 * force participants to reveal honest beliefs about trustworthiness through
 * skin-in-the-game stakes.
 *
 * @module @mycelix/sdk/innovations/trust-markets
 */

import { CalibrationEngine } from '../../calibration/engine';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicPosition,
  getRecommendedResolution,
  type ResolutionMechanism,
} from '../../integrations/epistemic-markets';

import type { HappId } from '../../bridge/cross-happ';
import type { CalibrationReport } from '../../calibration/types';


// =============================================================================
// TRUST MARKET TYPES
// =============================================================================

/**
 * A market for predicting trust/reputation outcomes
 */
export interface TrustMarket {
  id: string;
  /** Subject whose trust is being predicted */
  subjectDid: string;
  /** The trust claim being traded */
  claim: TrustClaim;
  /** Market mechanism */
  mechanism: TrustMarketMechanism;
  /** Current market state */
  state: TrustMarketState;
  /** Resolution configuration */
  resolution: TrustResolutionConfig;
  /** Market metadata */
  metadata: TrustMarketMetadata;
  /** Creation timestamp */
  createdAt: number;
  /** Resolution timestamp (if resolved) */
  resolvedAt?: number;
}

/**
 * A claim about future trust/reputation
 */
export interface TrustClaim {
  /** Type of trust claim */
  type: TrustClaimType;
  /** Target threshold (e.g., 0.7 for "reputation > 0.7") */
  threshold: number;
  /** hApps to consider for aggregated reputation */
  contextHapps: HappId[];
  /** Duration in milliseconds */
  durationMs: number;
  /** Condition operator */
  operator: 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'between';
  /** Secondary threshold for 'between' operator */
  upperThreshold?: number;
}

export type TrustClaimType =
  | 'reputation_maintenance'     // Will reputation stay above threshold?
  | 'reputation_improvement'     // Will reputation increase by X%?
  | 'reputation_recovery'        // Will reputation recover from low point?
  | 'cross_happ_consistency'     // Will reputation be consistent across hApps?
  | 'calibration_accuracy'       // Will calibration score remain good?
  | 'byzantine_probability'      // What's probability subject acts Byzantine?
  | 'trust_convergence';         // Will network converge on trust assessment?

/**
 * Market mechanism for trust trading
 */
export interface TrustMarketMechanism {
  type: 'LMSR' | 'CDA' | 'Parimutuel' | 'TrustWeighted';
  /** LMSR liquidity parameter */
  liquidityParameter?: number;
  /** Subsidy pool for market making */
  subsidyPool?: number;
  /** Current share quantities per outcome */
  outcomeShares: Map<TrustOutcome, number>;
  /** Order book for CDA */
  orderBook?: {
    bids: TrustOrder[];
    asks: TrustOrder[];
  };
}

export type TrustOutcome = 'yes' | 'no' | 'partial';

export interface TrustOrder {
  id: string;
  participantId: string;
  outcome: TrustOutcome;
  price: number;
  quantity: number;
  timestamp: number;
  /** Participant's own reputation (affects order weight) */
  participantReputation: number;
}

/**
 * Current state of a trust market
 */
export interface TrustMarketState {
  status: 'open' | 'trading' | 'resolving' | 'resolved' | 'disputed';
  /** Current implied probabilities from market prices */
  impliedProbabilities: Map<TrustOutcome, number>;
  /** Total volume traded */
  totalVolume: number;
  /** Number of unique traders */
  uniqueTraders: number;
  /** Current best prices */
  bestBid?: number;
  bestAsk?: number;
  /** Market maker inventory */
  makerInventory: Map<TrustOutcome, number>;
  /** Historical price snapshots */
  priceHistory: Array<{
    timestamp: number;
    prices: Map<TrustOutcome, number>;
  }>;
}

/**
 * Resolution configuration
 */
export interface TrustResolutionConfig {
  /** How to resolve the market */
  mechanism: ResolutionMechanism;
  /** Minimum oracles required */
  minOracles: number;
  /** Minimum MATL score for oracles */
  minOracleMatl: number;
  /** Consensus threshold */
  consensusThreshold: number;
  /** Byzantine tolerance */
  byzantineTolerance: number;
  /** Resolution data sources */
  dataSources: TrustDataSource[];
}

export interface TrustDataSource {
  type: 'matl_score' | 'calibration_report' | 'cross_happ_query' | 'byzantine_flags';
  happId?: HappId;
  weight: number;
}

export interface TrustMarketMetadata {
  title: string;
  description: string;
  tags: string[];
  epistemicPosition: EpistemicPosition;
  /** Creator's stake (skin in the game) */
  creatorStake: number;
  /** Minimum stake to participate */
  minStake: number;
}

// =============================================================================
// TRUST POSITION & STAKES
// =============================================================================

/**
 * A participant's position in a trust market
 */
export interface TrustPosition {
  marketId: string;
  participantId: string;
  /** Shares held per outcome */
  shares: Map<TrustOutcome, number>;
  /** Total amount staked */
  totalStaked: number;
  /** Average entry prices */
  averageEntryPrices: Map<TrustOutcome, number>;
  /** Multi-dimensional stake breakdown */
  stakeBreakdown: MultiDimensionalTrustStake;
  /** P&L if resolved now */
  unrealizedPnL: number;
  /** Timestamp of last trade */
  lastTradeAt: number;
}

/**
 * Multi-dimensional stake for trust markets
 * Includes reputation stake - you risk your own trust when betting on others
 */
export interface MultiDimensionalTrustStake {
  /** Monetary stake (tokens, currency) */
  monetary: {
    amount: number;
    currency: string;
  };
  /** Reputation stake - % of own reputation at risk */
  reputation: {
    stakePercentage: number;
    currentReputation: number;
    atRisk: number;
  };
  /** Social stake - visibility of position */
  social: {
    visibility: 'anonymous' | 'pseudonymous' | 'public';
    linkedIdentity?: string;
  };
  /** Commitment stake - actions if wrong */
  commitment?: {
    ifWrong: CommitmentAction[];
  };
}

export interface CommitmentAction {
  type: 'investigation' | 'apology' | 'donation' | 'mentorship' | 'review';
  details: string;
  value: number; // Estimated value in base currency
}

// =============================================================================
// CALIBRATION FEEDBACK LOOP
// =============================================================================

/**
 * Links trust market outcomes to calibration engine
 * Market prices become calibration targets
 */
export interface CalibrationFeedbackLoop {
  /** Market ID */
  marketId: string;
  /** Calibration domain */
  domain: string;
  /** Pre-market confidence (stated) */
  preMarketConfidence: number;
  /** Market-implied confidence */
  marketImpliedConfidence: number;
  /** Post-resolution actual outcome */
  actualOutcome?: number;
  /** Calibration adjustment factor */
  adjustmentFactor?: number;
  /** Wisdom extracted from resolution */
  wisdom?: TrustWisdom;
}

export interface TrustWisdom {
  /** What we learned if market was correct */
  ifMarketCorrect: string;
  /** What we learned if market was wrong */
  ifMarketWrong: string;
  /** Meta-lesson about trust prediction */
  metaLesson: string;
  /** Activation conditions for future markets */
  activationConditions: WisdomActivation[];
}

export type WisdomActivation =
  | { type: 'similar_subject'; subjectPattern: string }
  | { type: 'similar_claim'; claimType: TrustClaimType }
  | { type: 'market_conditions'; conditions: string }
  | { type: 'time_based'; afterMs: number };

// =============================================================================
// TRUST MARKET SERVICE
// =============================================================================

/**
 * Global service statistics
 */
export interface TrustServiceStats {
  totalMarkets: number;
  activeMarkets: number;
  resolvedMarkets: number;
  totalVolume: number;
}

/**
 * Service for managing trust prediction markets
 */
export class TrustMarketService {
  private markets: Map<string, TrustMarket> = new Map();
  private positions: Map<string, TrustPosition[]> = new Map();
  private feedbackLoops: Map<string, CalibrationFeedbackLoop> = new Map();
  private calibrationEngine: CalibrationEngine;

  constructor(calibrationEngine?: CalibrationEngine) {
    this.calibrationEngine = calibrationEngine ?? new CalibrationEngine();
  }

  // ---------------------------------------------------------------------------
  // Global Statistics
  // ---------------------------------------------------------------------------

  /**
   * Get global service statistics
   */
  getStats(): TrustServiceStats {
    const allMarkets = Array.from(this.markets.values());
    const activeMarkets = allMarkets.filter(
      (m) => m.state.status === 'open' || m.state.status === 'trading'
    );
    const resolvedMarkets = allMarkets.filter(
      (m) => m.state.status === 'resolved'
    );
    const totalVolume = allMarkets.reduce(
      (sum, m) => sum + m.state.totalVolume,
      0
    );

    return {
      totalMarkets: allMarkets.length,
      activeMarkets: activeMarkets.length,
      resolvedMarkets: resolvedMarkets.length,
      totalVolume,
    };
  }

  /**
   * Get all active (open/trading) markets
   */
  getActiveMarkets(): TrustMarket[] {
    return this.listOpenMarkets();
  }

  /**
   * Get markets for a specific subject
   * @alias getMarketsBySubject
   */
  getMarketsForSubject(subjectDid: string): TrustMarket[] {
    return this.getMarketsBySubject(subjectDid);
  }

  /**
   * Get a participant's position in a specific market
   */
  getPosition(marketId: string, participantId: string): TrustPosition | undefined {
    const positions = this.positions.get(participantId) ?? [];
    return positions.find((p) => p.marketId === marketId);
  }

  // ---------------------------------------------------------------------------
  // Market Creation
  // ---------------------------------------------------------------------------

  /**
   * Create a new trust market
   *
   * @example
   * ```typescript
   * const market = await trustMarkets.createMarket({
   *   subjectDid: 'did:mycelix:alice',
   *   claim: {
   *     type: 'reputation_maintenance',
   *     threshold: 0.7,
   *     operator: 'gte',
   *     contextHapps: ['finance', 'governance', 'identity'],
   *     durationMs: 30 * 24 * 60 * 60 * 1000, // 30 days
   *   },
   *   title: "Will Alice's reputation stay >= 0.7?",
   *   creatorStake: 100,
   * });
   * ```
   */
  async createMarket(input: CreateTrustMarketInput): Promise<TrustMarket> {
    const id = `trust-market-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    // Determine epistemic position based on claim type
    const epistemicPosition = this.classifyClaimEpistemics(input.claim);

    // Select resolution mechanism based on epistemics
    const resolutionMechanism = getRecommendedResolution(epistemicPosition);

    // Handle legacy input formats
    const mechanismType = input.mechanismType ?? input.mechanism ?? 'LMSR';
    const description = input.description ?? input.metadata?.description ?? '';
    const tags = input.tags ?? input.metadata?.tags ?? [];
    const title = input.title ?? `Trust market for ${input.subjectDid}`;
    const creatorStake = input.creatorStake ?? 100;

    const market: TrustMarket = {
      id,
      subjectDid: input.subjectDid,
      claim: input.claim,
      mechanism: {
        type: mechanismType,
        liquidityParameter: input.liquidityParameter ?? 100,
        subsidyPool: input.subsidyPool ?? 1000,
        outcomeShares: new Map([
          ['yes', 0],
          ['no', 0],
          ['partial', 0],
        ]),
      },
      state: {
        status: 'open',
        impliedProbabilities: new Map([
          ['yes', 0.5],
          ['no', 0.5],
          ['partial', 0],
        ]),
        totalVolume: 0,
        uniqueTraders: 0,
        makerInventory: new Map(),
        priceHistory: [],
      },
      resolution: {
        mechanism: resolutionMechanism,
        minOracles: input.minOracles ?? 5,
        minOracleMatl: input.minOracleMatl ?? 0.7,
        consensusThreshold: 0.67,
        byzantineTolerance: 0.34,
        dataSources: this.getDefaultDataSources(input.claim),
      },
      metadata: {
        title,
        description,
        tags,
        epistemicPosition,
        creatorStake,
        minStake: input.minStake ?? 10,
      },
      createdAt: Date.now(),
    };

    this.markets.set(id, market);

    // Initialize calibration feedback loop
    this.initializeFeedbackLoop(market);

    return market;
  }

  /**
   * Classify epistemic position of a trust claim
   */
  private classifyClaimEpistemics(claim: TrustClaim): EpistemicPosition {
    // Trust claims are partially verifiable through on-chain data
    let empirical: EmpiricalLevel;
    let normative: NormativeLevel;
    let materiality: MaterialityLevel;

    switch (claim.type) {
      case 'reputation_maintenance':
      case 'reputation_improvement':
      case 'reputation_recovery':
        // Cryptographically verifiable through MATL scores
        empirical = EmpiricalLevel.Cryptographic;
        normative = NormativeLevel.Network;
        materiality = MaterialityLevel.Persistent;
        break;

      case 'cross_happ_consistency':
        // Requires cross-hApp verification
        empirical = EmpiricalLevel.PrivateVerify;
        normative = NormativeLevel.Network;
        materiality = MaterialityLevel.Temporal;
        break;

      case 'calibration_accuracy':
        // Measurable through calibration engine
        empirical = EmpiricalLevel.Measurable;
        normative = NormativeLevel.Communal;
        materiality = MaterialityLevel.Persistent;
        break;

      case 'byzantine_probability':
        // Testimonial + statistical
        empirical = EmpiricalLevel.Testimonial;
        normative = NormativeLevel.Network;
        materiality = MaterialityLevel.Temporal;
        break;

      case 'trust_convergence':
        // Network consensus
        empirical = EmpiricalLevel.Cryptographic;
        normative = NormativeLevel.Universal;
        materiality = MaterialityLevel.Foundational;
        break;

      default:
        empirical = EmpiricalLevel.Testimonial;
        normative = NormativeLevel.Communal;
        materiality = MaterialityLevel.Temporal;
    }

    return { empirical, normative, materiality };
  }

  /**
   * Get default data sources for resolution
   */
  private getDefaultDataSources(claim: TrustClaim): TrustDataSource[] {
    const sources: TrustDataSource[] = [
      { type: 'matl_score', weight: 0.4 },
    ];

    // Add cross-hApp queries for each context hApp
    for (const happId of claim.contextHapps) {
      sources.push({
        type: 'cross_happ_query',
        happId,
        weight: 0.3 / claim.contextHapps.length,
      });
    }

    if (claim.type === 'calibration_accuracy') {
      sources.push({ type: 'calibration_report', weight: 0.3 });
    }

    if (claim.type === 'byzantine_probability') {
      sources.push({ type: 'byzantine_flags', weight: 0.3 });
    }

    return sources;
  }

  /**
   * Initialize calibration feedback loop for market
   */
  private initializeFeedbackLoop(market: TrustMarket): void {
    const feedbackLoop: CalibrationFeedbackLoop = {
      marketId: market.id,
      domain: `trust:${market.claim.type}`,
      preMarketConfidence: 0.5, // Uninformed prior
      marketImpliedConfidence: 0.5,
    };

    this.feedbackLoops.set(market.id, feedbackLoop);
  }

  // ---------------------------------------------------------------------------
  // Trading Operations
  // ---------------------------------------------------------------------------

  /**
   * Submit a trade to a trust market
   *
   * @example
   * ```typescript
   * const trade = await trustMarkets.submitTrade({
   *   marketId: market.id,
   *   participantId: 'did:mycelix:bob',
   *   outcome: 'yes',
   *   shares: 10,
   *   stake: {
   *     monetary: { amount: 50, currency: 'MUSD' },
   *     reputation: { stakePercentage: 0.05 }, // 5% of own reputation
   *   },
   * });
   * ```
   */
  async submitTrade(input: SubmitTrustTradeInput): Promise<TrustTradeResult> {
    const market = this.markets.get(input.marketId);
    if (!market) {
      throw new Error(`Market not found: ${input.marketId}`);
    }

    if (market.state.status !== 'open' && market.state.status !== 'trading') {
      throw new Error(`Market not open for trading: ${market.state.status}`);
    }

    // Handle legacy traderId
    const participantId = input.participantId ?? input.traderId;
    if (!participantId) {
      throw new Error('participantId or traderId is required');
    }

    // Normalize stake format (support simplified format from tests)
    const normalizedStake = this.normalizeStake(input.stake);

    // Validate minimum stake
    const stakeValue = this.calculateStakeValue(normalizedStake);
    if (stakeValue < market.metadata.minStake) {
      throw new Error(`Stake ${stakeValue} below minimum ${market.metadata.minStake}`);
    }

    // Calculate trade price based on mechanism
    const price = this.calculatePrice(market, input.outcome, input.shares);

    // Update market state
    const currentShares = market.mechanism.outcomeShares.get(input.outcome) ?? 0;
    market.mechanism.outcomeShares.set(input.outcome, currentShares + input.shares);

    // Update implied probabilities (LMSR formula)
    this.updateImpliedProbabilities(market);

    // Update volume and traders
    market.state.totalVolume += stakeValue;
    market.state.uniqueTraders++; // Simplified - should track unique

    // Record price snapshot
    market.state.priceHistory.push({
      timestamp: Date.now(),
      prices: new Map(market.state.impliedProbabilities),
    });

    // Update calibration feedback loop with new market-implied confidence
    const feedbackLoop = this.feedbackLoops.get(market.id);
    if (feedbackLoop) {
      feedbackLoop.marketImpliedConfidence =
        market.state.impliedProbabilities.get('yes') ?? 0.5;
    }

    // Update or create position
    this.updatePosition(market.id, participantId, {
      outcome: input.outcome,
      shares: input.shares,
      stake: normalizedStake,
    });

    // Set market to trading status
    market.state.status = 'trading';

    return {
      success: true,
      tradeId: `trade-${Date.now()}`,
      executedPrice: price,
      shares: input.shares,
      totalCost: price * input.shares,
      newImpliedProbability: market.state.impliedProbabilities.get(input.outcome) ?? 0,
      newShares: input.shares, // Added for test compatibility
    };
  }

  /**
   * Calculate price using LMSR (Logarithmic Market Scoring Rule)
   */
  private calculatePrice(
    market: TrustMarket,
    outcome: TrustOutcome,
    shares: number
  ): number {
    if (market.mechanism.type !== 'LMSR') {
      // Simplified: return current implied probability
      return market.state.impliedProbabilities.get(outcome) ?? 0.5;
    }

    const b = market.mechanism.liquidityParameter ?? 100;
    const q = market.mechanism.outcomeShares;

    // LMSR cost function: C(q) = b * log(sum(exp(q_i / b)))
    const sumExpBefore = Array.from(q.values()).reduce(
      (sum, qty) => sum + Math.exp(qty / b),
      0
    );

    // Cost after trade
    const qAfter = new Map(q);
    qAfter.set(outcome, (qAfter.get(outcome) ?? 0) + shares);
    const sumExpAfter = Array.from(qAfter.values()).reduce(
      (sum, qty) => sum + Math.exp(qty / b),
      0
    );

    const costBefore = b * Math.log(sumExpBefore);
    const costAfter = b * Math.log(sumExpAfter);

    return (costAfter - costBefore) / shares;
  }

  /**
   * Update implied probabilities from share quantities
   */
  private updateImpliedProbabilities(market: TrustMarket): void {
    if (market.mechanism.type !== 'LMSR') return;

    const b = market.mechanism.liquidityParameter ?? 100;
    const q = market.mechanism.outcomeShares;

    // LMSR probability: p_i = exp(q_i / b) / sum(exp(q_j / b))
    const sumExp = Array.from(q.values()).reduce(
      (sum, qty) => sum + Math.exp(qty / b),
      0
    );

    for (const [outcome, qty] of q.entries()) {
      const prob = Math.exp(qty / b) / sumExp;
      market.state.impliedProbabilities.set(outcome, prob);
    }
  }

  /**
   * Calculate total stake value including reputation risk
   */
  private calculateStakeValue(stake: Partial<MultiDimensionalTrustStake>): number {
    let total = 0;

    if (stake.monetary) {
      total += stake.monetary.amount;
    }

    if (stake.reputation) {
      // Reputation stake: 1% of reputation ≈ 100 units
      const repValue = stake.reputation.stakePercentage * 10000;
      total += repValue;
    }

    if (stake.social?.visibility === 'public') {
      total += 500; // Public visibility premium
    }

    if (stake.commitment?.ifWrong) {
      for (const action of stake.commitment.ifWrong) {
        total += action.value;
      }
    }

    return total;
  }

  /**
   * Normalize stake from simplified format to full format
   */
  private normalizeStake(
    stake: Partial<MultiDimensionalTrustStake> | SimplifiedStake
  ): Partial<MultiDimensionalTrustStake> {
    // Check if it's already in the full format
    if (stake.monetary && typeof stake.monetary === 'object' && 'amount' in stake.monetary) {
      return stake as Partial<MultiDimensionalTrustStake>;
    }

    // Convert from simplified format
    const simplified = stake as SimplifiedStake;
    const normalized: Partial<MultiDimensionalTrustStake> = {};

    if (simplified.monetary !== undefined) {
      normalized.monetary = {
        amount: simplified.monetary,
        currency: 'MUSD',
      };
    }

    if (simplified.reputation !== undefined) {
      normalized.reputation = {
        stakePercentage: simplified.reputation,
        currentReputation: 0.8,
        atRisk: simplified.reputation * 0.8,
      };
    }

    return normalized;
  }

  /**
   * Update participant position
   */
  private updatePosition(
    marketId: string,
    participantId: string,
    trade: { outcome: TrustOutcome; shares: number; stake: Partial<MultiDimensionalTrustStake> }
  ): void {
    const positions = this.positions.get(participantId) ?? [];

    let position = positions.find((p) => p.marketId === marketId);
    if (!position) {
      position = {
        marketId,
        participantId,
        shares: new Map(),
        totalStaked: 0,
        averageEntryPrices: new Map(),
        stakeBreakdown: {
          monetary: { amount: 0, currency: 'MUSD' },
          reputation: { stakePercentage: 0, currentReputation: 0.5, atRisk: 0 },
          social: { visibility: 'anonymous' },
        },
        unrealizedPnL: 0,
        lastTradeAt: Date.now(),
      };
      positions.push(position);
    }

    // Update shares
    const currentShares = position.shares.get(trade.outcome) ?? 0;
    position.shares.set(trade.outcome, currentShares + trade.shares);

    // Update stake breakdown
    if (trade.stake.monetary) {
      position.stakeBreakdown.monetary.amount += trade.stake.monetary.amount;
    }
    if (trade.stake.reputation) {
      position.stakeBreakdown.reputation.stakePercentage +=
        trade.stake.reputation.stakePercentage;
    }

    position.totalStaked += this.calculateStakeValue(trade.stake);
    position.lastTradeAt = Date.now();

    this.positions.set(participantId, positions);
  }

  // ---------------------------------------------------------------------------
  // Resolution & Calibration
  // ---------------------------------------------------------------------------

  /**
   * Resolve a trust market based on actual outcomes
   *
   * @example
   * ```typescript
   * // With explicit outcome
   * const resolution = await trustMarkets.resolveMarket(market.id, 'yes', {
   *   actualReputation: 0.75,
   * });
   *
   * // Auto-determine outcome from data
   * const resolution = await trustMarkets.resolveMarket(market.id, {
   *   actualReputation: 0.75,
   *   oracleVotes: [...],
   * });
   * ```
   */
  async resolveMarket(
    marketId: string,
    outcomeOrData: TrustOutcome | TrustResolutionData,
    resolutionDataArg?: TrustResolutionData
  ): Promise<TrustMarketResolution> {
    const market = this.markets.get(marketId);
    if (!market) {
      throw new Error(`Market not found: ${marketId}`);
    }

    market.state.status = 'resolving';

    // Handle both call signatures:
    // resolveMarket(id, outcome, data) - 3 args
    // resolveMarket(id, data) - 2 args
    let outcome: TrustOutcome;
    let resolutionData: TrustResolutionData;

    if (typeof outcomeOrData === 'string') {
      // 3-arg form: outcome is explicit
      outcome = outcomeOrData;
      resolutionData = resolutionDataArg ?? {};
    } else {
      // 2-arg form: determine outcome from data
      resolutionData = outcomeOrData;
      outcome = this.determineOutcome(market, resolutionData);
    }

    // Calculate payouts
    const payouts = this.calculatePayouts(market, outcome);

    // Update calibration feedback loop
    await this.updateCalibrationFeedback(market, outcome, resolutionData);

    // Extract wisdom from resolution
    const wisdom = this.extractWisdom(market, outcome, resolutionData);

    // Update market state
    market.state.status = 'resolved';
    market.resolvedAt = Date.now();

    return {
      marketId,
      outcome,
      payouts,
      calibrationUpdate: this.feedbackLoops.get(marketId),
      wisdom,
      resolutionConfidence: this.calculateResolutionConfidence(resolutionData),
    };
  }

  /**
   * Determine outcome based on resolution data
   */
  private determineOutcome(
    market: TrustMarket,
    data: TrustResolutionData
  ): TrustOutcome {
    const { claim } = market;
    const actualValue = data.actualReputation ?? data.aggregatedScore ?? 0;

    switch (claim.operator) {
      case 'gt':
        return actualValue > claim.threshold ? 'yes' : 'no';
      case 'gte':
        return actualValue >= claim.threshold ? 'yes' : 'no';
      case 'lt':
        return actualValue < claim.threshold ? 'yes' : 'no';
      case 'lte':
        return actualValue <= claim.threshold ? 'yes' : 'no';
      case 'eq':
        return Math.abs(actualValue - claim.threshold) < 0.01 ? 'yes' : 'no';
      case 'between':
        const upper = claim.upperThreshold ?? 1;
        if (actualValue >= claim.threshold && actualValue <= upper) {
          return 'yes';
        } else if (
          actualValue >= claim.threshold - 0.1 ||
          actualValue <= upper + 0.1
        ) {
          return 'partial';
        }
        return 'no';
      default:
        return 'no';
    }
  }

  /**
   * Calculate payouts for all positions
   */
  private calculatePayouts(
    market: TrustMarket,
    outcome: TrustOutcome
  ): Map<string, TrustPayout> {
    const payouts = new Map<string, TrustPayout>();

    for (const [participantId, positions] of this.positions.entries()) {
      const position = positions.find((p) => p.marketId === market.id);
      if (!position) continue;

      const winningShares = position.shares.get(outcome) ?? 0;
      const totalShares = Array.from(position.shares.values()).reduce(
        (sum, s) => sum + s,
        0
      );

      // Calculate monetary payout
      const monetaryPayout =
        winningShares > 0 ? (position.totalStaked * winningShares) / totalShares : 0;

      // Calculate reputation adjustment
      const wasCorrect = winningShares > 0;
      const reputationAdjustment = wasCorrect
        ? position.stakeBreakdown.reputation.stakePercentage * 0.5 // Reward: gain half of stake
        : -position.stakeBreakdown.reputation.stakePercentage; // Penalty: lose full stake

      payouts.set(participantId, {
        participantId,
        monetaryPayout,
        reputationAdjustment,
        wasCorrect,
        sharesHeld: totalShares,
        winningShares,
      });
    }

    return payouts;
  }

  /**
   * Update calibration engine with market outcome
   *
   * This is the key innovation: market prices become calibration targets
   */
  private async updateCalibrationFeedback(
    market: TrustMarket,
    outcome: TrustOutcome,
    data: TrustResolutionData
  ): Promise<void> {
    const feedbackLoop = this.feedbackLoops.get(market.id);
    if (!feedbackLoop) return;

    // Record the resolution in calibration engine
    const marketConfidence = feedbackLoop.marketImpliedConfidence;
    const actualOutcome = outcome === 'yes' ? 1 : outcome === 'partial' ? 0.5 : 0;

    feedbackLoop.actualOutcome = actualOutcome;

    // Calculate adjustment factor
    // If market said 0.8 and outcome was 1.0, adjustment = 1.0/0.8 = 1.25 (underconfident)
    // If market said 0.8 and outcome was 0.0, adjustment = 0.0/0.8 = 0.0 (overconfident)
    const adjustmentFactor =
      marketConfidence > 0.01 ? actualOutcome / marketConfidence : 1.0;

    feedbackLoop.adjustmentFactor = adjustmentFactor;

    // Record in calibration engine for domain
    await this.calibrationEngine.recordResolution(
      {
        id: market.id,
        confidence: marketConfidence,
        domain: feedbackLoop.domain,
      },
      actualOutcome > 0.5 ? 'correct' : 'incorrect',
      [
        {
          type: 'trust_market',
          reference: `trust-market:${market.id}`,
          description: `Market implied ${marketConfidence}, actual ${data.actualReputation}. ${market.state.uniqueTraders} traders, volume ${market.state.totalVolume}`,
        },
      ]
    );
  }

  /**
   * Extract wisdom from market resolution
   */
  private extractWisdom(
    market: TrustMarket,
    outcome: TrustOutcome,
    data: TrustResolutionData
  ): TrustWisdom {
    const marketConfidence =
      this.feedbackLoops.get(market.id)?.marketImpliedConfidence ?? 0.5;
    const wasCorrect = outcome === 'yes';
    const wasOverconfident = marketConfidence > 0.7 && !wasCorrect;
    const wasUnderconfident = marketConfidence < 0.3 && wasCorrect;

    let metaLesson: string;
    if (wasOverconfident) {
      metaLesson = `Market was overconfident (${(marketConfidence * 100).toFixed(1)}% vs actual ${outcome}). Consider: Were there warning signs ignored? Were stakes sufficient to reveal true beliefs?`;
    } else if (wasUnderconfident) {
      metaLesson = `Market was underconfident (${(marketConfidence * 100).toFixed(1)}% vs actual ${outcome}). Consider: Was relevant information not reaching the market?`;
    } else {
      metaLesson = `Market was well-calibrated for ${market.claim.type} claims on ${market.subjectDid.slice(0, 20)}...`;
    }

    return {
      ifMarketCorrect: wasCorrect
        ? `Trust prediction validated: ${market.claim.type} claims can be accurately predicted with ${market.state.uniqueTraders} participants`
        : `Market participants overestimated trust stability - need more diverse information sources`,
      ifMarketWrong: !wasCorrect
        ? `Unexpected ${outcome}: Actual reputation ${data.actualReputation?.toFixed(2)} vs threshold ${market.claim.threshold}`
        : `Market correctly predicted outcome despite low confidence`,
      metaLesson,
      activationConditions: [
        { type: 'similar_subject', subjectPattern: market.subjectDid.slice(0, 15) },
        { type: 'similar_claim', claimType: market.claim.type },
      ],
    };
  }

  /**
   * Calculate confidence in resolution
   */
  private calculateResolutionConfidence(data: TrustResolutionData): number {
    let confidence = 0.5;

    // More oracle votes = higher confidence
    if (data.oracleVotes && data.oracleVotes.length >= 5) {
      const agreementRate =
        data.oracleVotes.filter((v) => v.outcome === data.oracleVotes![0].outcome)
          .length / data.oracleVotes.length;
      confidence = Math.max(confidence, agreementRate);
    }

    // Multiple data sources = higher confidence
    if (data.crossHappScores && Object.keys(data.crossHappScores).length >= 3) {
      confidence = Math.min(1, confidence + 0.1);
    }

    return confidence;
  }

  // ---------------------------------------------------------------------------
  // Query Methods
  // ---------------------------------------------------------------------------

  getMarket(id: string): TrustMarket | undefined {
    return this.markets.get(id);
  }

  listOpenMarkets(): TrustMarket[] {
    return Array.from(this.markets.values()).filter(
      (m) => m.state.status === 'open' || m.state.status === 'trading'
    );
  }

  getMarketsBySubject(subjectDid: string): TrustMarket[] {
    return Array.from(this.markets.values()).filter(
      (m) => m.subjectDid === subjectDid
    );
  }

  getPositions(participantId: string): TrustPosition[] {
    return this.positions.get(participantId) ?? [];
  }

  getCalibrationFeedback(marketId: string): CalibrationFeedbackLoop | undefined {
    return this.feedbackLoops.get(marketId);
  }

  /**
   * Get market statistics for analytics
   */
  getMarketStats(marketId: string): TrustMarketStats | undefined {
    const market = this.markets.get(marketId);
    if (!market) return undefined;

    const priceHistory = market.state.priceHistory;
    const yesHistory = priceHistory.map((p) => p.prices.get('yes') ?? 0.5);

    return {
      marketId,
      totalVolume: market.state.totalVolume,
      uniqueTraders: market.state.uniqueTraders,
      currentYesPrice: market.state.impliedProbabilities.get('yes') ?? 0.5,
      priceVolatility: this.calculateVolatility(yesHistory),
      avgTradeSize: market.state.totalVolume / Math.max(1, priceHistory.length),
      durationRemaining:
        market.claim.durationMs - (Date.now() - market.createdAt),
      feedbackLoopActive: this.feedbackLoops.has(marketId),
    };
  }

  private calculateVolatility(prices: number[]): number {
    if (prices.length < 2) return 0;
    const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
    const variance =
      prices.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / prices.length;
    return Math.sqrt(variance);
  }
}

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

export interface CreateTrustMarketInput {
  subjectDid: string;
  claim: TrustClaim;
  title?: string;
  description?: string;
  tags?: string[];
  creatorStake?: number;
  minStake?: number;
  /** @deprecated Use mechanismType */
  mechanism?: 'LMSR' | 'CDA' | 'Parimutuel' | 'TrustWeighted';
  mechanismType?: 'LMSR' | 'CDA' | 'Parimutuel' | 'TrustWeighted';
  liquidityParameter?: number;
  subsidyPool?: number;
  minOracles?: number;
  minOracleMatl?: number;
  /** Legacy resolution config (ignored, uses epistemic-based resolution) */
  resolutionConfig?: unknown;
  /** Legacy metadata format */
  metadata?: {
    description?: string;
    tags?: string[];
  };
}

/**
 * Simplified stake format (for test compatibility)
 */
export interface SimplifiedStake {
  monetary?: number;
  reputation?: number;
  commitment?: unknown[];
}

export interface SubmitTrustTradeInput {
  marketId: string;
  /** @deprecated Use participantId */
  traderId?: string;
  participantId?: string;
  outcome: TrustOutcome;
  shares: number;
  stake: Partial<MultiDimensionalTrustStake> | SimplifiedStake;
}

export interface TrustTradeResult {
  success: boolean;
  tradeId: string;
  executedPrice: number;
  shares: number;
  totalCost: number;
  newImpliedProbability: number;
  /** Alias for shares (test compatibility) */
  newShares: number;
}

export interface TrustResolutionData {
  actualReputation?: number;
  aggregatedScore?: number;
  crossHappScores?: Record<HappId, number>;
  oracleVotes?: Array<{
    oracleId: string;
    outcome: TrustOutcome;
    confidence: number;
    matlScore: number;
  }>;
  calibrationReport?: CalibrationReport;
  byzantineFlags?: number;
  /** Legacy fields for test compatibility */
  dataPoints?: number;
  sources?: string[];
  timestamp?: number;
}

export interface TrustPayout {
  participantId: string;
  monetaryPayout: number;
  reputationAdjustment: number;
  wasCorrect: boolean;
  sharesHeld: number;
  winningShares: number;
}

export interface TrustMarketResolution {
  marketId: string;
  outcome: TrustOutcome;
  payouts: Map<string, TrustPayout>;
  calibrationUpdate?: CalibrationFeedbackLoop;
  wisdom: TrustWisdom;
  resolutionConfidence: number;
}

export interface TrustMarketStats {
  marketId: string;
  totalVolume: number;
  uniqueTraders: number;
  currentYesPrice: number;
  priceVolatility: number;
  avgTradeSize: number;
  durationRemaining: number;
  feedbackLoopActive: boolean;
}

// =============================================================================
// EXPORTS
// =============================================================================

export { CalibrationEngine };
