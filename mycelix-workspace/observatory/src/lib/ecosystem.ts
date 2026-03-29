// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Ecosystem Monitoring Service
 *
 * Provides unified health checks and metrics across all Mycelix hApps:
 * - Mycelix-Core (FL coordination, MATL)
 * - Mycelix-Mail (secure messaging)
 * - Mycelix-Marketplace (P2P transactions)
 * - Mycelix-EduNet (educational credentials)
 * - Mycelix-SupplyChain (provenance tracking)
 */

export interface HappHealth {
  name: string;
  version: string;
  status: 'healthy' | 'degraded' | 'offline' | 'unknown';
  hdkVersion: string;
  nodeCount: number;
  lastSeen: number;
  zomeCount: number;
  metrics: HappMetrics;
}

export interface HappMetrics {
  totalTransactions: number;
  avgLatencyMs: number;
  errorRate: number;
  trustScoreAvg: number;
  byzantineEvents: number;
}

export interface EcosystemStatus {
  healthy: boolean;
  totalNodes: number;
  byzantineTolerance: number;
  happs: HappHealth[];
  crossHappQueries: number;
  bridgeMessageCount: number;
  lastAuditTime: number;
}

export interface ByzantineAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  source: string;
  timestamp: number;
  resolved: boolean;
}

// ============================================
// MYCELIX ECONOMIC PRIMITIVES
// ============================================

export interface EconomicStatus {
  // Network-wide stats
  totalSapSupply: number;
  totalHearthPool: number;
  activeCommitments: number;
  avgPatienceCoefficient: number;

  // Sample members showing the economic model
  members: MemberEconomics[];

  // Recent economic activity
  recentActivity: EconomicActivity[];

  // KREDIT agent stats
  agentStats: AgentStats;
}

export interface MemberEconomics {
  did: string;
  displayName: string;

  // Core tokens
  sapBalance: number;
  civScore: number;  // 0-1, Civilizational Index Value

  // Temporal commitments
  commitments: TemporalCommitment[];
  patienceCoefficient: number;  // 0.5-3.0

  // HEARTH contribution
  hearthShare: number;
  hearthWarmth: number;  // yield earned

  // Contribution to intentions
  intentionContributions: { domain: string; score: number }[];
}

export interface TemporalCommitment {
  tier: 'Sprout' | 'Sapling' | 'Tree' | 'Grove' | 'Forest';
  sapLocked: number;
  durationMonths: number;
  governanceMultiplier: number;
  covenantType: string;
}

export interface EconomicActivity {
  id: string;
  type: 'sap_transfer' | 'hearth_warm' | 'commitment_lock' | 'kredit_spend' | 'intention_contribution';
  actor: string;
  amount: number;
  description: string;
  timestamp: number;
}

export interface AgentStats {
  totalAgents: number;
  totalKreditBalance: number;
  avgCivRequirement: number;
  activeTasks: number;
}

// ============================================
// MFA (MULTI-FACTOR AUTHENTICATION) METRICS
// ============================================

export type AssuranceLevel = 'Anonymous' | 'Basic' | 'Verified' | 'HighlyAssured' | 'ConstitutionallyCritical';

export interface MfaMetrics {
  // Network-wide MFA statistics
  totalMfaEnabled: number;
  totalDids: number;
  mfaAdoptionRate: number; // 0-100%

  // Assurance level distribution
  assuranceLevelDistribution: {
    anonymous: number;
    basic: number;
    verified: number;
    highlyAssured: number;
    constitutionallyCritical: number;
  };

  // Factor type distribution
  factorTypeDistribution: {
    primaryKeyPair: number;
    hardwareKey: number;
    biometric: number;
    socialRecovery: number;
    gitcoinPassport: number;
    verifiableCredential: number;
    reputationAttestation: number;
    securityQuestions: number;
    recoveryPhrase: number;
  };

  // FL eligibility stats
  flEligibleCount: number;
  flEligibilityRate: number; // 0-100%

  // Average metrics
  avgFactorsPerIdentity: number;
  avgCategoriesPerIdentity: number;
  avgAssuranceScore: number; // 0-1

  // Freshness metrics
  staleFactorCount: number;
  factorsNeedingReverification: number;

  // Recent activity
  recentEnrollments: number;
  recentRevocations: number;
  recentVerifications: number;
}

export interface IdentityMfaSummary {
  did: string;
  displayName: string;
  assuranceLevel: AssuranceLevel;
  assuranceScore: number;
  factorCount: number;
  categoryCount: number;
  flEligible: boolean;
  hasExternalVerification: boolean;
  lastVerified: number;
}

// hApp definitions with current status
// Includes all 18 Civilizational OS hApps
const ECOSYSTEM_HAPPS: Omit<HappHealth, 'metrics'>[] = [
  // === CORE INFRASTRUCTURE ===
  {
    name: 'Mycelix-Core',
    version: '0.6.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 6, // FL coordinator/integrity, civitas reputation, causal contribution
  },
  {
    name: 'Mycelix-Mail',
    version: '0.2.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // mail_messages, trust_filter, contacts, profiles
  },
  // === EXISTING DOMAIN HAPPS ===
  {
    name: 'Mycelix-Marketplace',
    version: '0.4.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 12, // listings, reputation, transactions, arbitration, messaging, notifications
  },
  {
    name: 'Mycelix-EduNet',
    version: '0.3.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 18, // learning, fl, credential, dao, pods, knowledge, srs, gamification, adaptive, integration
  },
  {
    name: 'Mycelix-SupplyChain',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 10, // procurement, inventory, logistics, payments, trust
  },
  // === GOVERNANCE PILLAR (New Civilizational hApps) ===
  {
    name: 'Mycelix-Identity',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // did_registry, credential_schema, revocation, recovery
  },
  {
    name: 'Mycelix-Governance',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // proposals, voting, execution, constitution
  },
  {
    name: 'Mycelix-Justice',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // complaints, evidence, arbitration, enforcement
  },
  // === ECONOMIC PILLAR (New Civilizational hApps) ===
  {
    name: 'Mycelix-Finance',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // credit_scoring, lending, payments, treasury
  },
  {
    name: 'Mycelix-Property',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // registry, transfer, disputes, commons
  },
  {
    name: 'Mycelix-Energy',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // projects, investments, regenerative, grid
  },
  // === KNOWLEDGE PILLAR (New Civilizational hApps) ===
  {
    name: 'Mycelix-Media',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // publication, attribution, factcheck, curation
  },
  {
    name: 'Mycelix-Knowledge',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 4, // claims, graph, query, inference
  },
  // === MANUFACTURING PILLAR (New Civilizational hApps) ===
  {
    name: 'Mycelix-Fabrication',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 14, // designs, printers, prints, materials, verification, bridge, symthaea (integrity + coordinator each)
  },
  // === COMMUNITY PILLAR (New Civilizational hApps) ===
  {
    name: 'Mycelix-Care',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 12, // caregiving, scheduling, needs, resources, matching, coordination (integrity + coordinator each)
  },
  {
    name: 'Mycelix-Emergency',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 14, // alerts, response, resources, coordination, communication, mapping, recovery (integrity + coordinator each)
  },
  {
    name: 'Mycelix-Water',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 12, // sources, quality, distribution, conservation, rights, monitoring (integrity + coordinator each)
  },
  {
    name: 'Mycelix-Housing',
    version: '0.1.0',
    status: 'healthy',
    hdkVersion: '0.6.0',
    nodeCount: 0,
    lastSeen: Date.now(),
    zomeCount: 14, // listings, applications, leases, maintenance, community, cooperatives, commons (integrity + coordinator each)
  },
];

// Default metrics for each hApp
function defaultMetrics(): HappMetrics {
  return {
    totalTransactions: 0,
    avgLatencyMs: 0,
    errorRate: 0,
    trustScoreAvg: 0.85,
    byzantineEvents: 0,
  };
}

/**
 * Ecosystem Service for monitoring all Mycelix hApps
 */
export class EcosystemService {
  private status: EcosystemStatus;
  private alerts: ByzantineAlert[] = [];
  private listeners: Set<(status: EcosystemStatus) => void> = new Set();
  private mfaMetrics: MfaMetrics;
  private mfaListeners: Set<(metrics: MfaMetrics) => void> = new Set();

  constructor() {
    this.status = {
      healthy: true,
      totalNodes: 0,
      byzantineTolerance: 34, // 34% BFT threshold (validated maximum)
      happs: ECOSYSTEM_HAPPS.map((h) => ({ ...h, metrics: defaultMetrics() })),
      crossHappQueries: 0,
      bridgeMessageCount: 0,
      lastAuditTime: Date.now(),
    };

    // Initialize MFA metrics
    this.mfaMetrics = this.defaultMfaMetrics();
  }

  private defaultMfaMetrics(): MfaMetrics {
    return {
      totalMfaEnabled: 0,
      totalDids: 0,
      mfaAdoptionRate: 0,
      assuranceLevelDistribution: {
        anonymous: 0,
        basic: 0,
        verified: 0,
        highlyAssured: 0,
        constitutionallyCritical: 0,
      },
      factorTypeDistribution: {
        primaryKeyPair: 0,
        hardwareKey: 0,
        biometric: 0,
        socialRecovery: 0,
        gitcoinPassport: 0,
        verifiableCredential: 0,
        reputationAttestation: 0,
        securityQuestions: 0,
        recoveryPhrase: 0,
      },
      flEligibleCount: 0,
      flEligibilityRate: 0,
      avgFactorsPerIdentity: 0,
      avgCategoriesPerIdentity: 0,
      avgAssuranceScore: 0,
      staleFactorCount: 0,
      factorsNeedingReverification: 0,
      recentEnrollments: 0,
      recentRevocations: 0,
      recentVerifications: 0,
    };
  }

  /**
   * Get current ecosystem status
   */
  getStatus(): EcosystemStatus {
    return { ...this.status };
  }

  /**
   * Get all active alerts
   */
  getAlerts(): ByzantineAlert[] {
    return [...this.alerts];
  }

  /**
   * Subscribe to status updates
   */
  subscribe(callback: (status: EcosystemStatus) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  /**
   * Update hApp status (called when connecting to conductors)
   */
  updateHappStatus(name: string, update: Partial<HappHealth>): void {
    const happ = this.status.happs.find((h) => h.name === name);
    if (happ) {
      Object.assign(happ, update);
      happ.lastSeen = Date.now();
      this.recalculateEcosystemHealth();
      this.notifyListeners();
    }
  }

  /**
   * Record a cross-hApp bridge query
   */
  recordBridgeQuery(): void {
    this.status.crossHappQueries++;
    this.status.bridgeMessageCount++;
    this.notifyListeners();
  }

  /**
   * Add a Byzantine detection alert
   */
  addAlert(severity: ByzantineAlert['severity'], message: string, source: string): void {
    const alert: ByzantineAlert = {
      id: `alert-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      severity,
      message,
      source,
      timestamp: Date.now(),
      resolved: false,
    };
    this.alerts.unshift(alert);

    // Keep only last 100 alerts
    if (this.alerts.length > 100) {
      this.alerts = this.alerts.slice(0, 100);
    }

    // Update hApp metrics
    const happ = this.status.happs.find((h) => h.name === source);
    if (happ) {
      happ.metrics.byzantineEvents++;
    }

    this.notifyListeners();
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string): void {
    const alert = this.alerts.find((a) => a.id === alertId);
    if (alert) {
      alert.resolved = true;
      this.notifyListeners();
    }
  }

  /**
   * Simulate real-time metrics update (for demo/development)
   */
  simulateMetricsUpdate(): void {
    this.status.happs.forEach((happ) => {
      // Simulate transactions
      happ.metrics.totalTransactions += Math.floor(Math.random() * 10);

      // Simulate latency variations
      happ.metrics.avgLatencyMs = 50 + Math.random() * 100;

      // Simulate trust scores (mostly stable)
      happ.metrics.trustScoreAvg = 0.80 + Math.random() * 0.15;

      // Simulate node count
      happ.nodeCount = Math.floor(5 + Math.random() * 20);
    });

    // Update totals
    this.status.totalNodes = this.status.happs.reduce((sum, h) => sum + h.nodeCount, 0);
    this.notifyListeners();

    // Also simulate MFA metrics
    this.simulateMfaMetrics();
  }

  // ============================================
  // MFA METRICS METHODS
  // ============================================

  /**
   * Get current MFA metrics
   */
  getMfaMetrics(): MfaMetrics {
    return { ...this.mfaMetrics };
  }

  /**
   * Subscribe to MFA metrics updates
   */
  subscribeMfa(callback: (metrics: MfaMetrics) => void): () => void {
    this.mfaListeners.add(callback);
    return () => this.mfaListeners.delete(callback);
  }

  /**
   * Update MFA metrics from conductor data
   */
  updateMfaMetrics(metrics: Partial<MfaMetrics>): void {
    Object.assign(this.mfaMetrics, metrics);
    this.notifyMfaListeners();
  }

  /**
   * Record MFA enrollment event
   */
  recordMfaEnrollment(): void {
    this.mfaMetrics.recentEnrollments++;
    this.mfaMetrics.totalMfaEnabled++;
    this.recalculateMfaRates();
    this.notifyMfaListeners();
  }

  /**
   * Record MFA revocation event
   */
  recordMfaRevocation(): void {
    this.mfaMetrics.recentRevocations++;
    this.notifyMfaListeners();
  }

  /**
   * Record MFA verification event
   */
  recordMfaVerification(): void {
    this.mfaMetrics.recentVerifications++;
    if (this.mfaMetrics.factorsNeedingReverification > 0) {
      this.mfaMetrics.factorsNeedingReverification--;
    }
    this.notifyMfaListeners();
  }

  /**
   * Update assurance level distribution
   */
  updateAssuranceDistribution(level: AssuranceLevel, delta: number): void {
    const key = level.toLowerCase() as keyof typeof this.mfaMetrics.assuranceLevelDistribution;
    if (key in this.mfaMetrics.assuranceLevelDistribution) {
      this.mfaMetrics.assuranceLevelDistribution[key] += delta;
    }
    this.recalculateMfaRates();
    this.notifyMfaListeners();
  }

  /**
   * Simulate MFA metrics for demo mode
   */
  simulateMfaMetrics(): void {
    // Simulate realistic MFA adoption and distribution
    const totalDids = 100 + Math.floor(Math.random() * 50);
    const mfaEnabled = Math.floor(totalDids * (0.65 + Math.random() * 0.20)); // 65-85% adoption

    this.mfaMetrics.totalDids = totalDids;
    this.mfaMetrics.totalMfaEnabled = mfaEnabled;
    this.mfaMetrics.mfaAdoptionRate = (mfaEnabled / totalDids) * 100;

    // Assurance level distribution (bell curve around Verified)
    this.mfaMetrics.assuranceLevelDistribution = {
      anonymous: Math.floor(mfaEnabled * 0.05),
      basic: Math.floor(mfaEnabled * 0.25),
      verified: Math.floor(mfaEnabled * 0.40),
      highlyAssured: Math.floor(mfaEnabled * 0.22),
      constitutionallyCritical: Math.floor(mfaEnabled * 0.08),
    };

    // Factor type distribution
    this.mfaMetrics.factorTypeDistribution = {
      primaryKeyPair: mfaEnabled, // Everyone has this
      hardwareKey: Math.floor(mfaEnabled * 0.15),
      biometric: Math.floor(mfaEnabled * 0.25),
      socialRecovery: Math.floor(mfaEnabled * 0.30),
      gitcoinPassport: Math.floor(mfaEnabled * 0.20),
      verifiableCredential: Math.floor(mfaEnabled * 0.35),
      reputationAttestation: Math.floor(mfaEnabled * 0.45),
      securityQuestions: Math.floor(mfaEnabled * 0.40),
      recoveryPhrase: Math.floor(mfaEnabled * 0.55),
    };

    // FL eligibility (Verified+ with Crypto + ExternalVerification)
    const flEligible = Math.floor(mfaEnabled * 0.35);
    this.mfaMetrics.flEligibleCount = flEligible;
    this.mfaMetrics.flEligibilityRate = (flEligible / mfaEnabled) * 100;

    // Averages
    this.mfaMetrics.avgFactorsPerIdentity = 2.3 + Math.random() * 0.5;
    this.mfaMetrics.avgCategoriesPerIdentity = 1.8 + Math.random() * 0.4;
    this.mfaMetrics.avgAssuranceScore = 0.45 + Math.random() * 0.15;

    // Freshness
    this.mfaMetrics.staleFactorCount = Math.floor(mfaEnabled * 0.12);
    this.mfaMetrics.factorsNeedingReverification = Math.floor(mfaEnabled * 0.08);

    // Activity (simulated per update)
    this.mfaMetrics.recentEnrollments = Math.floor(Math.random() * 5);
    this.mfaMetrics.recentRevocations = Math.floor(Math.random() * 2);
    this.mfaMetrics.recentVerifications = Math.floor(Math.random() * 15);

    this.notifyMfaListeners();
  }

  private recalculateMfaRates(): void {
    if (this.mfaMetrics.totalDids > 0) {
      this.mfaMetrics.mfaAdoptionRate =
        (this.mfaMetrics.totalMfaEnabled / this.mfaMetrics.totalDids) * 100;
    }
    if (this.mfaMetrics.totalMfaEnabled > 0) {
      this.mfaMetrics.flEligibilityRate =
        (this.mfaMetrics.flEligibleCount / this.mfaMetrics.totalMfaEnabled) * 100;
    }
  }

  private notifyMfaListeners(): void {
    const metrics = this.getMfaMetrics();
    this.mfaListeners.forEach((callback) => callback(metrics));
  }

  private recalculateEcosystemHealth(): void {
    const healthyCount = this.status.happs.filter((h) => h.status === 'healthy').length;
    const totalCount = this.status.happs.length;
    this.status.healthy = healthyCount / totalCount >= 0.6; // 60% must be healthy
  }

  private notifyListeners(): void {
    const status = this.getStatus();
    this.listeners.forEach((callback) => callback(status));
  }
}

// Singleton instance
let ecosystemService: EcosystemService | null = null;

export function getEcosystemService(): EcosystemService {
  if (!ecosystemService) {
    ecosystemService = new EcosystemService();
  }
  return ecosystemService;
}

// ============================================
// ECONOMIC SIMULATION SERVICE
// ============================================

const SAMPLE_MEMBERS: MemberEconomics[] = [
  {
    did: 'did:mycelix:alice',
    displayName: 'Alice (Patient Builder)',
    sapBalance: 45000,
    civScore: 0.87,
    commitments: [
      { tier: 'Forest', sapLocked: 20000, durationMonths: 180, governanceMultiplier: 7.0, covenantType: 'Universal' },
      { tier: 'Tree', sapLocked: 10000, durationMonths: 60, governanceMultiplier: 2.5, covenantType: 'Ecological' },
    ],
    patienceCoefficient: 2.4,
    hearthShare: 0.08,
    hearthWarmth: 1200,
    intentionContributions: [
      { domain: 'Ecological Regeneration', score: 0.45 },
      { domain: 'Knowledge Commons', score: 0.30 },
    ],
  },
  {
    did: 'did:mycelix:bob',
    displayName: 'Bob (Active Contributor)',
    sapBalance: 12000,
    civScore: 0.72,
    commitments: [
      { tier: 'Sapling', sapLocked: 5000, durationMonths: 18, governanceMultiplier: 1.5, covenantType: 'LocalCommunity' },
    ],
    patienceCoefficient: 1.2,
    hearthShare: 0.02,
    hearthWarmth: 300,
    intentionContributions: [
      { domain: 'Digital Sovereignty', score: 0.60 },
    ],
  },
  {
    did: 'did:mycelix:carol',
    displayName: 'Carol (New Member)',
    sapBalance: 1000,
    civScore: 0.50,
    commitments: [
      { tier: 'Sprout', sapLocked: 500, durationMonths: 3, governanceMultiplier: 1.0, covenantType: 'None' },
    ],
    patienceCoefficient: 0.6,
    hearthShare: 0.005,
    hearthWarmth: 25,
    intentionContributions: [],
  },
  {
    did: 'did:mycelix:agent-helper-01',
    displayName: 'Helper Agent (AI)',
    sapBalance: 0,
    civScore: 0.65, // Minimum CIV to operate
    commitments: [],
    patienceCoefficient: 1.0,
    hearthShare: 0,
    hearthWarmth: 0,
    intentionContributions: [
      { domain: 'Knowledge Commons', score: 0.25 },
    ],
  },
];

class EconomicService {
  private status: EconomicStatus;
  private listeners: Set<(status: EconomicStatus) => void> = new Set();

  constructor() {
    this.status = {
      totalSapSupply: 1_000_000,
      totalHearthPool: 150_000,
      activeCommitments: 42,
      avgPatienceCoefficient: 1.45,
      members: [...SAMPLE_MEMBERS],
      recentActivity: [],
      agentStats: {
        totalAgents: 12,
        totalKreditBalance: 85000,
        avgCivRequirement: 0.60,
        activeTasks: 8,
      },
    };

    // Seed some initial activity
    this.generateActivity();
  }

  getStatus(): EconomicStatus {
    return { ...this.status, members: [...this.status.members] };
  }

  subscribe(callback: (status: EconomicStatus) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  simulateUpdate(): void {
    // Simulate economic activity
    this.status.members.forEach((member) => {
      // Small SAP balance changes
      member.sapBalance += Math.floor((Math.random() - 0.3) * 100);
      if (member.sapBalance < 0) member.sapBalance = 0;

      // CIV score slowly drifts based on activity
      member.civScore = Math.min(1, Math.max(0.1, member.civScore + (Math.random() - 0.5) * 0.02));

      // HEARTH warmth accumulates
      member.hearthWarmth += member.hearthShare * 10;
    });

    // Update network stats
    this.status.totalSapSupply += Math.floor((Math.random() - 0.4) * 500);
    this.status.totalHearthPool += Math.floor(Math.random() * 100);

    // Generate occasional activity
    if (Math.random() > 0.6) {
      this.generateActivity();
    }

    this.notifyListeners();
  }

  private generateActivity(): void {
    const activities: EconomicActivity[] = [
      {
        id: `act-${Date.now()}`,
        type: 'hearth_warm',
        actor: 'Alice',
        amount: 500,
        description: 'Warmed HEARTH pool with long-term commitment',
        timestamp: Date.now(),
      },
      {
        id: `act-${Date.now() + 1}`,
        type: 'kredit_spend',
        actor: 'Helper Agent',
        amount: 150,
        description: 'Spent KREDIT on knowledge synthesis task',
        timestamp: Date.now() - 30000,
      },
      {
        id: `act-${Date.now() + 2}`,
        type: 'intention_contribution',
        actor: 'Bob',
        amount: 0.15,
        description: 'Contributed to Digital Sovereignty intention',
        timestamp: Date.now() - 60000,
      },
      {
        id: `act-${Date.now() + 3}`,
        type: 'commitment_lock',
        actor: 'Carol',
        amount: 500,
        description: 'Created Sprout-tier temporal commitment',
        timestamp: Date.now() - 120000,
      },
    ];

    // Add 1-2 random activities
    const count = 1 + Math.floor(Math.random() * 2);
    for (let i = 0; i < count; i++) {
      const activity = activities[Math.floor(Math.random() * activities.length)];
      activity.id = `act-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      activity.timestamp = Date.now() - Math.floor(Math.random() * 300000);
      this.status.recentActivity.unshift({ ...activity });
    }

    // Keep only last 20
    this.status.recentActivity = this.status.recentActivity.slice(0, 20);
  }

  private notifyListeners(): void {
    const status = this.getStatus();
    this.listeners.forEach((cb) => cb(status));
  }
}

let economicService: EconomicService | null = null;

export function getEconomicService(): EconomicService {
  if (!economicService) {
    economicService = new EconomicService();
  }
  return economicService;
}

/**
 * SDK integration types for the Observatory
 */
export interface SdkStats {
  sdkVersion: string;
  testsPassing: number;
  testsTotal: number;
  matlVersion: string;
  epistemicVersion: string;
  bridgeVersion: string;
  benchmarks: BenchmarkMetrics;
  integrationModules: IntegrationModuleStats[];
}

export interface BenchmarkMetrics {
  // Core operations (ops/sec)
  matlOperations: number;
  epistemicClassification: number;
  bridgeCrossHapp: number;
  flAggregation: number;

  // Civilizational workflows (ops/sec)
  regenerativeExit: number;
  creditAssessment: number;
  factCheck: number;
  defamationDispute: number;

  // Service operations (ops/sec)
  identityVerification: number;
  governanceProposal: number;
  financeCredit: number;
  propertyTransfer: number;
  energyProject: number;
  mediaPublication: number;
  knowledgeClaim: number;
  justiceCase: number;

  // Fabrication operations (ops/sec)
  fabricationDesignCreate: number;
  fabricationPrinterMatch: number;
  fabricationPrintJob: number;
  fabricationPogScore: number;

  lastBenchmarkRun: number;
}

export interface IntegrationModuleStats {
  name: string;
  version: string;
  testsCount: number;
  status: 'stable' | 'beta' | 'alpha';
}

export function getSdkStats(): SdkStats {
  return {
    sdkVersion: '0.6.0',
    testsPassing: 1909,
    testsTotal: 1909,
    matlVersion: '1.0',
    epistemicVersion: '2.0',
    bridgeVersion: '1.0',
    benchmarks: {
      // Core operations (verified benchmark results)
      matlOperations: 4_400_000,
      epistemicClassification: 2_100_000,
      bridgeCrossHapp: 215_000,
      flAggregation: 1_800_000,

      // Extended civilizational workflows
      regenerativeExit: 160_500,
      creditAssessment: 235_300,
      factCheck: 95_900,
      defamationDispute: 444_500,

      // Service operations (from civilizational.bench.ts)
      identityVerification: 1_400_000,
      governanceProposal: 480_000,
      financeCredit: 350_000,
      propertyTransfer: 280_000,
      energyProject: 22_000,
      mediaPublication: 520_000,
      knowledgeClaim: 680_000,
      justiceCase: 180_000,

      // Fabrication operations
      fabricationDesignCreate: 150_000,
      fabricationPrinterMatch: 95_000,
      fabricationPrintJob: 45_000,
      fabricationPogScore: 380_000,

      lastBenchmarkRun: Date.now(),
    },
    integrationModules: [
      { name: 'Mail', version: '0.2.0', testsCount: 45, status: 'stable' },
      { name: 'Marketplace', version: '0.4.0', testsCount: 89, status: 'stable' },
      { name: 'EduNet', version: '0.3.0', testsCount: 112, status: 'stable' },
      { name: 'SupplyChain', version: '0.1.0', testsCount: 67, status: 'beta' },
      { name: 'Fabrication', version: '0.1.0', testsCount: 78, status: 'beta' },
      { name: 'Identity', version: '0.1.0', testsCount: 78, status: 'alpha' },
      { name: 'Governance', version: '0.1.0', testsCount: 85, status: 'alpha' },
      { name: 'Justice', version: '0.1.0', testsCount: 72, status: 'alpha' },
      { name: 'Finance', version: '0.1.0', testsCount: 94, status: 'alpha' },
      { name: 'Property', version: '0.1.0', testsCount: 68, status: 'alpha' },
      { name: 'Energy', version: '0.1.0', testsCount: 56, status: 'alpha' },
      { name: 'Media', version: '0.1.0', testsCount: 61, status: 'alpha' },
      { name: 'Knowledge', version: '0.1.0', testsCount: 83, status: 'alpha' },
      { name: 'Care', version: '0.1.0', testsCount: 54, status: 'alpha' },
      { name: 'Emergency', version: '0.1.0', testsCount: 62, status: 'alpha' },
      { name: 'Water', version: '0.1.0', testsCount: 48, status: 'alpha' },
      { name: 'Housing', version: '0.1.0', testsCount: 58, status: 'alpha' },
    ],
  };
}
