/**
 * @mycelix/sdk DeSci (Decentralized Science) Integration
 *
 * hApp-specific adapter for Mycelix-DeSci providing:
 * - Peer review reputation tracking
 * - Research publication with verifiable credentials
 * - Funding transparency and milestone verification
 * - Cross-hApp researcher identity verification
 * - Federated Learning for research collaboration
 *
 * @packageDocumentation
 * @module integrations/desci
 * @see {@link DeSciService} - Main service class
 * @see {@link getDeSciService} - Singleton accessor
 *
 * @example Basic research publication
 * ```typescript
 * import { getDeSciService } from '@mycelix/sdk/integrations/desci';
 *
 * const desci = getDeSciService();
 *
 * // Register a researcher
 * const researcher = desci.registerResearcher({
 *   id: 'researcher-001',
 *   name: 'Dr. Alice Smith',
 *   orcid: '0000-0002-1234-5678',
 *   institution: 'University of Example',
 *   fields: ['neuroscience', 'machine-learning'],
 * });
 *
 * // Publish research
 * const publication = desci.publishResearch({
 *   id: 'pub-001',
 *   title: 'Novel Findings in Neural Networks',
 *   authors: [researcher.id],
 *   abstract: 'We present groundbreaking research...',
 *   doi: '10.1234/example.001',
 *   publishedAt: Date.now(),
 *   peerReviewed: false,
 * });
 * ```
 */

import { LocalBridge, createReputationQuery } from '../../bridge/index.js';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClassification,
} from '../../epistemic/index.js';
import { FLCoordinator, AggregationMethod } from '../../fl/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  type ReputationScore,
} from '../../matl/index.js';

// ============================================================================
// DeSci-Specific Types
// ============================================================================

/**
 * Researcher profile
 */
export interface Researcher {
  id: string;
  name: string;
  orcid?: string; // ORCID identifier
  institution?: string;
  fields: string[];
  hIndex?: number;
  verifiedIdentity: boolean;
  joinedAt: number;
}

/**
 * Research publication
 */
export interface Publication {
  id: string;
  title: string;
  authors: string[]; // Researcher IDs
  abstract: string;
  doi?: string;
  arxivId?: string;
  publishedAt: number;
  peerReviewed: boolean;
  reviewCount: number;
  citationCount: number;
  epistemicClassification?: EpistemicClassification;
  dataAvailable: boolean;
  codeAvailable: boolean;
  preregistered: boolean;
}

/**
 * Peer review submission
 */
export interface PeerReview {
  id: string;
  publicationId: string;
  reviewerId: string;
  submittedAt: number;
  recommendation: 'accept' | 'minor_revision' | 'major_revision' | 'reject';
  confidence: number; // 1-5
  qualityScore: number; // 1-10
  comments: string;
  blind: boolean; // Double-blind review
  verified: boolean;
}

/**
 * Research funding grant
 */
export interface ResearchGrant {
  id: string;
  title: string;
  principalInvestigator: string; // Researcher ID
  coInvestigators: string[];
  fundingAgency: string;
  amount: number;
  currency: string;
  startDate: number;
  endDate: number;
  milestones: GrantMilestone[];
  status: 'proposed' | 'active' | 'completed' | 'terminated';
}

/**
 * Grant milestone
 */
export interface GrantMilestone {
  id: string;
  title: string;
  description: string;
  dueDate: number;
  completedDate?: number;
  deliverables: string[];
  verified: boolean;
  verifierId?: string;
}

/**
 * Researcher reputation profile
 */
export interface ResearcherProfile {
  researcher: Researcher;
  reputation: ReputationScore;
  trustScore: number;
  publicationCount: number;
  reviewCount: number;
  citationCount: number;
  hIndex: number;
  grantCount: number;
  totalFunding: number;
  collaboratorCount: number;
  verified: boolean;
}

/**
 * Research collaboration
 */
export interface Collaboration {
  id: string;
  title: string;
  participants: string[]; // Researcher IDs
  institution: string[];
  startDate: number;
  endDate?: number;
  publications: string[];
  grants: string[];
  status: 'active' | 'completed' | 'paused';
}

// ============================================================================
// DeSci Service
// ============================================================================

/**
 * DeSciService - Decentralized science research management
 *
 * @remarks
 * This service provides:
 * - Transparent peer review with reputation tracking
 * - Verifiable research credentials
 * - Grant milestone verification
 * - Open science data sharing
 * - Federated learning for research collaboration
 *
 * @example
 * ```typescript
 * const desci = new DeSciService();
 *
 * // Register researcher and publish
 * const researcher = desci.registerResearcher({...});
 * const pub = desci.publishResearch({...});
 *
 * // Submit peer review
 * desci.submitPeerReview({
 *   publicationId: pub.id,
 *   reviewerId: 'reviewer-001',
 *   recommendation: 'accept',
 *   confidence: 4,
 *   qualityScore: 8,
 *   comments: 'Excellent methodology...',
 * });
 * ```
 */
export class DeSciService {
  private researchers: Map<string, Researcher> = new Map();
  private publications: Map<string, Publication> = new Map();
  private reviews: Map<string, PeerReview[]> = new Map();
  private grants: Map<string, ResearchGrant> = new Map();
  private collaborations: Map<string, Collaboration> = new Map();
  private researcherReputations: Map<string, ReputationScore> = new Map();
  private bridge: LocalBridge;
  private flCoordinator: FLCoordinator;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('desci');

    this.flCoordinator = new FLCoordinator({
      minParticipants: 5,
      aggregationMethod: AggregationMethod.TrustWeighted,
      byzantineTolerance: 0.34,
    });
  }

  /**
   * Register a new researcher
   *
   * @param input - Researcher registration data
   * @returns Registered researcher with verification status
   *
   * @remarks
   * - Researchers with ORCID get verified identity
   * - Initial reputation is neutral (0.5)
   */
  registerResearcher(input: Omit<Researcher, 'verifiedIdentity' | 'joinedAt'>): Researcher {
    const researcher: Researcher = {
      ...input,
      verifiedIdentity: !!input.orcid, // Verified if has ORCID
      joinedAt: Date.now(),
    };

    this.researchers.set(researcher.id, researcher);
    this.researcherReputations.set(researcher.id, createReputation(researcher.id));

    // Store in bridge for cross-hApp queries
    this.bridge.setReputation(
      'desci',
      researcher.id,
      this.researcherReputations.get(researcher.id)!
    );

    return researcher;
  }

  /**
   * Publish research with epistemic classification
   *
   * @param input - Publication data
   * @returns Published research with classification
   *
   * @remarks
   * - Peer reviewed publications boost author reputation
   * - Open data/code increases epistemic level
   * - Preregistered studies get highest empirical classification
   */
  publishResearch(
    input: Omit<Publication, 'reviewCount' | 'citationCount' | 'epistemicClassification'>
  ): Publication {
    // Calculate epistemic classification
    let empirical = EmpiricalLevel.E1_Testimonial;
    let normative = NormativeLevel.N1_Communal;
    const materiality = MaterialityLevel.M2_Persistent;

    if (input.preregistered) {
      empirical = EmpiricalLevel.E3_Cryptographic;
    } else if (input.dataAvailable && input.codeAvailable) {
      empirical = EmpiricalLevel.E2_PrivateVerify;
    }

    if (input.peerReviewed) {
      normative = NormativeLevel.N2_Network;
    }

    const publication: Publication = {
      ...input,
      reviewCount: 0,
      citationCount: 0,
      epistemicClassification: {
        empirical,
        normative,
        materiality,
      },
    };

    this.publications.set(publication.id, publication);
    this.reviews.set(publication.id, []);

    // Update author reputations
    for (const authorId of input.authors) {
      let rep = this.researcherReputations.get(authorId) || createReputation(authorId);
      rep = recordPositive(rep); // Publishing is positive
      this.researcherReputations.set(authorId, rep);
    }

    return publication;
  }

  /**
   * Submit a peer review
   *
   * @param input - Review submission data
   * @returns Submitted review
   *
   * @remarks
   * - Quality reviews boost reviewer reputation
   * - Accepting high-quality papers is positive
   * - Reviews are verified if reviewer is verified
   */
  submitPeerReview(input: Omit<PeerReview, 'id' | 'submittedAt' | 'verified'>): PeerReview {
    const publication = this.publications.get(input.publicationId);
    if (!publication) {
      throw new Error(`Publication not found: ${input.publicationId}`);
    }

    const reviewer = this.researchers.get(input.reviewerId);
    const review: PeerReview = {
      ...input,
      id: `review-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      submittedAt: Date.now(),
      verified: reviewer?.verifiedIdentity || false,
    };

    const pubReviews = this.reviews.get(input.publicationId) || [];
    pubReviews.push(review);
    this.reviews.set(input.publicationId, pubReviews);

    // Update publication review count
    publication.reviewCount++;
    if (pubReviews.length >= 2 && pubReviews.every((r) => r.recommendation !== 'reject')) {
      publication.peerReviewed = true;
    }
    this.publications.set(input.publicationId, publication);

    // Update reviewer reputation
    let rep = this.researcherReputations.get(input.reviewerId) || createReputation(input.reviewerId);
    if (input.confidence >= 3 && input.comments.length >= 100) {
      rep = recordPositive(rep); // Quality review
    }
    this.researcherReputations.set(input.reviewerId, rep);

    return review;
  }

  /**
   * Create a research grant
   *
   * @param input - Grant details
   * @returns Created grant
   */
  createGrant(input: Omit<ResearchGrant, 'status'>): ResearchGrant {
    const grant: ResearchGrant = {
      ...input,
      status: 'proposed',
    };

    this.grants.set(grant.id, grant);
    return grant;
  }

  /**
   * Verify a grant milestone
   *
   * @param grantId - Grant ID
   * @param milestoneId - Milestone ID
   * @param verifierId - Researcher verifying the milestone
   * @returns Whether verification succeeded
   */
  verifyMilestone(grantId: string, milestoneId: string, verifierId: string): boolean {
    const grant = this.grants.get(grantId);
    if (!grant) {
      throw new Error(`Grant not found: ${grantId}`);
    }

    const milestone = grant.milestones.find((m) => m.id === milestoneId);
    if (!milestone) {
      throw new Error(`Milestone not found: ${milestoneId}`);
    }

    const verifier = this.researchers.get(verifierId);
    if (!verifier?.verifiedIdentity) {
      throw new Error('Verifier must have verified identity');
    }

    milestone.verified = true;
    milestone.verifierId = verifierId;
    milestone.completedDate = Date.now();

    this.grants.set(grantId, grant);

    // Boost PI reputation for verified milestone
    let rep =
      this.researcherReputations.get(grant.principalInvestigator) ||
      createReputation(grant.principalInvestigator);
    rep = recordPositive(rep);
    this.researcherReputations.set(grant.principalInvestigator, rep);

    return true;
  }

  /**
   * Record a citation
   *
   * @param citingPubId - Publication doing the citing
   * @param citedPubId - Publication being cited
   */
  recordCitation(_citingPubId: string, citedPubId: string): void {
    const citedPub = this.publications.get(citedPubId);
    if (!citedPub) {
      throw new Error(`Publication not found: ${citedPubId}`);
    }

    citedPub.citationCount++;
    this.publications.set(citedPubId, citedPub);

    // Boost cited authors' reputation
    for (const authorId of citedPub.authors) {
      let rep = this.researcherReputations.get(authorId) || createReputation(authorId);
      rep = recordPositive(rep);
      this.researcherReputations.set(authorId, rep);
    }
  }

  /**
   * Get comprehensive researcher profile
   */
  getResearcherProfile(researcherId: string): ResearcherProfile {
    const researcher = this.researchers.get(researcherId);
    if (!researcher) {
      throw new Error(`Researcher not found: ${researcherId}`);
    }

    const reputation = this.researcherReputations.get(researcherId) || createReputation(researcherId);

    // Calculate statistics
    let publicationCount = 0;
    let citationCount = 0;
    const collaborators = new Set<string>();

    for (const pub of this.publications.values()) {
      if (pub.authors.includes(researcherId)) {
        publicationCount++;
        citationCount += pub.citationCount;
        pub.authors.forEach((a) => {
          if (a !== researcherId) collaborators.add(a);
        });
      }
    }

    let reviewCount = 0;
    for (const reviews of this.reviews.values()) {
      reviewCount += reviews.filter((r) => r.reviewerId === researcherId).length;
    }

    let grantCount = 0;
    let totalFunding = 0;
    for (const grant of this.grants.values()) {
      if (
        grant.principalInvestigator === researcherId ||
        grant.coInvestigators.includes(researcherId)
      ) {
        grantCount++;
        totalFunding += grant.amount;
      }
    }

    // Simple h-index calculation
    const papers = Array.from(this.publications.values())
      .filter((p) => p.authors.includes(researcherId))
      .map((p) => p.citationCount)
      .sort((a, b) => b - a);

    let hIndex = 0;
    for (let i = 0; i < papers.length; i++) {
      if (papers[i] >= i + 1) {
        hIndex = i + 1;
      } else {
        break;
      }
    }

    return {
      researcher,
      reputation,
      trustScore: reputationValue(reputation),
      publicationCount,
      reviewCount,
      citationCount,
      hIndex,
      grantCount,
      totalFunding,
      collaboratorCount: collaborators.size,
      verified: researcher.verifiedIdentity && reputationValue(reputation) >= 0.7,
    };
  }

  /**
   * Get publication with reviews
   */
  getPublicationWithReviews(publicationId: string): {
    publication: Publication;
    reviews: PeerReview[];
    averageQuality: number;
  } {
    const publication = this.publications.get(publicationId);
    if (!publication) {
      throw new Error(`Publication not found: ${publicationId}`);
    }

    const reviews = this.reviews.get(publicationId) || [];
    const averageQuality =
      reviews.length > 0
        ? reviews.reduce((sum, r) => sum + r.qualityScore, 0) / reviews.length
        : 0;

    return { publication, reviews, averageQuality };
  }

  /**
   * Check if researcher is trustworthy
   */
  isResearcherTrustworthy(researcherId: string, threshold = 0.7): boolean {
    const reputation = this.researcherReputations.get(researcherId);
    if (!reputation) return false;
    return isTrustworthy(reputation, threshold);
  }

  /**
   * Query researcher reputation from other hApps
   */
  queryExternalReputation(researcherId: string): void {
    const query = createReputationQuery('desci', researcherId);
    this.bridge.send('identity', query);
    this.bridge.send('edunet', query);
    this.bridge.send('knowledge', query);
  }

  /**
   * Create research collaboration
   */
  createCollaboration(input: Omit<Collaboration, 'publications' | 'grants' | 'status'>): Collaboration {
    const collaboration: Collaboration = {
      ...input,
      publications: [],
      grants: [],
      status: 'active',
    };

    this.collaborations.set(collaboration.id, collaboration);
    return collaboration;
  }

  /**
   * Get FL coordinator for collaborative models
   */
  getFLCoordinator(): FLCoordinator {
    return this.flCoordinator;
  }
}

// ============================================================================
// Exports
// ============================================================================

export {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  AggregationMethod,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
};

// Default service instance
let defaultService: DeSciService | null = null;

/**
 * Get the default DeSci service instance
 */
export function getDeSciService(): DeSciService {
  if (!defaultService) {
    defaultService = new DeSciService();
  }
  return defaultService;
}
