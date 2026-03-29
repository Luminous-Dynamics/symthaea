// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * GraphQL Layer for Mycelix SDK
 *
 * Provides a unified query interface across all 8 Civilizational OS hApps.
 * Enables flexible data access and frontend integration.
 */

// @ts-ignore
import { buildSchema, graphql, type GraphQLSchema, type ExecutionResult } from 'graphql';

// ============================================================================
// Schema Definition
// ============================================================================

const typeDefs = `
  # Scalar types
  scalar DateTime
  scalar JSON

  # ============================================================================
  # Core Types
  # ============================================================================

  type MATLScore {
    merit: Float!
    alignment: Float!
    trackRecord: Float!
    longevity: Float!
    composite: Float!
  }

  type EpistemicProfile {
    empiricalLevel: Int!
    normativeLevel: Int!
    mythicLevel: Int!
    confidence: Float
  }

  # ============================================================================
  # Identity Types
  # ============================================================================

  type Identity {
    did: String!
    publicKey: String!
    displayName: String
    verificationLevel: VerificationLevel!
    attestations: [Attestation!]!
    reputation: MATLScore
    createdAt: DateTime!
  }

  enum VerificationLevel {
    UNVERIFIED
    SELF_ATTESTED
    PEER_VERIFIED
    INSTITUTIONAL
    GOVERNMENTAL
  }

  type Attestation {
    attesterId: String!
    type: String!
    value: String!
    timestamp: DateTime!
  }

  # ============================================================================
  # Finance Types
  # ============================================================================

  type Wallet {
    id: String!
    ownerId: String!
    type: WalletType!
    balances: [Balance!]!
    createdAt: DateTime!
  }

  enum WalletType {
    PERSONAL
    BUSINESS
    COOPERATIVE
    ESCROW
  }

  type Balance {
    currency: String!
    amount: Float!
  }

  type Transaction {
    id: String!
    fromWallet: String!
    toWallet: String!
    amount: Float!
    currency: String!
    status: TransactionStatus!
    memo: String
    timestamp: DateTime!
  }

  enum TransactionStatus {
    PENDING
    CONFIRMED
    FAILED
    REVERSED
  }

  type CreditScore {
    did: String!
    score: Float!
    matlComponent: Float!
    paymentHistory: Float!
    utilizationRatio: Float!
    lastUpdated: DateTime!
  }

  # ============================================================================
  # Property Types
  # ============================================================================

  type Asset {
    id: String!
    type: AssetType!
    name: String!
    description: String
    owners: [Ownership!]!
    valuation: Float
    location: Location
    createdAt: DateTime!
  }

  enum AssetType {
    REAL_ESTATE
    VEHICLE
    EQUIPMENT
    INTELLECTUAL
    DIGITAL
  }

  type Ownership {
    ownerId: String!
    percentage: Float!
    type: OwnershipType!
  }

  enum OwnershipType {
    SOLE
    JOINT
    COOPERATIVE
    COMMONS
  }

  type Location {
    lat: Float!
    lng: Float!
    address: String
  }

  # ============================================================================
  # Energy Types
  # ============================================================================

  type EnergyParticipant {
    id: String!
    did: String!
    type: ParticipantType!
    sources: [String!]!
    capacityKwh: Float!
    location: Location
    credits: [EnergyCredit!]!
  }

  enum ParticipantType {
    PRODUCER
    CONSUMER
    PROSUMER
  }

  type EnergyCredit {
    id: String!
    amountKwh: Float!
    source: String!
    timestamp: DateTime!
    verified: Boolean!
  }

  type EnergyTrade {
    id: String!
    sellerId: String!
    buyerId: String!
    amountKwh: Float!
    pricePerKwh: Float!
    source: String!
    settlementStatus: String!
    timestamp: DateTime!
  }

  # ============================================================================
  # Media Types
  # ============================================================================

  type Content {
    id: String!
    authorDid: String!
    type: ContentType!
    title: String!
    description: String
    contentHash: String!
    storageUri: String!
    license: String!
    tags: [String!]!
    publishedAt: DateTime!
    moderationStatus: ModerationStatus!
  }

  enum ContentType {
    ARTICLE
    IMAGE
    VIDEO
    AUDIO
    DATASET
    SOFTWARE
    MIXED
  }

  enum ModerationStatus {
    PENDING
    APPROVED
    FLAGGED
    REMOVED
  }

  # ============================================================================
  # Governance Types
  # ============================================================================

  type DAO {
    id: String!
    name: String!
    description: String
    members: [DAOMember!]!
    proposals: [Proposal!]!
    treasury: Wallet
    createdAt: DateTime!
  }

  type DAOMember {
    did: String!
    role: String!
    votingPower: Int!
    delegatedPower: Int!
    joinedAt: DateTime!
  }

  type Proposal {
    id: String!
    daoId: String!
    title: String!
    description: String!
    proposerId: String!
    status: ProposalStatus!
    votingPeriodHours: Int!
    quorumPercentage: Float!
    approvesWeight: Int!
    rejectsWeight: Int!
    abstainsWeight: Int!
    createdAt: DateTime!
    expiresAt: DateTime!
  }

  enum ProposalStatus {
    DRAFT
    ACTIVE
    PASSED
    REJECTED
    EXECUTED
    CANCELLED
  }

  # ============================================================================
  # Justice Types
  # ============================================================================

  type Case {
    id: String!
    complainantId: String!
    respondentId: String!
    title: String!
    description: String!
    category: String!
    status: CaseStatus!
    phase: CasePhase!
    evidence: [Evidence!]!
    decision: Decision
    filedAt: DateTime!
  }

  enum CaseStatus {
    OPEN
    IN_PROGRESS
    RESOLVED
    DISMISSED
    APPEALED
  }

  enum CasePhase {
    FILED
    MEDIATION
    ARBITRATION
    APPEAL
    ENFORCEMENT
    CLOSED
  }

  type Evidence {
    id: String!
    submitterId: String!
    type: String!
    title: String!
    description: String!
    hash: String!
    timestamp: DateTime!
  }

  type Decision {
    outcome: String!
    reasoning: String!
    remedies: [Remedy!]!
    decidedAt: DateTime!
  }

  type Remedy {
    type: String!
    targetId: String!
    description: String!
    amount: Float
    deadline: DateTime
    completed: Boolean!
  }

  # ============================================================================
  # Knowledge Types
  # ============================================================================

  type KnowledgeClaim {
    id: String!
    authorId: String!
    title: String!
    content: String!
    epistemic: EpistemicProfile!
    endorsements: Int!
    challenges: Int!
    status: ClaimStatus!
    tags: [String!]!
    evidence: [ClaimEvidence!]!
    createdAt: DateTime!
  }

  enum ClaimStatus {
    PENDING
    VERIFIED
    CONTESTED
    REFUTED
  }

  type ClaimEvidence {
    id: String!
    submitterId: String!
    type: String!
    description: String!
    sourceUrl: String
    timestamp: DateTime!
  }

  # ============================================================================
  # Query Root
  # ============================================================================

  type Query {
    # Identity queries
    identity(did: String!): Identity
    identities(filter: IdentityFilter): [Identity!]!

    # Finance queries
    wallet(id: String!): Wallet
    walletsByOwner(ownerId: String!): [Wallet!]!
    transaction(id: String!): Transaction
    transactions(filter: TransactionFilter): [Transaction!]!
    creditScore(did: String!): CreditScore

    # Property queries
    asset(id: String!): Asset
    assets(filter: AssetFilter): [Asset!]!
    assetsByOwner(ownerId: String!): [Asset!]!

    # Energy queries
    energyParticipant(id: String!): EnergyParticipant
    energyParticipants(filter: EnergyFilter): [EnergyParticipant!]!
    energyTrades(filter: TradeFilter): [EnergyTrade!]!

    # Media queries
    content(id: String!): Content
    contentList(filter: ContentFilter): [Content!]!
    contentByAuthor(authorDid: String!): [Content!]!

    # Governance queries
    dao(id: String!): DAO
    daos: [DAO!]!
    proposal(id: String!): Proposal
    proposals(daoId: String, status: ProposalStatus): [Proposal!]!

    # Justice queries
    case(id: String!): Case
    cases(filter: CaseFilter): [Case!]!
    casesByParty(partyId: String!): [Case!]!

    # Knowledge queries
    claim(id: String!): KnowledgeClaim
    claims(filter: ClaimFilter): [KnowledgeClaim!]!
    claimsByAuthor(authorId: String!): [KnowledgeClaim!]!

    # Cross-domain queries
    entityReputation(did: String!): MATLScore
    relatedEntities(did: String!): RelatedEntities
  }

  # ============================================================================
  # Filter Inputs
  # ============================================================================

  input IdentityFilter {
    verificationLevel: VerificationLevel
    minReputation: Float
  }

  input TransactionFilter {
    fromWallet: String
    toWallet: String
    currency: String
    status: TransactionStatus
    fromDate: DateTime
    toDate: DateTime
    limit: Int
  }

  input AssetFilter {
    type: AssetType
    ownerId: String
    minValuation: Float
    maxValuation: Float
    limit: Int
  }

  input EnergyFilter {
    type: ParticipantType
    source: String
    minCapacity: Float
    limit: Int
  }

  input TradeFilter {
    sellerId: String
    buyerId: String
    source: String
    fromDate: DateTime
    toDate: DateTime
    limit: Int
  }

  input ContentFilter {
    type: ContentType
    authorDid: String
    license: String
    tags: [String!]
    status: ModerationStatus
    limit: Int
  }

  input CaseFilter {
    category: String
    status: CaseStatus
    phase: CasePhase
    limit: Int
  }

  input ClaimFilter {
    status: ClaimStatus
    minEmpirical: Int
    tags: [String!]
    limit: Int
  }

  # ============================================================================
  # Cross-Domain Types
  # ============================================================================

  type RelatedEntities {
    did: String!
    assets: [Asset!]!
    wallets: [Wallet!]!
    content: [Content!]!
    proposals: [Proposal!]!
    cases: [Case!]!
    claims: [KnowledgeClaim!]!
  }

  # ============================================================================
  # Mutation Root
  # ============================================================================

  type Mutation {
    # Identity mutations
    createIdentity(publicKey: String!, displayName: String): Identity!
    attestTrust(attesterId: String!, targetId: String!): Attestation!

    # Finance mutations
    createWallet(ownerId: String!, type: WalletType!): Wallet!
    transfer(fromWallet: String!, toWallet: String!, amount: Float!, currency: String!, memo: String): Transaction!

    # Property mutations
    registerAsset(type: AssetType!, name: String!, description: String, ownerId: String!, ownershipType: OwnershipType!): Asset!
    transferAsset(assetId: String!, fromOwner: String!, toOwner: String!, percentage: Float!): Asset!

    # Energy mutations
    registerParticipant(did: String!, type: ParticipantType!, sources: [String!]!, capacityKwh: Float!): EnergyParticipant!
    submitReading(participantId: String!, productionKwh: Float!, consumptionKwh: Float!, source: String!): EnergyCredit!
    tradeEnergy(sellerId: String!, buyerId: String!, amountKwh: Float!, source: String!, pricePerKwh: Float!): EnergyTrade!

    # Media mutations
    publishContent(authorDid: String!, type: ContentType!, title: String!, description: String, contentHash: String!, storageUri: String!, license: String!, tags: [String!]): Content!
    reportContent(contentId: String!, reporterId: String!, reason: String!, category: String!): Boolean!

    # Governance mutations
    createProposal(daoId: String!, title: String!, description: String!, proposerId: String!, votingPeriodHours: Int!, quorumPercentage: Float!): Proposal!
    castVote(proposalId: String!, voterId: String!, choice: String!, weight: Int!): Boolean!

    # Justice mutations
    fileCase(complainantId: String!, respondentId: String!, title: String!, description: String!, category: String!): Case!
    submitEvidence(caseId: String!, submitterId: String!, type: String!, title: String!, description: String!, hash: String!): Evidence!

    # Knowledge mutations
    submitClaim(authorId: String!, title: String!, content: String!, empirical: Int!, normative: Int!, tags: [String!]): KnowledgeClaim!
    endorseClaim(claimId: String!, endorserId: String!): Boolean!
  }

  # ============================================================================
  # Subscription Root
  # ============================================================================

  type Subscription {
    # Real-time updates
    transactionCreated(walletId: String): Transaction!
    proposalUpdated(daoId: String): Proposal!
    caseUpdated(caseId: String): Case!
    claimEndorsed(claimId: String): KnowledgeClaim!
    energyTradeCreated(participantId: String): EnergyTrade!
  }
`;

// ============================================================================
// Schema Builder
// ============================================================================

let cachedSchema: GraphQLSchema | null = null;

/**
 * Get the GraphQL schema
 */
export function getSchema(): GraphQLSchema {
  if (!cachedSchema) {
    cachedSchema = buildSchema(typeDefs);
  }
  return cachedSchema;
}

/**
 * Get the schema type definitions as a string
 */
export function getTypeDefs(): string {
  return typeDefs;
}

// ============================================================================
// Resolver Context
// ============================================================================

export interface ResolverContext {
  // Service instances (injected by the application)
  identityService?: any;
  financeService?: any;
  propertyService?: any;
  energyService?: any;
  mediaService?: any;
  governanceService?: any;
  justiceService?: any;
  knowledgeService?: any;

  // Current user context
  currentUser?: {
    did: string;
    permissions: string[];
  };
}

// ============================================================================
// Resolver Factory
// ============================================================================

export interface Resolvers {
  Query: Record<string, (args: any, context: ResolverContext) => any>;
  Mutation: Record<string, (args: any, context: ResolverContext) => any>;
}

/**
 * Create resolvers that delegate to service instances
 */
export function createResolvers(): Resolvers {
  return {
    Query: {
      // Identity
      identity: ({ did }: { did: string }, ctx: ResolverContext) =>
        ctx.identityService?.getProfile(did),

      identities: ({ filter }: { filter?: any }, ctx: ResolverContext) =>
        ctx.identityService?.listProfiles(filter) ?? [],

      // Finance
      wallet: ({ id }: { id: string }, ctx: ResolverContext) => ctx.financeService?.getWallet(id),

      walletsByOwner: ({ ownerId }: { ownerId: string }, ctx: ResolverContext) =>
        ctx.financeService?.getWalletsByOwner(ownerId) ?? [],

      creditScore: ({ did }: { did: string }, ctx: ResolverContext) =>
        ctx.financeService?.getCreditScore(did),

      // Property
      asset: ({ id }: { id: string }, ctx: ResolverContext) => ctx.propertyService?.getAsset(id),

      assets: ({ filter }: { filter?: any }, ctx: ResolverContext) =>
        ctx.propertyService?.listAssets(filter) ?? [],

      // Energy
      energyParticipant: ({ id }: { id: string }, ctx: ResolverContext) =>
        ctx.energyService?.getParticipant(id),

      energyTrades: ({ filter }: { filter?: any }, ctx: ResolverContext) =>
        ctx.energyService?.listTrades(filter) ?? [],

      // Media
      content: ({ id }: { id: string }, ctx: ResolverContext) => ctx.mediaService?.getContent(id),

      contentList: ({ filter }: { filter?: any }, ctx: ResolverContext) =>
        ctx.mediaService?.listContent(filter) ?? [],

      // Governance
      dao: ({ id }: { id: string }, ctx: ResolverContext) => ctx.governanceService?.getDAO(id),

      proposal: ({ id }: { id: string }, ctx: ResolverContext) =>
        ctx.governanceService?.getProposal(id),

      proposals: ({ daoId, status }: { daoId?: string; status?: string }, ctx: ResolverContext) =>
        ctx.governanceService?.listProposals({ daoId, status }) ?? [],

      // Justice
      case: ({ id }: { id: string }, ctx: ResolverContext) => ctx.justiceService?.getCase(id),

      cases: ({ filter }: { filter?: any }, ctx: ResolverContext) =>
        ctx.justiceService?.listCases(filter) ?? [],

      // Knowledge
      claim: ({ id }: { id: string }, ctx: ResolverContext) => ctx.knowledgeService?.getClaim(id),

      claims: ({ filter }: { filter?: any }, ctx: ResolverContext) =>
        ctx.knowledgeService?.listClaims(filter) ?? [],

      // Cross-domain
      entityReputation: ({ did }: { did: string }, ctx: ResolverContext) =>
        ctx.identityService?.getReputation(did),

      relatedEntities: ({ did }: { did: string }, ctx: ResolverContext) => ({
        did,
        assets: ctx.propertyService?.getAssetsByOwner(did) ?? [],
        wallets: ctx.financeService?.getWalletsByOwner(did) ?? [],
        content: ctx.mediaService?.getContentByAuthor(did) ?? [],
        proposals: ctx.governanceService?.getProposalsByMember(did) ?? [],
        cases: ctx.justiceService?.getCasesByParty(did) ?? [],
        claims: ctx.knowledgeService?.getClaimsByAuthor(did) ?? [],
      }),
    },

    Mutation: {
      // Identity
      createIdentity: ({ publicKey, displayName }: any, ctx: ResolverContext) =>
        ctx.identityService?.createIdentity(publicKey, displayName),

      attestTrust: ({ attesterId, targetId }: any, ctx: ResolverContext) =>
        ctx.identityService?.attestTrust(attesterId, targetId),

      // Finance
      createWallet: ({ ownerId, type }: any, ctx: ResolverContext) =>
        ctx.financeService?.createWallet(ownerId, type),

      transfer: ({ fromWallet, toWallet, amount, currency, memo }: any, ctx: ResolverContext) =>
        ctx.financeService?.transfer(fromWallet, toWallet, amount, currency, memo),

      // Property
      registerAsset: (args: any, ctx: ResolverContext) =>
        ctx.propertyService?.registerAsset(
          args.type,
          args.name,
          args.description,
          args.ownerId,
          args.ownershipType
        ),

      // Energy
      registerParticipant: (args: any, ctx: ResolverContext) =>
        ctx.energyService?.registerParticipant(args.did, args.type, args.sources, args.capacityKwh),

      tradeEnergy: (args: any, ctx: ResolverContext) =>
        ctx.energyService?.tradeEnergy(
          args.sellerId,
          args.buyerId,
          args.amountKwh,
          args.source,
          args.pricePerKwh
        ),

      // Media
      publishContent: (args: any, ctx: ResolverContext) =>
        ctx.mediaService?.publishContent(
          args.authorDid,
          args.type,
          args.title,
          args.description,
          args.contentHash,
          args.storageUri,
          args.license,
          args.tags
        ),

      // Governance
      createProposal: (args: any, ctx: ResolverContext) =>
        ctx.governanceService?.createProposal(args),

      castVote: (args: any, ctx: ResolverContext) => ctx.governanceService?.castVote(args),

      // Justice
      fileCase: (args: any, ctx: ResolverContext) =>
        ctx.justiceService?.fileCase(
          args.complainantId,
          args.respondentId,
          args.title,
          args.description,
          args.category
        ),

      submitEvidence: (args: any, ctx: ResolverContext) =>
        ctx.justiceService?.submitEvidence(
          args.caseId,
          args.submitterId,
          args.type,
          args.title,
          args.description,
          args.hash
        ),

      // Knowledge
      submitClaim: (args: any, ctx: ResolverContext) =>
        ctx.knowledgeService?.submitClaim(
          args.authorId,
          args.title,
          args.content,
          args.empirical,
          args.normative,
          args.tags
        ),

      endorseClaim: ({ claimId, endorserId }: any, ctx: ResolverContext) => {
        ctx.knowledgeService?.endorseClaim(claimId, endorserId);
        return true;
      },
    },
  };
}

// ============================================================================
// Query Executor
// ============================================================================

/**
 * Execute a GraphQL query
 */
export async function executeQuery(
  query: string,
  variables?: Record<string, any>,
  context?: ResolverContext
): Promise<ExecutionResult> {
  const schema = getSchema();
  const resolvers = createResolvers();

  // Create root value from resolvers
  const rootValue = {
    ...resolvers.Query,
    ...resolvers.Mutation,
  };

  return graphql({
    schema,
    source: query,
    rootValue,
    contextValue: context ?? {},
    variableValues: variables,
  });
}

// ============================================================================
// Convenience Query Builders
// ============================================================================

export const queries = {
  /** Get identity with reputation */
  getIdentityWithReputation: `
    query GetIdentity($did: String!) {
      identity(did: $did) {
        did
        displayName
        verificationLevel
        reputation {
          merit
          alignment
          trackRecord
          longevity
          composite
        }
      }
    }
  `,

  /** Get all entities related to a DID */
  getRelatedEntities: `
    query GetRelatedEntities($did: String!) {
      relatedEntities(did: $did) {
        did
        assets { id name type valuation }
        wallets { id type balances { currency amount } }
        content { id title type publishedAt }
        proposals { id title status }
        cases { id title status phase }
        claims { id title status endorsements }
      }
    }
  `,

  /** Get active proposals for a DAO */
  getActiveProposals: `
    query GetActiveProposals($daoId: String!) {
      proposals(daoId: $daoId, status: ACTIVE) {
        id
        title
        description
        proposerId
        approvesWeight
        rejectsWeight
        expiresAt
      }
    }
  `,

  /** Get energy trades for a participant */
  getEnergyTrades: `
    query GetEnergyTrades($participantId: String!) {
      energyTrades(filter: { sellerId: $participantId }) {
        id
        buyerId
        amountKwh
        pricePerKwh
        source
        timestamp
      }
    }
  `,
};

// ============================================================================
// Exports
// ============================================================================

export { graphql, type ExecutionResult, type GraphQLSchema };
