/**
 * Commons Zome Client
 *
 * Handles shared resource management and commons governance.
 *
 * @module @mycelix/sdk/integrations/property/zomes/commons
 */

import { PropertySdkError } from '../types';

import type {
  Commons,
  CommonsType,
  CommonsAccessLevel,
  CommonsMember,
  CommonsUsage,
  CreateCommonsInput,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Commons client
 */
export interface CommonsClientConfig {
  /** Role ID for the property DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: CommonsClientConfig = {
  roleId: 'property',
  zomeName: 'property',
};

/**
 * Client for commons and shared resource operations
 *
 * @example
 * ```typescript
 * const commons = new CommonsClient(holochainClient);
 *
 * // Create a land commons
 * const landCommons = await commons.createCommons({
 *   id: 'community-garden',
 *   name: 'Community Garden',
 *   description: 'Shared garden space for community members',
 *   commons_type: 'Land',
 *   dao_id: 'garden-dao',
 *   governance_rules: JSON.stringify({
 *     maxPlots: 50,
 *     plotSize: '10x10',
 *     seasonalFee: 25,
 *   }),
 *   created_by: 'did:mycelix:founder',
 * });
 *
 * // Join the commons
 * await commons.joinCommons('community-garden', 'did:mycelix:member');
 *
 * // Record usage
 * await commons.recordUsage({
 *   commons_id: 'community-garden',
 *   user_did: 'did:mycelix:member',
 *   purpose: 'Weekly plot maintenance',
 * });
 * ```
 */
export class CommonsClient {
  private readonly config: CommonsClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<CommonsClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      throw new PropertySdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Extract entry from a Holochain record
   */
  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new PropertySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Commons Operations
  // ============================================================================

  /**
   * Create a new commons
   *
   * @param input - Commons creation parameters
   * @returns The created commons
   */
  async createCommons(input: CreateCommonsInput): Promise<Commons> {
    const record = await this.call<HolochainRecord>('create_commons', input);
    return this.extractEntry<Commons>(record);
  }

  /**
   * Get a commons by ID
   *
   * @param commonsId - Commons identifier
   * @returns The commons or null
   */
  async getCommons(commonsId: string): Promise<Commons | null> {
    const record = await this.call<HolochainRecord | null>('get_commons', commonsId);
    if (!record) return null;
    return this.extractEntry<Commons>(record);
  }

  /**
   * Get commons by DAO
   *
   * @param daoId - DAO identifier
   * @returns Array of commons
   */
  async getCommonsByDao(daoId: string): Promise<Commons[]> {
    const records = await this.call<HolochainRecord[]>('get_commons_by_dao', daoId);
    return records.map(r => this.extractEntry<Commons>(r));
  }

  /**
   * Get commons by type
   *
   * @param commonsType - Commons type
   * @returns Array of commons
   */
  async getCommonsByType(commonsType: CommonsType): Promise<Commons[]> {
    const records = await this.call<HolochainRecord[]>('get_commons_by_type', commonsType);
    return records.map(r => this.extractEntry<Commons>(r));
  }

  /**
   * Update governance rules
   *
   * @param commonsId - Commons identifier
   * @param rules - New rules (JSON string)
   * @returns Updated commons
   */
  async updateRules(commonsId: string, rules: string): Promise<Commons> {
    const record = await this.call<HolochainRecord>('update_commons_rules', {
      commons_id: commonsId,
      governance_rules: rules,
    });
    return this.extractEntry<Commons>(record);
  }

  // ============================================================================
  // Membership Operations
  // ============================================================================

  /**
   * Join a commons
   *
   * @param commonsId - Commons identifier
   * @param memberDid - Member's DID
   * @param accessLevel - Access level (default: Use)
   * @returns Updated commons
   */
  async joinCommons(
    commonsId: string,
    memberDid: string,
    accessLevel: CommonsAccessLevel = 'Use'
  ): Promise<Commons> {
    const record = await this.call<HolochainRecord>('join_commons', {
      commons_id: commonsId,
      member_did: memberDid,
      access_level: accessLevel,
    });
    return this.extractEntry<Commons>(record);
  }

  /**
   * Leave a commons
   *
   * @param commonsId - Commons identifier
   * @param memberDid - Member's DID
   * @returns Updated commons
   */
  async leaveCommons(commonsId: string, memberDid: string): Promise<Commons> {
    const record = await this.call<HolochainRecord>('leave_commons', {
      commons_id: commonsId,
      member_did: memberDid,
    });
    return this.extractEntry<Commons>(record);
  }

  /**
   * Update member access level
   *
   * @param commonsId - Commons identifier
   * @param memberDid - Member's DID
   * @param accessLevel - New access level
   * @returns Updated commons
   */
  async updateMemberAccess(
    commonsId: string,
    memberDid: string,
    accessLevel: CommonsAccessLevel
  ): Promise<Commons> {
    const record = await this.call<HolochainRecord>('update_member_access', {
      commons_id: commonsId,
      member_did: memberDid,
      access_level: accessLevel,
    });
    return this.extractEntry<Commons>(record);
  }

  /**
   * Get member info
   *
   * @param commonsId - Commons identifier
   * @param memberDid - Member's DID
   * @returns Member info or null
   */
  async getMember(commonsId: string, memberDid: string): Promise<CommonsMember | null> {
    return this.call<CommonsMember | null>('get_commons_member', {
      commons_id: commonsId,
      member_did: memberDid,
    });
  }

  /**
   * Get all members of a commons
   *
   * @param commonsId - Commons identifier
   * @returns Array of members
   */
  async getMembers(commonsId: string): Promise<CommonsMember[]> {
    return this.call<CommonsMember[]>('get_commons_members', commonsId);
  }

  /**
   * Award contribution credits to a member
   *
   * @param commonsId - Commons identifier
   * @param memberDid - Member's DID
   * @param credits - Credits to award
   * @param reason - Reason for credits
   * @returns Updated member
   */
  async awardCredits(
    commonsId: string,
    memberDid: string,
    credits: number,
    reason?: string
  ): Promise<CommonsMember> {
    return this.call<CommonsMember>('award_contribution_credits', {
      commons_id: commonsId,
      member_did: memberDid,
      credits,
      reason,
    });
  }

  // ============================================================================
  // Asset Operations
  // ============================================================================

  /**
   * Add an asset to commons
   *
   * @param commonsId - Commons identifier
   * @param assetId - Asset identifier
   * @returns Updated commons
   */
  async addAsset(commonsId: string, assetId: string): Promise<Commons> {
    const record = await this.call<HolochainRecord>('add_asset_to_commons', {
      commons_id: commonsId,
      asset_id: assetId,
    });
    return this.extractEntry<Commons>(record);
  }

  /**
   * Remove an asset from commons
   *
   * @param commonsId - Commons identifier
   * @param assetId - Asset identifier
   * @returns Updated commons
   */
  async removeAsset(commonsId: string, assetId: string): Promise<Commons> {
    const record = await this.call<HolochainRecord>('remove_asset_from_commons', {
      commons_id: commonsId,
      asset_id: assetId,
    });
    return this.extractEntry<Commons>(record);
  }

  // ============================================================================
  // Usage Tracking
  // ============================================================================

  /**
   * Record usage of commons
   *
   * @param input - Usage parameters
   * @returns Usage record
   */
  async recordUsage(input: {
    commons_id: string;
    user_did: string;
    purpose: string;
  }): Promise<CommonsUsage> {
    const record = await this.call<HolochainRecord>('record_commons_usage', input);
    return this.extractEntry<CommonsUsage>(record);
  }

  /**
   * End usage session
   *
   * @param usageId - Usage record ID
   * @returns Updated usage record
   */
  async endUsage(usageId: string): Promise<CommonsUsage> {
    const record = await this.call<HolochainRecord>('end_commons_usage', usageId);
    return this.extractEntry<CommonsUsage>(record);
  }

  /**
   * Get usage history for a commons
   *
   * @param commonsId - Commons identifier
   * @param limit - Maximum records
   * @returns Array of usage records
   */
  async getUsageHistory(commonsId: string, limit: number = 50): Promise<CommonsUsage[]> {
    const records = await this.call<HolochainRecord[]>('get_commons_usage_history', {
      commons_id: commonsId,
      limit,
    });
    return records.map(r => this.extractEntry<CommonsUsage>(r));
  }

  /**
   * Get usage by member
   *
   * @param commonsId - Commons identifier
   * @param memberDid - Member's DID
   * @returns Array of usage records
   */
  async getUsageByMember(commonsId: string, memberDid: string): Promise<CommonsUsage[]> {
    const records = await this.call<HolochainRecord[]>('get_usage_by_member', {
      commons_id: commonsId,
      member_did: memberDid,
    });
    return records.map(r => this.extractEntry<CommonsUsage>(r));
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Check if DID is a member
   *
   * @param commonsId - Commons identifier
   * @param did - DID to check
   * @returns True if member
   */
  async isMember(commonsId: string, did: string): Promise<boolean> {
    const member = await this.getMember(commonsId, did);
    return member !== null;
  }

  /**
   * Check if member has access level
   *
   * @param commonsId - Commons identifier
   * @param did - Member's DID
   * @param requiredLevel - Required access level
   * @returns True if member has required access
   */
  async hasAccess(
    commonsId: string,
    did: string,
    requiredLevel: CommonsAccessLevel
  ): Promise<boolean> {
    const member = await this.getMember(commonsId, did);
    if (!member) return false;

    const levels: CommonsAccessLevel[] = ['Read', 'Use', 'Contribute', 'Manage', 'Admin'];
    const memberIndex = levels.indexOf(member.access_level);
    const requiredIndex = levels.indexOf(requiredLevel);

    return memberIndex >= requiredIndex;
  }

  /**
   * Get commons type description
   *
   * @param commonsType - Commons type
   * @returns Human-readable description
   */
  getTypeDescription(commonsType: CommonsType): string {
    const descriptions: Record<CommonsType, string> = {
      Land: 'Shared land for community use',
      Water: 'Water resources and rights',
      Forest: 'Forest and timber commons',
      Infrastructure: 'Shared infrastructure',
      Digital: 'Digital commons (software, data)',
      Knowledge: 'Knowledge and intellectual commons',
      Other: 'Other shared resources',
    };
    return descriptions[commonsType];
  }

  /**
   * Get access level description
   *
   * @param level - Access level
   * @returns Human-readable description
   */
  getAccessLevelDescription(level: CommonsAccessLevel): string {
    const descriptions: Record<CommonsAccessLevel, string> = {
      Read: 'Can view commons information',
      Use: 'Can use commons resources',
      Contribute: 'Can add to commons',
      Manage: 'Can manage resources and members',
      Admin: 'Full administrative access',
    };
    return descriptions[level];
  }
}
