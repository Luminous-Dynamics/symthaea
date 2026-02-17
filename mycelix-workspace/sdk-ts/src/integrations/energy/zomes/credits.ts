/**
 * Credits Zome Client
 *
 * Handles energy credits and carbon certificates.
 *
 * @module @mycelix/sdk/integrations/energy/zomes/credits
 */

import { EnergySdkError } from '../types';

import type {
  EnergyCredit,
  IssueCreditInput,
  TransferCreditInput,
  CarbonCertificate,
  RetireCarbonInput,
  EnergySource,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Credits client
 */
export interface CreditsClientConfig {
  roleId: string;
  zomeName: string;
}

const DEFAULT_CONFIG: CreditsClientConfig = {
  roleId: 'energy',
  zomeName: 'credits',
};

/**
 * Client for energy credits and carbon certificate operations
 *
 * @example
 * ```typescript
 * const credits = new CreditsClient(holochainClient);
 *
 * // Issue energy credits
 * const credit = await credits.issueCredit({
 *   holder_did: 'did:mycelix:producer',
 *   amount_kwh: 100,
 *   source: 'Solar',
 *   transferable: true,
 * });
 *
 * // Transfer credits
 * await credits.transferCredit({
 *   credit_id: credit.id,
 *   recipient_did: 'did:mycelix:buyer',
 * });
 *
 * // Get carbon certificates
 * const certs = await credits.getCarbonCertificates('did:mycelix:producer');
 * ```
 */
export class CreditsClient {
  private readonly config: CreditsClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<CreditsClientConfig> = {}
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
      throw new EnergySdkError(
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
      throw new EnergySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Energy Credit Operations
  // ============================================================================

  /**
   * Issue energy credits
   *
   * @param input - Credit issuance parameters
   * @returns The created credit
   */
  async issueCredit(input: IssueCreditInput): Promise<EnergyCredit> {
    const carbonOffsetKg = this.calculateCarbonOffset(input.amount_kwh, input.source);
    const record = await this.call<HolochainRecord>('issue_credit', {
      ...input,
      carbon_offset_kg: carbonOffsetKg,
      issued_at: Date.now() * 1000,
      used: false,
      transferable: input.transferable ?? true,
    });
    return this.extractEntry<EnergyCredit>(record);
  }

  /**
   * Get a credit by ID
   *
   * @param creditId - Credit identifier
   * @returns The credit or null if not found
   */
  async getCredit(creditId: string): Promise<EnergyCredit | null> {
    const record = await this.call<HolochainRecord | null>('get_credit', creditId);
    if (!record) return null;
    return this.extractEntry<EnergyCredit>(record);
  }

  /**
   * Get credits for a holder
   *
   * @param holderDid - Holder's DID
   * @returns Array of credits
   */
  async getCredits(holderDid: string): Promise<EnergyCredit[]> {
    const records = await this.call<HolochainRecord[]>('get_credits', holderDid);
    return records.map(r => this.extractEntry<EnergyCredit>(r));
  }

  /**
   * Get available (unused, not expired) credits
   *
   * @param holderDid - Holder's DID
   * @returns Array of available credits
   */
  async getAvailableCredits(holderDid: string): Promise<EnergyCredit[]> {
    const credits = await this.getCredits(holderDid);
    const now = Date.now() * 1000;
    return credits.filter(c =>
      !c.used &&
      (!c.expires_at || c.expires_at > now)
    );
  }

  /**
   * Get credits by energy source
   *
   * @param holderDid - Holder's DID
   * @param source - Energy source
   * @returns Array of credits
   */
  async getCreditsBySource(holderDid: string, source: EnergySource): Promise<EnergyCredit[]> {
    const credits = await this.getCredits(holderDid);
    return credits.filter(c => c.source === source);
  }

  /**
   * Transfer a credit to another holder
   *
   * @param input - Transfer parameters
   * @returns The updated credit
   */
  async transferCredit(input: TransferCreditInput): Promise<EnergyCredit> {
    const record = await this.call<HolochainRecord>('transfer_credit', input);
    return this.extractEntry<EnergyCredit>(record);
  }

  /**
   * Use a credit (mark as consumed)
   *
   * @param creditId - Credit identifier
   * @returns The updated credit
   */
  async useCredit(creditId: string): Promise<EnergyCredit> {
    const record = await this.call<HolochainRecord>('use_credit', creditId);
    return this.extractEntry<EnergyCredit>(record);
  }

  /**
   * Get total available credit balance
   *
   * @param holderDid - Holder's DID
   * @returns Total available kWh
   */
  async getCreditBalance(holderDid: string): Promise<number> {
    const credits = await this.getAvailableCredits(holderDid);
    return credits.reduce((sum, c) => sum + c.amount_kwh, 0);
  }

  // ============================================================================
  // Carbon Certificate Operations
  // ============================================================================

  /**
   * Get carbon certificates for a holder
   *
   * @param holderDid - Holder's DID
   * @returns Array of carbon certificates
   */
  async getCarbonCertificates(holderDid: string): Promise<CarbonCertificate[]> {
    const records = await this.call<HolochainRecord[]>('get_carbon_credits', holderDid);
    return records.map(r => this.extractEntry<CarbonCertificate>(r));
  }

  /**
   * Get active (not retired) carbon certificates
   *
   * @param holderDid - Holder's DID
   * @returns Array of active certificates
   */
  async getActiveCarbonCertificates(holderDid: string): Promise<CarbonCertificate[]> {
    const certs = await this.getCarbonCertificates(holderDid);
    const now = Date.now() * 1000;
    return certs.filter(c =>
      !c.retired &&
      (!c.expires_at || c.expires_at > now)
    );
  }

  /**
   * Transfer a carbon certificate
   *
   * @param certificateId - Certificate identifier
   * @param recipientDid - Recipient's DID
   * @returns The updated certificate
   */
  async transferCarbonCertificate(
    certificateId: string,
    recipientDid: string
  ): Promise<CarbonCertificate> {
    const record = await this.call<HolochainRecord>('transfer_carbon_credits', {
      certificate_id: certificateId,
      recipient_did: recipientDid,
    });
    return this.extractEntry<CarbonCertificate>(record);
  }

  /**
   * Retire a carbon certificate (permanently remove from circulation)
   *
   * @param input - Retirement parameters
   * @returns The retired certificate
   */
  async retireCarbonCertificate(input: RetireCarbonInput): Promise<CarbonCertificate> {
    const record = await this.call<HolochainRecord>('retire_carbon_credits', input);
    return this.extractEntry<CarbonCertificate>(record);
  }

  /**
   * Get total carbon offset for a holder
   *
   * @param holderDid - Holder's DID
   * @returns Total carbon offset in kg CO2
   */
  async getTotalCarbonOffset(holderDid: string): Promise<number> {
    const certs = await this.getCarbonCertificates(holderDid);
    return certs.reduce((sum, c) => sum + c.amount_kg_co2, 0);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick credit issuance
   *
   * @param holderDid - Holder's DID
   * @param amountKwh - Amount in kWh
   * @param source - Energy source
   * @returns The created credit
   */
  async quickIssue(
    holderDid: string,
    amountKwh: number,
    source: EnergySource
  ): Promise<EnergyCredit> {
    return this.issueCredit({
      holder_did: holderDid,
      amount_kwh: amountKwh,
      source,
      transferable: true,
    });
  }

  /**
   * Calculate carbon offset for energy production
   *
   * @param amountKwh - Energy amount in kWh
   * @param source - Energy source
   * @returns Carbon offset in kg CO2
   */
  calculateCarbonOffset(amountKwh: number, source: EnergySource): number {
    if (source === 'Grid') return 0;
    const gridEmissionsFactor = 0.4; // kg CO2 per kWh (US average)
    return amountKwh * gridEmissionsFactor;
  }

  /**
   * Check if credit is transferable
   *
   * @param credit - The credit
   * @returns True if transferable
   */
  isTransferable(credit: EnergyCredit): boolean {
    if (!credit.transferable) return false;
    if (credit.used) return false;
    if (credit.expires_at && credit.expires_at < Date.now() * 1000) return false;
    return true;
  }

  /**
   * Check if credit is usable
   *
   * @param credit - The credit
   * @returns True if usable
   */
  isUsable(credit: EnergyCredit): boolean {
    if (credit.used) return false;
    if (credit.expires_at && credit.expires_at < Date.now() * 1000) return false;
    return true;
  }

  /**
   * Get credits expiring soon
   *
   * @param holderDid - Holder's DID
   * @param withinDays - Days until expiration (default 30)
   * @returns Array of expiring credits
   */
  async getExpiringCredits(holderDid: string, withinDays: number = 30): Promise<EnergyCredit[]> {
    const credits = await this.getAvailableCredits(holderDid);
    const deadline = (Date.now() + withinDays * 24 * 60 * 60 * 1000) * 1000;
    return credits.filter(c => c.expires_at && c.expires_at < deadline);
  }

  /**
   * Get credit summary for a holder
   *
   * @param holderDid - Holder's DID
   * @returns Credit summary
   */
  async getCreditSummary(holderDid: string): Promise<{
    totalKwh: number;
    availableKwh: number;
    usedKwh: number;
    expiredKwh: number;
    carbonOffsetKg: number;
    bySource: Record<EnergySource, number>;
  }> {
    const credits = await this.getCredits(holderDid);
    const now = Date.now() * 1000;

    let totalKwh = 0;
    let availableKwh = 0;
    let usedKwh = 0;
    let expiredKwh = 0;
    let carbonOffsetKg = 0;
    const bySource: Record<string, number> = {};

    for (const credit of credits) {
      totalKwh += credit.amount_kwh;
      carbonOffsetKg += credit.carbon_offset_kg;

      if (credit.used) {
        usedKwh += credit.amount_kwh;
      } else if (credit.expires_at && credit.expires_at < now) {
        expiredKwh += credit.amount_kwh;
      } else {
        availableKwh += credit.amount_kwh;
      }

      bySource[credit.source] = (bySource[credit.source] || 0) + credit.amount_kwh;
    }

    return {
      totalKwh,
      availableKwh,
      usedKwh,
      expiredKwh,
      carbonOffsetKg,
      bySource: bySource as Record<EnergySource, number>,
    };
  }
}
