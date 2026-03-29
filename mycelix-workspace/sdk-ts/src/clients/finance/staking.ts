// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Staking Zome Client
 *
 * Tri-asset staking with cryptographic verification, slashing,
 * escrow, and cross-chain proofs.
 *
 * @module @mycelix/sdk/clients/finance/staking
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  TriAssetStake,
  CreateStakeInput,
  UpdateTrustInput,
  SlashStakeInput,
  SlashingEvent,
  CrossChainLockProof,
  SubmitProofInput,
  CryptoEscrow,
  CreateCryptoEscrowInput,
  RevealPreimageInput,
  AddSignatureInput,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'staking';

/**
 * Staking Client for tri-asset staking and escrow
 *
 * Provides staking with:
 * - Tri-asset stakes (MYC, ETH, USDC) with K-Vector trust weighting
 * - Cross-chain proof verification
 * - Slashing with cryptographic evidence
 * - Multi-condition escrow
 *
 * @example
 * ```typescript
 * const staking = new StakingClient(client);
 *
 * // Create a tri-asset stake
 * const stake = await staking.createStake({
 *   staker_did: 'did:mycelix:alice',
 *   myc_amount: BigInt('1000000000000000000'), // 1 MYC
 *   eth_amount: BigInt('500000000000000000'),  // 0.5 ETH
 *   usdc_amount: BigInt('1000000000'),         // 1000 USDC
 *   kvector_trust_score: 0.8,
 *   myc_lock_proof: new Uint8Array([...]),
 *   eth_lock_proof: new Uint8Array([...]),
 *   usdc_lock_proof: new Uint8Array([...]),
 * });
 *
 * // Create a crypto escrow
 * const escrow = await staking.createEscrow({
 *   depositor_did: 'did:mycelix:alice',
 *   beneficiary_did: 'did:mycelix:bob',
 *   myc_amount: BigInt('100000000000000000'),
 *   eth_amount: BigInt(0),
 *   usdc_amount: BigInt(0),
 *   purpose: 'Service payment',
 *   conditions: [{ HashLock: { hash: hashBytes, hash_type: 'Sha256' } }],
 *   required_conditions: 1,
 *   multisig_signers: [],
 * });
 * ```
 */
export class StakingClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Staking
  // ===========================================================================

  /**
   * Create a new tri-asset stake
   *
   * Requires cross-chain proofs for all three assets.
   * Stake weight is calculated based on K-Vector trust score.
   */
  async createStake(input: CreateStakeInput): Promise<TriAssetStake> {
    const record = await this.callZomeOnce<HolochainRecord>('create_stake', input);
    return this.extractEntry<TriAssetStake>(record);
  }

  /**
   * Begin unbonding a stake (21-day unbonding period)
   */
  async beginUnbonding(stakeId: string): Promise<TriAssetStake> {
    const record = await this.callZomeOnce<HolochainRecord>('begin_unbonding', stakeId);
    return this.extractEntry<TriAssetStake>(record);
  }

  /**
   * Complete withdrawal after unbonding period
   */
  async withdrawStake(stakeId: string): Promise<TriAssetStake> {
    const record = await this.callZomeOnce<HolochainRecord>('withdraw_stake', stakeId);
    return this.extractEntry<TriAssetStake>(record);
  }

  /**
   * Update stake K-Vector trust score
   */
  async updateStakeTrust(input: UpdateTrustInput): Promise<TriAssetStake> {
    const record = await this.callZomeOnce<HolochainRecord>('update_stake_trust', input);
    return this.extractEntry<TriAssetStake>(record);
  }

  /**
   * Get all stakes for a staker
   */
  async getStakerStakes(stakerDid: string): Promise<TriAssetStake[]> {
    const records = await this.callZome<HolochainRecord[]>('get_staker_stakes', stakerDid);
    return records.map((r) => this.extractEntry<TriAssetStake>(r));
  }

  /**
   * Get all active stakes
   */
  async getActiveStakes(): Promise<TriAssetStake[]> {
    const records = await this.callZome<HolochainRecord[]>('get_active_stakes', null);
    return records.map((r) => this.extractEntry<TriAssetStake>(r));
  }

  /**
   * Calculate total weighted stake in the network
   */
  async getTotalWeightedStake(): Promise<number> {
    return this.callZome<number>('get_total_weighted_stake', null);
  }

  // ===========================================================================
  // Slashing
  // ===========================================================================

  /**
   * Slash a stake with cryptographic evidence
   */
  async slashStake(input: SlashStakeInput): Promise<SlashingEvent> {
    const record = await this.callZomeOnce<HolochainRecord>('slash_stake', input);
    return this.extractEntry<SlashingEvent>(record);
  }

  // ===========================================================================
  // Cross-Chain Proofs
  // ===========================================================================

  /**
   * Submit a cross-chain lock proof
   *
   * Proofs expire in 24 hours and must be refreshed.
   */
  async submitCrossChainProof(input: SubmitProofInput): Promise<CrossChainLockProof> {
    const record = await this.callZomeOnce<HolochainRecord>('submit_cross_chain_proof', input);
    return this.extractEntry<CrossChainLockProof>(record);
  }

  // ===========================================================================
  // Escrow
  // ===========================================================================

  /**
   * Create a crypto escrow with release conditions
   *
   * Supports multiple release conditions:
   * - Hash lock (reveal preimage)
   * - Time lock (release after timestamp)
   * - Multi-sig (threshold signatures)
   * - Oracle attestation
   */
  async createEscrow(input: CreateCryptoEscrowInput): Promise<CryptoEscrow> {
    const record = await this.callZomeOnce<HolochainRecord>('create_escrow', input);
    return this.extractEntry<CryptoEscrow>(record);
  }

  /**
   * Reveal hash preimage to satisfy hash-lock condition
   */
  async revealHashPreimage(input: RevealPreimageInput): Promise<CryptoEscrow> {
    const record = await this.callZomeOnce<HolochainRecord>('reveal_hash_preimage', input);
    return this.extractEntry<CryptoEscrow>(record);
  }

  /**
   * Add multi-sig signature to escrow
   */
  async addEscrowSignature(input: AddSignatureInput): Promise<CryptoEscrow> {
    const record = await this.callZomeOnce<HolochainRecord>('add_escrow_signature', input);
    return this.extractEntry<CryptoEscrow>(record);
  }

  /**
   * Release escrow to beneficiary (when conditions are met)
   */
  async releaseEscrow(escrowId: string): Promise<CryptoEscrow> {
    const record = await this.callZomeOnce<HolochainRecord>('release_escrow', escrowId);
    return this.extractEntry<CryptoEscrow>(record);
  }

  /**
   * Get escrows where caller is depositor
   */
  async getDepositorEscrows(depositorDid: string): Promise<CryptoEscrow[]> {
    const records = await this.callZome<HolochainRecord[]>('get_depositor_escrows', depositorDid);
    return records.map((r) => this.extractEntry<CryptoEscrow>(r));
  }

  /**
   * Get escrows where caller is beneficiary
   */
  async getBeneficiaryEscrows(beneficiaryDid: string): Promise<CryptoEscrow[]> {
    const records = await this.callZome<HolochainRecord[]>(
      'get_beneficiary_escrows',
      beneficiaryDid
    );
    return records.map((r) => this.extractEntry<CryptoEscrow>(r));
  }
}
