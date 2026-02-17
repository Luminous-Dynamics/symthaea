/**
 * @mycelix/sdk Energy Integration - Legacy Types and Service
 *
 * @deprecated This module contains legacy types and services that are kept for backward compatibility.
 * New code should use MycelixEnergyClient and types from './types.js' instead.
 *
 * @packageDocumentation
 * @module integrations/energy/legacy
 */

import { LocalBridge } from '../../bridge/index.js';
import {
  createReputation,
  recordPositive,
  type ReputationScore,
} from '../../matl/index.js';
import { EnergyValidators } from '../../utils/validation.js';

// ============================================================================
// Legacy Types (kept for backward compatibility)
// ============================================================================

/** @deprecated Use types from './types.js' instead */
export type LegacyEnergySource = 'solar' | 'wind' | 'hydro' | 'battery' | 'grid' | 'other_renewable';

/** @deprecated Use types from './types.js' instead */
export type ParticipantRole = 'producer' | 'consumer' | 'prosumer' | 'storage' | 'grid_operator';

/** @deprecated Use types from './types.js' instead */
export type EnergyTransactionType = 'sale' | 'purchase' | 'donation' | 'grid_feed' | 'storage';

/** @deprecated Use EnergyParticipant from './types.js' instead */
export interface LegacyEnergyParticipant {
  id: string;
  did: string;
  role: ParticipantRole;
  sources: LegacyEnergySource[];
  capacity: number;
  reputation: ReputationScore;
  totalProduced: number;
  totalConsumed: number;
  carbonOffset: number;
  location?: { lat: number; lng: number };
  registeredAt: number;
}

/** @deprecated Use ProductionReading from './types.js' instead */
export interface LegacyEnergyReading {
  id: string;
  participantId: string;
  timestamp: number;
  production: number;
  consumption: number;
  netExport: number;
  source: LegacyEnergySource;
  verified: boolean;
}

/** @deprecated Use EnergyTrade from './types.js' instead */
export interface LegacyEnergyTransaction {
  id: string;
  type: EnergyTransactionType;
  sellerId: string;
  buyerId: string;
  amount: number;
  pricePerKwh?: number;
  totalPrice?: number;
  source: LegacyEnergySource;
  carbonCredits: number;
  timestamp: number;
  settlementStatus: 'pending' | 'settled' | 'disputed';
}

/** @deprecated Use GridBalanceRequest from './types.js' instead */
export interface LegacyGridBalanceRequest {
  id: string;
  requesterId: string;
  type: 'supply' | 'demand';
  amountNeeded: number;
  maxPrice?: number;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  expiresAt: number;
  fulfilled: boolean;
  fulfilledBy?: string[];
}

/** @deprecated Use EnergyCredit from './types.js' instead */
export interface LegacyEnergyCredit {
  id: string;
  holderId: string;
  amount: number;
  source: LegacyEnergySource;
  carbonOffset: number;
  issuedAt: number;
  expiresAt?: number;
  transferable: boolean;
}

// ============================================================================
// Legacy Service (kept for backward compatibility)
// ============================================================================

/**
 * Legacy Energy service for distributed grid management
 * @deprecated Use MycelixEnergyClient instead
 */
export class EnergyService {
  private participants = new Map<string, LegacyEnergyParticipant>();
  private readings: LegacyEnergyReading[] = [];
  private transactions: LegacyEnergyTransaction[] = [];
  private balanceRequests = new Map<string, LegacyGridBalanceRequest>();
  private credits = new Map<string, LegacyEnergyCredit[]>();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('energy');
  }

  registerParticipant(
    did: string,
    role: ParticipantRole,
    sources: LegacyEnergySource[],
    capacity: number,
    location?: { lat: number; lng: number }
  ): LegacyEnergyParticipant {
    const id = `participant-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const participant: LegacyEnergyParticipant = {
      id,
      did,
      role,
      sources,
      capacity,
      reputation: createReputation(did),
      totalProduced: 0,
      totalConsumed: 0,
      carbonOffset: 0,
      location,
      registeredAt: Date.now(),
    };

    this.participants.set(id, participant);
    this.credits.set(id, []);
    return participant;
  }

  submitReading(
    participantId: string,
    production: number,
    consumption: number,
    source: LegacyEnergySource
  ): LegacyEnergyReading {
    EnergyValidators.reading(production, consumption);

    const participant = this.participants.get(participantId);
    if (!participant) throw new Error('Participant not found');

    const reading: LegacyEnergyReading = {
      id: `reading-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      participantId,
      timestamp: Date.now(),
      production,
      consumption,
      netExport: production - consumption,
      source,
      verified: false,
    };

    this.readings.push(reading);

    participant.totalProduced += production;
    participant.totalConsumed += consumption;

    if (source !== 'grid') {
      const carbonPerKwh = 0.5;
      participant.carbonOffset += production * carbonPerKwh;
    }

    if (reading.netExport > 0) {
      this.issueCredit(participantId, reading.netExport, source);
    }

    return reading;
  }

  issueCredit(holderId: string, amount: number, source: LegacyEnergySource): LegacyEnergyCredit {
    const credit: LegacyEnergyCredit = {
      id: `credit-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      holderId,
      amount,
      source,
      carbonOffset: source !== 'grid' ? amount * 0.5 : 0,
      issuedAt: Date.now(),
      transferable: true,
    };

    const holderCredits = this.credits.get(holderId) || [];
    holderCredits.push(credit);
    this.credits.set(holderId, holderCredits);

    return credit;
  }

  tradeEnergy(
    sellerId: string,
    buyerId: string,
    amount: number,
    source: LegacyEnergySource,
    pricePerKwh?: number
  ): LegacyEnergyTransaction {
    if (pricePerKwh !== undefined) {
      EnergyValidators.trade(pricePerKwh, amount);
    }

    const seller = this.participants.get(sellerId);
    const buyer = this.participants.get(buyerId);

    if (!seller || !buyer) throw new Error('Participant not found');

    const sellerCredits = this.credits.get(sellerId) || [];
    const availableCredits = sellerCredits
      .filter((c) => c.transferable && c.source === source)
      .reduce((sum, c) => sum + c.amount, 0);

    if (availableCredits < amount) {
      throw new Error('Insufficient energy credits');
    }

    const transaction: LegacyEnergyTransaction = {
      id: `tx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      type: 'sale',
      sellerId,
      buyerId,
      amount,
      pricePerKwh,
      totalPrice: pricePerKwh ? amount * pricePerKwh : undefined,
      source,
      carbonCredits: source !== 'grid' ? amount * 0.5 : 0,
      timestamp: Date.now(),
      settlementStatus: 'settled',
    };

    this.transactions.push(transaction);
    this.transferCredits(sellerId, buyerId, amount, source);

    seller.reputation = recordPositive(seller.reputation);
    buyer.reputation = recordPositive(buyer.reputation);

    return transaction;
  }

  requestGridBalance(
    requesterId: string,
    type: 'supply' | 'demand',
    amountNeeded: number,
    maxPrice?: number,
    urgency: LegacyGridBalanceRequest['urgency'] = 'medium'
  ): LegacyGridBalanceRequest {
    const expiryHours = urgency === 'critical' ? 1 : urgency === 'high' ? 4 : urgency === 'medium' ? 12 : 24;

    const request: LegacyGridBalanceRequest = {
      id: `balance-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      requesterId,
      type,
      amountNeeded,
      maxPrice,
      urgency,
      expiresAt: Date.now() + expiryHours * 60 * 60 * 1000,
      fulfilled: false,
    };

    this.balanceRequests.set(request.id, request);
    return request;
  }

  getParticipant(participantId: string): LegacyEnergyParticipant | undefined {
    return this.participants.get(participantId);
  }

  getCredits(participantId: string): LegacyEnergyCredit[] {
    return this.credits.get(participantId) || [];
  }

  getGridStats(): {
    totalProduction: number;
    totalConsumption: number;
    netBalance: number;
    participantCount: number;
    carbonOffset: number;
  } {
    let totalProduction = 0;
    let totalConsumption = 0;
    let carbonOffset = 0;

    for (const participant of this.participants.values()) {
      totalProduction += participant.totalProduced;
      totalConsumption += participant.totalConsumed;
      carbonOffset += participant.carbonOffset;
    }

    return {
      totalProduction,
      totalConsumption,
      netBalance: totalProduction - totalConsumption,
      participantCount: this.participants.size,
      carbonOffset,
    };
  }

  private transferCredits(fromId: string, toId: string, amount: number, source: LegacyEnergySource): void {
    const fromCredits = this.credits.get(fromId) || [];
    const toCredits = this.credits.get(toId) || [];

    let remaining = amount;
    const newFromCredits: LegacyEnergyCredit[] = [];

    for (const credit of fromCredits) {
      if (credit.source !== source || !credit.transferable || remaining <= 0) {
        newFromCredits.push(credit);
        continue;
      }

      if (credit.amount <= remaining) {
        credit.holderId = toId;
        toCredits.push(credit);
        remaining -= credit.amount;
      } else {
        const transferCredit = { ...credit, id: `credit-${Date.now()}`, amount: remaining, holderId: toId };
        toCredits.push(transferCredit);
        credit.amount -= remaining;
        newFromCredits.push(credit);
        remaining = 0;
      }
    }

    this.credits.set(fromId, newFromCredits);
    this.credits.set(toId, toCredits);
  }
}

// Legacy singleton
let instance: EnergyService | null = null;

/** @deprecated Use MycelixEnergyClient instead */
export function getEnergyService(): EnergyService {
  if (!instance) instance = new EnergyService();
  return instance;
}

/** @deprecated Use MycelixEnergyClient instead */
export function resetEnergyService(): void {
  instance = null;
}
