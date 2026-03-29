// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Energy Validated Clients
 */

import { z } from 'zod';

import { MycelixError, ErrorCode } from '../errors.js';

import {
  ProjectsClient,
  ParticipantsClient,
  TradingClient,
  CreditsClient,
  InvestmentsClient,
  type ZomeCallable,
  type HolochainRecord,
  type EnergyProject,
  type RegisterProjectInput,
  type EnergyParticipant,
  type RegisterParticipantInput,
  type EnergyTrade,
  type TradeInput,
  type EnergyCredit,
  type Investment,
  type InvestInput,
  type EnergySource,
  type ParticipantType,
  type ProjectStatus,
} from './index.js';

const didSchema = z
  .string()
  .refine((val) => val.startsWith('did:'), { message: 'Must be a valid DID' });
const energySourceSchema = z.enum([
  'Solar',
  'Wind',
  'Hydro',
  'Nuclear',
  'Geothermal',
  'Storage',
  'Other',
]);
const projectStatusSchema = z.enum([
  'Proposed',
  'Planning',
  'Development',
  'Construction',
  'Operational',
  'Decommissioned',
]);
const participantTypeSchema = z.enum(['Producer', 'Consumer', 'Prosumer', 'Operator', 'Investor']);

const locationSchema = z.object({
  lat: z.number().min(-90).max(90),
  lng: z.number().min(-180).max(180),
  address: z.string().optional(),
});

const registerProjectInputSchema = z.object({
  name: z.string().min(1).max(200),
  description: z.string().min(1),
  source: energySourceSchema,
  capacity_kw: z.number().positive(),
  location: locationSchema,
  investment_goal: z.number().positive().optional(),
});

const registerParticipantInputSchema = z.object({
  type_: participantTypeSchema,
  sources: z.array(energySourceSchema).min(1),
  capacity_kwh: z.number().min(0),
  location: locationSchema.optional(),
});

const tradeInputSchema = z.object({
  buyer: didSchema,
  amount_kwh: z.number().positive(),
  source: energySourceSchema,
  price_per_kwh: z.number().positive(),
  currency: z.string().min(1),
});

const investInputSchema = z.object({
  project_id: z.string().min(1),
  amount: z.number().positive(),
  currency: z.string().min(1),
});

function validateOrThrow<T>(schema: z.ZodSchema<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errors = result.error.issues.map((e) => `${e.path.join('.')}: ${e.message}`).join('; ');
    throw new MycelixError(
      `Validation failed for ${context}: ${errors}`,
      ErrorCode.INVALID_ARGUMENT
    );
  }
  return result.data;
}

export class ValidatedProjectsClient {
  private client: ProjectsClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new ProjectsClient(zomeClient);
  }

  async registerProject(input: RegisterProjectInput): Promise<HolochainRecord<EnergyProject>> {
    validateOrThrow(registerProjectInputSchema, input, 'registerProject input');
    return this.client.registerProject(input);
  }

  async getProject(projectId: string): Promise<HolochainRecord<EnergyProject> | null> {
    validateOrThrow(z.string().min(1), projectId, 'projectId');
    return this.client.getProject(projectId);
  }

  async getProjectsBySource(source: EnergySource): Promise<HolochainRecord<EnergyProject>[]> {
    validateOrThrow(energySourceSchema, source, 'source');
    return this.client.getProjectsBySource(source);
  }

  async getProjectsByStatus(status: ProjectStatus): Promise<HolochainRecord<EnergyProject>[]> {
    validateOrThrow(projectStatusSchema, status, 'status');
    return this.client.getProjectsByStatus(status);
  }

  async searchProjects(query: {
    source?: EnergySource;
    status?: ProjectStatus;
    minCapacity?: number;
  }): Promise<HolochainRecord<EnergyProject>[]> {
    return this.client.searchProjects(query);
  }
}

export class ValidatedParticipantsClient {
  private client: ParticipantsClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new ParticipantsClient(zomeClient);
  }

  async register(input: RegisterParticipantInput): Promise<HolochainRecord<EnergyParticipant>> {
    validateOrThrow(registerParticipantInputSchema, input, 'register input');
    return this.client.register(input);
  }

  async getParticipant(did: string): Promise<HolochainRecord<EnergyParticipant> | null> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getParticipant(did);
  }

  async getParticipantsByType(
    type_: ParticipantType
  ): Promise<HolochainRecord<EnergyParticipant>[]> {
    validateOrThrow(participantTypeSchema, type_, 'type_');
    return this.client.getParticipantsByType(type_);
  }

  async updateCapacity(capacityKwh: number): Promise<HolochainRecord<EnergyParticipant>> {
    validateOrThrow(z.number().min(0), capacityKwh, 'capacityKwh');
    return this.client.updateCapacity(capacityKwh);
  }
}

export class ValidatedTradingClient {
  private client: TradingClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new TradingClient(zomeClient);
  }

  async createTrade(input: TradeInput): Promise<HolochainRecord<EnergyTrade>> {
    validateOrThrow(tradeInputSchema, input, 'createTrade input');
    return this.client.createTrade(input);
  }

  async acceptTrade(tradeId: string): Promise<HolochainRecord<EnergyTrade>> {
    validateOrThrow(z.string().min(1), tradeId, 'tradeId');
    return this.client.acceptTrade(tradeId);
  }

  async confirmDelivery(tradeId: string): Promise<HolochainRecord<EnergyTrade>> {
    validateOrThrow(z.string().min(1), tradeId, 'tradeId');
    return this.client.confirmDelivery(tradeId);
  }

  async getTradesByParticipant(did: string): Promise<HolochainRecord<EnergyTrade>[]> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getTradesByParticipant(did);
  }

  async getOpenTrades(source?: EnergySource): Promise<HolochainRecord<EnergyTrade>[]> {
    if (source !== undefined) validateOrThrow(energySourceSchema, source, 'source');
    return this.client.getOpenTrades(source);
  }
}

export class ValidatedCreditsClient {
  private client: CreditsClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new CreditsClient(zomeClient);
  }

  async issueCredit(
    participantDid: string,
    amountKwh: number,
    source: EnergySource
  ): Promise<HolochainRecord<EnergyCredit>> {
    validateOrThrow(didSchema, participantDid, 'participantDid');
    validateOrThrow(z.number().positive(), amountKwh, 'amountKwh');
    validateOrThrow(energySourceSchema, source, 'source');
    return this.client.issueCredit(participantDid, amountKwh, source);
  }

  async getCredits(participantDid: string): Promise<HolochainRecord<EnergyCredit>[]> {
    validateOrThrow(didSchema, participantDid, 'participantDid');
    return this.client.getCredits(participantDid);
  }

  async useCredit(creditId: string): Promise<HolochainRecord<EnergyCredit>> {
    validateOrThrow(z.string().min(1), creditId, 'creditId');
    return this.client.useCredit(creditId);
  }
}

export class ValidatedInvestmentsClient {
  private client: InvestmentsClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new InvestmentsClient(zomeClient);
  }

  async invest(input: InvestInput): Promise<HolochainRecord<Investment>> {
    validateOrThrow(investInputSchema, input, 'invest input');
    return this.client.invest(input);
  }

  async getInvestmentsByProject(projectId: string): Promise<HolochainRecord<Investment>[]> {
    validateOrThrow(z.string().min(1), projectId, 'projectId');
    return this.client.getInvestmentsByProject(projectId);
  }

  async getInvestmentsByInvestor(investorDid: string): Promise<HolochainRecord<Investment>[]> {
    validateOrThrow(didSchema, investorDid, 'investorDid');
    return this.client.getInvestmentsByInvestor(investorDid);
  }
}

export function createValidatedEnergyClients(client: ZomeCallable) {
  return {
    projects: new ValidatedProjectsClient(client),
    participants: new ValidatedParticipantsClient(client),
    trading: new ValidatedTradingClient(client),
    credits: new ValidatedCreditsClient(client),
    investments: new ValidatedInvestmentsClient(client),
  };
}
