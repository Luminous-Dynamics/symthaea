/**
 * @mycelix/sdk Property Validated Clients
 */

import { z } from 'zod';

import { MycelixError, ErrorCode } from '../errors.js';

import {
  RegistryClient,
  TransferClient,
  LienClient,
  CommonsClient,
  type ZomeCallable,
  type HolochainRecord,
  type Asset,
  type RegisterAssetInput,
  type TitleTransfer,
  type InitiateTransferInput,
  type Lien,
  type FileLienInput,
  type Commons,
  type CreateCommonsInput,
  type AssetType,
} from './index.js';

const didSchema = z
  .string()
  .refine((val) => val.startsWith('did:'), { message: 'Must be a valid DID' });
const assetTypeSchema = z.enum([
  'RealEstate',
  'Vehicle',
  'Equipment',
  'Intellectual',
  'Digital',
  'Other',
]);
const ownershipTypeSchema = z.enum(['Sole', 'Joint', 'Fractional', 'Commons']);
const lienTypeSchema = z.enum(['Mortgage', 'Tax', 'Mechanic', 'Judgment', 'Other']);
const commonsTypeSchema = z.enum(['Land', 'Water', 'Forest', 'Infrastructure', 'Digital', 'Other']);

const geoLocationSchema = z.object({
  lat: z.number().min(-90).max(90),
  lng: z.number().min(-180).max(180),
  address: z.string().optional(),
  geohash: z.string().optional(),
});

const registerAssetInputSchema = z.object({
  type_: assetTypeSchema,
  name: z.string().min(1, 'Name is required').max(200),
  description: z.string().min(1, 'Description is required'),
  ownership_type: ownershipTypeSchema,
  location: geoLocationSchema.optional(),
  valuation: z.number().min(0).optional(),
  valuation_currency: z.string().optional(),
  metadata: z.string().optional(),
});

const initiateTransferInputSchema = z.object({
  asset_id: z.string().min(1),
  to_owner: didSchema,
  percentage: z.number().min(0).max(100),
  consideration: z.number().min(0).optional(),
  currency: z.string().optional(),
  use_escrow: z.boolean().optional(),
});

const fileLienInputSchema = z.object({
  asset_id: z.string().min(1),
  type_: lienTypeSchema,
  amount: z.number().positive(),
  currency: z.string().min(1),
});

const createCommonsInputSchema = z.object({
  name: z.string().min(1).max(200),
  description: z.string().min(1),
  type_: commonsTypeSchema,
  managing_dao: z.string().min(1),
  boundary: z.array(geoLocationSchema).optional(),
  rules: z.string().min(1),
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

export class ValidatedRegistryClient {
  private client: RegistryClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new RegistryClient(zomeClient);
  }

  async registerAsset(input: RegisterAssetInput): Promise<HolochainRecord<Asset>> {
    validateOrThrow(registerAssetInputSchema, input, 'registerAsset input');
    return this.client.registerAsset(input);
  }

  async getAsset(assetId: string): Promise<HolochainRecord<Asset> | null> {
    validateOrThrow(z.string().min(1), assetId, 'assetId');
    return this.client.getAsset(assetId);
  }

  async getAssetsByOwner(ownerDid: string): Promise<HolochainRecord<Asset>[]> {
    validateOrThrow(didSchema, ownerDid, 'ownerDid');
    return this.client.getAssetsByOwner(ownerDid);
  }

  async getAssetsByType(assetType: AssetType): Promise<HolochainRecord<Asset>[]> {
    validateOrThrow(assetTypeSchema, assetType, 'assetType');
    return this.client.getAssetsByType(assetType);
  }

  async updateAsset(
    assetId: string,
    updates: Partial<RegisterAssetInput>
  ): Promise<HolochainRecord<Asset>> {
    validateOrThrow(z.string().min(1), assetId, 'assetId');
    return this.client.updateAsset(assetId, updates);
  }

  async getAssetHistory(assetId: string): Promise<HolochainRecord<Asset>[]> {
    validateOrThrow(z.string().min(1), assetId, 'assetId');
    return this.client.getAssetHistory(assetId);
  }
}

export class ValidatedTransferClient {
  private client: TransferClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new TransferClient(zomeClient);
  }

  async initiateTransfer(input: InitiateTransferInput): Promise<HolochainRecord<TitleTransfer>> {
    validateOrThrow(initiateTransferInputSchema, input, 'initiateTransfer input');
    return this.client.initiateTransfer(input);
  }

  async acceptTransfer(transferId: string): Promise<HolochainRecord<TitleTransfer>> {
    validateOrThrow(z.string().min(1), transferId, 'transferId');
    return this.client.acceptTransfer(transferId);
  }

  async cancelTransfer(transferId: string): Promise<HolochainRecord<TitleTransfer>> {
    validateOrThrow(z.string().min(1), transferId, 'transferId');
    return this.client.cancelTransfer(transferId);
  }

  async getTransfer(transferId: string): Promise<HolochainRecord<TitleTransfer> | null> {
    validateOrThrow(z.string().min(1), transferId, 'transferId');
    return this.client.getTransfer(transferId);
  }

  async getTransfersForAsset(assetId: string): Promise<HolochainRecord<TitleTransfer>[]> {
    validateOrThrow(z.string().min(1), assetId, 'assetId');
    return this.client.getTransfersForAsset(assetId);
  }

  async getPendingTransfers(did: string): Promise<HolochainRecord<TitleTransfer>[]> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getPendingTransfers(did);
  }
}

export class ValidatedLienClient {
  private client: LienClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new LienClient(zomeClient);
  }

  async fileLien(input: FileLienInput): Promise<HolochainRecord<Lien>> {
    validateOrThrow(fileLienInputSchema, input, 'fileLien input');
    return this.client.fileLien(input);
  }

  async getLien(lienId: string): Promise<HolochainRecord<Lien> | null> {
    validateOrThrow(z.string().min(1), lienId, 'lienId');
    return this.client.getLien(lienId);
  }

  async getLiensForAsset(assetId: string): Promise<HolochainRecord<Lien>[]> {
    validateOrThrow(z.string().min(1), assetId, 'assetId');
    return this.client.getLiensForAsset(assetId);
  }

  async satisfyLien(lienId: string): Promise<HolochainRecord<Lien>> {
    validateOrThrow(z.string().min(1), lienId, 'lienId');
    return this.client.satisfyLien(lienId);
  }

  async releaseLien(lienId: string): Promise<HolochainRecord<Lien>> {
    validateOrThrow(z.string().min(1), lienId, 'lienId');
    return this.client.releaseLien(lienId);
  }
}

export class ValidatedCommonsClient {
  private client: CommonsClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new CommonsClient(zomeClient);
  }

  async createCommons(input: CreateCommonsInput): Promise<HolochainRecord<Commons>> {
    validateOrThrow(createCommonsInputSchema, input, 'createCommons input');
    return this.client.createCommons(input);
  }

  async getCommons(commonsId: string): Promise<HolochainRecord<Commons> | null> {
    validateOrThrow(z.string().min(1), commonsId, 'commonsId');
    return this.client.getCommons(commonsId);
  }

  async getCommonsByDAO(daoId: string): Promise<HolochainRecord<Commons>[]> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    return this.client.getCommonsByDAO(daoId);
  }

  async joinCommons(commonsId: string): Promise<HolochainRecord<Commons>> {
    validateOrThrow(z.string().min(1), commonsId, 'commonsId');
    return this.client.joinCommons(commonsId);
  }

  async updateRules(commonsId: string, rules: string): Promise<HolochainRecord<Commons>> {
    validateOrThrow(z.string().min(1), commonsId, 'commonsId');
    validateOrThrow(z.string().min(1), rules, 'rules');
    return this.client.updateRules(commonsId, rules);
  }
}

export function createValidatedPropertyClients(client: ZomeCallable) {
  return {
    registry: new ValidatedRegistryClient(client),
    transfers: new ValidatedTransferClient(client),
    liens: new ValidatedLienClient(client),
    commons: new ValidatedCommonsClient(client),
  };
}
