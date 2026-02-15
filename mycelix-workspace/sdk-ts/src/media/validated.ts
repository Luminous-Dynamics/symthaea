/**
 * @mycelix/sdk Media Validated Clients
 */

import { z } from 'zod';

import { MycelixError, ErrorCode } from '../errors.js';

import {
  PublicationClient,
  AttributionClient,
  FactCheckClient,
  CurationClient,
  type ZomeCallable,
  type HolochainRecord,
  type Content,
  type PublishContentInput,
  type Attribution,
  type AddAttributionInput,
  type FactCheck,
  type SubmitFactCheckInput,
  type CurationList,
  type CreateCurationListInput,
  type Endorsement,
  type ContentType,
  type VerificationStatus,
} from './index.js';

const didSchema = z
  .string()
  .refine((val) => val.startsWith('did:'), { message: 'Must be a valid DID' });
const contentTypeSchema = z.enum([
  'Article',
  'Opinion',
  'Investigation',
  'Review',
  'Analysis',
  'Interview',
  'Report',
  'Editorial',
  'Other',
]);
const licenseTypeSchema = z.enum([
  'CC0',
  'CCBY',
  'CCBYSA',
  'CCBYNC',
  'CCBYNCSA',
  'AllRightsReserved',
  'Custom',
]);
const verificationStatusSchema = z.enum([
  'Unverified',
  'Pending',
  'Verified',
  'Disputed',
  'Debunked',
]);
const attributionRoleSchema = z.enum(['Author', 'Editor', 'Contributor', 'Source', 'Translator']);

const epistemicScoreSchema = z.object({
  empirical: z.number().min(0).max(1),
  normative: z.number().min(0).max(1),
  metaphorical: z.number().min(0).max(1),
});

const publishContentInputSchema = z.object({
  type_: contentTypeSchema,
  title: z.string().min(1).max(500),
  description: z.string().optional(),
  content_hash: z.string().min(32),
  storage_uri: z.string().min(1),
  license: licenseTypeSchema,
  custom_license: z.string().optional(),
  tags: z.array(z.string()).optional(),
});

const addAttributionInputSchema = z.object({
  content_id: z.string().min(1),
  contributor: didSchema,
  role: attributionRoleSchema,
  description: z.string().optional(),
  royalty_percentage: z.number().min(0).max(100).optional(),
});

const submitFactCheckInputSchema = z.object({
  content_id: z.string().min(1),
  verdict: verificationStatusSchema,
  reasoning: z.string().min(20),
  sources: z.array(z.string()).min(1),
  epistemic_score: epistemicScoreSchema,
});

const createCurationListInputSchema = z.object({
  name: z.string().min(1).max(200),
  description: z.string().min(1),
  content_ids: z.array(z.string()).optional(),
  public_: z.boolean(),
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

export class ValidatedPublicationClient {
  private client: PublicationClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new PublicationClient(zomeClient);
  }

  async publish(input: PublishContentInput): Promise<HolochainRecord<Content>> {
    validateOrThrow(publishContentInputSchema, input, 'publish input');
    return this.client.publish(input);
  }

  async getContent(contentId: string): Promise<HolochainRecord<Content> | null> {
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.getContent(contentId);
  }

  async getContentByAuthor(authorDid: string): Promise<HolochainRecord<Content>[]> {
    validateOrThrow(didSchema, authorDid, 'authorDid');
    return this.client.getContentByAuthor(authorDid);
  }

  async getContentByTag(tag: string): Promise<HolochainRecord<Content>[]> {
    validateOrThrow(z.string().min(1), tag, 'tag');
    return this.client.getContentByTag(tag);
  }

  async getContentByType(type_: ContentType): Promise<HolochainRecord<Content>[]> {
    validateOrThrow(contentTypeSchema, type_, 'type_');
    return this.client.getContentByType(type_);
  }

  async updateContent(
    contentId: string,
    updates: Partial<PublishContentInput>
  ): Promise<HolochainRecord<Content>> {
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.updateContent(contentId, updates);
  }

  async recordView(contentId: string): Promise<void> {
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.recordView(contentId);
  }
}

export class ValidatedAttributionClient {
  private client: AttributionClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new AttributionClient(zomeClient);
  }

  async addAttribution(input: AddAttributionInput): Promise<HolochainRecord<Attribution>> {
    validateOrThrow(addAttributionInputSchema, input, 'addAttribution input');
    return this.client.addAttribution(input);
  }

  async getAttributions(contentId: string): Promise<HolochainRecord<Attribution>[]> {
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.getAttributions(contentId);
  }

  async getContributorWorks(contributorDid: string): Promise<HolochainRecord<Attribution>[]> {
    validateOrThrow(didSchema, contributorDid, 'contributorDid');
    return this.client.getContributorWorks(contributorDid);
  }
}

export class ValidatedFactCheckClient {
  private client: FactCheckClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new FactCheckClient(zomeClient);
  }

  async submitFactCheck(input: SubmitFactCheckInput): Promise<HolochainRecord<FactCheck>> {
    validateOrThrow(submitFactCheckInputSchema, input, 'submitFactCheck input');
    return this.client.submitFactCheck(input);
  }

  async getFactChecks(contentId: string): Promise<HolochainRecord<FactCheck>[]> {
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.getFactChecks(contentId);
  }

  async getFactChecksByChecker(checkerDid: string): Promise<HolochainRecord<FactCheck>[]> {
    validateOrThrow(didSchema, checkerDid, 'checkerDid');
    return this.client.getFactChecksByChecker(checkerDid);
  }

  async getContentVerificationStatus(contentId: string): Promise<VerificationStatus> {
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.getContentVerificationStatus(contentId);
  }
}

export class ValidatedCurationClient {
  private client: CurationClient;
  constructor(zomeClient: ZomeCallable) {
    this.client = new CurationClient(zomeClient);
  }

  async createList(input: CreateCurationListInput): Promise<HolochainRecord<CurationList>> {
    validateOrThrow(createCurationListInputSchema, input, 'createList input');
    return this.client.createList(input);
  }

  async getList(listId: string): Promise<HolochainRecord<CurationList> | null> {
    validateOrThrow(z.string().min(1), listId, 'listId');
    return this.client.getList(listId);
  }

  async getListsByCurator(curatorDid: string): Promise<HolochainRecord<CurationList>[]> {
    validateOrThrow(didSchema, curatorDid, 'curatorDid');
    return this.client.getListsByCurator(curatorDid);
  }

  async addToList(listId: string, contentId: string): Promise<HolochainRecord<CurationList>> {
    validateOrThrow(z.string().min(1), listId, 'listId');
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.addToList(listId, contentId);
  }

  async removeFromList(listId: string, contentId: string): Promise<HolochainRecord<CurationList>> {
    validateOrThrow(z.string().min(1), listId, 'listId');
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.removeFromList(listId, contentId);
  }

  async endorse(
    contentId: string,
    weight: number,
    comment?: string
  ): Promise<HolochainRecord<Endorsement>> {
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    validateOrThrow(z.number().min(0).max(1), weight, 'weight');
    if (comment !== undefined) validateOrThrow(z.string(), comment, 'comment');
    return this.client.endorse(contentId, weight, comment);
  }

  async getEndorsements(contentId: string): Promise<HolochainRecord<Endorsement>[]> {
    validateOrThrow(z.string().min(1), contentId, 'contentId');
    return this.client.getEndorsements(contentId);
  }
}

export function createValidatedMediaClients(client: ZomeCallable) {
  return {
    publication: new ValidatedPublicationClient(client),
    attribution: new ValidatedAttributionClient(client),
    factcheck: new ValidatedFactCheckClient(client),
    curation: new ValidatedCurationClient(client),
  };
}
