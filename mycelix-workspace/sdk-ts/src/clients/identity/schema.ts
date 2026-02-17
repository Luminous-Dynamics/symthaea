/**
 * Credential Schema Client
 *
 * Client for credential schema management in Mycelix-Identity.
 * Enables schema creation, discovery, and endorsement.
 *
 * @module @mycelix/sdk/clients/identity/schema
 */

import { DEFAULT_IDENTITY_CLIENT_CONFIG } from './types.js';
import { ZomeClient } from '../../core/zome-client.js';

import type {
  CredentialSchema,
  SchemaEndorsement,
  UpdateSchemaInput,
  EndorseSchemaInput,
  IdentityClientConfig,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';


/**
 * Schema record wrapper
 */
export interface SchemaRecord {
  hash: Uint8Array;
  schema: CredentialSchema;
}

/**
 * Endorsement record wrapper
 */
export interface EndorsementRecord {
  hash: Uint8Array;
  endorsement: SchemaEndorsement;
}

/**
 * Credential Schema Client
 *
 * Provides credential schema management including creation, updates,
 * endorsements, and discovery.
 *
 * @example
 * ```typescript
 * const schemaClient = new SchemaClient(appClient);
 *
 * // Create a new schema
 * const schema = await schemaClient.createSchema({
 *   id: 'mycelix:schema:employment:v1',
 *   name: 'Employment Credential',
 *   description: 'Proof of employment',
 *   version: '1.0.0',
 *   author: 'did:mycelix:acme123',
 *   schema: JSON.stringify({
 *     type: 'object',
 *     properties: {
 *       employer: { type: 'string' },
 *       role: { type: 'string' },
 *       startDate: { type: 'string', format: 'date' },
 *     },
 *     required: ['employer', 'role', 'startDate'],
 *   }),
 *   required_fields: ['employer', 'role', 'startDate'],
 *   optional_fields: ['endDate', 'department'],
 *   credential_type: 'EmploymentCredential',
 *   revocable: true,
 *   active: true,
 * });
 *
 * // Endorse a schema
 * await schemaClient.endorseSchema({
 *   schema_id: 'mycelix:schema:employment:v1',
 *   endorser_did: 'did:mycelix:validator456',
 *   trust_level: 0.9,
 *   comment: 'Well-structured schema for employment verification',
 * });
 * ```
 */
export class SchemaClient extends ZomeClient {
  protected readonly zomeName = 'credential_schema';

  constructor(client: AppClient, config: Partial<IdentityClientConfig> = {}) {
    const mergedConfig = { ...DEFAULT_IDENTITY_CLIENT_CONFIG, ...config };
    super(client, {
      roleName: mergedConfig.roleName,
      timeout: mergedConfig.timeout,
    });
  }

  // ==========================================================================
  // SCHEMA MANAGEMENT
  // ==========================================================================

  /**
   * Create a new credential schema
   *
   * Schemas define the structure and validation rules for credentials.
   *
   * @param schema - Schema definition
   * @returns Created schema record
   */
  async createSchema(schema: CredentialSchema): Promise<SchemaRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('create_schema', schema);
    return this.recordToSchemaRecord(record);
  }

  /**
   * Get a schema by its ID
   *
   * @param schemaId - Unique schema identifier
   * @returns Schema record or null if not found
   */
  async getSchema(schemaId: string): Promise<SchemaRecord | null> {
    const record = await this.callZome<HolochainRecord | null>('get_schema', schemaId);
    return record ? this.recordToSchemaRecord(record) : null;
  }

  /**
   * Get all schemas by author
   *
   * @param authorDid - DID of the schema author
   * @returns Array of schema records
   */
  async getSchemasByAuthor(authorDid: string): Promise<SchemaRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_schemas_by_author', authorDid);
    return records.map((r) => this.recordToSchemaRecord(r));
  }

  /**
   * Update a schema
   *
   * Creates a new version of the schema with updated fields.
   *
   * @param input - Update parameters
   * @returns Updated schema record
   */
  async updateSchema(input: UpdateSchemaInput): Promise<SchemaRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('update_schema', input);
    return this.recordToSchemaRecord(record);
  }

  /**
   * List all active schemas
   *
   * @returns Array of active schema records
   */
  async listActiveSchemas(): Promise<SchemaRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('list_active_schemas', null);
    return records.map((r) => this.recordToSchemaRecord(r));
  }

  // ==========================================================================
  // SCHEMA ENDORSEMENT
  // ==========================================================================

  /**
   * Endorse a schema
   *
   * Trusted entities can endorse schemas to indicate quality and reliability.
   *
   * @param input - Endorsement parameters
   * @returns Created endorsement record
   */
  async endorseSchema(input: EndorseSchemaInput): Promise<EndorsementRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('endorse_schema', input);
    return this.recordToEndorsementRecord(record);
  }

  /**
   * Get all endorsements for a schema
   *
   * @param schemaId - ID of the schema
   * @returns Array of endorsement records
   */
  async getSchemaEndorsements(schemaId: string): Promise<EndorsementRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_schema_endorsements', schemaId);
    return records.map((r) => this.recordToEndorsementRecord(r));
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private recordToSchemaRecord(record: HolochainRecord): SchemaRecord {
    const schema = this.extractEntry<CredentialSchema>(record);
    return {
      hash: this.getActionHash(record),
      schema,
    };
  }

  private recordToEndorsementRecord(record: HolochainRecord): EndorsementRecord {
    const endorsement = this.extractEntry<SchemaEndorsement>(record);
    return {
      hash: this.getActionHash(record),
      endorsement,
    };
  }
}
