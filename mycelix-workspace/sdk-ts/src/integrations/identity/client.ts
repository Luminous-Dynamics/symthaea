// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Identity Client
 *
 * Unified client for the Mycelix-Identity hApp providing access to all zomes.
 *
 * @example
 * ```typescript
 * import { MycelixIdentityClient } from '@mycelix/sdk/integrations/identity';
 * import { AppWebsocket } from '@holochain/client';
 *
 * // Connect directly with SDK
 * const identity = await MycelixIdentityClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Or use an existing client
 * const client = await AppWebsocket.connect({ url: 'ws://localhost:8888' });
 * const identity = MycelixIdentityClient.fromClient(client);
 *
 * // Create a DID
 * const myDid = await identity.did.createDid();
 * console.log(`Created: ${myDid.document.id}`);
 *
 * // Issue a credential
 * const credential = await identity.credentials.issueCredential({
 *   subject_did: 'did:mycelix:bob...',
 *   schema_id: 'mycelix:schema:employment',
 *   claims: { role: 'Engineer' },
 * });
 * ```
 */

import { type AppClient, AppWebsocket, encodeHashToBase64 } from '@holochain/client';

import { IdentitySdkError, IdentitySdkErrorCode, DEFAULT_IDENTITY_CONFIG } from './types.js';
import { DidRegistryClient } from './zomes/did-registry.js';
import { MfaClient } from './zomes/mfa.js';
import { VerifiableCredentialClient } from './zomes/verifiable-credential.js';

import { IdentityBridgeClient } from './index.js';

import type { IdentityConfig } from './types.js';

/**
 * Full Identity SDK configuration
 */
export interface MycelixIdentityConfig extends IdentityConfig {
  /** Holochain app ID */
  appId?: string;
  /** WebSocket URL for conductor */
  url?: string;
}

const DEFAULT_FULL_CONFIG: Required<MycelixIdentityConfig> = {
  ...DEFAULT_IDENTITY_CONFIG,
  appId: 'mycelix-identity',
  url: 'ws://localhost:8888',
};

/**
 * Helper to encode agent pub key for logging
 */
function encodeAgentPubKey(pubKey: Uint8Array): string {
  return encodeHashToBase64(pubKey);
}

/**
 * MycelixIdentityClient
 *
 * Unified client for the Mycelix-Identity hApp providing access to all zomes.
 */
export class MycelixIdentityClient {
  /**
   * DID Registry - Create and manage DID documents
   */
  public readonly did: DidRegistryClient;

  /**
   * Verifiable Credentials - Issue, verify, and present credentials
   */
  public readonly credentials: VerifiableCredentialClient;

  /**
   * Identity Bridge - Cross-hApp identity federation
   */
  public readonly bridge: IdentityBridgeClient;

  /**
   * MFA - Multi-Factor Authentication
   */
  public readonly mfa: MfaClient;

  /**
   * The underlying Holochain client
   */
  public readonly client: AppClient;

  /**
   * Configuration used to create this client
   */
  public readonly config: Required<MycelixIdentityConfig>;

  private constructor(client: AppClient, config: Required<MycelixIdentityConfig>) {
    this.client = client;
    this.config = config;

    // Initialize zome clients
    this.did = new DidRegistryClient(client, config.roleName);
    this.credentials = new VerifiableCredentialClient(client, config.roleName);
    this.mfa = new MfaClient(client, config.roleName);

    // Bridge uses the shared MycelixClient wrapper - adapting here
    // This is a temporary bridge until we unify the client patterns
    this.bridge = new IdentityBridgeClient({
      callZome: async (params: any) => {
        return client.callZome({
          role_name: params.role_name,
          zome_name: params.zome_name,
          fn_name: params.fn_name,
          payload: params.payload,
        });
      },
    } as never);
  }

  /**
   * Create a MycelixIdentityClient from an existing AppClient
   *
   * @param client - Existing Holochain client
   * @param config - Optional configuration overrides
   * @returns Configured MycelixIdentityClient
   */
  static fromClient(
    client: AppClient,
    config: Partial<MycelixIdentityConfig> = {}
  ): MycelixIdentityClient {
    const fullConfig: Required<MycelixIdentityConfig> = {
      ...DEFAULT_FULL_CONFIG,
      ...config,
    };

    return new MycelixIdentityClient(client, fullConfig);
  }

  /**
   * Connect to Holochain and create a MycelixIdentityClient
   *
   * @param config - Connection configuration
   * @returns Connected MycelixIdentityClient
   * @throws IdentitySdkError if connection fails
   */
  static async connect(config: Partial<MycelixIdentityConfig> = {}): Promise<MycelixIdentityClient> {
    const fullConfig: Required<MycelixIdentityConfig> = {
      ...DEFAULT_FULL_CONFIG,
      ...config,
    };

    let client: AppClient;
    let attempts = 0;

    while (attempts < fullConfig.retry.maxAttempts) {
      try {
        client = await AppWebsocket.connect({
          url: new URL(fullConfig.url),
        });

        // Verify connection by getting app info
        const appInfo = await client.appInfo();
        if (!appInfo) {
          throw new Error('Failed to get app info');
        }

        if (fullConfig.debug) {
          console.log(`[identity-sdk] Connected to ${fullConfig.appId}`);
          console.log(
            `[identity-sdk] Agent: ${encodeAgentPubKey(client.myPubKey).slice(0, 16)}...`
          );
        }

        return new MycelixIdentityClient(client, fullConfig);
      } catch (error) {
        attempts++;
        const message = error instanceof Error ? error.message : String(error);

        if (attempts >= fullConfig.retry.maxAttempts) {
          throw new IdentitySdkError(
            IdentitySdkErrorCode.CONNECTION_FAILED,
            `Failed to connect after ${attempts} attempts: ${message}`,
            { url: fullConfig.url, appId: fullConfig.appId }
          );
        }

        // Exponential backoff
        const delay =
          fullConfig.retry.delayMs *
          Math.pow(fullConfig.retry.backoffMultiplier, attempts - 1);

        if (fullConfig.debug) {
          console.log(
            `[identity-sdk] Connection attempt ${attempts} failed, retrying in ${delay}ms...`
          );
        }

        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }

    // TypeScript requires this, but it's unreachable
    throw new IdentitySdkError(IdentitySdkErrorCode.CONNECTION_FAILED, 'Connection failed');
  }

  /**
   * Get the current agent's public key
   *
   * @returns Agent public key as Uint8Array
   */
  getAgentPubKey(): Uint8Array {
    return this.client.myPubKey;
  }

  /**
   * Get the current agent's DID
   *
   * Convenience method that returns the DID string for the current agent.
   *
   * @returns DID string or null if no DID exists
   */
  async getMyDid(): Promise<string | null> {
    const didRecord = await this.did.getMyDid();
    return didRecord?.document.id ?? null;
  }

  /**
   * Check if connected to Holochain
   *
   * @returns true if connection is healthy
   */
  async isConnected(): Promise<boolean> {
    try {
      const appInfo = await this.client.appInfo();
      return appInfo !== null;
    } catch {
      return false;
    }
  }

  /**
   * Get a summary of all zome capabilities
   *
   * Useful for debugging and introspection.
   */
  getCapabilities(): {
    did: string[];
    credentials: string[];
    bridge: string[];
    mfa: string[];
  } {
    return {
      did: [
        'createDid',
        'getMyDid',
        'getDidDocument',
        'resolveDid',
        'isDidActive',
        'updateDidDocument',
        'deactivateDid',
        'getDidDeactivation',
        'addServiceEndpoint',
        'removeServiceEndpoint',
        'addVerificationMethod',
      ],
      credentials: [
        'issueCredential',
        'getCredential',
        'getCredentialsIssuedBy',
        'getCredentialsForSubject',
        'getMyIssuedCredentials',
        'getMyCredentials',
        'verifyCredential',
        'isCredentialRevoked',
        'getCredentialStatus',
        'createPresentation',
        'createDerivedCredential',
        'requestCredential',
        'getPendingRequests',
        'updateRequestStatus',
      ],
      bridge: [
        'registerHapp',
        'getRegisteredHapps',
        'queryIdentity',
        'reportReputation',
        'getReputation',
        'broadcastEvent',
        'getRecentEvents',
        'getEventsByDid',
        'getEventsByType',
      ],
      mfa: [
        'getMfaState',
        'createMfaState',
        'enrollFactor',
        'removeFactor',
        'generateChallenge',
        'verifyFactor',
        'getAssuranceLevel',
        'checkFlEligibility',
        'getEnrollmentHistory',
        'getFactorsByType',
        'getMfaSummary',
      ],
    };
  }
}

// Default export
export default MycelixIdentityClient;
