/**
 * @mycelix/sdk Vue 3 Composables for Identity Module
 *
 * Provides Vue 3 Composition API composables for the Identity hApp integration.
 * Works with Vue 3.3+ and Nuxt 3.
 *
 * @packageDocumentation
 * @module vue/identity
 */

import type { UseQueryReturn, UseMutationReturn, UseQueryOptions } from './index.js';
import type {
  DidDocument,
  VerificationMethod,
  ServiceEndpoint,
  CredentialSchema,
  SchemaCategory,
  RevocationStatus,
  RecoveryConfig,
  VoteDecision,
  HolochainRecord,
} from '../identity/index.js';

// ============================================================================
// DID Composables
// ============================================================================

export interface UseDidDocumentOptions extends UseQueryOptions {
  /** Include verification methods */
  includeVerification?: boolean;
  /** Include service endpoints */
  includeServices?: boolean;
}

/**
 * Composable to create a new DID
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useCreateDid } from '@mycelix/sdk/vue';
 *
 * const { mutate, loading, data } = useCreateDid();
 *
 * async function createIdentity() {
 *   const result = await mutate();
 *   console.log('Created DID:', result.entry.Present.id);
 * }
 * </script>
 * ```
 */
export function useCreateDid(): UseMutationReturn<HolochainRecord<DidDocument>, void> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get current user's DID document
 */
export function useMyDid(
  _options?: UseDidDocumentOptions
): UseQueryReturn<HolochainRecord<DidDocument> | null> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to resolve a DID to its document
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useResolveDid } from '@mycelix/sdk/vue';
 * import { watchEffect } from 'vue';
 *
 * const props = defineProps<{ did: string }>();
 * const { data, loading, error } = useResolveDid(props.did);
 *
 * watchEffect(() => {
 *   if (data.value) {
 *     console.log('Resolved:', data.value.entry.Present);
 *   }
 * });
 * </script>
 * ```
 */
export function useResolveDid(
  _did: string,
  _options?: UseDidDocumentOptions
): UseQueryReturn<HolochainRecord<DidDocument> | null> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get DID document by agent public key
 */
export function useDidDocument(
  _agentPubKey: string,
  _options?: UseDidDocumentOptions
): UseQueryReturn<HolochainRecord<DidDocument> | null> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to check if a DID is active
 */
export function useIsDidActive(_did: string): UseQueryReturn<boolean> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to update DID document
 */
export function useUpdateDidDocument(): UseMutationReturn<
  HolochainRecord<DidDocument>,
  {
    verification_method?: VerificationMethod[];
    authentication?: string[];
    service?: ServiceEndpoint[];
  }
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to add a service endpoint
 */
export function useAddServiceEndpoint(): UseMutationReturn<
  HolochainRecord<DidDocument>,
  ServiceEndpoint
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to add a verification method
 */
export function useAddVerificationMethod(): UseMutationReturn<
  HolochainRecord<DidDocument>,
  VerificationMethod
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to deactivate a DID
 */
export function useDeactivateDid(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

// ============================================================================
// Credential Schema Composables
// ============================================================================

export interface UseSchemaOptions extends UseQueryOptions {
  /** Include endorsement count */
  includeEndorsements?: boolean;
}

/**
 * Composable to register a credential schema
 */
export function useRegisterSchema(): UseMutationReturn<
  HolochainRecord<CredentialSchema>,
  CredentialSchema
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get a schema by ID
 */
export function useSchema(
  _schemaId: string,
  _options?: UseSchemaOptions
): UseQueryReturn<HolochainRecord<CredentialSchema> | null> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get schemas by category
 */
export function useSchemasByCategory(
  _category: SchemaCategory,
  _options?: UseSchemaOptions
): UseQueryReturn<HolochainRecord<CredentialSchema>[]> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get schemas by author
 */
export function useSchemasByAuthor(
  _authorDid: string,
  _options?: UseSchemaOptions
): UseQueryReturn<HolochainRecord<CredentialSchema>[]> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to endorse a schema
 */
export function useEndorseSchema(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

// ============================================================================
// Revocation Composables
// ============================================================================

/**
 * Composable to revoke a credential
 */
export function useRevokeCredential(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    credential_id: string;
    reason: string;
    status: RevocationStatus;
  }
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to check credential revocation status
 */
export function useCheckRevocation(
  _credentialId: string
): UseQueryReturn<HolochainRecord<unknown> | null> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get revocations by issuer
 */
export function useRevocationsByIssuer(
  _issuerDid: string
): UseQueryReturn<HolochainRecord<unknown>[]> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to update revocation status
 */
export function useUpdateRevocationStatus(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    credential_id: string;
    new_status: RevocationStatus;
    reason: string;
  }
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

// ============================================================================
// Recovery Composables
// ============================================================================

/**
 * Composable to configure social recovery
 */
export function useConfigureRecovery(): UseMutationReturn<
  HolochainRecord<RecoveryConfig>,
  RecoveryConfig
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get recovery configuration
 */
export function useRecoveryConfig(
  _did: string
): UseQueryReturn<HolochainRecord<RecoveryConfig> | null> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to initiate recovery
 */
export function useInitiateRecovery(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    did: string;
    new_public_key: string;
    reason: string;
  }
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to vote on recovery request
 */
export function useVoteOnRecovery(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    request_id: string;
    decision: VoteDecision;
    reason?: string;
  }
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get recovery request
 */
export function useRecoveryRequest(
  _requestId: string
): UseQueryReturn<HolochainRecord<unknown> | null> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get recovery votes
 */
export function useRecoveryVotes(_requestId: string): UseQueryReturn<HolochainRecord<unknown>[]> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to complete recovery
 */
export function useCompleteRecovery(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

// ============================================================================
// Identity Bridge Composables
// ============================================================================

/**
 * Composable to register a hApp with identity bridge
 */
export function useRegisterHapp(): UseMutationReturn<
  unknown,
  {
    happ_name: string;
    capabilities: string[];
  }
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get registered hApps
 */
export function useRegisteredHapps(): UseQueryReturn<unknown[]> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to query identity from another hApp
 */
export function useQueryIdentity(): UseMutationReturn<
  unknown,
  {
    did: string;
    source_happ: string;
    requested_fields?: string[];
  }
> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get aggregated reputation for a DID
 */
export function useIdentityReputation(_did: string): UseQueryReturn<unknown> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}

/**
 * Composable to get identity events by DID
 */
export function useIdentityEvents(_did: string): UseQueryReturn<unknown[]> {
  throw new Error(
    'Vue composables require the @mycelix/vue package. ' +
      'Install it with: npm install @mycelix/vue'
  );
}
