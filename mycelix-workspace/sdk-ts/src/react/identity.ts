/**
 * @mycelix/sdk React Hooks for Identity Module
 *
 * Provides React hooks for the Identity hApp integration.
 * Works with React 18+ and supports Suspense, concurrent features.
 *
 * @packageDocumentation
 * @module react/identity
 */

import type { QueryState, MutationState } from './index.js';
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
// Identity DID Hooks
// ============================================================================

export interface UseDidDocumentOptions {
  /** Auto-fetch on mount */
  enabled?: boolean;
  /** Refetch interval in ms */
  refetchInterval?: number;
}

/**
 * Hook to create a new DID
 *
 * @example
 * ```tsx
 * function CreateIdentity() {
 *   const { mutate, loading, error, data } = useCreateDid();
 *
 *   return (
 *     <button onClick={() => mutate()} disabled={loading}>
 *       {loading ? 'Creating...' : 'Create DID'}
 *     </button>
 *   );
 * }
 * ```
 */
export function useCreateDid(): MutationState<HolochainRecord<DidDocument>, void> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get current user's DID document
 */
export function useMyDid(
  _options?: UseDidDocumentOptions
): QueryState<HolochainRecord<DidDocument> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to resolve a DID to its document
 *
 * @example
 * ```tsx
 * function DidResolver({ did }: { did: string }) {
 *   const { data, loading, error } = useResolveDid(did);
 *
 *   if (loading) return <Spinner />;
 *   if (error) return <Error error={error} />;
 *   if (!data) return <div>DID not found</div>;
 *
 *   return <DidDocumentView document={data.entry.Present} />;
 * }
 * ```
 */
export function useResolveDid(
  _did: string,
  _options?: UseDidDocumentOptions
): QueryState<HolochainRecord<DidDocument> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get DID document by agent public key
 */
export function useDidDocument(
  _agentPubKey: string,
  _options?: UseDidDocumentOptions
): QueryState<HolochainRecord<DidDocument> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to check if a DID is active
 */
export function useIsDidActive(_did: string): QueryState<boolean> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to update DID document
 */
export function useUpdateDidDocument(): MutationState<
  HolochainRecord<DidDocument>,
  {
    verification_method?: VerificationMethod[];
    authentication?: string[];
    service?: ServiceEndpoint[];
  }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to add a service endpoint
 */
export function useAddServiceEndpoint(): MutationState<
  HolochainRecord<DidDocument>,
  ServiceEndpoint
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to add a verification method
 */
export function useAddVerificationMethod(): MutationState<
  HolochainRecord<DidDocument>,
  VerificationMethod
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to deactivate a DID
 */
export function useDeactivateDid(): MutationState<HolochainRecord<unknown>, string> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// Credential Schema Hooks
// ============================================================================

export interface UseSchemaOptions {
  enabled?: boolean;
  refetchInterval?: number;
}

/**
 * Hook to register a credential schema
 */
export function useRegisterSchema(): MutationState<
  HolochainRecord<CredentialSchema>,
  CredentialSchema
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get a schema by ID
 */
export function useSchema(
  _schemaId: string,
  _options?: UseSchemaOptions
): QueryState<HolochainRecord<CredentialSchema> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get schemas by category
 */
export function useSchemasByCategory(
  _category: SchemaCategory,
  _options?: UseSchemaOptions
): QueryState<HolochainRecord<CredentialSchema>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get schemas by author
 */
export function useSchemasByAuthor(
  _authorDid: string,
  _options?: UseSchemaOptions
): QueryState<HolochainRecord<CredentialSchema>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to endorse a schema
 */
export function useEndorseSchema(): MutationState<HolochainRecord<unknown>, string> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// Revocation Hooks
// ============================================================================

/**
 * Hook to revoke a credential
 */
export function useRevokeCredential(): MutationState<
  HolochainRecord<unknown>,
  {
    credential_id: string;
    reason: string;
    status: RevocationStatus;
  }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to check credential revocation status
 */
export function useCheckRevocation(
  _credentialId: string
): QueryState<HolochainRecord<unknown> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get revocations by issuer
 */
export function useRevocationsByIssuer(_issuerDid: string): QueryState<HolochainRecord<unknown>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to update revocation status
 */
export function useUpdateRevocationStatus(): MutationState<
  HolochainRecord<unknown>,
  {
    credential_id: string;
    new_status: RevocationStatus;
    reason: string;
  }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// Recovery Hooks
// ============================================================================

/**
 * Hook to configure social recovery
 */
export function useConfigureRecovery(): MutationState<
  HolochainRecord<RecoveryConfig>,
  RecoveryConfig
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get recovery configuration
 */
export function useRecoveryConfig(
  _did: string
): QueryState<HolochainRecord<RecoveryConfig> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to initiate recovery
 */
export function useInitiateRecovery(): MutationState<
  HolochainRecord<unknown>,
  {
    did: string;
    new_public_key: string;
    reason: string;
  }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to vote on recovery request
 */
export function useVoteOnRecovery(): MutationState<
  HolochainRecord<unknown>,
  {
    request_id: string;
    decision: VoteDecision;
    reason?: string;
  }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get recovery request
 */
export function useRecoveryRequest(
  _requestId: string
): QueryState<HolochainRecord<unknown> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get recovery votes
 */
export function useRecoveryVotes(_requestId: string): QueryState<HolochainRecord<unknown>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to complete recovery
 */
export function useCompleteRecovery(): MutationState<HolochainRecord<unknown>, string> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// Identity Bridge Hooks
// ============================================================================

/**
 * Hook to register a hApp with identity bridge
 */
export function useRegisterHapp(): MutationState<
  unknown,
  {
    happ_name: string;
    capabilities: string[];
  }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get registered hApps
 */
export function useRegisteredHapps(): QueryState<unknown[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to query identity from another hApp
 */
export function useQueryIdentity(): MutationState<
  unknown,
  {
    did: string;
    source_happ: string;
    requested_fields?: string[];
  }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get aggregated reputation for a DID
 */
export function useIdentityReputation(_did: string): QueryState<unknown> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get identity events by DID
 */
export function useIdentityEvents(_did: string): QueryState<unknown[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// MFA (Multi-Factor Authentication) Hooks
// ============================================================================

import type {
  MfaState,
  EnrolledFactor,
  FactorType,
  AssuranceLevel,
  EnrollFactorInput,
  FactorProof,
  VerificationChallenge,
  MfaVerificationResult,
  FlEligibilityResult,
  FactorEnrollment,
} from '../integrations/identity/index.js';

export interface UseMfaOptions {
  /** Auto-fetch on mount */
  enabled?: boolean;
  /** Refetch interval in ms */
  refetchInterval?: number;
}

/**
 * Hook to get MFA state for a DID
 *
 * @example
 * ```tsx
 * function MfaStatus({ did }: { did: string }) {
 *   const { data, loading, error } = useMfaState(did);
 *
 *   if (loading) return <Spinner />;
 *   if (error) return <Error error={error} />;
 *   if (!data) return <div>No MFA configured</div>;
 *
 *   return (
 *     <div>
 *       <p>Assurance Level: {data.assuranceLevel}</p>
 *       <p>Factors: {data.factors.length}</p>
 *       <p>FL Eligible: {data.flEligible ? 'Yes' : 'No'}</p>
 *     </div>
 *   );
 * }
 * ```
 */
export function useMfaState(
  _did: string,
  _options?: UseMfaOptions
): QueryState<MfaState | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to create initial MFA state for a DID
 */
export function useCreateMfaState(): MutationState<
  MfaState,
  { did: string; primaryKeyHash: string }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to enroll a new authentication factor
 *
 * @example
 * ```tsx
 * function EnrollHardwareKey({ did }: { did: string }) {
 *   const { mutate, loading, error, data } = useEnrollFactor();
 *
 *   const handleEnroll = () => {
 *     mutate({
 *       did,
 *       input: {
 *         factorType: 'HardwareKey',
 *         metadata: { keyName: 'YubiKey 5' }
 *       }
 *     });
 *   };
 *
 *   return (
 *     <button onClick={handleEnroll} disabled={loading}>
 *       {loading ? 'Enrolling...' : 'Enroll Hardware Key'}
 *     </button>
 *   );
 * }
 * ```
 */
export function useEnrollFactor(): MutationState<
  EnrolledFactor,
  { did: string; input: EnrollFactorInput }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to remove an enrolled factor
 */
export function useRemoveFactor(): MutationState<
  MfaState,
  { did: string; factorId: string }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to generate a verification challenge for a factor
 */
export function useGenerateChallenge(): MutationState<
  VerificationChallenge,
  { did: string; factorId: string }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to verify a factor with proof
 *
 * @example
 * ```tsx
 * function VerifyFactor({ did, factorId }: { did: string; factorId: string }) {
 *   const { mutate: generateChallenge, data: challenge } = useGenerateChallenge();
 *   const { mutate: verify, loading, data: result } = useVerifyFactor();
 *
 *   const handleVerify = async () => {
 *     // First generate a challenge
 *     const challenge = await generateChallenge({ did, factorId });
 *
 *     // Then verify with proof (e.g., from WebAuthn)
 *     verify({
 *       did,
 *       factorId,
 *       proof: {
 *         type: 'webauthn',
 *         authenticatorData: '...',
 *         clientDataHash: '...',
 *         signature: '...'
 *       }
 *     });
 *   };
 *
 *   return (
 *     <button onClick={handleVerify} disabled={loading}>
 *       Verify Factor
 *     </button>
 *   );
 * }
 * ```
 */
export function useVerifyFactor(): MutationState<
  MfaVerificationResult,
  { did: string; factorId: string; proof: FactorProof }
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get current assurance level for a DID
 */
export function useAssuranceLevel(
  _did: string,
  _options?: UseMfaOptions
): QueryState<{ level: AssuranceLevel; score: number }> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to check FL (Federated Learning) eligibility
 *
 * @example
 * ```tsx
 * function FlEligibility({ did }: { did: string }) {
 *   const { data, loading } = useFlEligibility(did);
 *
 *   if (loading) return <Spinner />;
 *   if (!data) return null;
 *
 *   return (
 *     <div>
 *       <h3>FL Eligibility: {data.eligible ? 'Yes' : 'No'}</h3>
 *       <p>Score: {data.currentScore} / {data.requiredScore}</p>
 *       <ul>
 *         {data.requirements.map(req => (
 *           <li key={req.name}>
 *             {req.met ? '✓' : '✗'} {req.description}
 *           </li>
 *         ))}
 *       </ul>
 *     </div>
 *   );
 * }
 * ```
 */
export function useFlEligibility(
  _did: string,
  _options?: UseMfaOptions
): QueryState<FlEligibilityResult> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get factor enrollment history
 */
export function useEnrollmentHistory(
  _did: string,
  _limit?: number,
  _options?: UseMfaOptions
): QueryState<FactorEnrollment[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get factors by type
 */
export function useFactorsByType(
  _did: string,
  _factorType: FactorType,
  _options?: UseMfaOptions
): QueryState<EnrolledFactor[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get MFA summary (lightweight state)
 */
export function useMfaSummary(
  _did: string,
  _options?: UseMfaOptions
): QueryState<{
  factorCount: number;
  assuranceLevel: AssuranceLevel;
  flEligible: boolean;
  categories: string[];
} | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}
