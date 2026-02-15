/**
 * Mycelix FHE - Fully Homomorphic Encryption
 *
 * Privacy-preserving computation on encrypted data.
 * Enables encrypted voting, private analytics, and secure aggregation.
 *
 * Note: The default provider is a simulation for development.
 * For production, integrate with TFHE-rs, Microsoft SEAL, or OpenFHE.
 *
 * @example Encrypted Voting
 * ```typescript
 * import { fhe } from '@mycelix/sdk';
 *
 * // Create a voting client
 * const client = await fhe.createVotingClient();
 *
 * // Create encrypted votes
 * const vote1 = await client.createEncryptedVote('proposal-1', 'voter-1', 1);
 * const vote2 = await client.createEncryptedVote('proposal-1', 'voter-2', 1);
 * const vote3 = await client.createEncryptedVote('proposal-1', 'voter-3', 0);
 *
 * // Aggregate votes (still encrypted)
 * const aggregation = await client.aggregateVotes([vote1, vote2, vote3]);
 *
 * // Decrypt final result
 * const sum = await client.decrypt(aggregation.sum);
 * console.log(`Total yes votes: ${sum[0]}`); // 2
 * ```
 *
 * @example Secure Aggregation for Federated Learning
 * ```typescript
 * import { fhe } from '@mycelix/sdk';
 *
 * // Create aggregator
 * const aggregator = await fhe.initializeSecureAggregator({
 *   threshold: 3,
 *   totalParticipants: 5,
 *   roundTimeout: 60000,
 * });
 *
 * // Start a round
 * const round = aggregator.startRound();
 *
 * // Participants submit encrypted gradients
 * await aggregator.submitValue(round.roundId, 'participant-1', [0.1, 0.2, 0.3]);
 * await aggregator.submitValue(round.roundId, 'participant-2', [0.2, 0.3, 0.1]);
 * await aggregator.submitValue(round.roundId, 'participant-3', [0.15, 0.25, 0.2]);
 *
 * // Aggregate and finalize
 * const aggregation = await aggregator.aggregate(round.roundId);
 * const result = await aggregator.finalizeResult(aggregation!);
 * console.log('Aggregated gradients:', result?.result);
 * ```
 *
 * @example Secret Sharing
 * ```typescript
 * import { fhe } from '@mycelix/sdk';
 *
 * // Create 3-of-5 threshold scheme
 * const sharing = fhe.createSecretSharing(3, 5);
 *
 * // Split a secret
 * const secret = BigInt(12345);
 * const shares = sharing.split(secret);
 *
 * // Any 3 shares can reconstruct
 * const reconstructed = sharing.reconstruct(shares.slice(0, 3));
 * console.log(reconstructed === secret); // true
 * ```
 *
 * @module fhe
 */

// Types
export {
  type FHEScheme,
  type SecurityLevel,
  type FHEParams,
  type FHEKeyPair,
  type Ciphertext,
  type Plaintext,
  type HomomorphicOp,
  type EncryptedResult,
  type EncryptedBatch,
  type EncryptedVote,
  type EncryptedAggregation,
  type EncryptedQuery,
  type PSIResult,
  type FHEProvider,
  type FHEPreset,
  FHE_PRESETS,
} from './types.js';

// FHE Client
export {
  FHEClient,
  createFHEClient,
  initializeFHEClient,
  createVotingClient,
  createAnalyticsClient,
  type FHEClientConfig,
} from './client.js';

// Secure Aggregation
export {
  SecureAggregator,
  ShamirSecretSharing,
  createSecureAggregator,
  initializeSecureAggregator,
  createSecretSharing,
  type SecureAggregationConfig,
  type AggregationRound,
  type ParticipantShare,
  type AggregatedResult,
  type SecretShare,
} from './secure-aggregation.js';
