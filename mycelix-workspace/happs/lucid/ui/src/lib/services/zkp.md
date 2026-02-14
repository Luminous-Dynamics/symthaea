# ZKP Service API Documentation

TypeScript service for zero-knowledge proofs in the LUCID UI.

## Overview

The ZKP service provides a high-level interface for generating and verifying zero-knowledge proofs. It automatically handles:

- Tauri backend communication
- Simulation fallbacks when Tauri is unavailable
- Error handling and recovery

## Installation

The service is included in the LUCID UI package:

```typescript
import {
  isZkpAvailable,
  generateAnonymousBeliefProof,
  generateReputationRangeProof,
  verifyProof,
  proveBeliefAuthorship,
  proveEligibility,
} from '$services/zkp';
```

## API Reference

### `isZkpAvailable(): Promise<boolean>`

Check if the ZKP backend is available.

```typescript
const available = await isZkpAvailable();
if (available) {
  console.log('ZKP proofs will use native Rust implementation');
} else {
  console.log('ZKP proofs will use simulation mode');
}
```

### `generateAnonymousBeliefProof(beliefHash, agentSecret): Promise<AnonymousBeliefProofOutput>`

Generate a proof that you authored a belief without revealing your identity.

**Parameters:**
- `beliefHash: string` - Hex-encoded hash of the belief content
- `agentSecret: string` - Hex-encoded secret key

**Returns:**
```typescript
interface AnonymousBeliefProofOutput {
  proof_json: string;         // Serialized proof
  author_commitment: string;  // Hex-encoded commitment
  belief_hash: string;        // Echo of input hash
}
```

**Example:**
```typescript
const proof = await generateAnonymousBeliefProof(
  'a1b2c3d4...', // belief hash
  'e5f6g7h8...'  // your secret
);

// Share proof.proof_json and proof.author_commitment
// Keep your secret private!
```

### `generateReputationRangeProof(actualReputation, minThreshold): Promise<ReputationRangeProofOutput>`

Prove your reputation exceeds a threshold without revealing the exact value.

**Parameters:**
- `actualReputation: number` - Your actual reputation score
- `minThreshold: number` - Minimum threshold to prove

**Returns:**
```typescript
interface ReputationRangeProofOutput {
  proof_json: string;            // Serialized proof
  reputation_commitment: string; // Hex-encoded commitment
  min_threshold: number;         // The proven threshold
}
```

**Example:**
```typescript
// Prove reputation >= 100 (actual: 150)
const proof = await generateReputationRangeProof(150, 100);

// Verifier only learns: reputation >= 100
// They don't learn: actual value is 150
```

### `verifyProof(proofType, proofJson): Promise<VerifyProofOutput>`

Verify a proof from another participant.

**Parameters:**
- `proofType: 'anonymous_belief' | 'reputation_range'`
- `proofJson: string` - The serialized proof

**Returns:**
```typescript
interface VerifyProofOutput {
  valid: boolean;
  error: string | null;
}
```

**Example:**
```typescript
const result = await verifyProof('anonymous_belief', receivedProofJson);

if (result.valid) {
  console.log('Proof verified!');
} else {
  console.error('Invalid proof:', result.error);
}
```

### `proveBeliefAuthorship(beliefContent, secretKey): Promise<BeliefAuthorshipResult>`

High-level function that handles the complete authorship proof flow.

**Parameters:**
- `beliefContent: string` - The actual belief text
- `secretKey: string` - Your secret key (hex-encoded)

**Returns:**
```typescript
interface BeliefAuthorshipResult {
  proof: string;       // Serialized proof JSON
  commitment: string;  // Author commitment
  beliefHash: string;  // Hash of the belief
}
```

**Example:**
```typescript
const { proof, commitment, beliefHash } = await proveBeliefAuthorship(
  'Decentralized systems are more resilient',
  mySecretKey
);

// Submit to collective:
await shareBelief({
  content: 'Decentralized systems are more resilient',
  content_hash: beliefHash,
  author_commitment: commitment,
  authorship_proof: proof,
  // ... other fields
});
```

### `proveEligibility(actualReputation, requiredThreshold): Promise<EligibilityResult>`

Prove you meet a reputation requirement for participation.

**Parameters:**
- `actualReputation: number` - Your current reputation
- `requiredThreshold: number` - Required minimum

**Returns:**
```typescript
interface EligibilityResult {
  proof: string;         // Serialized proof JSON
  commitment: string;    // Reputation commitment
  meetsThreshold: boolean; // Whether you qualify
}
```

**Example:**
```typescript
// Check eligibility for voting (requires 50 reputation)
const { proof, commitment, meetsThreshold } = await proveEligibility(
  myReputation,
  50
);

if (meetsThreshold) {
  // Submit vote with proof
  await voteOnBelief(beliefHash, voteType, {
    eligibility_proof: proof,
    reputation_commitment: commitment,
  });
}
```

## Simulation Mode

When Tauri is unavailable (e.g., in browser-only mode), the service automatically falls back to simulation:

- Proofs are generated with deterministic mock data
- Verification always succeeds for well-formed proofs
- Useful for development and testing

To force simulation mode:

```typescript
// In tests
vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn().mockRejectedValue(new Error('Not available')),
}));
```

## Integration with Collective Sensemaking

### Anonymous Belief Sharing

```typescript
import { shareBelief } from '$services/collective-sensemaking';
import { proveBeliefAuthorship } from '$services/zkp';

async function shareAnonymously(thought: Thought, secretKey: string) {
  // Generate authorship proof
  const { proof, commitment, beliefHash } = await proveBeliefAuthorship(
    thought.content,
    secretKey
  );

  // Share with proof attached
  return await shareBelief(thought, {
    authorship_proof: proof,
    author_commitment: commitment,
  });
}
```

### Reputation-Gated Voting

```typescript
import { voteOnBelief } from '$services/collective-sensemaking';
import { proveEligibility } from '$services/zkp';

async function voteWithProof(
  beliefHash: ActionHash,
  voteType: ValidationVoteType,
  myReputation: number
) {
  const VOTING_THRESHOLD = 50;

  const { proof, meetsThreshold } = await proveEligibility(
    myReputation,
    VOTING_THRESHOLD
  );

  if (!meetsThreshold) {
    throw new Error('Insufficient reputation to vote');
  }

  return await voteOnBelief(beliefHash, voteType, undefined, {
    eligibility_proof: proof,
  });
}
```

## Error Handling

The service handles errors gracefully:

```typescript
try {
  const proof = await generateAnonymousBeliefProof(hash, secret);
} catch (error) {
  if (error.message.includes('Invalid')) {
    // Input validation error
  } else if (error.message.includes('generation failed')) {
    // Proof generation error
  } else {
    // Unknown error, falls back to simulation
  }
}
```

## Security Best Practices

1. **Secret Storage**: Never store secrets in localStorage or sessionStorage. Use secure storage solutions.

2. **Proof Freshness**: Generate new proofs for each action. Don't reuse proofs.

3. **Verification**: Always verify proofs received from others before trusting claims.

4. **Logging**: Never log secrets or full proofs in production.

## Testing

```typescript
import { describe, it, expect, vi } from 'vitest';
import { generateAnonymousBeliefProof } from '$services/zkp';

describe('ZKP Service', () => {
  it('should generate proof in simulation mode', async () => {
    const proof = await generateAnonymousBeliefProof(
      'test-hash',
      'test-secret'
    );

    expect(proof).toHaveProperty('proof_json');
    expect(proof).toHaveProperty('author_commitment');
  });
});
```
