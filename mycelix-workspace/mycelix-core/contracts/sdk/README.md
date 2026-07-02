# Mycelix SDK

TypeScript SDK for interacting with Mycelix smart contracts on Ethereum.

## Installation

```bash
npm install @mycelix/sdk
# or
yarn add @mycelix/sdk
```

## Quick Start

```typescript
import { MycelixClient } from '@mycelix/sdk';

// Read-only client (no private key needed)
const client = new MycelixClient({ network: 'sepolia' });

// Full client with write capabilities
const client = new MycelixClient({
  network: 'sepolia',
  privateKey: process.env.PRIVATE_KEY
});
```

## Live Contracts (Sepolia)

| Contract | Address |
|----------|---------|
| Registry | `0x556b810371e3d8D9E5753117514F03cC6C93b835` |
| Reputation | `0xf3B343888a9b82274cEfaa15921252DB6c5f48C9` |
| Payment | `0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB` |

## API Reference

### MycelixClient

The main client that provides access to all contract interactions.

```typescript
const client = new MycelixClient({
  network: 'sepolia',      // or 'local' for development
  privateKey: '0x...',     // optional, required for write operations
  provider: customProvider // optional, use custom provider
});

// Get connected wallet address
client.getAddress(); // '0x...'

// Get wallet balance
await client.getBalance(); // bigint

// Get block explorer URLs
client.getTxUrl('0x...');      // 'https://sepolia.etherscan.io/tx/0x...'
client.getAddressUrl('0x...'); // 'https://sepolia.etherscan.io/address/0x...'
```

### Registry Client

Manage Decentralized Identifiers (DIDs).

```typescript
// Register a new DID
const tx = await client.registry.registerDID(
  'did:mycelix:alice',           // DID string
  'ipfs://QmYourMetadataHash'    // Metadata URI
);
await tx.wait();

// Look up a DID
const info = await client.registry.getDID('did:mycelix:alice');
console.log(info);
// {
//   owner: '0x...',
//   registeredAt: 1704700000n,
//   updatedAt: 1704700000n,
//   metadataHash: '0x...',
//   revoked: false
// }

// Update DID metadata
await client.registry.updateMetadata(
  'did:mycelix:alice',
  'ipfs://QmNewMetadataHash'
);

// Revoke a DID
await client.registry.revokeDID('did:mycelix:alice');

// Get registration fee
const fee = await client.registry.getRegistrationFee(); // bigint

// Get total registered DIDs
const total = await client.registry.getTotalDids(); // bigint
```

### Reputation Client

Query and submit reputation scores.

```typescript
// Get reputation for an address
const rep = await client.reputation.getReputation('0x...');
console.log(rep);
// {
//   score: 100n,
//   submissions: 5n,
//   lastUpdate: 1704700000n
// }

// Submit reputation (requires authorized submitter)
await client.reputation.submitReputation(
  '0xSubjectAddress',           // Address being rated
  '0xContextHash',              // Context identifier (bytes32)
  85,                           // Score (-100 to 100)
  '0x'                          // Optional proof data
);

// Anchor a Merkle root for batch updates
await client.reputation.anchorMerkleRoot(
  '0xMerkleRoot',               // Root hash
  42                            // Epoch number
);
```

### Payment Client

Create and manage escrow payments for FL contributions.

```typescript
import { ethers } from 'ethers';

// Create an escrow payment
const amount = ethers.parseEther('0.1');
const releaseTime = Math.floor(Date.now() / 1000) + 86400; // 24 hours

const tx = await client.payment.createPayment(
  '0xRecipientAddress',         // Who receives the payment
  amount,                       // Amount in wei
  releaseTime,                  // Unix timestamp when releasable
  ethers.ZeroHash               // Optional condition hash
);
await tx.wait();

// Get payment details
const payment = await client.payment.getPayment(1);
console.log(payment);
// {
//   sender: '0x...',
//   recipient: '0x...',
//   amount: 100000000000000000n,
//   releaseTime: 1704786400n,
//   released: false,
//   refunded: false
// }

// Release payment to recipient (after releaseTime)
await client.payment.releasePayment(1);

// Refund payment to sender (if conditions not met)
await client.payment.refundPayment(1);
```

## CLI Usage

The SDK includes a command-line interface for quick interactions.

```bash
# Show network info
npm run cli -- info

# Look up a DID
npm run cli -- lookup did:mycelix:alice

# Query reputation
npm run cli -- reputation 0x1234...

# With private key for write operations
PRIVATE_KEY=0x... npm run cli -- register did:mycelix:bob ipfs://QmMetadata

# Create escrow payment
PRIVATE_KEY=0x... npm run cli -- payment create 0xRecipient 0.01

# Get payment info
npm run cli -- payment info 1
```

## Networks

### Sepolia (Default)

```typescript
const client = new MycelixClient({ network: 'sepolia' });
```

### Local Development

Start a local Anvil node:

```bash
anvil
```

Then connect:

```typescript
const client = new MycelixClient({ network: 'local' });
```

### Custom Network

```typescript
const client = new MycelixClient({
  network: {
    chainId: 1337,
    rpcUrl: 'http://localhost:8545',
    contracts: {
      registry: '0x...',
      reputation: '0x...',
      payment: '0x...'
    },
    blockExplorer: 'http://localhost'
  }
});
```

## Development

```bash
# Install dependencies
npm install

# Run demo
npm run demo

# Run CLI
npm run cli -- help

# Build
npm run build
```

## Error Handling

```typescript
try {
  await client.registry.registerDID('did:mycelix:test', 'ipfs://...');
} catch (error) {
  if (error.message.includes('DID already exists')) {
    console.log('This DID is already registered');
  } else if (error.message.includes('Insufficient funds')) {
    console.log('Not enough ETH for registration fee');
  } else {
    throw error;
  }
}
```

## Types

```typescript
import type {
  MycelixClient,
  RegistryClient,
  ReputationClient,
  PaymentClient,
  NetworkConfig,
  ClientConfig
} from '@mycelix/sdk';
```

## License

MIT
