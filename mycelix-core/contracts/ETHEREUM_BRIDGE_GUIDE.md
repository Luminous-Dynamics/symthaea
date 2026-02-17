# Mycelix Ethereum/Polygon Bridge Guide

This guide covers the Mycelix smart contracts and SDK for bridging Holochain hApps to Ethereum and Polygon networks.

## Overview

The Mycelix Ethereum Bridge provides three core contracts:

1. **ReputationAnchor** - Anchors PoGQ (Proof of Genuine Quality) reputation scores on-chain using Merkle proofs
2. **PaymentRouter** - Routes payments with split distribution, escrow, and dispute resolution
3. **MycelixRegistry** - Registry for DIDs (Decentralized Identifiers) and hApp contract addresses

## Contract Addresses

### Polygon Mumbai Testnet (Chain ID: 80001)

```
ReputationAnchor: [To be deployed]
PaymentRouter: [To be deployed]
MycelixRegistry: [To be deployed]
```

### Polygon Amoy Testnet (Chain ID: 80002)

```
ReputationAnchor: [To be deployed]
PaymentRouter: [To be deployed]
MycelixRegistry: [To be deployed]
```

### Polygon Mainnet (Chain ID: 137)

```
ReputationAnchor: [To be deployed]
PaymentRouter: [To be deployed]
MycelixRegistry: [To be deployed]
```

## Installation

### Smart Contracts

```bash
cd Mycelix-Core/contracts

# Install dependencies (inside nix shell)
forge install

# Build contracts
forge build

# Run tests
forge test
```

### TypeScript SDK

```bash
cd mycelix-workspace/sdk-eth

npm install
npm run build
```

## Deployment

### Environment Setup

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Configure environment variables:
```bash
# Your deployer private key
PRIVATE_KEY=your_private_key_here

# For production deployments
OWNER_ADDRESS=0x...
ANCHOR_ADDRESS=0x...
FEE_RECIPIENT=0x...

# RPC URLs
POLYGON_MUMBAI_RPC_URL=https://rpc-mumbai.maticvigil.com
POLYGON_AMOY_RPC_URL=https://rpc-amoy.polygon.technology

# For contract verification
POLYGONSCAN_API_KEY=your_api_key
```

### Deploy to Mumbai Testnet

```bash
# Load environment
source .env

# Deploy (dry run)
forge script script/Deploy.s.sol:DeployMumbai \
  --rpc-url $POLYGON_MUMBAI_RPC_URL \
  -vvvv

# Deploy with broadcast
forge script script/Deploy.s.sol:DeployMumbai \
  --rpc-url $POLYGON_MUMBAI_RPC_URL \
  --broadcast \
  --verify

# Or for Amoy
forge script script/Deploy.s.sol:DeployAmoy \
  --rpc-url $POLYGON_AMOY_RPC_URL \
  --broadcast \
  --verify
```

## Usage Examples

### TypeScript SDK

```typescript
import {
  MycelixEthBridge,
  parseScore,
  NETWORK_CONFIGS,
  SupportedNetwork
} from "@mycelix/sdk-eth";
import { ethers } from "ethers";

// Setup
const provider = new ethers.JsonRpcProvider(
  NETWORK_CONFIGS[SupportedNetwork.PolygonMumbai].rpcUrl
);
const signer = new ethers.Wallet(process.env.PRIVATE_KEY!, provider);

const bridge = new MycelixEthBridge({
  provider,
  signer,
  contracts: {
    reputationAnchor: "0x...",
    paymentRouter: "0x...",
    mycelixRegistry: "0x...",
  },
});
```

### Verify Reputation

```typescript
// Verify an agent's reputation using a Merkle proof
const result = await bridge.verifyReputation({
  agent: "0x1234567890123456789012345678901234567890",
  score: parseScore(850), // 850 out of 1000
  nonce: 1n,
  proof: [
    "0x1111111111111111111111111111111111111111111111111111111111111111",
    "0x2222222222222222222222222222222222222222222222222222222222222222",
  ],
});

console.log(`Reputation valid: ${result.valid}`);
console.log(`Score: ${result.score}`);
```

### Route Payment

```typescript
// Route payment to multiple recipients
const payment = await bridge.routePayment(
  [
    { recipient: "0xArtist...", shareBps: 7000 },   // 70%
    { recipient: "0xLabel...", shareBps: 2000 },    // 20%
    { recipient: "0xPlatform...", shareBps: 1000 }, // 10%
  ],
  ethers.ZeroAddress, // Native token (MATIC)
  ethers.parseEther("10")
);

console.log(`Payment ID: ${payment.paymentId}`);
console.log(`Transaction: ${payment.transactionHash}`);
```

### Create Escrow

```typescript
// Create an escrow for a marketplace transaction
const escrow = await bridge.createEscrow(
  "0xPayee...",           // Payee address
  ethers.ZeroAddress,      // Native token
  ethers.parseEther("5"),  // Amount
  7 * 24 * 60 * 60,        // 7 days duration
  "0x...",                 // Metadata IPFS hash
);

console.log(`Escrow ID: ${escrow.escrowId}`);
console.log(`Releases at: ${new Date(Number(escrow.releaseTime) * 1000)}`);

// Later: release the escrow
await bridge.releaseEscrow(escrow.escrowId);
```

### Register DID

```typescript
// Register a DID (links Holochain agent to Ethereum address)
await bridge.registerDID(
  "did:mycelix:uhCAk123456789...",
  "0x..." // IPFS hash of metadata
);

// Resolve a DID
const resolved = await bridge.resolveDID("did:mycelix:uhCAk123456789...");
console.log(`Owner: ${resolved.owner}`);
console.log(`Active: ${resolved.isActive}`);
```

## Gas Cost Estimates

Estimated gas costs on Polygon (at 30 gwei gas price):

| Operation | Gas Units | Cost (MATIC) |
|-----------|-----------|--------------|
| Store Reputation Root | ~50,000 | ~0.0015 |
| Verify Reputation (view) | 0 | Free |
| Verify and Record | ~65,000 | ~0.002 |
| Route Payment (2 recipients) | ~120,000 | ~0.0036 |
| Route Payment (10 recipients) | ~350,000 | ~0.0105 |
| Create Escrow | ~150,000 | ~0.0045 |
| Release Escrow | ~80,000 | ~0.0024 |
| Open Dispute | ~100,000 | ~0.003 |
| Resolve Dispute | ~100,000 | ~0.003 |
| Register DID | ~100,000 | ~0.003 |
| Update DID Owner | ~55,000 | ~0.00165 |
| Register hApp | ~80,000 | ~0.0024 |

*Note: Actual costs vary with network congestion and gas prices.*

## Security Considerations

### Access Control

The contracts use OpenZeppelin's AccessControl for role-based permissions:

- **DEFAULT_ADMIN_ROLE**: Can grant/revoke roles
- **ANCHOR_ROLE** (ReputationAnchor): Can store reputation roots
- **GOVERNANCE_ROLE**: Can update fees, pause contracts, etc.
- **ARBITRATOR_ROLE** (PaymentRouter): Can resolve disputes
- **ROUTER_ROLE** (PaymentRouter): Can initiate payments (for trusted contracts)
- **HAPP_REGISTRAR_ROLE** (Registry): Can register hApps

### Security Features

1. **Reentrancy Protection**: All state-changing functions use ReentrancyGuard
2. **Pausable**: Contracts can be paused in emergencies
3. **Safe Token Transfers**: Uses OpenZeppelin's SafeERC20
4. **Merkle Proofs**: Efficient and secure verification of reputation data
5. **Grace Period**: Smooth transition when updating reputation roots
6. **Token Whitelist**: Only approved tokens can be used for payments

### Best Practices

- Store private keys securely (hardware wallets for production)
- Verify contract addresses before interacting
- Use appropriate gas limits
- Monitor for unusual activity
- Keep contracts upgraded with latest security patches

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Holochain Network                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ PoGQ Engine  │  │ Marketplace  │  │   Music      │           │
│  │  (Scores)    │  │   hApp       │  │   hApp       │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Mycelix Bridge SDK                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ anchorRep()  │  │ routePay()   │  │ registerDID()│           │
│  │ verifyRep()  │  │ escrow()     │  │ resolve()    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Polygon Network                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Reputation   │  │  Payment     │  │   Mycelix    │           │
│  │   Anchor     │  │   Router     │  │   Registry   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Testing

### Run Contract Tests

```bash
cd Mycelix-Core/contracts

# Run all tests
forge test

# Run with verbosity
forge test -vvv

# Run specific test file
forge test --match-path test/ReputationAnchor.t.sol

# Run with gas reporting
forge test --gas-report

# Run with coverage
forge coverage
```

### Run SDK Tests

```bash
cd mycelix-workspace/sdk-eth

npm test

# With coverage
npm run test:coverage
```

## Upgradeability

The current contracts are not upgradeable. For production deployments, consider:

1. Using OpenZeppelin's UUPS or Transparent Proxy patterns
2. Implementing migration scripts for data transfer
3. Using a timelock for governance actions

## Support

- **Documentation**: [docs.mycelix.net](https://docs.mycelix.net)
- **GitHub Issues**: [github.com/Luminous-Dynamics/Mycelix-Core](https://github.com/Luminous-Dynamics/Mycelix-Core)
- **Discord**: [discord.gg/mycelix](https://discord.gg/mycelix)

## License

MIT License - see LICENSE file for details.

---

## Deployment Status (2026-01-08)

### Contracts Tested and Ready

| Contract | Status | Tests |
|----------|--------|-------|
| ReputationAnchor | Ready | 19/19 passed |
| PaymentRouter | Ready | 20/20 passed |
| MycelixRegistry | Ready | 29/29 passed |

**Total: 68 tests passing**

### Deployment Pending

Deployment to Polygon Amoy requires:
1. Testnet POL tokens (approx. 0.25 POL)
2. Get tokens from [Polygon Amoy Faucet](https://faucet.polygon.technology/)

See `DEPLOYMENT_READY.md` for complete deployment instructions.

### Compilation Details

- **Solidity Version:** 0.8.24
- **Foundry Version:** 1.4.4
- **Dependencies:**
  - OpenZeppelin Contracts v5.0.2
  - Forge-std v1.9.0
  - Murky (Merkle tree library)

### Gas Estimates (Polygon Amoy)

| Contract | Gas | Est. Cost |
|----------|-----|-----------|
| ReputationAnchor | ~1.8M | ~0.055 POL |
| PaymentRouter | ~1.9M | ~0.060 POL |
| MycelixRegistry | ~3.1M | ~0.095 POL |
| **Total** | **~6.8M** | **~0.21 POL** |

