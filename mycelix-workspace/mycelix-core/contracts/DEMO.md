# Mycelix Demo Walkthrough

A step-by-step guide to interacting with the Mycelix smart contracts on Sepolia testnet.

## Prerequisites

1. **MetaMask** or compatible Web3 wallet installed
2. **Sepolia ETH** for gas fees (get from [Sepolia PoW Faucet](https://sepolia-faucet.pk910.de/))
3. **Node.js 18+** installed

## Quick Demo (CLI)

### 1. Setup

```bash
cd contracts/sdk
npm install
```

### 2. View Network Info

```bash
npm run cli -- info
```

Expected output:
```
=== Mycelix Network Info ===

Network:     sepolia
Chain ID:    11155111
Contracts:
  Registry:   0x556b810371e3d8D9E5753117514F03cC6C93b835
  Reputation: 0xf3B343888a9b82274cEfaa15921252DB6c5f48C9
  Payment:    0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB

Registry Stats:
  Total DIDs: 1
  Reg. Fee:   0.0 ETH
```

### 3. Register a DID

```bash
# Set your private key (use a testnet wallet!)
export PRIVATE_KEY=0x...

# Register a new DID
npm run cli -- register did:mycelix:alice ipfs://QmYourMetadataHash
```

Expected output:
```
Registering DID: did:mycelix:alice
Metadata URI:    ipfs://QmYourMetadataHash

Transaction:     0x...
Explorer:        https://sepolia.etherscan.io/tx/0x...

Waiting for confirmation...
Confirmed:       Block 12345678
Gas Used:        123456

DID registered successfully!
```

### 4. Look Up a DID

```bash
npm run cli -- lookup did:mycelix:alice
```

Expected output:
```
Looking up: did:mycelix:alice

Owner:       0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c
Registered:  2026-01-08T18:47:00.000Z
Updated:     2026-01-08T18:47:00.000Z
Metadata:    0x1234...
Revoked:     false
```

### 5. Query Reputation

```bash
npm run cli -- reputation 0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c
```

Expected output:
```
Querying reputation for: 0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c

Score:       0
Submissions: 0
Last Update: Never
```

### 6. Create an Escrow Payment

```bash
export PRIVATE_KEY=0x...
npm run cli -- payment create 0xRecipientAddress 0.001
```

Expected output:
```
Creating escrow payment:
  To:          0xRecipientAddress
  Amount:      0.001 ETH
  Release:     2026-01-08T19:47:00.000Z

Transaction:   0x...
Confirmed:     Block 12345679

Payment created successfully!
```

---

## Interactive Demo (dApp)

### 1. Start the dApp

```bash
cd contracts/dapp
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

### 2. Connect Wallet

1. Click **"Connect Wallet"**
2. Approve the MetaMask connection
3. If on wrong network, click **"Switch to Sepolia"**

### 3. Register a DID

1. Enter a DID identifier (e.g., `did:mycelix:my-unique-id`)
2. Enter a metadata URI (e.g., `ipfs://QmYourHash`)
3. Click **"Register DID"**
4. Confirm the transaction in MetaMask
5. Wait for confirmation

### 4. Look Up a DID

1. Scroll to **"Lookup DID"** section
2. Enter a DID to search for
3. Click **"Lookup"**
4. View the owner, registration date, and status

---

## SDK Demo (TypeScript)

```typescript
import { MycelixClient } from '@mycelix/sdk';

async function demo() {
  // Initialize client
  const client = new MycelixClient({
    network: 'sepolia',
    privateKey: process.env.PRIVATE_KEY
  });

  console.log('Wallet:', client.getAddress());

  // Check balance
  const balance = await client.getBalance();
  console.log('Balance:', ethers.formatEther(balance), 'ETH');

  // Get network stats
  const totalDids = await client.registry.getTotalDids();
  console.log('Total DIDs:', totalDids.toString());

  // Register a DID
  const did = `did:mycelix:demo-${Date.now()}`;
  const tx = await client.registry.registerDID(did, 'ipfs://QmDemo');
  console.log('TX:', tx.hash);

  await tx.wait();
  console.log('DID registered!');

  // Verify registration
  const info = await client.registry.getDID(did);
  console.log('Owner:', info.owner);
  console.log('Revoked:', info.revoked);
}

demo().catch(console.error);
```

Run with:
```bash
cd contracts/sdk
PRIVATE_KEY=0x... npm run demo -- register
```

---

## Contract Addresses (Sepolia)

| Contract | Address | Explorer |
|----------|---------|----------|
| Registry | `0x556b810371e3d8D9E5753117514F03cC6C93b835` | [View](https://sepolia.etherscan.io/address/0x556b810371e3d8D9E5753117514F03cC6C93b835) |
| Reputation | `0xf3B343888a9b82274cEfaa15921252DB6c5f48C9` | [View](https://sepolia.etherscan.io/address/0xf3B343888a9b82274cEfaa15921252DB6c5f48C9) |
| Payment | `0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB` | [View](https://sepolia.etherscan.io/address/0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB) |

All contracts verified on [Sourcify](https://sourcify.dev/#/lookup/0x556b810371e3d8D9E5753117514F03cC6C93b835).

---

## Troubleshooting

### "Insufficient funds"
Get Sepolia ETH from the [PoW Faucet](https://sepolia-faucet.pk910.de/).

### "DID already exists"
Each DID can only be registered once. Use a unique identifier.

### "Transaction failed"
Check that you're on Sepolia network and have enough ETH for gas.

### "Network error"
The RPC endpoint may be rate-limited. Try again in a few seconds.

---

## Next Steps

1. **Explore the code**: `contracts/src/` contains the Solidity contracts
2. **Run tests**: `cd contracts/sdk && npx tsx src/test.ts`
3. **Build for production**: `cd contracts/dapp && npm run build`
4. **Read the docs**: [SDK README](./sdk/README.md) | [Contracts README](./README.md)
