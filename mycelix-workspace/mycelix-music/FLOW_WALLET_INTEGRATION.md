# Flow Blockchain Wallet Integration Research

## Overview
Flow is a blockchain designed for high-performance NFTs and games. Mycelix Music could benefit from Flow's low transaction costs and fast finality for music streaming micropayments.

## Current Status
- **Privy**: Does not natively support Flow blockchain yet
- **Alternative**: Need custom Flow wallet integration
- **Use Case**: Lower transaction costs for streaming ($0.01 per stream)

## Flow Wallet Options

### 1. Flow Client Library (FCL) - Official Solution ✅ Recommended

**Description**: Flow's official wallet connection library, similar to Web3.js/ethers.js

**Features**:
- Supports all major Flow wallets (Blocto, Dapper, Lilico, NuFi)
- Built-in authentication flow
- Transaction signing
- Account creation

**Installation**:
```bash
npm install @onflow/fcl @onflow/types
```

**Basic Setup**:
```typescript
// lib/flow.ts
import * as fcl from '@onflow/fcl';

// Configure FCL
fcl.config({
  'accessNode.api': 'https://rest-mainnet.onflow.org', // Mainnet
  // 'accessNode.api': 'https://rest-testnet.onflow.org', // Testnet
  'discovery.wallet': 'https://fcl-discovery.onflow.org/authn', // Wallet discovery
  'app.detail.title': 'Mycelix Music',
  'app.detail.icon': 'https://music.mycelix.net/logo.png',
});

// Authenticate user
export async function authenticate() {
  return await fcl.authenticate();
}

// Get current user
export function getCurrentUser() {
  return fcl.currentUser;
}

// Sign out
export async function unauthenticate() {
  return await fcl.unauthenticate();
}

// Execute transaction
export async function sendTransaction(cadence: string, args: any[]) {
  const txId = await fcl.mutate({
    cadence,
    args,
    limit: 100,
  });

  return fcl.tx(txId).onceSealed();
}
```

**Integration with Mycelix**:
```typescript
// Example: Pay for stream on Flow
import * as fcl from '@onflow/fcl';
import * as t from '@onflow/types';

const CODE = `
import FlowToken from 0x1654653399040a61

transaction(amount: UFix64, recipient: Address) {
  let vault: @FlowToken.Vault

  prepare(signer: AuthAccount) {
    let vaultRef = signer.borrow<&FlowToken.Vault>(from: /storage/flowTokenVault)
      ?? panic("Could not borrow reference to the owner's Vault!")

    self.vault <- vaultRef.withdraw(amount: amount)
  }

  execute {
    let recipient = getAccount(recipient)
    let receiverRef = recipient.getCapability(/public/flowTokenReceiver)
      .borrow<&{FungibleToken.Receiver}>()
      ?? panic("Could not borrow receiver reference to the recipient's Vault")

    receiverRef.deposit(from: <-self.vault)
  }
}
`;

export async function payForStream(artistAddress: string, amount: string) {
  const txId = await fcl.mutate({
    cadence: CODE,
    args: (arg, t) => [
      arg(amount, t.UFix64),
      arg(artistAddress, t.Address),
    ],
    payer: fcl.authz,
    proposer: fcl.authz,
    authorizations: [fcl.authz],
    limit: 100,
  });

  return fcl.tx(txId).onceSealed();
}
```

### 2. Blocto Wallet SDK - Easiest UX

**Description**: Blocto provides a seamless wallet experience with email/social login

**Features**:
- Email login (no seed phrases!)
- Social login (Google, Apple, etc.)
- Fiat on-ramp
- Multi-chain support (Flow, Ethereum, Polygon, etc.)

**Installation**:
```bash
npm install @blocto/fcl
```

**Setup**:
```typescript
import * as fcl from '@onflow/fcl';

fcl.config({
  'accessNode.api': 'https://rest-mainnet.onflow.org',
  'discovery.wallet': 'https://wallet.blocto.app/api/flow/authn',
});
```

**Pros**:
- Best UX for non-crypto users
- No seed phrase management
- Built-in fiat on-ramp
- Free account creation

**Cons**:
- Custodial wallet (Blocto holds keys)
- Less decentralized

### 3. Dapper Wallet - Created by Flow Team

**Description**: Official wallet from Dapper Labs (Flow creators)

**Features**:
- Credit card payments
- Used by NBA Top Shot, NFL All Day
- Optimized for NFTs and collectibles
- Custodial model

**Setup**: Works via FCL wallet discovery

**Pros**:
- Official Flow wallet
- Credit card support
- Proven at scale

**Cons**:
- Custodial
- Requires KYC for fiat

### 4. Custom Mycelix Wallet for Flow

**Concept**: Build a custom embedded wallet specifically for Mycelix Music on Flow

**Architecture**:
```
User Account (Social Login)
  ↓
Mycelix Custodial Service
  ↓
Flow Account (managed keys)
  ↓
Smart Contracts (streaming, payments)
```

**Implementation Approach**:

1. **Account Creation**:
```typescript
// Server-side account creation
import { ec as EC } from 'elliptic';
const ec = new EC('p256');

export async function createFlowAccount(userId: string) {
  // Generate key pair
  const keyPair = ec.genKeyPair();
  const privateKey = keyPair.getPrivate('hex');
  const publicKey = keyPair.getPublic('hex');

  // Store encrypted private key in database
  await storeEncryptedKey(userId, privateKey);

  // Create Flow account via service account
  const account = await createAccountOnChain(publicKey);

  return {
    address: account.address,
    publicKey: publicKey,
  };
}
```

2. **Transaction Signing**:
```typescript
// Server-side signing service
export async function signTransaction(userId: string, transaction: any) {
  const privateKey = await getDecryptedKey(userId);

  // Sign transaction
  const signature = signWithKey(privateKey, transaction);

  return signature;
}
```

3. **Security Considerations**:
- Encrypt private keys at rest (AES-256)
- Use hardware security modules (HSM) for production
- Implement transaction approval flow
- Rate limiting and fraud detection
- Multi-sig for large transactions

**Pros**:
- Full control over UX
- No external wallet required
- Seamless onboarding
- Can integrate fiat directly

**Cons**:
- Custodial (security responsibility)
- More development work
- Regulatory considerations
- Key management complexity

## Recommended Approach

### Phase 1: FCL with Blocto (Immediate)
Use Flow Client Library with Blocto wallet for:
- Quick integration
- Best UX for mainstream users
- No custody concerns (users control keys via Blocto)

```typescript
// Quick integration example
import * as fcl from '@onflow/fcl';

fcl.config({
  'accessNode.api': 'https://rest-mainnet.onflow.org',
  'discovery.wallet': 'https://wallet.blocto.app/api/flow/authn',
});

// In your component
const { authenticate, currentUser, unauthenticate } = useFlowAuth();
```

### Phase 2: Hybrid Approach (3-6 months)
- FCL for crypto-native users (MetaMask, Ledger, etc.)
- Blocto for mainstream users (email/social)
- Custom embedded wallet for power users who want Mycelix-only account

### Phase 3: Full Mycelix Wallet (6-12 months)
- Custom Flow wallet fully integrated into Mycelix
- Fiat on-ramp integration
- Multi-chain support (Flow + EVM chains)
- Advanced features (subscriptions, automated tipping, etc.)

## Technical Comparison

| Feature | FCL + Blocto | Dapper | Custom Wallet |
|---------|-------------|--------|---------------|
| Integration Time | 2-3 days | 2-3 days | 4-6 weeks |
| User Onboarding | Email/Social | Credit Card | Email/Social |
| Custody | User | Dapper | Mycelix |
| Transaction Fees | ~$0.001 | ~$0.001 | ~$0.001 |
| Fiat On-ramp | Via Blocto | Built-in | Custom |
| Regulatory | Low risk | KYC required | High risk |
| Development Cost | Low | Low | High |
| Maintenance | Low | Low | High |

## Implementation Steps

### Step 1: Add Flow Support to Project
```bash
npm install @onflow/fcl @onflow/types
```

### Step 2: Create Flow Configuration
```typescript
// lib/flow/config.ts
import * as fcl from '@onflow/fcl';

export function configureFlow() {
  fcl.config({
    'accessNode.api': process.env.NEXT_PUBLIC_FLOW_ACCESS_NODE,
    'discovery.wallet': process.env.NEXT_PUBLIC_FLOW_WALLET_DISCOVERY,
    'app.detail.title': 'Mycelix Music',
    'app.detail.icon': 'https://music.mycelix.net/logo.png',
  });
}
```

### Step 3: Create React Hooks
```typescript
// hooks/useFlowAuth.ts
import { useState, useEffect } from 'react';
import * as fcl from '@onflow/fcl';

export function useFlowAuth() {
  const [user, setUser] = useState({ loggedIn: false, addr: null });

  useEffect(() => {
    fcl.currentUser.subscribe(setUser);
  }, []);

  const login = () => fcl.authenticate();
  const logout = () => fcl.unauthenticate();

  return { user, login, logout };
}
```

### Step 4: Update UI
```typescript
// components/FlowWalletButton.tsx
import { useFlowAuth } from '@/hooks/useFlowAuth';

export function FlowWalletButton() {
  const { user, login, logout } = useFlowAuth();

  return user.loggedIn ? (
    <button onClick={logout}>
      Disconnect Flow ({user.addr})
    </button>
  ) : (
    <button onClick={login}>
      Connect Flow Wallet
    </button>
  );
}
```

## Cost Comparison: Flow vs EVM

| Operation | Flow | Gnosis Chain | Ethereum Mainnet |
|-----------|------|--------------|------------------|
| Stream Payment ($0.01) | ~$0.0001 | ~$0.001 | ~$0.50 |
| Tip Artist ($1.00) | ~$0.0001 | ~$0.001 | ~$1.00 |
| NFT Mint | ~$0.01 | ~$0.10 | ~$50 |
| Account Creation | Free | ~$0.10 | ~$10 |

**Why Flow wins for Mycelix**:
- 10x-100x cheaper than Gnosis
- 1000x-5000x cheaper than Ethereum
- Sub-second finality
- Built for high-volume microtransactions

## Next Steps

1. ✅ Research Flow wallet options (DONE)
2. ⏳ Deploy test smart contracts to Flow Testnet
3. ⏳ Integrate FCL + Blocto for wallet connection
4. ⏳ Create payment flow for streaming
5. ⏳ Test with real users
6. ⏳ Deploy to Flow Mainnet

## Resources

- [Flow Documentation](https://developers.flow.com/)
- [FCL Guide](https://developers.flow.com/tools/fcl-js)
- [Blocto SDK](https://docs.blocto.app/)
- [Flow Playground](https://play.onflow.org/)
- [Cadence Language](https://developers.flow.com/cadence)

## Questions to Resolve

1. **Should we support both EVM and Flow?**
   - Yes for maximum flexibility
   - Users choose based on preference/cost

2. **Custody model?**
   - Start non-custodial (Blocto)
   - Add embedded wallet later for convenience

3. **What about existing EVM contracts?**
   - Keep them on Gnosis
   - Gradually migrate to Flow for new features

4. **Fiat on-ramp?**
   - Use Blocto's built-in for Phase 1
   - Custom solution (Stripe, Moonpay) for Phase 2
