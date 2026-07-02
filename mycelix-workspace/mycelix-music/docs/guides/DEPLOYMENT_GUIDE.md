# Mycelix Music: Deployment & Testing Guide

## Prerequisites

Before deploying, ensure you have:

```bash
# Development tools
- Node.js 20+
- Foundry (for Solidity)
- Holochain dev tools
- Ceramic CLI

# Wallets & Keys
- Private key for deploying contracts
- XDAI on Gnosis Chiado testnet (get from faucet)
- Web3.Storage API key (free)
- Privy API key (free tier)
```

## Phase 1: Local Development Setup

### Step 1: Clone and Install

```bash
# Clone repository
git clone https://github.com/mycelix/mycelix-music
cd mycelix-music

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
```

### Step 2: Configure Environment

```bash
# .env file
PRIVATE_KEY=your_private_key
RPC_URL=https://rpc.chiadochain.net
WEB3_STORAGE_TOKEN=your_token
PRIVY_APP_ID=your_app_id
CERAMIC_NODE=https://ceramic-clay.3boxlabs.com

# Frontend
NEXT_PUBLIC_ROUTER_ADDRESS=  # Fill after deployment
NEXT_PUBLIC_FLOW_TOKEN_ADDRESS=  # Fill after deployment
```

### Step 3: Run Local Tests

```bash
# Test smart contracts
cd contracts
forge test -vvv

# Expected output:
# ✓ testRegisterSong (gas: 185432)
# ✓ testProcessPayment (gas: 201345)
# ✓ testPayPerStreamStrategy (gas: 176234)
# ✓ testGiftEconomyStrategy (gas: 189123)
# All tests passed!

# Test TypeScript SDK
cd ../packages/sdk
npm test

# Test React components
cd ../../apps/web
npm test
```

## Phase 2: Testnet Deployment

### Step 1: Deploy FLOW Token (ERC20)

```bash
cd contracts

# Deploy mock FLOW token for testing
forge script script/DeployFlowToken.s.sol \
  --rpc-url $RPC_URL \
  --private-key $PRIVATE_KEY \
  --broadcast

# Save the deployed address
export FLOW_TOKEN_ADDRESS=<deployed_address>
```

```solidity
// script/DeployFlowToken.s.sol
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/FlowToken.sol";

contract DeployFlowToken is Script {
    function run() external {
        vm.startBroadcast();

        FlowToken token = new FlowToken();
        console.log("FLOW Token deployed at:", address(token));

        // Mint 1M FLOW for testing
        token.mint(msg.sender, 1_000_000 ether);

        vm.stopBroadcast();
    }
}
```

### Step 2: Deploy CGC Token (Gift Economy)

```bash
forge script script/DeployCGCToken.s.sol \
  --rpc-url $RPC_URL \
  --private-key $PRIVATE_KEY \
  --broadcast

export CGC_TOKEN_ADDRESS=<deployed_address>
```

### Step 3: Deploy Economic Strategy Router

```bash
forge script script/DeployRouter.s.sol \
  --rpc-url $RPC_URL \
  --private-key $PRIVATE_KEY \
  --broadcast \
  --verify

export ROUTER_ADDRESS=<deployed_address>
```

```solidity
// script/DeployRouter.s.sol
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/EconomicStrategyRouter.sol";

contract DeployRouter is Script {
    function run() external {
        vm.startBroadcast();

        address flowToken = vm.envAddress("FLOW_TOKEN_ADDRESS");

        EconomicStrategyRouter router = new EconomicStrategyRouter(flowToken);
        console.log("Router deployed at:", address(router));

        vm.stopBroadcast();
    }
}
```

### Step 4: Deploy Strategy Implementations

```bash
# Deploy Pay Per Stream Strategy
forge script script/DeployPayPerStream.s.sol \
  --rpc-url $RPC_URL \
  --private-key $PRIVATE_KEY \
  --broadcast

export PAY_PER_STREAM_ADDRESS=<deployed_address>

# Deploy Gift Economy Strategy
forge script script/DeployGiftEconomy.s.sol \
  --rpc-url $RPC_URL \
  --private-key $PRIVATE_KEY \
  --broadcast

export GIFT_ECONOMY_ADDRESS=<deployed_address>
```

### Step 5: Register Strategies with Router

```bash
# Create registration script
cat > scripts/register-strategies.js << 'EOF'
const { ethers } = require('ethers');

async function main() {
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);

  const router = new ethers.Contract(
    process.env.ROUTER_ADDRESS,
    ['function registerStrategy(bytes32 strategyId, address strategyAddress) external'],
    wallet
  );

  // Register Pay Per Stream
  const tx1 = await router.registerStrategy(
    ethers.id('pay-per-stream-v1'),
    process.env.PAY_PER_STREAM_ADDRESS
  );
  await tx1.wait();
  console.log('✓ Pay Per Stream strategy registered');

  // Register Gift Economy
  const tx2 = await router.registerStrategy(
    ethers.id('gift-economy-v1'),
    process.env.GIFT_ECONOMY_ADDRESS
  );
  await tx2.wait();
  console.log('✓ Gift Economy strategy registered');

  console.log('All strategies registered successfully!');
}

main().catch(console.error);
EOF

node scripts/register-strategies.js
```

### Step 6: Verify Deployment

```bash
# Check that everything is connected
cat > scripts/verify-deployment.js << 'EOF'
const { ethers } = require('ethers');

async function main() {
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);

  const router = new ethers.Contract(
    process.env.ROUTER_ADDRESS,
    [
      'function registeredStrategies(bytes32) view returns (address)',
      'function flowToken() view returns (address)'
    ],
    provider
  );

  // Verify strategies are registered
  const ppsAddress = await router.registeredStrategies(ethers.id('pay-per-stream-v1'));
  const geAddress = await router.registeredStrategies(ethers.id('gift-economy-v1'));
  const flowToken = await router.flowToken();

  console.log('\n🎉 Deployment Verification:');
  console.log('─────────────────────────────');
  console.log('✓ Router:', process.env.ROUTER_ADDRESS);
  console.log('✓ FLOW Token:', flowToken);
  console.log('✓ Pay Per Stream:', ppsAddress);
  console.log('✓ Gift Economy:', geAddress);
  console.log('─────────────────────────────\n');

  if (ppsAddress === ethers.ZeroAddress || geAddress === ethers.ZeroAddress) {
    throw new Error('Strategies not registered correctly!');
  }

  console.log('✅ All contracts deployed and registered successfully!');
}

main().catch(console.error);
EOF

node scripts/verify-deployment.js
```

## Phase 3: End-to-End Testing

### Test 1: Artist Uploads Song with Pay Per Stream

```bash
cat > scripts/test-pay-per-stream.js << 'EOF'
const { ethers } = require('ethers');
const { EconomicStrategySDK } = require('@mycelix/sdk');

async function main() {
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const artistWallet = new ethers.Wallet(process.env.ARTIST_PRIVATE_KEY, provider);
  const listenerWallet = new ethers.Wallet(process.env.LISTENER_PRIVATE_KEY, provider);

  const sdk = new EconomicStrategySDK(
    provider,
    process.env.ROUTER_ADDRESS,
    artistWallet
  );

  console.log('🎸 Test: Pay Per Stream Strategy');
  console.log('─────────────────────────────────\n');

  // Step 1: Artist registers song
  console.log('Step 1: Artist registers song...');
  const songId = 'test-song-' + Date.now();

  const receipt = await sdk.registerSong(songId, {
    strategyId: 'pay-per-stream-v1',
    paymentModel: 'pay_per_stream',
    distributionSplits: [
      { recipient: await artistWallet.getAddress(), basisPoints: 9500, role: 'artist' },
      { recipient: process.env.ROUTER_ADDRESS, basisPoints: 500, role: 'protocol' }
    ],
    minimumPayment: 0.01,
  });

  console.log('✓ Song registered. TX:', receipt.hash);

  // Step 2: Listener mints FLOW tokens
  console.log('\nStep 2: Listener gets FLOW tokens...');
  const flowToken = new ethers.Contract(
    process.env.FLOW_TOKEN_ADDRESS,
    ['function mint(address to, uint256 amount) external'],
    artistWallet  // Use artist wallet which has mint permissions
  );

  const mintTx = await flowToken.mint(await listenerWallet.getAddress(), ethers.parseEther('100'));
  await mintTx.wait();
  console.log('✓ Listener received 100 FLOW');

  // Step 3: Listener streams song
  console.log('\nStep 3: Listener streams song...');
  const listenerSDK = new EconomicStrategySDK(
    provider,
    process.env.ROUTER_ADDRESS,
    listenerWallet
  );

  const initialBalance = await flowToken.balanceOf(await artistWallet.getAddress());

  const streamReceipt = await listenerSDK.streamSong(songId, '0.01');
  console.log('✓ Stream paid. TX:', streamReceipt.transactionHash);

  // Step 4: Verify artist received payment
  const finalBalance = await flowToken.balanceOf(await artistWallet.getAddress());
  const earned = ethers.formatEther(finalBalance - initialBalance);

  console.log('\n💰 Payment Results:');
  console.log('─────────────────────────────────');
  console.log('Listener paid: 0.01 FLOW');
  console.log('Artist received:', earned, 'FLOW');
  console.log('Protocol fee:', (0.01 - parseFloat(earned)).toFixed(4), 'FLOW');
  console.log('─────────────────────────────────\n');

  if (Math.abs(parseFloat(earned) - 0.0095) > 0.0001) {
    throw new Error('Payment split incorrect!');
  }

  console.log('✅ Pay Per Stream test PASSED!\n');
}

main().catch(console.error);
EOF

node scripts/test-pay-per-stream.js
```

### Test 2: Artist Uploads with Gift Economy

```bash
cat > scripts/test-gift-economy.js << 'EOF'
const { ethers } = require('ethers');
const { EconomicStrategySDK } = require('@mycelix/sdk');

async function main() {
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const artistWallet = new ethers.Wallet(process.env.ARTIST_PRIVATE_KEY, provider);
  const listenerWallet = new ethers.Wallet(process.env.LISTENER_PRIVATE_KEY, provider);

  const sdk = new EconomicStrategySDK(
    provider,
    process.env.ROUTER_ADDRESS,
    artistWallet
  );

  console.log('🎁 Test: Gift Economy Strategy');
  console.log('─────────────────────────────────\n');

  // Step 1: Artist registers song
  console.log('Step 1: Artist registers song with gift economy...');
  const songId = 'gift-song-' + Date.now();

  const receipt = await sdk.registerSong(songId, {
    strategyId: 'gift-economy-v1',
    paymentModel: 'gift_economy',
    distributionSplits: [
      { recipient: await artistWallet.getAddress(), basisPoints: 9500, role: 'artist' },
      { recipient: process.env.ROUTER_ADDRESS, basisPoints: 500, role: 'protocol' }
    ],
    acceptsGifts: true,
    minimumPayment: 0,
  });

  console.log('✓ Song registered with gift economy. TX:', receipt.hash);

  // Step 2: Listener streams for FREE
  console.log('\nStep 2: Listener streams for FREE...');
  const listenerSDK = new EconomicStrategySDK(
    provider,
    process.env.ROUTER_ADDRESS,
    listenerWallet
  );

  const streamReceipt = await listenerSDK.streamSong(songId, '0');  // FREE!
  console.log('✓ Free stream recorded. TX:', streamReceipt.transactionHash);

  // Step 3: Check listener's CGC balance
  console.log('\nStep 3: Check listener rewards...');
  const cgcBalance = await listenerSDK.getListenerCGCBalance(
    await artistWallet.getAddress(),
    await listenerWallet.getAddress()
  );

  console.log('\n🎁 Listener Rewards:');
  console.log('─────────────────────────────────');
  console.log('CGC earned:', ethers.formatEther(cgcBalance));
  console.log('Status: Listener got PAID to listen!');
  console.log('─────────────────────────────────\n');

  // Step 4: Optional tip
  console.log('Step 4: Listener sends optional tip...');
  const tipReceipt = await listenerSDK.tipArtist(songId, '1');
  console.log('✓ Tip sent. TX:', tipReceipt.transactionHash);

  console.log('✅ Gift Economy test PASSED!\n');
}

main().catch(console.error);
EOF

node scripts/test-gift-economy.js
```

### Test 3: Artist Changes Strategy

```bash
cat > scripts/test-strategy-change.js << 'EOF'
const { ethers } = require('ethers');
const { EconomicStrategySDK } = require('@mycelix/sdk');

async function main() {
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const artistWallet = new ethers.Wallet(process.env.ARTIST_PRIVATE_KEY, provider);

  const sdk = new EconomicStrategySDK(
    provider,
    process.env.ROUTER_ADDRESS,
    artistWallet
  );

  console.log('🔄 Test: Strategy Change');
  console.log('─────────────────────────────────\n');

  // Start with gift economy
  const songId = 'change-test-' + Date.now();

  console.log('Step 1: Register with gift economy...');
  await sdk.registerSong(songId, {
    strategyId: 'gift-economy-v1',
    paymentModel: 'gift_economy',
    distributionSplits: [
      { recipient: await artistWallet.getAddress(), basisPoints: 10000, role: 'artist' }
    ],
  });
  console.log('✓ Song using gift economy');

  // Change to pay per stream
  console.log('\nStep 2: Change to pay per stream...');
  const changeTx = await sdk.changeSongStrategy(songId, 'pay-per-stream-v1');
  await changeTx.wait();
  console.log('✓ Strategy changed to pay per stream');

  // Verify change
  const router = new ethers.Contract(
    process.env.ROUTER_ADDRESS,
    ['function songStrategy(bytes32) view returns (address)'],
    provider
  );

  const currentStrategy = await router.songStrategy(ethers.id(songId));
  const expectedStrategy = process.env.PAY_PER_STREAM_ADDRESS;

  if (currentStrategy !== expectedStrategy) {
    throw new Error('Strategy change failed!');
  }

  console.log('\n✅ Strategy change test PASSED!\n');
}

main().catch(console.error);
EOF

node scripts/test-strategy-change.js
```

## Phase 4: Frontend Deployment

### Step 1: Build Frontend

```bash
cd apps/web

# Update environment with deployed addresses
echo "NEXT_PUBLIC_ROUTER_ADDRESS=$ROUTER_ADDRESS" >> .env.local
echo "NEXT_PUBLIC_FLOW_TOKEN_ADDRESS=$FLOW_TOKEN_ADDRESS" >> .env.local

# Build
npm run build

# Test locally
npm run dev
# Open http://localhost:3000
```

### Step 2: Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod

# Set environment variables
vercel env add NEXT_PUBLIC_ROUTER_ADDRESS production
vercel env add NEXT_PUBLIC_FLOW_TOKEN_ADDRESS production

# Redeploy with env vars
vercel --prod
```

### Step 3: Manual Test Flow

1. **Connect Wallet:** Visit deployed app, connect MetaMask
2. **Upload Song:** Go to /upload, choose audio file
3. **Choose Strategy:** Use wizard to select gift economy
4. **Publish:** Deploy economic config on-chain
5. **Share Link:** Copy song URL
6. **Test as Listener:** Open in incognito, stream song
7. **Verify Payment:** Check artist wallet balance

## Phase 5: Production Checklist

Before going to mainnet:

```bash
# ✅ Smart Contract Audits
- [ ] Get contracts audited by professional firm
- [ ] Run Slither static analysis
- [ ] Test with 10K+ mock transactions

# ✅ Frontend Security
- [ ] Add rate limiting
- [ ] Implement CSRF protection
- [ ] Set up monitoring (Sentry)

# ✅ Economic Testing
- [ ] Test all strategy combinations
- [ ] Verify split calculations to 18 decimals
- [ ] Test edge cases (0 payments, max payments)

# ✅ Documentation
- [ ] API documentation complete
- [ ] Artist onboarding guide written
- [ ] Video tutorials recorded

# ✅ Legal
- [ ] DMCA compliance implemented
- [ ] Terms of service written
- [ ] Privacy policy published
```

## Monitoring & Maintenance

### Set Up Monitoring

```typescript
// apps/web/src/lib/monitoring.ts

import * as Sentry from '@sentry/nextjs';

export function setupMonitoring() {
  Sentry.init({
    dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
    tracesSampleRate: 1.0,
  });

  // Track economic events
  window.addEventListener('payment-processed', (event) => {
    Sentry.captureMessage('Payment processed', {
      level: 'info',
      extra: event.detail,
    });
  });
}
```

### Set Up Alerts

```bash
# Set up alerts for:
- Failed transactions (>5% failure rate)
- Gas price spikes (>100 gwei)
- Contract balance low (<1 ETH)
- API errors (>1% error rate)
```

---

## Summary: Deployment Timeline

**Week 1:** Local development and testing
**Week 2:** Testnet deployment and testing
**Week 3:** Frontend deployment and integration testing
**Week 4:** Beta launch with 10 artists
**Month 2-3:** Iterate based on feedback
**Month 4:** Security audit
**Month 5:** Mainnet deployment

**Estimated Costs:**
- Smart contract deployment: $50 (testnet) / $500 (mainnet)
- Frontend hosting: $0 (Vercel free tier) / $20/mo (pro)
- Security audit: $15-25K
- Infrastructure: $200/month (RPC, IPFS, Ceramic)

**Total First Year:** ~$30K for professional launch

🚀 **You're ready to launch a modular economic music platform!**
