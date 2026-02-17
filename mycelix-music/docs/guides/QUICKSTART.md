# 🚀 Mycelix Music: 30-Minute Quick Start

Get the entire Mycelix Music platform running locally in under 30 minutes.

---

## Prerequisites

```bash
# Required
- Docker & Docker Compose
- Node.js 20+
- Foundry (Solidity toolkit)
- Git

# Optional (for full stack)
- NixOS or Nix package manager
- Rust 1.70+ (for Holochain)
```

---

## Step 1: Clone and Setup (5 minutes)

```bash
# Navigate to project
cd /srv/luminous-dynamics/04-infinite-play/core/mycelix-music

# Install dependencies
npm install

# Install Foundry (if not already installed)
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Create environment file
cp .env.example .env
```

### Configure .env

```bash
# .env file
# ==========

# Blockchain
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80  # Anvil default
RPC_URL=http://localhost:8545
CHAIN_ID=31337

# Storage
WEB3_STORAGE_TOKEN=get_from_https://web3.storage
IPFS_GATEWAY=https://w3s.link/ipfs

# DKG (Ceramic)
CERAMIC_URL=http://localhost:7007
CERAMIC_ADMIN_SEED=your_test_seed_here

# Authentication
PRIVY_APP_ID=get_from_https://privy.io
PRIVY_APP_SECRET=your_secret_here

# Frontend
NEXT_PUBLIC_ROUTER_ADDRESS=  # Will be filled after deployment
NEXT_PUBLIC_FLOW_TOKEN_ADDRESS=  # Will be filled after deployment
```

---

## Step 2: Start Local Blockchain (2 minutes)

```bash
# Terminal 1: Start Anvil (local Ethereum node)
anvil --block-time 1

# This gives you:
# - RPC: http://localhost:8545
# - Chain ID: 31337
# - 10 pre-funded accounts
# - Account #0 private key: 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```

---

## Step 3: Deploy Smart Contracts (3 minutes)

```bash
# Terminal 2: Deploy contracts
cd contracts

# Compile contracts
forge build

# Deploy to local network
forge script script/DeployLocal.s.sol \
  --rpc-url http://localhost:8545 \
  --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
  --broadcast

# Output will show deployed addresses:
# FLOW Token: 0x5FbDB2315678afecb367f032d93F642f64180aa3
# EconomicStrategyRouter: 0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512
# PayPerStreamStrategy: 0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0
# GiftEconomyStrategy: 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9

# Copy these addresses to .env
echo "NEXT_PUBLIC_FLOW_TOKEN_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3" >> ../.env
echo "NEXT_PUBLIC_ROUTER_ADDRESS=0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512" >> ../.env
```

---

## Step 4: Start Backend Services (5 minutes)

```bash
# Terminal 3: Start services with Docker Compose
cd ..
docker-compose up -d

# This starts:
# - PostgreSQL (catalog indexer)
# - Redis (caching)
# - Ceramic (DKG node)
# - IPFS (local storage)

# Check services are running
docker-compose ps

# Should see all services "Up"
```

---

## Step 5: Seed Test Data (5 minutes)

```bash
# Terminal 2: Load test fixtures
npm run seed:local

# This creates:
# - 3 test artists (DJ Nova, The Echoes, Orchestra)
# - 10 test songs across different economic models
# - 5 test listeners with FLOW tokens
# - Sample playlists

# Output:
# ✓ Created artist: DJ Nova (gift economy)
# ✓ Created artist: The Echoes (pay per stream)
# ✓ Created artist: Symphony Orchestra (patronage)
# ✓ Uploaded 10 songs to IPFS
# ✓ Created 10 DKG claims
# ✓ Registered 10 songs with router
# ✓ Funded 5 listener accounts with 100 FLOW each
```

---

## Step 6: Start Frontend (5 minutes)

```bash
# Terminal 4: Start Next.js dev server
cd apps/web
npm run dev

# Frontend running at: http://localhost:3000
```

---

## Step 7: Test the Platform (5 minutes)

### As an Artist

1. **Visit**: http://localhost:3000
2. **Click**: "Upload Song"
3. **Connect Wallet**: Use MetaMask with Anvil account
   - Network: Localhost 8545
   - Account: Import with private key from Anvil
4. **Upload**: Select audio file (MP3/FLAC)
5. **Choose Strategy**: Select "Gift Economy"
6. **Set Splits**:
   - You: 95%
   - Protocol: 5%
7. **Deploy**: Click "Register Song"
8. **Success**: Your song is now on the platform!

### As a Listener

1. **Visit**: http://localhost:3000/discover
2. **Browse**: See all uploaded songs
3. **Play**: Click on DJ Nova's song (gift economy)
4. **Notice**: Your CGC balance increases! (You earned tokens for listening)
5. **Tip**: Click "Tip Artist" → Send 1 FLOW
6. **Verify**: Check artist's wallet → Payment received instantly!

---

## Verification Checklist

```bash
# ✓ Blockchain running
curl -X POST http://localhost:8545 -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# ✓ Contracts deployed
cast call 0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512 "flowToken()" \
  --rpc-url http://localhost:8545

# ✓ Frontend accessible
curl http://localhost:3000

# ✓ Backend services
docker-compose ps | grep Up | wc -l  # Should be 4

# ✓ Test song uploaded
cast call $ROUTER_ADDRESS "songStrategy(bytes32)" $(cast keccak "test-song-1") \
  --rpc-url http://localhost:8545
```

---

## Common Issues & Fixes

### Issue: "Port already in use"

```bash
# Find and kill process
lsof -ti:3000 | xargs kill -9  # Frontend
lsof -ti:8545 | xargs kill -9  # Anvil
lsof -ti:5432 | xargs kill -9  # PostgreSQL

# Restart services
docker-compose down && docker-compose up -d
```

### Issue: "Contract deployment failed"

```bash
# Ensure Anvil is running
ps aux | grep anvil

# Check RPC connection
curl http://localhost:8545

# Redeploy
cd contracts
forge clean
forge script script/DeployLocal.s.sol --broadcast
```

### Issue: "Frontend can't connect to contracts"

```bash
# Verify .env has correct addresses
cat .env | grep ADDRESS

# Restart frontend
cd apps/web
npm run dev
```

### Issue: "IPFS upload failing"

```bash
# Check Web3.Storage token
echo $WEB3_STORAGE_TOKEN

# Get free token at: https://web3.storage
# Add to .env and restart
```

---

## Next Steps

### Test Different Economic Models

```bash
# Create pay-per-stream song
npm run seed:pay-per-stream

# Create patronage subscription
npm run seed:patronage

# Test listener rewards
npm run test:gift-economy
```

### Deploy to Testnet

```bash
# See DEPLOYMENT_GUIDE.md for:
# - Gnosis Chiado testnet setup
# - Contract verification
# - Frontend deployment to Vercel
```

### Explore the Code

```bash
# Smart contracts
cd contracts/src
cat EconomicStrategyRouter.sol

# SDK
cd packages/sdk/src
cat economic-strategies.ts

# Frontend
cd apps/web/src/components
cat EconomicStrategyWizard.tsx
```

---

## Development Workflow

### Making Changes

```bash
# 1. Edit smart contract
vim contracts/src/strategies/MyCustomStrategy.sol

# 2. Test locally
cd contracts
forge test

# 3. Redeploy
forge script script/DeployLocal.s.sol --broadcast

# 4. Update frontend
cd apps/web
# Code uses new contract automatically (address from .env)
npm run dev
```

### Adding a New Economic Strategy

```bash
# 1. Create contract
cp contracts/src/strategies/PayPerStreamStrategy.sol \
   contracts/src/strategies/MyNewStrategy.sol

# 2. Edit implementation
vim contracts/src/strategies/MyNewStrategy.sol

# 3. Add deployment script
vim contracts/script/DeployLocal.s.sol
# Add: MyNewStrategy strategy = new MyNewStrategy(flowToken, router);

# 4. Register with router
# In DeployLocal.s.sol:
# router.registerStrategy(keccak256("my-new-strategy-v1"), address(strategy));

# 5. Add to SDK presets
vim packages/sdk/src/economic-strategies.ts
# Add to PRESET_STRATEGIES object

# 6. Test end-to-end
npm run test:new-strategy
```

---

## Monitoring & Debugging

### View Logs

```bash
# Backend services
docker-compose logs -f

# Frontend
cd apps/web
npm run dev  # Logs to terminal

# Blockchain
# Anvil logs show all transactions in terminal
```

### Query Contract State

```bash
# Get song strategy
cast call $ROUTER_ADDRESS "songStrategy(bytes32)" \
  $(cast keccak "test-song-1") \
  --rpc-url http://localhost:8545

# Get FLOW balance
cast call $FLOW_TOKEN_ADDRESS "balanceOf(address)" \
  $ARTIST_ADDRESS \
  --rpc-url http://localhost:8545

# Get payment history
cast call $ROUTER_ADDRESS "getPaymentHistory(bytes32)" \
  $(cast keccak "test-song-1") \
  --rpc-url http://localhost:8545
```

### Performance Testing

```bash
# Load test with 100 concurrent listeners
npm run test:load

# Measure streaming latency
npm run test:streaming-performance

# Check P2P success rate
npm run test:p2p-metrics
```

---

## Clean Up

```bash
# Stop all services
docker-compose down

# Stop Anvil
pkill -f anvil

# Stop frontend
# Ctrl+C in terminal

# Clean build artifacts
forge clean
rm -rf node_modules
npm cache clean --force

# Start fresh
npm install
docker-compose up -d
anvil &
```

---

## What's Running?

| Service | Port | Purpose | URL |
|---------|------|---------|-----|
| Frontend | 3000 | Next.js UI | http://localhost:3000 |
| Backend API | 3100 | REST endpoints | http://localhost:3100 |
| Anvil | 8545 | Local blockchain | http://localhost:8545 |
| PostgreSQL | 5432 | Catalog DB | postgres://localhost:5432/mycelix_music |
| Redis | 6379 | Caching | redis://localhost:6379 |
| IPFS | 5001 | File storage | http://localhost:5001 |
| Ceramic | 7007 | DKG node | http://localhost:7007 |

---

## Success! 🎉

You now have a fully functional decentralized music platform running locally!

**What you can do:**
- Upload songs with different economic models
- Stream music and earn CGC tokens as a listener
- Send tips to artists (instant payment!)
- Create playlists
- Test the governance system
- Experiment with new economic strategies

**Next steps:**
- Read [IMPLEMENTATION_EXAMPLE.md](./IMPLEMENTATION_EXAMPLE.md) for detailed flows
- Deploy to testnet: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- Add features: [ECONOMIC_MODULES_ARCHITECTURE.md](./ECONOMIC_MODULES_ARCHITECTURE.md)

---

**Questions?** Check the [README.md](./README.md) or open an issue!

**Ready for production?** See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
