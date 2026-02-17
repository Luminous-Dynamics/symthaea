# Polygon Amoy Testnet Deployment

## Status: Ready to Deploy (Awaiting POL)

### Pre-requisites Verified
- [x] Contracts compile successfully
- [x] Deployment script simulated
- [x] Estimated gas: ~6.9M gas (~0.226 POL)
- [ ] Wallet funded with POL

### Wallet Details
- **Address**: `0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c`
- **Required**: ~0.3 POL (with buffer)
- **Current Balance**: 0 POL

### Getting Testnet POL

#### Option 1: Official Polygon Faucet
1. Visit https://faucet.polygon.technology/
2. Connect wallet or paste address
3. Select "Amoy" network
4. Request POL

#### Option 2: Alchemy Faucet
1. Visit https://www.alchemy.com/faucets/polygon-amoy
2. Create Alchemy account (free)
3. Request POL to address

#### Option 3: QuickNode Faucet
1. Visit https://faucet.quicknode.com/polygon/amoy
2. Request POL

### Deployment Steps

Once wallet is funded:

```bash
cd contracts

# Verify balance
nix-shell shell.nix --run "cast balance 0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c --rpc-url https://rpc-amoy.polygon.technology --ether"

# Deploy
./deploy-amoy.sh

# Or manually:
nix-shell shell.nix --run "source .env && forge script script/Deploy.s.sol:DeployAmoy --rpc-url https://rpc-amoy.polygon.technology --broadcast -vvv"
```

### Expected Addresses (deterministic)
Based on simulation:
- ReputationAnchor: `0xf3B343888a9b82274cEfaa15921252DB6c5f48C9`
- PaymentRouter: `0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB`
- MycelixRegistry: `0x556b810371e3d8D9E5753117514F03cC6C93b835`

### Post-Deployment
1. Save deployment info to `deployments/amoy-deployment.json`
2. Verify contracts on Polygonscan
3. Update SDK configuration

### Existing Deployment
Contracts are already deployed on **Sepolia** and verified:
- See `sepolia-deployment.json` for details
- Explorer: https://sepolia.etherscan.io/address/0xf3B343888a9b82274cEfaa15921252DB6c5f48C9
