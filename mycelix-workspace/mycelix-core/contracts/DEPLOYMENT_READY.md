# Mycelix Smart Contracts - Deployment Ready

## Status: READY FOR DEPLOYMENT

All contracts have been compiled and tested successfully.

### Test Results
- **Total Tests:** 68
- **Passed:** 68
- **Failed:** 0

### Contracts
1. **ReputationAnchor.sol** - Merkle tree-based reputation anchoring
2. **PaymentRouter.sol** - Multi-party payment routing with escrow
3. **MycelixRegistry.sol** - DID registry and hApp management

### Estimated Deployment Costs (Polygon Amoy)
- **Gas Required:** ~6,829,477 gas units
- **Estimated POL:** ~0.21 POL (at ~31 gwei gas price)

---

## Deployment Instructions

### Prerequisites

1. **Get Testnet POL:**
   - Visit the [Polygon Amoy Faucet](https://faucet.polygon.technology/)
   - Request test POL for your deployer address
   - You need at least 0.25 POL for deployment + some buffer

2. **Configure Environment:**
   ```bash
   cd /home/tstoltz/Luminous-Dynamics/Mycelix-Core/contracts
   cp .env.example .env
   ```
   
   Edit `.env` with your values:
   ```
   PRIVATE_KEY=0x<your_private_key>
   OWNER_ADDRESS=<your_address>
   ANCHOR_ADDRESS=<anchor_node_address>
   FEE_RECIPIENT=<fee_recipient_address>
   POLYGON_AMOY_RPC_URL=https://rpc-amoy.polygon.technology
   ```

### Deploy to Polygon Amoy

```bash
# Enter the development shell
cd /home/tstoltz/Luminous-Dynamics/Mycelix-Core/contracts
nix-shell

# Source environment variables
source .env

# Simulate deployment first (recommended)
forge script script/Deploy.s.sol:DeployAmoy \
  --rpc-url $POLYGON_AMOY_RPC_URL \
  -vvv

# Deploy for real
forge script script/Deploy.s.sol:DeployAmoy \
  --rpc-url $POLYGON_AMOY_RPC_URL \
  --broadcast \
  -vvv
```

### Deploy to Polygon Mainnet (Production)

```bash
# Use the generic deploy script with full configuration
forge script script/Deploy.s.sol:DeployMycelix \
  --rpc-url $POLYGON_RPC_URL \
  --broadcast \
  --verify \
  -vvv
```

---

## Contract Addresses

After deployment, the script will output contract addresses like:
```json
{
  "network": "polygon-amoy",
  "chainId": 80002,
  "contracts": {
    "reputationAnchor": "0x...",
    "paymentRouter": "0x...",
    "mycelixRegistry": "0x..."
  }
}
```

Update `ETHEREUM_BRIDGE_GUIDE.md` with these addresses after deployment.

---

## Verification

After deployment, verify contracts on Polygonscan:
```bash
forge verify-contract <CONTRACT_ADDRESS> src/ReputationAnchor.sol:ReputationAnchor \
  --chain-id 80002 \
  --etherscan-api-key $POLYGONSCAN_API_KEY
```

---

## Security Notes

1. **NEVER** commit real private keys to git
2. Use hardware wallets for mainnet deployment
3. The test private key in `.env` is Foundry's default - DO NOT use for real funds
4. Consider using a multi-sig for production owner address

---

## Files Modified During Setup

- `src/MycelixRegistry.sol` - Fixed duplicate identifier (DIDRevoked event/error)
- `test/MycelixRegistry.t.sol` - Updated error expectations
- `test/PaymentRouter.t.sol` - Updated error expectations
- `test/ReputationAnchor.t.sol` - Fixed test timing and error expectations
- `shell.nix` - Created minimal nix shell for Foundry

---

## Build Commands Reference

```bash
# Compile contracts
forge build

# Run tests
forge test

# Run tests with verbosity
forge test -vvv

# Gas report
forge test --gas-report

# Generate ABI
forge inspect ReputationAnchor abi > abi/ReputationAnchor.json
forge inspect PaymentRouter abi > abi/PaymentRouter.json
forge inspect MycelixRegistry abi > abi/MycelixRegistry.json
```

---

**Last Updated:** 2026-01-08
**Compiler Version:** Solc 0.8.24
**Foundry Version:** 1.4.4
