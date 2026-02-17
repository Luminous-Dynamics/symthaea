# Mycelix Polygon Amoy Testnet Deployment

Quick deployment guide - everything is pre-configured and ready.

## Deployment Wallet

```
Address: 0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c
```

This is a fresh testnet wallet generated specifically for this deployment.

## Step 1: Fund the Wallet

Get testnet POL from the Polygon faucet:

1. Go to: https://faucet.polygon.technology/
2. Select **Amoy** testnet
3. Paste the wallet address: `0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c`
4. Complete the captcha and request tokens
5. Wait for the transaction to confirm (usually < 1 minute)

**Required amount:** ~0.35 POL (the faucet typically provides 0.5 POL)

## Step 2: Deploy

Run one command:

```bash
cd /home/tstoltz/Luminous-Dynamics/Mycelix-Core/contracts
nix-shell -p foundry --run "./deploy-amoy.sh"
```

That's it! The script will:
- Verify wallet balance is sufficient
- Build the contracts
- Deploy to Polygon Amoy (chain ID: 80002)
- Output contract addresses and SDK configuration

## What Gets Deployed

| Contract | Description |
|----------|-------------|
| ReputationAnchor | Node reputation tracking and anchoring |
| PaymentRouter | Fee distribution and payment routing |
| MycelixRegistry | Service registration and discovery |

## Deployment Details

- **Network:** Polygon Amoy Testnet
- **Chain ID:** 80002
- **RPC URL:** https://rpc-amoy.polygon.technology
- **Block Explorer:** https://amoy.polygonscan.com
- **Estimated Gas:** ~6.8M gas units
- **Estimated Cost:** ~0.30 POL

## Verification (Optional)

To verify contracts on Polygonscan:

1. Get a free API key from https://polygonscan.com/apis
2. Add to `.env`:
   ```
   POLYGONSCAN_API_KEY=your_api_key_here
   ```
3. Deploy with verification:
   ```bash
   nix-shell -p foundry --run "./deploy-amoy.sh --verify"
   ```

## After Deployment

The script outputs JSON configuration for the SDK:

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

Copy these addresses to your client configuration.

## Troubleshooting

### "INSUFFICIENT BALANCE"
Request more POL from the faucet. You need at least 0.35 POL.

### "RPC Error"
The default RPC may be rate-limited. Alternative RPCs:
- https://polygon-amoy.g.alchemy.com/v2/YOUR_KEY (Alchemy)
- https://polygon-amoy.infura.io/v3/YOUR_KEY (Infura)

### "Nonce too low"
Previous transaction may be pending. Wait a minute and retry.

## Dry Run (Simulation Only)

To simulate without broadcasting:

```bash
cd /home/tstoltz/Luminous-Dynamics/Mycelix-Core/contracts
nix-shell -p foundry --run "source .env && forge script script/Deploy.s.sol:DeployAmoy --rpc-url \$POLYGON_AMOY_RPC_URL -vvv"
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Check balance | `nix-shell -p foundry --run "cast balance 0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c --rpc-url https://rpc-amoy.polygon.technology"` |
| Deploy | `nix-shell -p foundry --run "./deploy-amoy.sh"` |
| Dry run | See above |
| View on explorer | https://amoy.polygonscan.com/address/0x8aDfb26E4596cd6ba8dC31943cEEF4AC77261C1c |
