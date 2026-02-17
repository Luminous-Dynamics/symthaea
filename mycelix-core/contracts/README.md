# Mycelix Smart Contracts

Ethereum smart contracts for the Mycelix network, providing on-chain anchoring for decentralized identity, reputation, and payments.

## Live Deployment (Sepolia)

| Contract | Address | Verified |
|----------|---------|----------|
| **MycelixRegistry** | [`0x556b810371e3d8D9E5753117514F03cC6C93b835`](https://sepolia.etherscan.io/address/0x556b810371e3d8D9E5753117514F03cC6C93b835) | [Sourcify](https://sourcify.dev/#/lookup/0x556b810371e3d8D9E5753117514F03cC6C93b835) |
| **ReputationAnchor** | [`0xf3B343888a9b82274cEfaa15921252DB6c5f48C9`](https://sepolia.etherscan.io/address/0xf3B343888a9b82274cEfaa15921252DB6c5f48C9) | [Sourcify](https://sourcify.dev/#/lookup/0xf3B343888a9b82274cEfaa15921252DB6c5f48C9) |
| **PaymentRouter** | [`0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB`](https://sepolia.etherscan.io/address/0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB) | [Sourcify](https://sourcify.dev/#/lookup/0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB) |

## Contracts

### MycelixRegistry.sol
Central registry for Decentralized Identifiers (DIDs) and metadata.

- Register DIDs with associated metadata hashes
- Update metadata for existing DIDs
- Revoke DIDs when needed
- Query DID ownership and status

### ReputationAnchor.sol
On-chain reputation scoring system with Merkle proof support.

- Submit individual reputation scores
- Anchor Merkle roots for batch reputation updates
- Query reputation by address and context
- Byzantine-resistant through PoGQ integration

### PaymentRouter.sol
Escrow payment system for FL contributions.

- Create time-locked escrow payments
- Release payments upon condition fulfillment
- Refund mechanism for failed conditions
- Split payment distribution support

## Quick Start

### Prerequisites

- [Foundry](https://getfoundry.sh/) installed
- Node.js 18+ (for SDK)

### Build & Test

```bash
# Install Foundry dependencies
forge install

# Build contracts
forge build

# Run tests
forge test

# Run tests with verbosity
forge test -vvv
```

### Deploy to Sepolia

```bash
# Set environment variables
export PRIVATE_KEY=0x...
export SEPOLIA_RPC_URL=https://sepolia.drpc.org

# Deploy
forge script script/Deploy.s.sol:DeploySepolia \
  --rpc-url $SEPOLIA_RPC_URL \
  --broadcast \
  --verify

# Or use the deployment script
./deploy-sepolia.sh
```

### Deploy Locally (Anvil)

```bash
# Terminal 1: Start Anvil
anvil

# Terminal 2: Deploy
forge script script/Deploy.s.sol:DeployLocal \
  --rpc-url http://127.0.0.1:8545 \
  --broadcast
```

## SDK

TypeScript SDK for contract interactions:

```bash
cd sdk
npm install
npm run cli -- info
```

See [sdk/README.md](./sdk/README.md) for full documentation.

## dApp

React-based web interface:

```bash
cd dapp
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## Project Structure

```
contracts/
├── src/                    # Solidity contracts
│   ├── MycelixRegistry.sol
│   ├── ReputationAnchor.sol
│   └── PaymentRouter.sol
├── script/                 # Deployment scripts
│   └── Deploy.s.sol
├── test/                   # Contract tests
├── sdk/                    # TypeScript SDK
│   ├── src/
│   │   ├── index.ts       # Main SDK
│   │   ├── cli.ts         # CLI tool
│   │   └── demo.ts        # Demo script
│   └── config.ts          # Network configs
├── dapp/                   # React dApp
├── abi/                    # Generated ABIs
├── deployments/            # Deployment records
└── foundry.toml            # Foundry config
```

## Configuration

### foundry.toml

```toml
[profile.default]
src = "src"
out = "out"
libs = ["lib"]
solc = "0.8.24"
optimizer = true
optimizer_runs = 200
via_ir = true
```

### Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Required variables:
- `PRIVATE_KEY` - Deployer wallet private key
- `SEPOLIA_RPC_URL` - Sepolia RPC endpoint
- `ETHERSCAN_API_KEY` - For contract verification (optional)

## Documentation

- [ETHEREUM_BRIDGE_GUIDE.md](./ETHEREUM_BRIDGE_GUIDE.md) - Comprehensive integration guide
- [DEPLOYMENT_READY.md](./DEPLOYMENT_READY.md) - Deployment checklist
- [sdk/README.md](./sdk/README.md) - SDK documentation

## License

MIT
