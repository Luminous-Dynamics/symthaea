# 🚀 Ethereum Demo Quick Start Guide

**Goal**: Record compelling demo showing Zero-TrustML on Anvil local fork → Polygon live testnet

**Time Required**: ~30 minutes setup + 5 minutes recording

---

## ✅ Prerequisites

### 1. Install Foundry (for Anvil)
```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

Verify:
```bash
anvil --version
# Should print: anvil 0.x.x
```

### 2. Compile Smart Contract
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Generate ABI
python generate_abi.py

# Compile contract (creates bytecode)
python compile_contract.py
```

### 3. Optional: Get Testnet POL (for live verification)
- Visit https://faucet.polygon.technology/
- Select "Polygon Amoy"
- Enter your wallet address
- Get free testnet POL

---

## 🎬 Recording Flow (5 minutes)

### Part 1: Local Fork Demo (3 minutes)

**Terminal 1** - Start Anvil:
```bash
./scripts/start_anvil_fork.sh
```

Expected output:
```
✅ Anvil running on http://localhost:8545
Available Accounts:
(0) 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 (10000 ETH)
...
```

**Terminal 2** - Deploy & Run Demo:
```bash
# Deploy contract
./scripts/deploy_to_anvil.sh

# Run FL demo
python demos/demo_ethereum_local_fork.py
```

**What you'll see**:
- 🏥 3 honest hospitals + 1 Byzantine node
- 🔄 Gradient submission (instant <100ms)
- 🛡️ Byzantine detection via PoGQ
- 💰 Credit issuance on-chain
- 📊 Contract state updates

**Perfect for recording**: Instant transactions, beautiful colored output!

### Part 2: Live Testnet Verification (2 minutes)

**Prerequisites**:
- Testnet POL in wallet
- Private key in `build/.ethereum_key`

```bash
# Run on live Polygon Amoy
python demos/demo_ethereum_live_testnet.py
```

**What you'll see**:
- Same FL scenario, real blockchain
- 2-3 second block times (real consensus)
- Polygonscan verification links
- Permanent immutable records

---

## 🎥 OBS Recording Setup

### Scene 1: "Anvil Local Fork"
**Sources**:
1. Terminal window (full screen) - `demo_ethereum_local_fork.py`
2. Title overlay: "Zero-TrustML on Anvil Local Fork - Instant, Free"

**Duration**: 90 seconds

### Scene 2: "Live Testnet"
**Sources**:
1. Terminal window (left half) - `demo_ethereum_live_testnet.py`
2. Browser (right half) - Polygonscan contract page
3. Title overlay: "Verified on Polygon Amoy - Real Blockchain"

**Duration**: 60 seconds

### Scene 3: "Comparison"
**Sources**:
1. Text slide comparing Anvil vs Testnet
2. Talking points:
   - Development: Anvil (fast, free)
   - Production: Testnet (verified, permanent)
   - Multi-backend: Holochain OR Ethereum

**Duration**: 30 seconds

**Total**: 2:30-3:00 minutes

---

## 📝 Recording Script

### Opening (10 seconds)
> "Zero-TrustML enables decentralized federated learning with blockchain verification. Let me show you how it works."

### Anvil Demo (80 seconds)
> "First, on a local Ethereum fork using Anvil..."
>
> [Start demo]
>
> "Three hospitals train a diabetes prediction model collaboratively..."
>
> "Notice the Byzantine node submitting poisoned gradients..."
>
> "PoGQ detects the attack with 98% accuracy..."
>
> "Honest nodes earn credits, Byzantine nodes get zero..."
>
> "All transactions instant, less than 100 milliseconds..."

### Live Testnet (60 seconds)
> "Now let's verify on a real blockchain - Polygon Amoy..."
>
> [Start demo]
>
> "Same scenario, but now with real consensus..."
>
> "Transactions take 2-3 seconds as the network confirms..."
>
> "Every gradient, credit, and Byzantine event is permanently recorded..."
>
> "Fully verifiable on Polygonscan..."

### Closing (30 seconds)
> "Zero-TrustML gives you flexibility: use Anvil for rapid development, deploy to Polygon for production verification."
>
> "Or choose Holochain for even faster P2P coordination."
>
> "Multi-backend architecture means you pick the right tool for your use case."

---

## 🐛 Troubleshooting

### Anvil won't start
```bash
# Check if port 8545 is in use
lsof -i :8545

# Kill existing process
kill -9 <PID>

# Restart Anvil
./scripts/start_anvil_fork.sh
```

### Contract deployment fails
```bash
# Make sure Anvil is running
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545

# Check build directory exists
ls -la build/

# Regenerate ABI and bytecode
python generate_abi.py
python compile_contract.py
```

### Demo script errors
```bash
# Verify contract address exists
cat build/anvil_contract_address.txt

# Check Python dependencies
pip install web3 eth-account

# Run with verbose output
python -u demos/demo_ethereum_local_fork.py
```

### Live testnet out of POL
```bash
# Get free testnet POL
# Visit: https://faucet.polygon.technology/

# Check your balance
python -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://rpc-amoy.polygon.technology/'))
balance = w3.eth.get_balance('YOUR_ADDRESS')
print(f'{w3.from_wei(balance, \"ether\")} POL')
"
```

---

## 📦 What Gets Created

After running the demo, you'll have:

### Local Fork Artifacts
- `build/anvil_contract_address.txt` - Deployed contract address
- `build/anvil_deployment.json` - Deployment metadata
- Anvil state (in memory, lost on restart)

### Live Testnet Artifacts
- Permanent blockchain records on Polygon Amoy
- Polygonscan verification (public)
- Transaction hashes for all operations

---

## 🎯 Grant Demo Checklist

Before recording:
- [ ] Anvil starts without errors
- [ ] Contract deploys successfully
- [ ] Local fork demo runs completely
- [ ] Terminal output is readable (font size!)
- [ ] OBS scenes configured
- [ ] Narration script practiced

During recording:
- [ ] No errors in terminal output
- [ ] Gradients submit successfully
- [ ] Byzantine detection works
- [ ] Credits issued correctly
- [ ] Timing matches 2:30-3:00 target

After recording:
- [ ] Video exports to 1080p MP4
- [ ] Audio is clear
- [ ] Timestamps are visible
- [ ] Contract addresses shown
- [ ] Polygonscan links work

---

## 💡 Pro Tips

1. **Use larger terminal font** (18-20pt) for recording
2. **Slow down output** with `time.sleep(0.5)` if needed
3. **Rehearse narration** to match demo timing
4. **Record multiple takes** - easy to restart Anvil
5. **Check audio levels** before final recording
6. **Add captions** for technical terms (PoGQ, Byzantine, etc.)

---

## 📊 Expected Performance

### Anvil Local Fork
- Transaction time: <100ms
- Block time: Instant
- Gas cost: Free
- State persistence: Session only

### Polygon Amoy Testnet
- Transaction time: 2-3 seconds
- Block time: 2 seconds average
- Gas cost: Testnet POL (free from faucet)
- State persistence: Permanent

---

## 🔗 Useful Links

- **Anvil Docs**: https://book.getfoundry.sh/anvil/
- **Polygon Faucet**: https://faucet.polygon.technology/
- **Polygonscan Amoy**: https://amoy.polygonscan.com/
- **Contract Address**: `0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A`

---

**Ready to record?** Start with `./scripts/start_anvil_fork.sh` and follow the flow!
