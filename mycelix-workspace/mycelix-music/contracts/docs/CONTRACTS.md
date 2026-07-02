# Mycelix Smart Contract Documentation

## Overview

The Mycelix platform uses a suite of smart contracts to handle music rights management, royalty distribution, and governance. All contracts are deployed on Ethereum mainnet and compatible L2 networks.

## Contract Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Governance Layer                         │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │ VoteDelegation   │  │ProposalBatcher │  │ Optimistic  │ │
│  │                  │  │                │  │ Governance  │ │
│  └──────────────────┘  └────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Protocol                            │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │ EconomicStrategy │  │  RoyaltySplit  │  │   Rights    │ │
│  │   RouterV2       │  │    Manager     │  │   Registry  │ │
│  └──────────────────┘  └────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Token Layer                              │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │    MYC Token     │  │   Music NFTs   │  │  Streaming  │ │
│  │    (ERC-20)      │  │   (ERC-1155)   │  │   Credits   │ │
│  └──────────────────┘  └────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Contract Reference

### EconomicStrategyRouterV2

The upgradeable router contract that manages royalty distribution strategies.

**Address (Mainnet):** `0x...` (TBD)
**Address (Base):** `0x...` (TBD)

#### Key Functions

```solidity
/// @notice Register a new distribution strategy
/// @param strategyId Unique identifier for the strategy
/// @param recipients Array of recipient addresses
/// @param shares Array of share percentages (basis points)
/// @param metadata IPFS hash of strategy metadata
function registerStrategy(
    bytes32 strategyId,
    address[] calldata recipients,
    uint96[] calldata shares,
    bytes32 metadata
) external;

/// @notice Execute a royalty distribution
/// @param strategyId The strategy to use
/// @param amount Total amount to distribute
function distribute(bytes32 strategyId, uint256 amount) external;

/// @notice Get strategy details
/// @param strategyId The strategy identifier
/// @return Strategy struct with all details
function getStrategy(bytes32 strategyId) external view returns (Strategy memory);
```

#### Events

```solidity
event StrategyRegistered(bytes32 indexed strategyId, address indexed creator);
event DistributionExecuted(bytes32 indexed strategyId, uint256 amount);
event RecipientUpdated(bytes32 indexed strategyId, address indexed recipient, uint96 share);
```

#### Errors

```solidity
error InvalidRecipients();      // Recipients array empty or mismatched
error SharesExceedTotal();      // Shares sum > 10000 basis points
error StrategyNotFound();       // Strategy doesn't exist
error UnauthorizedCaller();     // Caller not strategy owner
error DistributionFailed();     // Transfer failed
```

---

### VoteDelegation

Advanced vote delegation with partial delegation support for the Mycelix DAO.

#### Key Functions

```solidity
/// @notice Create or update a delegation
/// @param delegate The address to delegate to
/// @param percentage Basis points (10000 = 100%)
/// @param expiry Delegation expiry timestamp
function createDelegation(
    address delegate,
    uint96 percentage,
    uint64 expiry
) external;

/// @notice Create delegation via EIP-712 signature (gasless)
function createDelegationBySig(DelegationSignature calldata sig) external;

/// @notice Revoke a delegation
function revokeDelegation(address delegate) external;

/// @notice Get effective voting power
/// @param voter The voter address
/// @param baseVotes The voter's base voting power
function getEffectiveVotes(
    address voter,
    uint256 baseVotes
) external view returns (uint256);
```

#### Delegation Struct

```solidity
struct Delegation {
    address delegate;    // Delegate address
    uint96 percentage;   // Basis points (10000 = 100%)
    uint64 expiry;       // Unix timestamp
    bool active;         // Is delegation active
}
```

#### Gasless Delegation (EIP-712)

```solidity
struct DelegationSignature {
    address delegator;
    address delegate;
    uint96 percentage;
    uint64 expiry;
    uint256 nonce;
    uint256 deadline;
    bytes signature;
}

// Domain separator
bytes32 DOMAIN_SEPARATOR = keccak256(
    "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
);

// Type hash
bytes32 DELEGATION_TYPEHASH = keccak256(
    "Delegation(address delegator,address delegate,uint96 percentage,uint64 expiry,uint256 nonce,uint256 deadline)"
);
```

---

### ProposalBatcher

Batch multiple governance proposals into a single meta-proposal.

#### Key Functions

```solidity
/// @notice Create a batch of proposals
function createBatch(
    address[][] memory targets,
    uint256[][] memory values,
    bytes[][] memory calldatas,
    string[] memory descriptions,
    string memory batchDescription
) external returns (uint256 batchId, uint256[] memory proposalIds);

/// @notice Execute all proposals in a batch
function executeBatch(
    uint256 batchId,
    address[][] memory targets,
    uint256[][] memory values,
    bytes[][] memory calldatas,
    bytes32[] memory descriptionHashes
) external;
```

---

### OptimisticGovernance

Allows optimistic execution of proposals with a challenge period.

#### State Machine

```
         ┌──────────┐
         │ Pending  │
         └────┬─────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌──────────┐      ┌───────────┐
│Challenged│      │ Executed  │
└────┬─────┘      └───────────┘
     │
     ▼
┌──────────┐
│Cancelled │
└──────────┘
```

#### Key Functions

```solidity
/// @notice Create an optimistic proposal
function createOptimisticProposal(
    address[] memory targets,
    uint256[] memory values,
    bytes[] memory calldatas,
    string memory description
) external returns (bytes32 proposalHash);

/// @notice Challenge a pending proposal (requires stake)
function challengeProposal(bytes32 proposalHash) external payable;

/// @notice Execute after challenge period
function executeOptimistic(
    address[] memory targets,
    uint256[] memory values,
    bytes[] memory calldatas,
    bytes32 descriptionHash
) external;
```

#### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `challengePeriod` | 2 days | Time window for challenges |
| `minChallengeStake` | 1 ETH | Minimum stake to challenge |

---

### GasOptimized Library

A library of gas-optimized utility functions.

#### Math Operations

```solidity
// Unchecked increment (saves ~20 gas per iteration)
function uncheckedInc(uint256 i) internal pure returns (uint256);

// Percentage calculation (basis points)
function percentage(uint256 amount, uint256 bps) internal pure returns (uint256);

// Integer square root (Babylonian method)
function sqrt(uint256 x) internal pure returns (uint256);
```

#### Bit Operations

```solidity
// Get bit at index
function getBit(uint256 bitmap, uint8 index) internal pure returns (bool);

// Set, clear, toggle bits
function setBit(uint256 bitmap, uint8 index) internal pure returns (uint256);
function clearBit(uint256 bitmap, uint8 index) internal pure returns (uint256);
function toggleBit(uint256 bitmap, uint8 index) internal pure returns (uint256);

// Population count
function popCount(uint256 x) internal pure returns (uint256);
```

#### Storage Packing

```solidity
// Pack two uint128 into one uint256
function pack128(uint128 a, uint128 b) internal pure returns (uint256);
function unpack128(uint256 packed) internal pure returns (uint128 a, uint128 b);

// Pack four uint64 into one uint256
function pack64(uint64 a, uint64 b, uint64 c, uint64 d) internal pure returns (uint256);
function unpack64(uint256 packed) internal pure returns (uint64, uint64, uint64, uint64);
```

---

## Gas Optimization Guide

### Best Practices Used

1. **Custom Errors** - Save ~200 gas vs require strings
   ```solidity
   error InvalidAmount();
   if (amount == 0) revert InvalidAmount();
   ```

2. **Unchecked Math** - Save ~100 gas when overflow impossible
   ```solidity
   unchecked { ++i; }  // Loop counters
   ```

3. **Storage Packing** - Pack related values into single slots
   ```solidity
   struct PackedData {
       uint64 timestamp;  // 8 bytes
       uint64 amount;     // 8 bytes
       uint64 rate;       // 8 bytes
       uint64 duration;   // 8 bytes
   }  // Total: 32 bytes = 1 slot
   ```

4. **Calldata over Memory** - Save ~60 gas per array element
   ```solidity
   function process(uint256[] calldata data) external;
   ```

5. **Short-Circuit Evaluation**
   ```solidity
   if (condition1 && expensiveCheck()) { ... }
   ```

### Gas Estimates

| Operation | Gas Cost |
|-----------|----------|
| Register Strategy | ~85,000 |
| Execute Distribution | ~45,000 + 21,000 per recipient |
| Create Delegation | ~50,000 |
| Create Optimistic Proposal | ~65,000 |
| Challenge Proposal | ~35,000 |

---

## Security Considerations

### Access Control

- All administrative functions use OpenZeppelin's `Ownable2Step`
- Time-locked upgrades via UUPS proxy pattern
- Multi-sig required for critical operations

### Reentrancy Protection

- All external calls protected with `nonReentrant` modifier
- Checks-Effects-Interactions pattern followed
- Pull payment pattern for withdrawals

### Upgrade Safety

- UUPS proxy pattern with `_authorizeUpgrade` hook
- Storage gaps for future-proofing
- Initializer functions properly guarded

### Audit Status

| Contract | Auditor | Date | Status |
|----------|---------|------|--------|
| EconomicStrategyRouterV2 | TBD | TBD | Pending |
| VoteDelegation | TBD | TBD | Pending |
| GasOptimized | Internal | 2024 | Reviewed |

---

## Deployment

### Prerequisites

```bash
# Install dependencies
forge install

# Set environment variables
export PRIVATE_KEY=<deployer-key>
export RPC_URL=<network-rpc>
export ETHERSCAN_API_KEY=<api-key>
```

### Deploy Commands

```bash
# Deploy to testnet
forge script script/Deploy.s.sol:DeployScript --rpc-url $RPC_URL --broadcast --verify

# Deploy specific contract
forge create src/governance/VoteDelegation.sol:VoteDelegation --rpc-url $RPC_URL --private-key $PRIVATE_KEY

# Verify contract
forge verify-contract <address> src/governance/VoteDelegation.sol:VoteDelegation --chain-id 1
```

### Upgrade Process

```bash
# 1. Deploy new implementation
forge create src/upgrades/EconomicStrategyRouterV2.sol:EconomicStrategyRouterV2 --rpc-url $RPC_URL

# 2. Propose upgrade via governance
# 3. Execute after timelock
```

---

## Testing

### Run Tests

```bash
# All tests
forge test

# Specific contract
forge test --match-contract GasOptimizedTest

# With verbosity
forge test -vvv

# Gas report
forge test --gas-report
```

### Coverage

```bash
forge coverage --report lcov
genhtml lcov.info -o coverage
```

---

## Integration Examples

### JavaScript/TypeScript

```typescript
import { ethers } from 'ethers';
import { EconomicStrategyRouterV2__factory } from './typechain';

const router = EconomicStrategyRouterV2__factory.connect(
  ROUTER_ADDRESS,
  signer
);

// Register a strategy
await router.registerStrategy(
  ethers.utils.id('my-strategy'),
  [artist, label, platform],
  [7000, 2000, 1000], // 70%, 20%, 10%
  ethers.utils.id('ipfs://...')
);

// Execute distribution
await router.distribute(
  ethers.utils.id('my-strategy'),
  ethers.utils.parseEther('100')
);
```

### Subgraph Query

```graphql
query GetDistributions($strategyId: String!) {
  distributions(
    where: { strategyId: $strategyId }
    orderBy: timestamp
    orderDirection: desc
  ) {
    id
    amount
    recipients {
      address
      share
    }
    timestamp
    transactionHash
  }
}
```

---

## Contact

- **Technical Issues:** contracts@mycelix.music
- **Security Reports:** security@mycelix.music
- **Discord:** [Mycelix Discord](https://discord.gg/mycelix)
