# Holochain Currency Exchange Architecture

**Version**: 1.0
**Date**: 2025-09-30
**Status**: Design Document - Reference Implementation

---

## Executive Summary

This document outlines a novel architecture for **inter-currency exchange between Holochain-based community currencies** using blockchain as a settlement layer. This pattern enables:

- Zero-cost, high-frequency transactions within communities (Holochain)
- Efficient price discovery and atomic swaps between communities (Blockchain)
- Seamless integration with external DeFi (USDC, ETH, etc.)

**Key Innovation**: Each technology layer does what it does best—Holochain for agent-centric economies, blockchain for global settlement.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture Overview](#architecture-overview)
3. [Technology Selection Rationale](#technology-selection-rationale)
4. [System Components](#system-components)
5. [Implementation Phases](#implementation-phases)
6. [Security Considerations](#security-considerations)
7. [Economic Model](#economic-model)
8. [Reference Implementation](#reference-implementation)
9. [Future Extensions](#future-extensions)

---

## Problem Statement

### Current Limitations

**Community Currencies Today**:
- **Blockchain-only**: High gas costs make micro-transactions impractical
- **Isolated**: Each currency exists in a silo with no interoperability
- **Centralized Exchanges**: Custody risk, KYC friction, censorship

**Holochain-only**:
- **No Cross-DHT Communication**: Different currencies can't directly interact
- **No Price Discovery**: No efficient mechanism for exchange rates
- **Limited Liquidity**: Hard to exchange for external assets (USDC, ETH)

### Our Solution

A **hybrid architecture** that combines:
- **Holochain**: For internal community transactions (zero cost, high frequency)
- **Blockchain**: For inter-community exchange (atomic swaps, price discovery)
- **Bridge Validators**: Trusted layer connecting both worlds

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DEFI LAYER                          │
│  Uniswap, Aave, Compound (USDC, ETH, DAI liquidity)             │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Bridges
┌──────────────────────┴──────────────────────────────────────────┐
│              BLOCKCHAIN EXCHANGE LAYER (Layer 2)                │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐         │
│  │  Orderbook  │  │ Liquidity    │  │  Atomic Swap    │         │
│  │    DEX      │  │    Pools     │  │    Contracts    │         │
│  └─────────────┘  └──────────────┘  └─────────────────┘         │
│                                                                 │
│  Technology: Polygon (low gas), Ethereum (security)             │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Bridge Validators (Multisig)
┌──────────────────────┴──────────────────────────────────────────┐
│           HOLOCHAIN CURRENCY LAYER (Layer 1)                    │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Zero-TrustML│  │  Compute     │  │   Storage    │           │
│  │  Credits     │  │  Credits     │  │   Credits    │           │
│  │  DHT         │  │  DHT         │  │   DHT        │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  TimeBank    │  │   Impact     │  │   Service    │           │
│  │   Hours      │  │   Tokens     │  │   Credits    │           │
│  │   DHT        │  │   DHT        │  │   DHT        │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                 │
│  Technology: Holochain (agent-centric, zero gas)                │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Example: Alice Swaps Zero-TrustML Credits → TimeBank Hours

```
1. Alice (Holochain)
   ↓ Posts swap intent: 100 Zero-TrustML Credits → 10 TimeBank Hours

2. Zero-TrustML Credits DHT
   ↓ Creates escrow entry, locks 100 credits

3. Bridge Validator Network
   ↓ Detects escrow, posts order to blockchain

4. Polygon DEX Smart Contract
   ↓ Orderbook matching: Bob accepts (opposite trade)

5. Bridge Validator Network
   ↓ Verifies match, signs atomic swap

6. Polygon DEX Smart Contract
   ↓ Executes swap, emits completion event

7. Bridge Validator Network
   ↓ Detects event, initiates DHT transfers

8. Zero-TrustML Credits DHT → Alice loses 100 credits
   TimeBank Hours DHT → Alice gains 10 hours

9. Alice (Holochain)
   ✓ Swap complete, can now use hours for services
```

---

## Technology Selection Rationale

### Why Holochain for Community Currencies?

| Feature | Holochain | Blockchain | Winner |
|---------|-----------|------------|--------|
| **Transaction Cost** | $0.00 | $0.01-$1.00 | ✅ Holochain |
| **Transaction Speed** | ~100ms | 12-15s | ✅ Holochain |
| **Scalability** | Linear (per agent) | Log(n) sharding | ✅ Holochain |
| **Privacy** | Private by default | Public | ✅ Holochain |
| **Customization** | Full (per DNA) | Limited (EVM) | ✅ Holochain |
| **Mutual Credit** | Native | Complex | ✅ Holochain |

**Use Case Fit**: Internal community transactions (high frequency, low cost)

### Why Blockchain for Exchange?

| Feature | Holochain | Blockchain | Winner |
|---------|-----------|------------|--------|
| **Cross-DHT Comms** | No native support | Native | ✅ Blockchain |
| **Price Discovery** | No orderbook | DEX orderbooks | ✅ Blockchain |
| **Global Liquidity** | Isolated | Composable DeFi | ✅ Blockchain |
| **Legal Recognition** | Emerging | Established | ✅ Blockchain |
| **External Bridges** | Manual | Native (Chainlink) | ✅ Blockchain |
| **Atomic Swaps** | Cross-DHT hard | Smart contracts | ✅ Blockchain |

**Use Case Fit**: Inter-community exchange (lower frequency, settlement layer)

### Why Polygon for Exchange Layer?

- **Low Gas Costs**: ~$0.01 per swap (vs $5-50 on Ethereum L1)
- **Fast Finality**: ~2 seconds (vs 12s on Ethereum)
- **EVM Compatible**: Can port contracts easily
- **Large Ecosystem**: Deep liquidity pools already exist
- **Ethereum Security**: Still inherits Ethereum's security model

**Alternative Considered**: Arbitrum, Optimism (also good choices)

---

## System Components

### Component 1: Holochain Currency DNAs

Each community currency is a separate Holochain DNA with customizable rules.

#### Zero-TrustML Credits DNA

```rust
// zomes/zerotrustml_credits/src/lib.rs

use hdk::prelude::*;

/// Zero-TrustML Credit entry
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Credit {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub earned_from: EarnReason,
    pub timestamp: Timestamp,
    pub verifiers: Vec<AgentPubKey>, // Peers who validated
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum EarnReason {
    QualityGradient { pogq_score: f64 },
    ByzantineDetection { caught_node: u32 },
    PeerValidation { validated_node: u32 },
    NetworkContribution { uptime_hours: u64 },
}

/// Credit balance query
#[hdk_extern]
pub fn get_balance(holder: AgentPubKey) -> ExternResult<u64> {
    let credits = query_credits_for_holder(holder.clone())?;
    Ok(credits.iter().map(|c| c.amount).sum())
}

/// Transfer credits
#[hdk_extern]
pub fn transfer(input: TransferInput) -> ExternResult<ActionHash> {
    // Validate sender has balance
    let sender_balance = get_balance(input.from.clone())?;
    if sender_balance < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient balance".into()
        )));
    }

    // Create debit entry
    let debit = Credit {
        holder: input.from.clone(),
        amount: input.amount,
        earned_from: EarnReason::Transfer {
            to: input.to.clone(),
        },
        timestamp: sys_time()?,
        verifiers: vec![],
    };

    // Create credit entry
    let credit = Credit {
        holder: input.to.clone(),
        amount: input.amount,
        earned_from: EarnReason::Transfer {
            from: input.from.clone(),
        },
        timestamp: sys_time()?,
        verifiers: vec![],
    };

    // Commit both (atomic)
    create_entry(&EntryTypes::Credit(debit))?;
    let credit_hash = create_entry(&EntryTypes::Credit(credit))?;

    Ok(credit_hash)
}

/// Bridge escrow (for exchange)
#[hdk_entry_helper]
pub struct BridgeEscrow {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub destination_chain: String, // "polygon"
    pub swap_intent: SwapIntent,
    pub lock_time: Timestamp,
    pub status: EscrowStatus,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SwapIntent {
    pub from_currency: String, // "zerotrustml_credits"
    pub to_currency: String,   // "timebank_hours"
    pub rate: f64,
    pub min_rate: f64,         // Slippage protection
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum EscrowStatus {
    Locked,
    Executed,
    Cancelled,
    Expired,
}

/// Create bridge escrow
#[hdk_extern]
pub fn create_bridge_escrow(input: BridgeEscrowInput) -> ExternResult<ActionHash> {
    // Validate balance
    let balance = get_balance(input.holder.clone())?;
    if balance < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient balance for escrow".into()
        )));
    }

    // Create escrow entry
    let escrow = BridgeEscrow {
        holder: input.holder,
        amount: input.amount,
        destination_chain: input.destination_chain,
        swap_intent: input.swap_intent,
        lock_time: sys_time()?,
        status: EscrowStatus::Locked,
    };

    let escrow_hash = create_entry(&EntryTypes::BridgeEscrow(escrow))?;

    // Emit signal for bridge validators
    emit_signal(Signal::EscrowCreated {
        escrow_hash: escrow_hash.clone(),
    })?;

    Ok(escrow_hash)
}

/// Validation rules
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(StoreEntry { entry, .. }) => {
            match entry {
                Entry::App(bytes) => {
                    match EntryTypes::deserialize_from_type(
                        entry_type,
                        bytes,
                    )? {
                        EntryTypes::Credit(credit) => {
                            // Validate credit issuance
                            match credit.earned_from {
                                EarnReason::QualityGradient { pogq_score } => {
                                    // Max 100 credits per gradient
                                    if credit.amount > 100 {
                                        return Ok(ValidateCallbackResult::Invalid(
                                            "Exceeds max reward per gradient".into()
                                        ));
                                    }
                                    // Must have good PoGQ score
                                    if pogq_score < 0.5 {
                                        return Ok(ValidateCallbackResult::Invalid(
                                            "PoGQ score too low for reward".into()
                                        ));
                                    }
                                    // Must have verifiers
                                    if credit.verifiers.len() < 3 {
                                        return Ok(ValidateCallbackResult::Invalid(
                                            "Insufficient verifiers".into()
                                        ));
                                    }
                                }
                                _ => {} // Other reasons have different rules
                            }
                            Ok(ValidateCallbackResult::Valid)
                        }
                        _ => Ok(ValidateCallbackResult::Valid),
                    }
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
```

**DNA Configuration** (`dna.yaml`):
```yaml
---
manifest_version: "1"
name: zerotrustml_credits
uid: "00000000-0000-0000-0000-000000000001"
properties:
  max_credit_per_gradient: 100
  min_verifiers: 3
  escrow_timeout_minutes: 60

zomes:
  - name: zerotrustml_credits
    bundled: target/wasm32-unknown-unknown/release/zerotrustml_credits.wasm
```

#### Compute Credits DNA (Similar Structure)

```rust
// For paying for GPU compute time
pub enum EarnReason {
    ProvidedGPU { hours: f64, gpu_type: String },
    ProvidedCPU { hours: f64, cores: u32 },
    ProvidedBandwidth { gb: f64 },
}
```

#### Storage Credits DNA (Similar Structure)

```rust
// For paying for model/gradient storage
pub enum EarnReason {
    ProvidedStorage { gb: f64, days: u32 },
    HostedModel { size_gb: f64, downloads: u32 },
}
```

---

### Component 2: Blockchain Exchange Layer

Smart contracts on Polygon for inter-currency swaps.

#### Exchange Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title HolochainCurrencyExchange
 * @dev Decentralized exchange for Holochain currencies
 */
contract HolochainCurrencyExchange {

    // Structs
    struct Order {
        address maker;              // Ethereum address of maker
        bytes holochainPubKey;      // Holochain public key
        string fromCurrency;        // "zerotrustml_credits"
        string toCurrency;          // "timebank_hours"
        uint256 amount;             // Amount to swap
        uint256 rate;               // Exchange rate (scaled 1e18)
        uint256 minRate;            // Min acceptable rate (slippage)
        uint256 expiry;             // Unix timestamp expiry
        bool active;                // Order status
    }

    struct Swap {
        uint256 order1Id;           // First order ID
        uint256 order2Id;           // Second order ID
        uint256 amount1;            // Amount from order 1
        uint256 amount2;            // Amount from order 2
        uint256 executionRate;      // Final rate
        uint256 timestamp;          // When executed
        bytes[] bridgeSignatures;   // Validator signatures
    }

    // State
    uint256 public nextOrderId = 1;
    uint256 public nextSwapId = 1;

    mapping(uint256 => Order) public orders;
    mapping(bytes32 => uint256[]) public orderbook; // Currency pair => order IDs
    mapping(uint256 => Swap) public swaps;

    // Bridge validator whitelist (multisig)
    mapping(address => bool) public bridgeValidators;
    uint256 public requiredValidatorSignatures = 3;

    // Events
    event OrderPosted(
        uint256 indexed orderId,
        address indexed maker,
        string fromCurrency,
        string toCurrency,
        uint256 amount,
        uint256 rate
    );

    event OrderCancelled(uint256 indexed orderId);

    event SwapExecuted(
        uint256 indexed swapId,
        uint256 order1Id,
        uint256 order2Id,
        uint256 amount1,
        uint256 amount2,
        uint256 rate
    );

    // Modifiers
    modifier onlyBridgeValidator() {
        require(bridgeValidators[msg.sender], "Not a bridge validator");
        _;
    }

    // Constructor
    constructor(address[] memory initialValidators) {
        for (uint256 i = 0; i < initialValidators.length; i++) {
            bridgeValidators[initialValidators[i]] = true;
        }
    }

    /**
     * @dev Post a new order
     */
    function postOrder(
        bytes memory holochainPubKey,
        string memory fromCurrency,
        string memory toCurrency,
        uint256 amount,
        uint256 rate,
        uint256 minRate,
        uint256 expiryMinutes
    ) external returns (uint256) {
        require(amount > 0, "Amount must be positive");
        require(rate > 0, "Rate must be positive");
        require(minRate <= rate, "Min rate exceeds rate");

        uint256 orderId = nextOrderId++;

        orders[orderId] = Order({
            maker: msg.sender,
            holochainPubKey: holochainPubKey,
            fromCurrency: fromCurrency,
            toCurrency: toCurrency,
            amount: amount,
            rate: rate,
            minRate: minRate,
            expiry: block.timestamp + (expiryMinutes * 60),
            active: true
        });

        // Add to orderbook
        bytes32 pairKey = keccak256(abi.encodePacked(fromCurrency, toCurrency));
        orderbook[pairKey].push(orderId);

        emit OrderPosted(orderId, msg.sender, fromCurrency, toCurrency, amount, rate);

        return orderId;
    }

    /**
     * @dev Cancel an order
     */
    function cancelOrder(uint256 orderId) external {
        Order storage order = orders[orderId];
        require(order.maker == msg.sender, "Not order maker");
        require(order.active, "Order not active");

        order.active = false;

        emit OrderCancelled(orderId);
    }

    /**
     * @dev Execute atomic swap between two orders
     * @notice Only callable by bridge validators with signatures
     */
    function executeSwap(
        uint256 order1Id,
        uint256 order2Id,
        uint256 amount1,
        uint256 amount2,
        bytes[] memory bridgeSignatures
    ) external onlyBridgeValidator returns (uint256) {
        // Validate orders
        Order storage order1 = orders[order1Id];
        Order storage order2 = orders[order2Id];

        require(order1.active, "Order 1 not active");
        require(order2.active, "Order 2 not active");
        require(order1.expiry > block.timestamp, "Order 1 expired");
        require(order2.expiry > block.timestamp, "Order 2 expired");

        // Validate currency pairs match (inverse)
        require(
            keccak256(bytes(order1.fromCurrency)) == keccak256(bytes(order2.toCurrency)) &&
            keccak256(bytes(order1.toCurrency)) == keccak256(bytes(order2.fromCurrency)),
            "Currency pair mismatch"
        );

        // Validate amounts
        require(amount1 <= order1.amount, "Exceeds order 1 amount");
        require(amount2 <= order2.amount, "Exceeds order 2 amount");

        // Calculate rate
        uint256 executionRate = (amount2 * 1e18) / amount1;

        // Validate rate within slippage
        require(executionRate >= order1.minRate, "Below order 1 min rate");
        require(executionRate <= order2.rate, "Exceeds order 2 rate");

        // Validate signatures
        require(
            bridgeSignatures.length >= requiredValidatorSignatures,
            "Insufficient validator signatures"
        );
        require(
            verifyBridgeSignatures(order1Id, order2Id, amount1, amount2, bridgeSignatures),
            "Invalid signatures"
        );

        // Update orders
        order1.amount -= amount1;
        order2.amount -= amount2;

        if (order1.amount == 0) order1.active = false;
        if (order2.amount == 0) order2.active = false;

        // Record swap
        uint256 swapId = nextSwapId++;
        swaps[swapId] = Swap({
            order1Id: order1Id,
            order2Id: order2Id,
            amount1: amount1,
            amount2: amount2,
            executionRate: executionRate,
            timestamp: block.timestamp,
            bridgeSignatures: bridgeSignatures
        });

        emit SwapExecuted(swapId, order1Id, order2Id, amount1, amount2, executionRate);

        return swapId;
    }

    /**
     * @dev Verify bridge validator signatures
     */
    function verifyBridgeSignatures(
        uint256 order1Id,
        uint256 order2Id,
        uint256 amount1,
        uint256 amount2,
        bytes[] memory signatures
    ) internal view returns (bool) {
        bytes32 messageHash = keccak256(abi.encodePacked(
            order1Id,
            order2Id,
            amount1,
            amount2,
            block.chainid
        ));

        bytes32 ethSignedHash = keccak256(abi.encodePacked(
            "\x19Ethereum Signed Message:\n32",
            messageHash
        ));

        uint256 validSignatures = 0;
        address[] memory signers = new address[](signatures.length);

        for (uint256 i = 0; i < signatures.length; i++) {
            address signer = recoverSigner(ethSignedHash, signatures[i]);

            // Check signer is validator and hasn't already signed
            if (bridgeValidators[signer] && !containsAddress(signers, signer)) {
                signers[i] = signer;
                validSignatures++;
            }
        }

        return validSignatures >= requiredValidatorSignatures;
    }

    /**
     * @dev Recover signer from signature
     */
    function recoverSigner(bytes32 hash, bytes memory signature)
        internal
        pure
        returns (address)
    {
        require(signature.length == 65, "Invalid signature length");

        bytes32 r;
        bytes32 s;
        uint8 v;

        assembly {
            r := mload(add(signature, 32))
            s := mload(add(signature, 64))
            v := byte(0, mload(add(signature, 96)))
        }

        return ecrecover(hash, v, r, s);
    }

    /**
     * @dev Check if array contains address
     */
    function containsAddress(address[] memory array, address addr)
        internal
        pure
        returns (bool)
    {
        for (uint256 i = 0; i < array.length; i++) {
            if (array[i] == addr) return true;
        }
        return false;
    }

    /**
     * @dev Get orderbook for currency pair
     */
    function getOrderbook(string memory fromCurrency, string memory toCurrency)
        external
        view
        returns (uint256[] memory)
    {
        bytes32 pairKey = keccak256(abi.encodePacked(fromCurrency, toCurrency));
        return orderbook[pairKey];
    }

    /**
     * @dev Get active orders for currency pair
     */
    function getActiveOrders(string memory fromCurrency, string memory toCurrency)
        external
        view
        returns (Order[] memory)
    {
        bytes32 pairKey = keccak256(abi.encodePacked(fromCurrency, toCurrency));
        uint256[] memory orderIds = orderbook[pairKey];

        // Count active orders
        uint256 activeCount = 0;
        for (uint256 i = 0; i < orderIds.length; i++) {
            if (orders[orderIds[i]].active && orders[orderIds[i]].expiry > block.timestamp) {
                activeCount++;
            }
        }

        // Build active orders array
        Order[] memory activeOrders = new Order[](activeCount);
        uint256 index = 0;
        for (uint256 i = 0; i < orderIds.length; i++) {
            Order memory order = orders[orderIds[i]];
            if (order.active && order.expiry > block.timestamp) {
                activeOrders[index] = order;
                index++;
            }
        }

        return activeOrders;
    }
}
```

#### Liquidity Pool Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title HolochainCurrencyPool
 * @dev Liquidity pool for Holochain currency pairs
 */
contract HolochainCurrencyPool {

    struct Pool {
        string currency1;
        string currency2;
        uint256 reserve1;       // Virtual reserves (not actual tokens)
        uint256 reserve2;
        uint256 totalShares;
        mapping(address => uint256) shares;
    }

    mapping(bytes32 => Pool) public pools;

    event PoolCreated(string currency1, string currency2);
    event LiquidityAdded(string currency1, string currency2, uint256 amount1, uint256 amount2);
    event LiquidityRemoved(string currency1, string currency2, uint256 amount1, uint256 amount2);
    event Swap(string fromCurrency, string toCurrency, uint256 amountIn, uint256 amountOut);

    /**
     * @dev Create liquidity pool for currency pair
     */
    function createPool(string memory currency1, string memory currency2) external {
        bytes32 poolKey = getPoolKey(currency1, currency2);
        require(pools[poolKey].totalShares == 0, "Pool exists");

        Pool storage pool = pools[poolKey];
        pool.currency1 = currency1;
        pool.currency2 = currency2;

        emit PoolCreated(currency1, currency2);
    }

    /**
     * @dev Add liquidity (called by bridge validators after Holochain escrow)
     */
    function addLiquidity(
        string memory currency1,
        string memory currency2,
        uint256 amount1,
        uint256 amount2,
        address provider
    ) external returns (uint256 shares) {
        bytes32 poolKey = getPoolKey(currency1, currency2);
        Pool storage pool = pools[poolKey];

        if (pool.totalShares == 0) {
            // First liquidity provider
            shares = sqrt(amount1 * amount2);
        } else {
            // Subsequent providers
            shares = min(
                (amount1 * pool.totalShares) / pool.reserve1,
                (amount2 * pool.totalShares) / pool.reserve2
            );
        }

        require(shares > 0, "Insufficient liquidity");

        pool.reserve1 += amount1;
        pool.reserve2 += amount2;
        pool.totalShares += shares;
        pool.shares[provider] += shares;

        emit LiquidityAdded(currency1, currency2, amount1, amount2);
    }

    /**
     * @dev Calculate output amount for swap (constant product formula)
     */
    function getAmountOut(
        string memory fromCurrency,
        string memory toCurrency,
        uint256 amountIn
    ) public view returns (uint256) {
        bytes32 poolKey = getPoolKey(fromCurrency, toCurrency);
        Pool storage pool = pools[poolKey];

        bool isForward = keccak256(bytes(pool.currency1)) == keccak256(bytes(fromCurrency));

        uint256 reserveIn = isForward ? pool.reserve1 : pool.reserve2;
        uint256 reserveOut = isForward ? pool.reserve2 : pool.reserve1;

        require(amountIn > 0, "Amount must be positive");
        require(reserveIn > 0 && reserveOut > 0, "Insufficient liquidity");

        // Constant product formula with 0.3% fee
        uint256 amountInWithFee = amountIn * 997;
        uint256 numerator = amountInWithFee * reserveOut;
        uint256 denominator = (reserveIn * 1000) + amountInWithFee;

        return numerator / denominator;
    }

    /**
     * @dev Get pool key
     */
    function getPoolKey(string memory currency1, string memory currency2)
        internal
        pure
        returns (bytes32)
    {
        // Normalize order (alphabetical)
        if (keccak256(bytes(currency1)) < keccak256(bytes(currency2))) {
            return keccak256(abi.encodePacked(currency1, currency2));
        } else {
            return keccak256(abi.encodePacked(currency2, currency1));
        }
    }

    // Helper functions
    function sqrt(uint256 x) internal pure returns (uint256) {
        if (x == 0) return 0;
        uint256 z = (x + 1) / 2;
        uint256 y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        return y;
    }

    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}
```

---

### Component 3: Bridge Validator Network

Python service that connects Holochain DHTs to blockchain.

```python
# bridge_validator.py

import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from holochain_client import HolochainClient
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_account.messages import encode_defunct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EscrowEvent:
    """Holochain escrow event"""
    escrow_hash: str
    holder: str
    amount: int
    from_currency: str
    to_currency: str
    rate: float
    min_rate: float
    timestamp: datetime


@dataclass
class SwapEvent:
    """Blockchain swap event"""
    swap_id: int
    order1_id: int
    order2_id: int
    amount1: int
    amount2: int
    rate: float
    timestamp: datetime


class HolochainBridgeValidator:
    """
    Bridge validator service connecting Holochain DHTs to Polygon

    Responsibilities:
    1. Monitor Holochain DHTs for escrow events
    2. Post orders to Polygon DEX
    3. Monitor Polygon for swap executions
    4. Execute transfers on Holochain DHTs
    5. Sign atomic swaps with other validators
    """

    def __init__(
        self,
        holochain_urls: Dict[str, str],
        polygon_rpc: str,
        exchange_address: str,
        validator_private_key: str,
        required_confirmations: int = 3,
    ):
        # Holochain clients (one per currency)
        self.holochain_clients: Dict[str, HolochainClient] = {
            currency: HolochainClient(url)
            for currency, url in holochain_urls.items()
        }

        # Web3 setup
        self.w3 = Web3(Web3.HTTPProvider(polygon_rpc))
        self.account = Account.from_key(validator_private_key)

        # Smart contract
        self.exchange: Contract = self.w3.eth.contract(
            address=exchange_address,
            abi=self._load_exchange_abi()
        )

        # State
        self.required_confirmations = required_confirmations
        self.processed_escrows: set = set()
        self.processed_swaps: set = set()

        logger.info(f"Bridge validator initialized: {self.account.address}")

    async def start(self):
        """Start validator services"""
        logger.info("Starting bridge validator services...")

        # Run both monitoring loops concurrently
        await asyncio.gather(
            self.monitor_holochain_escrows(),
            self.monitor_blockchain_swaps(),
        )

    async def monitor_holochain_escrows(self):
        """Monitor Holochain DHTs for new escrow events"""
        logger.info("Monitoring Holochain escrows...")

        while True:
            try:
                for currency, client in self.holochain_clients.items():
                    # Query pending escrows
                    escrows = await client.call_zome(
                        cell_id=client.cell_id,
                        zome_name=currency,
                        fn_name="query_pending_escrows",
                        payload={}
                    )

                    # Process new escrows
                    for escrow_data in escrows:
                        escrow = self._parse_escrow(escrow_data)

                        if escrow.escrow_hash in self.processed_escrows:
                            continue

                        logger.info(f"New escrow detected: {escrow.escrow_hash}")

                        # Post order to blockchain
                        await self.post_order_to_blockchain(escrow)

                        self.processed_escrows.add(escrow.escrow_hash)

                # Poll every 5 seconds
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error monitoring escrows: {e}")
                await asyncio.sleep(5)

    async def post_order_to_blockchain(self, escrow: EscrowEvent):
        """Post escrow as order on blockchain DEX"""
        try:
            # Convert Holochain pubkey to bytes
            holochain_pubkey = bytes.fromhex(escrow.holder)

            # Build transaction
            tx = self.exchange.functions.postOrder(
                holochain_pubkey,
                escrow.from_currency,
                escrow.to_currency,
                escrow.amount,
                int(escrow.rate * 1e18),
                int(escrow.min_rate * 1e18),
                60  # 60 minute expiry
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
            })

            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            logger.info(f"Posted order to blockchain: {tx_hash.hex()}")

            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt['status'] == 1:
                logger.info(f"Order confirmed: {tx_hash.hex()}")
            else:
                logger.error(f"Order failed: {tx_hash.hex()}")

        except Exception as e:
            logger.error(f"Error posting order: {e}")

    async def monitor_blockchain_swaps(self):
        """Monitor blockchain for executed swaps"""
        logger.info("Monitoring blockchain swaps...")

        # Get latest block
        latest_block = self.w3.eth.block_number

        while True:
            try:
                # Get new blocks
                current_block = self.w3.eth.block_number

                if current_block > latest_block:
                    # Check for SwapExecuted events
                    events = self.exchange.events.SwapExecuted.get_logs(
                        fromBlock=latest_block + 1,
                        toBlock=current_block
                    )

                    for event in events:
                        swap = self._parse_swap_event(event)

                        if swap.swap_id in self.processed_swaps:
                            continue

                        logger.info(f"New swap detected: {swap.swap_id}")

                        # Execute transfer on Holochain
                        await self.execute_holochain_transfer(swap)

                        self.processed_swaps.add(swap.swap_id)

                    latest_block = current_block

                # Poll every 2 seconds (Polygon block time)
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error monitoring swaps: {e}")
                await asyncio.sleep(2)

    async def execute_holochain_transfer(self, swap: SwapEvent):
        """Execute transfer on Holochain DHTs after swap"""
        try:
            # Get order details
            order1 = self.exchange.functions.orders(swap.order1_id).call()
            order2 = self.exchange.functions.orders(swap.order2_id).call()

            # Execute transfers on both DHTs
            await asyncio.gather(
                self._transfer_on_dht(
                    currency=order1[3],  # fromCurrency
                    from_holder=order1[1].hex(),  # holochain pubkey
                    to_holder=order2[1].hex(),
                    amount=swap.amount1
                ),
                self._transfer_on_dht(
                    currency=order2[3],  # fromCurrency
                    from_holder=order2[1].hex(),
                    to_holder=order1[1].hex(),
                    amount=swap.amount2
                )
            )

            logger.info(f"Holochain transfers completed for swap {swap.swap_id}")

        except Exception as e:
            logger.error(f"Error executing Holochain transfer: {e}")

    async def _transfer_on_dht(
        self,
        currency: str,
        from_holder: str,
        to_holder: str,
        amount: int
    ):
        """Execute transfer on specific Holochain DHT"""
        client = self.holochain_clients[currency]

        result = await client.call_zome(
            cell_id=client.cell_id,
            zome_name=currency,
            fn_name="execute_bridge_transfer",
            payload={
                "from_holder": from_holder,
                "to_holder": to_holder,
                "amount": amount,
            }
        )

        logger.info(f"Transfer executed on {currency} DHT: {amount}")
        return result

    async def sign_swap(
        self,
        order1_id: int,
        order2_id: int,
        amount1: int,
        amount2: int
    ) -> bytes:
        """Sign atomic swap as validator"""
        # Create message hash
        message_hash = Web3.solidity_keccak(
            ['uint256', 'uint256', 'uint256', 'uint256', 'uint256'],
            [order1_id, order2_id, amount1, amount2, self.w3.eth.chain_id]
        )

        # Sign with Ethereum account
        message = encode_defunct(message_hash)
        signature = self.account.sign_message(message)

        return signature.signature

    def _parse_escrow(self, data: dict) -> EscrowEvent:
        """Parse Holochain escrow data"""
        return EscrowEvent(
            escrow_hash=data['escrow_hash'],
            holder=data['holder'],
            amount=data['amount'],
            from_currency=data['swap_intent']['from_currency'],
            to_currency=data['swap_intent']['to_currency'],
            rate=data['swap_intent']['rate'],
            min_rate=data['swap_intent']['min_rate'],
            timestamp=datetime.fromtimestamp(data['lock_time'])
        )

    def _parse_swap_event(self, event) -> SwapEvent:
        """Parse blockchain swap event"""
        return SwapEvent(
            swap_id=event['args']['swapId'],
            order1_id=event['args']['order1Id'],
            order2_id=event['args']['order2Id'],
            amount1=event['args']['amount1'],
            amount2=event['args']['amount2'],
            rate=event['args']['rate'] / 1e18,
            timestamp=datetime.now()
        )

    def _load_exchange_abi(self) -> List[dict]:
        """Load exchange contract ABI"""
        # In production, load from file
        with open('HolochainCurrencyExchange.abi.json', 'r') as f:
            return json.load(f)


# Example usage
async def main():
    validator = HolochainBridgeValidator(
        holochain_urls={
            'zerotrustml_credits': 'ws://localhost:8888',
            'compute_credits': 'ws://localhost:8889',
            'storage_credits': 'ws://localhost:8890',
        },
        polygon_rpc='https://polygon-rpc.com',
        exchange_address='0x...',
        validator_private_key='0x...',
    )

    await validator.start()


if __name__ == '__main__':
    asyncio.run(main())
```

---

## Implementation Phases

### Phase 1: Holochain Audit Trail + Zero-TrustML Credits (2-3 hours)

**Goal**: Single working Holochain currency with audit trail

**Deliverables**:
- ✅ HDK 0.5 compatibility updates
- ✅ Zero-TrustML Credits DNA with validation rules
- ✅ Credit earning mechanisms (quality gradients, Byzantine detection)
- ✅ Audit trail query APIs
- ✅ Balance and transfer functions

**Test Criteria**:
- Can earn credits for quality gradients
- Can transfer credits peer-to-peer
- Can query full audit trail
- Validation rules enforce fairness

**Success Metric**: 100 test transactions on local conductor

---

### Phase 2: Multiple Holochain Currencies (1-2 weeks)

**Goal**: 3-5 currencies with different rules

**Deliverables**:
- ✅ Compute Credits DNA (pay for GPU/CPU)
- ✅ Storage Credits DNA (pay for storage)
- ✅ Impact Tokens DNA (social good measurement)
- ✅ TimeBank Hours DNA (time-based mutual credit)
- ✅ Service Credits DNA (peer services)

**Test Criteria**:
- Each currency has unique validation rules
- Can earn and spend in each currency
- Balances tracked independently

**Success Metric**: 5 active currencies, 1000+ transactions

---

### Phase 3: Blockchain DEX + Bridge (2-3 weeks)

**Goal**: Exchange layer with atomic swaps

**Deliverables**:
- ✅ HolochainCurrencyExchange.sol deployed to Polygon
- ✅ HolochainCurrencyPool.sol for liquidity
- ✅ Bridge validator network (3+ validators)
- ✅ Order posting from Holochain escrows
- ✅ Atomic swap execution

**Test Criteria**:
- Can post order from Holochain → Polygon
- Can execute swap on Polygon
- Bridge completes transfer on Holochain
- All or nothing (atomic)

**Success Metric**: 100 successful swaps between 2 currencies

---

### Phase 4: External Liquidity + DeFi (1-2 weeks)

**Goal**: Connect to existing DeFi ecosystem

**Deliverables**:
- ✅ USDC/Zero-TrustML Credits pool
- ✅ Uniswap V3 integration
- ✅ Chainlink price feeds
- ✅ External liquidity providers onboarded

**Test Criteria**:
- Can swap Zero-TrustML Credits → USDC
- Can add liquidity to pools
- Price oracles work correctly

**Success Metric**: $10K+ TVL in pools, 50+ swaps/day

---

## Security Considerations

### Holochain Layer Security

**Threat**: Malicious credit creation
**Mitigation**: Validation rules require peer verification (3+ validators)

**Threat**: Double-spend attacks
**Mitigation**: Holochain's CRDT ensures eventual consistency, double-spends detected

**Threat**: Sybil attacks (fake verifiers)
**Mitigation**: Reputation system, require established nodes as verifiers

### Blockchain Layer Security

**Threat**: Front-running attacks
**Mitigation**: Use FlashBots or private mempool for bridge transactions

**Threat**: Bridge validator collusion
**Mitigation**: 3+ validators required, multisig for critical operations

**Threat**: Smart contract exploits
**Mitigation**: Formal verification, audits by Trail of Bits or similar

### Bridge Security

**Threat**: Bridge validators sign invalid swaps
**Mitigation**:
- On-chain verification of Holochain state (Merkle proofs)
- Slashing for misbehavior
- DAO governance can rotate validators

**Threat**: Replay attacks
**Mitigation**: Include chain ID and nonce in signatures

**Threat**: Man-in-the-middle
**Mitigation**: TLS for all communications, signature verification

---

## Economic Model

### Transaction Fees

| Layer | Fee | Who Pays | Who Receives |
|-------|-----|----------|--------------|
| **Holochain** | $0.00 | N/A | N/A |
| **Blockchain DEX** | 0.3% | Taker | Liquidity providers |
| **Bridge** | 0.1% | Both sides | Validators |
| **External DeFi** | Variable | User | Protocol |

### Incentive Alignment

**For Communities**:
- Zero-cost internal transactions
- Control over currency rules
- Self-sovereign economics

**For Liquidity Providers**:
- Earn 0.3% on all swaps
- Earn bridge fees (0.1%)
- Early liquidity bonuses

**For Bridge Validators**:
- Earn 0.1% of swap volume
- Governance token rewards
- Priority access to new currencies

**For Users**:
- Freedom to hold multiple currencies
- Access to global liquidity
- No KYC friction for small amounts

---

## Reference Implementation

### Directory Structure

```
holochain-currency-exchange/
├── holochain/
│   ├── zerotrustml_credits/
│   │   ├── zomes/
│   │   │   └── zerotrustml_credits/
│   │   │       └── src/
│   │   │           ├── lib.rs
│   │   │           ├── credit.rs
│   │   │           ├── escrow.rs
│   │   │           └── validation.rs
│   │   └── dna.yaml
│   ├── compute_credits/
│   ├── storage_credits/
│   └── shared/
│       └── bridge_escrow/
├── blockchain/
│   ├── contracts/
│   │   ├── HolochainCurrencyExchange.sol
│   │   ├── HolochainCurrencyPool.sol
│   │   └── BridgeValidator.sol
│   ├── scripts/
│   │   ├── deploy.ts
│   │   └── verify.ts
│   └── test/
│       ├── Exchange.test.ts
│       └── Pool.test.ts
├── bridge/
│   ├── validator.py
│   ├── holochain_client.py
│   ├── blockchain_client.py
│   └── config.yaml
├── frontend/
│   ├── components/
│   │   ├── WalletConnect.tsx
│   │   ├── CurrencyBalance.tsx
│   │   ├── SwapInterface.tsx
│   │   └── OrderBook.tsx
│   └── hooks/
│       ├── useHolochain.ts
│       └── useBlockchain.ts
└── docs/
    ├── ARCHITECTURE.md (this file)
    ├── DEPLOYMENT.md
    └── API_REFERENCE.md
```

---

## Future Extensions

### Phase 5+: Advanced Features

1. **Cross-Chain Bridges** - Connect to other blockchains (Ethereum L2s, Cosmos, Polkadot)
2. **Algorithmic Market Makers** - Custom AMM curves for different currency types
3. **Governance DAOs** - Community governance for each currency
4. **Reputation-Weighted Fees** - Lower fees for high-reputation participants
5. **Privacy Layers** - zk-SNARKs for private swaps
6. **Prediction Markets** - Currency-based prediction markets
7. **Programmable Currencies** - Holochain currencies with complex rules (time decay, demurrage)
8. **NFT Integration** - Currencies backed by NFTs
9. **Cross-DHT Queries** - Aggregate data across multiple currencies
10. **Mobile Wallets** - React Native apps for iOS/Android

---

## Success Metrics

### Technical Metrics
- **Uptime**: 99.9%+ for bridge validators
- **Latency**: <5 seconds for escrow → order posting
- **Throughput**: 1000+ transactions/second per currency (Holochain)
- **Cost**: <$0.01 per swap (Polygon gas)

### Adoption Metrics
- **Active Currencies**: 10+ live currencies by Month 6
- **Transaction Volume**: $1M+ monthly by Month 12
- **Users**: 10,000+ unique addresses by Month 12
- **Liquidity**: $100K+ TVL in pools by Month 6

### Economic Metrics
- **Swap Volume**: $10K+ daily by Month 6
- **Liquidity Provider APY**: 10-20% average
- **Bridge Validator Revenue**: $1K+ monthly per validator
- **Fee Efficiency**: 90%+ lower fees than centralized exchanges

---

## Conclusion

This architecture provides a **production-ready blueprint** for building inter-currency exchanges between Holochain communities using blockchain as a settlement layer.

**Key Advantages**:
- ✅ Zero-cost internal transactions (Holochain)
- ✅ Efficient price discovery (Blockchain DEX)
- ✅ Atomic swaps (Bridge validators)
- ✅ Composable with DeFi (External liquidity)
- ✅ Sovereign currencies (Community rules)

**Next Steps**:
1. Implement Phase 1 (Zero-TrustML Credits DNA)
2. Test with 100+ local transactions
3. Deploy to Holochain devnet
4. Build Phase 2 (additional currencies)
5. Parallel: Smart contract development + audits

**Timeline**: 8-12 weeks for full implementation (Phases 1-4)

**Team**: 1-2 Holochain developers, 1 Solidity developer, 1 bridge engineer

---

**This architecture is ready for implementation. Phase 1 begins now.**

---

## Appendix: Comparison with Alternatives

### vs Centralized Exchanges (Coinbase, Binance)

| Feature | Holochain + Blockchain | Centralized Exchange |
|---------|----------------------|---------------------|
| Internal Tx Cost | $0.00 | $0.50-$2.00 |
| KYC Required | No (for P2P) | Yes (always) |
| Custody | Self-custody | Exchange custody |
| Downtime Risk | Distributed | Single point |
| Censorship Resistance | High | Low |

### vs Pure DEX (Uniswap, Curve)

| Feature | Holochain + Blockchain | Pure DEX |
|---------|----------------------|----------|
| Internal Tx Cost | $0.00 | $1-$50 (L1) |
| Scalability | Linear per currency | Limited by blockchain |
| Customization | Infinite (DNA rules) | Limited (EVM) |
| Privacy | Configurable | Public |

### vs Pure Holochain

| Feature | Holochain + Blockchain | Pure Holochain |
|---------|----------------------|----------------|
| Cross-DHT Swaps | Native (blockchain) | Complex/manual |
| Price Discovery | DEX orderbooks | No mechanism |
| External Liquidity | Native (DeFi) | Very hard |
| Legal Recognition | Blockchain helps | Uncertain |

**Conclusion**: Hybrid approach gets best of all worlds.

---

**END OF ARCHITECTURE DOCUMENT**

*This document is a living specification and will be updated as implementation progresses.*
