#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate ABI manually from ZeroTrustMLGradientStorage contract
This extracts the interface from the Solidity source
"""

import json
from pathlib import Path

# Complete ABI for ZeroTrustMLGradientStorage.sol
# Generated from the contract interface
CONTRACT_ABI = [
    # Constructor
    {
        "type": "constructor",
        "inputs": [],
        "stateMutability": "nonpayable"
    },

    # Storage Functions
    {
        "type": "function",
        "name": "storeGradient",
        "inputs": [
            {"name": "gradientId", "type": "string"},
            {"name": "nodeIdHash", "type": "bytes32"},
            {"name": "roundNum", "type": "uint256"},
            {"name": "gradientHash", "type": "string"},
            {"name": "pogqScore", "type": "uint256"},
            {"name": "zkpocVerified", "type": "bool"}
        ],
        "outputs": [],
        "stateMutability": "nonpayable"
    },
    {
        "type": "function",
        "name": "getGradient",
        "inputs": [
            {"name": "gradientId", "type": "string"}
        ],
        "outputs": [
            {"type": "string"},  # gradientId
            {"type": "bytes32"},  # nodeIdHash
            {"type": "uint256"},  # roundNum
            {"type": "string"},  # gradientHash
            {"type": "uint256"},  # pogqScore
            {"type": "bool"},  # zkpocVerified
            {"type": "uint256"},  # timestamp
            {"type": "address"}  # submitter
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "getGradientsByRound",
        "inputs": [
            {"name": "roundNum", "type": "uint256"}
        ],
        "outputs": [
            {"type": "string[]"}
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "gradientExists",
        "inputs": [
            {"name": "gradientId", "type": "string"}
        ],
        "outputs": [
            {"type": "bool"}
        ],
        "stateMutability": "view"
    },

    # Credit Functions
    {
        "type": "function",
        "name": "issueCredit",
        "inputs": [
            {"name": "holderHash", "type": "bytes32"},
            {"name": "amount", "type": "uint256"},
            {"name": "earnedFrom", "type": "string"}
        ],
        "outputs": [],
        "stateMutability": "nonpayable"
    },
    {
        "type": "function",
        "name": "getCreditBalance",
        "inputs": [
            {"name": "holderHash", "type": "bytes32"}
        ],
        "outputs": [
            {"type": "uint256"}
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "getCreditHistory",
        "inputs": [
            {"name": "holderHash", "type": "bytes32"}
        ],
        "outputs": [
            {
                "type": "tuple[]",
                "components": [
                    {"name": "holderHash", "type": "bytes32"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "earnedFrom", "type": "string"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "creditId", "type": "uint256"}
                ]
            }
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "getCreditCount",
        "inputs": [
            {"name": "holderHash", "type": "bytes32"}
        ],
        "outputs": [
            {"type": "uint256"}
        ],
        "stateMutability": "view"
    },

    # Byzantine Event Functions
    {
        "type": "function",
        "name": "logByzantineEvent",
        "inputs": [
            {"name": "nodeIdHash", "type": "bytes32"},
            {"name": "roundNum", "type": "uint256"},
            {"name": "detectionMethod", "type": "string"},
            {"name": "severity", "type": "string"},
            {"name": "details", "type": "string"}
        ],
        "outputs": [],
        "stateMutability": "nonpayable"
    },
    {
        "type": "function",
        "name": "getByzantineEvents",
        "inputs": [
            {"name": "nodeIdHash", "type": "bytes32"}
        ],
        "outputs": [
            {
                "type": "tuple[]",
                "components": [
                    {"name": "nodeIdHash", "type": "bytes32"},
                    {"name": "roundNum", "type": "uint256"},
                    {"name": "detectionMethod", "type": "string"},
                    {"name": "severity", "type": "string"},
                    {"name": "details", "type": "string"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "eventId", "type": "uint256"}
                ]
            }
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "getByzantineEventsByRound",
        "inputs": [
            {"name": "nodeIdHash", "type": "bytes32"},
            {"name": "roundNum", "type": "uint256"}
        ],
        "outputs": [
            {
                "type": "tuple[]",
                "components": [
                    {"name": "nodeIdHash", "type": "bytes32"},
                    {"name": "roundNum", "type": "uint256"},
                    {"name": "detectionMethod", "type": "string"},
                    {"name": "severity", "type": "string"},
                    {"name": "details", "type": "string"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "eventId", "type": "uint256"}
                ]
            }
        ],
        "stateMutability": "view"
    },

    # Reputation Functions
    {
        "type": "function",
        "name": "getReputation",
        "inputs": [
            {"name": "nodeIdHash", "type": "bytes32"}
        ],
        "outputs": [
            {"name": "totalGradientsSubmitted", "type": "uint256"},
            {"name": "totalCreditsEarned", "type": "uint256"},
            {"name": "byzantineEventCount", "type": "uint256"},
            {"name": "averagePogqScore", "type": "uint256"},
            {"name": "lastActivityTimestamp", "type": "uint256"}
        ],
        "stateMutability": "view"
    },

    # Statistics Functions
    {
        "type": "function",
        "name": "getStats",
        "inputs": [],
        "outputs": [
            {"type": "uint256"},  # totalGradients
            {"type": "uint256"},  # totalCreditsIssued
            {"type": "uint256"}   # totalByzantineEvents
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "getVersion",
        "inputs": [],
        "outputs": [
            {"type": "string"}
        ],
        "stateMutability": "view"
    },

    # Admin Functions
    {
        "type": "function",
        "name": "transferOwnership",
        "inputs": [
            {"name": "newOwner", "type": "address"}
        ],
        "outputs": [],
        "stateMutability": "nonpayable"
    },

    # Public Variables
    {
        "type": "function",
        "name": "owner",
        "inputs": [],
        "outputs": [
            {"type": "address"}
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "version",
        "inputs": [],
        "outputs": [
            {"type": "string"}
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "totalGradients",
        "inputs": [],
        "outputs": [
            {"type": "uint256"}
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "totalCreditsIssued",
        "inputs": [],
        "outputs": [
            {"type": "uint256"}
        ],
        "stateMutability": "view"
    },
    {
        "type": "function",
        "name": "totalByzantineEvents",
        "inputs": [],
        "outputs": [
            {"type": "uint256"}
        ],
        "stateMutability": "view"
    },

    # Events
    {
        "type": "event",
        "name": "GradientStored",
        "inputs": [
            {"name": "gradientId", "type": "string", "indexed": True},
            {"name": "nodeIdHash", "type": "bytes32", "indexed": True},
            {"name": "roundNum", "type": "uint256", "indexed": True},
            {"name": "pogqScore", "type": "uint256", "indexed": False},
            {"name": "zkpocVerified", "type": "bool", "indexed": False}
        ]
    },
    {
        "type": "event",
        "name": "CreditIssued",
        "inputs": [
            {"name": "holderHash", "type": "bytes32", "indexed": True},
            {"name": "amount", "type": "uint256", "indexed": False},
            {"name": "earnedFrom", "type": "string", "indexed": False},
            {"name": "creditId", "type": "uint256", "indexed": False}
        ]
    },
    {
        "type": "event",
        "name": "ByzantineEventLogged",
        "inputs": [
            {"name": "nodeIdHash", "type": "bytes32", "indexed": True},
            {"name": "roundNum", "type": "uint256", "indexed": True},
            {"name": "severity", "type": "string", "indexed": False},
            {"name": "eventId", "type": "uint256", "indexed": False}
        ]
    },
    {
        "type": "event",
        "name": "ReputationUpdated",
        "inputs": [
            {"name": "nodeIdHash", "type": "bytes32", "indexed": True},
            {"name": "totalGradients", "type": "uint256", "indexed": False},
            {"name": "totalCredits", "type": "uint256", "indexed": False},
            {"name": "byzantineEvents", "type": "uint256", "indexed": False}
        ]
    }
]


def main():
    """Generate ABI files"""
    print("📝 Generating ZeroTrustML Contract ABI")
    print("=" * 60)
    print()

    # Create build directory
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)

    # Save ABI JSON
    abi_file = build_dir / "ZeroTrustMLGradientStorage.abi.json"
    with open(abi_file, 'w') as f:
        json.dump(CONTRACT_ABI, f, indent=2)

    print(f"✅ ABI saved to: {abi_file}")

    # Save Python constant (with proper Python syntax for booleans)
    python_file = build_dir / "contract_abi.py"
    with open(python_file, 'w') as f:
        f.write('"""Generated ABI for ZeroTrustMLGradientStorage contract"""\n\n')
        # Use json.dumps but replace JSON booleans with Python booleans
        abi_str = json.dumps(CONTRACT_ABI, indent=2)
        abi_str = abi_str.replace(': true', ': True').replace(': false', ': False')
        f.write(f"CONTRACT_ABI = {abi_str}\n")

    print(f"✅ Python ABI saved to: {python_file}")

    # Print summary
    functions = [item for item in CONTRACT_ABI if item['type'] == 'function']
    events = [item for item in CONTRACT_ABI if item['type'] == 'event']

    print()
    print("=" * 60)
    print("📊 ABI SUMMARY")
    print("=" * 60)
    print(f"Total Functions: {len(functions)}")
    print(f"Total Events: {len(events)}")
    print()

    print("Public Functions:")
    for func in functions:
        name = func['name']
        inputs = len(func.get('inputs', []))
        state = func.get('stateMutability', 'nonpayable')
        print(f"  - {name}({inputs} params) [{state}]")

    print()
    print("Events:")
    for event in events:
        name = event['name']
        inputs = len(event.get('inputs', []))
        print(f"  - {name}({inputs} params)")

    print()
    print("=" * 60)
    print("🎉 ABI generation complete!")
    print()
    print("Next steps:")
    print("1. Deploy contract to Polygon Mumbai testnet")
    print("2. Update EthereumBackend._get_contract_abi() to use this ABI")
    print("3. Test with real blockchain")


if __name__ == "__main__":
    main()
