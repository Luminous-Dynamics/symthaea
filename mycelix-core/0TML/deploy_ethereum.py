#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Deploy ZeroTrustMLGradientStorage contract to Polygon Amoy Testnet

This script will:
1. Compile the Solidity contract (or use pre-compiled bytecode)
2. Connect to Polygon Amoy testnet (Mumbai replacement)
3. Deploy the contract using a test wallet
4. Save the deployed contract address
"""

import json
import sys
from pathlib import Path
from web3 import Web3
from eth_account import Account

# Polygon Amoy Testnet Configuration (Mumbai was deprecated April 2024)
AMOY_RPC = "https://rpc-amoy.polygon.technology/"
CHAIN_ID = 80002

def load_or_compile_contract():
    """
    Load contract ABI and bytecode

    Returns:
        tuple: (abi, bytecode) or (None, None) if compilation fails
    """
    print("📋 Loading contract ABI and bytecode...")

    # Load ABI
    abi_file = Path("build/ZeroTrustMLGradientStorage.abi.json")
    if not abi_file.exists():
        print("❌ ABI file not found. Run: python generate_abi.py")
        return None, None

    with open(abi_file, 'r') as f:
        abi = json.load(f)

    print(f"✅ ABI loaded: {len([item for item in abi if item['type'] == 'function'])} functions")

    # Try to load compiled bytecode
    bin_file = Path("build/ZeroTrustMLGradientStorage.bin")
    if bin_file.exists():
        with open(bin_file, 'r') as f:
            bytecode = f.read().strip()
        print(f"✅ Bytecode loaded: {len(bytecode)} bytes")
        return abi, bytecode

    # Try to compile using system solc
    print("\n📦 Bytecode not found. Attempting to compile...")
    try:
        import subprocess
        import tempfile

        # Read contract source
        contract_file = Path("contracts/ZeroTrustMLGradientStorage.sol")
        with open(contract_file, 'r') as f:
            contract_source = f.read()

        # Use system solc to compile
        print("   Using system solc compiler...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as tmp:
            tmp.write(contract_source)
            tmp_path = tmp.name

        try:
            # Compile with solc
            result = subprocess.run(
                ['solc', '--bin', '--optimize', tmp_path],
                capture_output=True,
                text=True,
                check=True
            )

            # Extract bytecode from output
            output_lines = result.stdout.strip().split('\n')
            bytecode = None
            for i, line in enumerate(output_lines):
                if 'Binary:' in line and i + 1 < len(output_lines):
                    bytecode = output_lines[i + 1].strip()
                    break

            if not bytecode:
                raise ValueError("Could not extract bytecode from solc output")

            # Save bytecode for future use
            with open(bin_file, 'w') as f:
                f.write(bytecode)

            print(f"✅ Contract compiled: {len(bytecode)} bytes")
            return abi, bytecode

        finally:
            # Clean up temp file
            import os
            os.unlink(tmp_path)

    except Exception as e:
        print(f"⚠️  Compilation failed: {e}")
        print("\n📝 MANUAL DEPLOYMENT INSTRUCTIONS:")
        print("=" * 60)
        print("Since automatic compilation failed, you can deploy manually:")
        print()
        print("1. Go to Remix IDE: https://remix.ethereum.org/")
        print("2. Create a new file: ZeroTrustMLGradientStorage.sol")
        print("3. Copy contents from: contracts/ZeroTrustMLGradientStorage.sol")
        print("4. Compile with Solidity 0.8.20")
        print("5. Switch to 'Deploy & Run Transactions' tab")
        print("6. Select 'Injected Provider - MetaMask'")
        print("7. Select 'Polygon Amoy Testnet' in MetaMask")
        print("8. Deploy the contract")
        print("9. Copy the deployed contract address")
        print("10. Save it to: build/ethereum_contract_address.txt")
        print()
        print("Need testnet POL? Get free tokens from:")
        print("  https://faucet.polygon.technology/")
        print("=" * 60)
        return None, None


def get_wallet():
    """
    Get wallet for deployment

    Returns:
        tuple: (account, private_key) or (None, None) if not available
    """
    print("\n🔑 Loading wallet...")

    # Check for environment variable
    import os
    private_key = os.environ.get('ETHEREUM_PRIVATE_KEY')

    if not private_key:
        # Check for key file
        key_file = Path("build/.ethereum_key")
        if key_file.exists():
            with open(key_file, 'r') as f:
                private_key = f.read().strip()

    if not private_key:
        print("⚠️  No wallet found!")
        print()
        print("To deploy automatically, you need a test wallet with Amoy POL:")
        print()
        print("1. Create a test wallet:")
        print("   from eth_account import Account")
        print("   account = Account.create()")
        print("   print(f'Address: {account.address}')")
        print("   print(f'Private Key: {account.key.hex()}')")
        print()
        print("2. Get free testnet POL:")
        print("   https://faucet.polygon.technology/")
        print()
        print("3. Save private key (NEVER use for real funds!):")
        print("   echo 'YOUR_PRIVATE_KEY' > build/.ethereum_key")
        print()
        print("OR set environment variable:")
        print("   export ETHEREUM_PRIVATE_KEY='YOUR_PRIVATE_KEY'")
        print()
        return None, None

    account = Account.from_key(private_key)
    print(f"✅ Wallet loaded: {account.address}")

    return account, private_key


def deploy_contract(w3, account, private_key, abi, bytecode):
    """
    Deploy contract to blockchain

    Args:
        w3: Web3 instance
        account: Account object
        private_key: Private key
        abi: Contract ABI
        bytecode: Contract bytecode

    Returns:
        str: Deployed contract address or None
    """
    print("\n🚀 Deploying contract...")

    # Check balance
    balance = w3.eth.get_balance(account.address)
    balance_pol = w3.from_wei(balance, 'ether')
    print(f"   Wallet balance: {balance_pol:.4f} POL")

    if balance == 0:
        print("❌ No POL in wallet! Get testnet POL from:")
        print("   https://faucet.polygon.technology/")
        return None

    # Create contract instance
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)

    # Estimate gas
    try:
        gas_estimate = Contract.constructor().estimate_gas({
            'from': account.address
        })
        print(f"   Estimated gas: {gas_estimate}")
    except Exception as e:
        print(f"⚠️  Gas estimation failed: {e}")
        gas_estimate = 3000000  # Fallback

    # Get gas price
    gas_price = w3.eth.gas_price
    print(f"   Gas price: {w3.from_wei(gas_price, 'gwei'):.2f} Gwei")

    # Build constructor transaction
    tx = Contract.constructor().build_transaction({
        'from': account.address,
        'gas': gas_estimate,
        'gasPrice': gas_price,
        'nonce': nonce,
        'chainId': CHAIN_ID
    })

    # Sign transaction
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)

    # Send transaction
    print("   Sending transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"   Transaction hash: {tx_hash.hex()}")

    # Wait for receipt
    print("   Waiting for confirmation...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)

    if tx_receipt['status'] == 1:
        contract_address = tx_receipt['contractAddress']
        print(f"✅ Contract deployed!")
        print(f"   Address: {contract_address}")
        print(f"   Block: {tx_receipt['blockNumber']}")
        print(f"   Gas used: {tx_receipt['gasUsed']}")

        # Save contract address
        address_file = Path("build/ethereum_contract_address.txt")
        with open(address_file, 'w') as f:
            f.write(f"{contract_address}\n")
        print(f"   Saved to: {address_file}")

        # Save deployment info
        deployment_info = {
            'contract_address': contract_address,
            'deployer': account.address,
            'chain_id': CHAIN_ID,
            'chain_name': 'Polygon Amoy Testnet',
            'block_number': tx_receipt['blockNumber'],
            'transaction_hash': tx_hash.hex(),
            'gas_used': tx_receipt['gasUsed']
        }

        info_file = Path("build/ethereum_deployment.json")
        with open(info_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        print(f"   Deployment info: {info_file}")

        return contract_address
    else:
        print("❌ Deployment failed!")
        return None


def main():
    """Main deployment flow"""
    print("🌐 ZeroTrustML Ethereum Deployment Script")
    print("=" * 60)
    print(f"Target: Polygon Amoy Testnet")
    print(f"RPC: {AMOY_RPC}")
    print(f"Chain ID: {CHAIN_ID}")
    print("=" * 60)

    # Load contract
    abi, bytecode = load_or_compile_contract()
    if not abi or not bytecode:
        print("\n⚠️  Cannot proceed without ABI and bytecode")
        print("Please follow the manual deployment instructions above.")
        return 1

    # Get wallet
    account, private_key = get_wallet()
    if not account:
        print("\n⚠️  Cannot proceed without wallet")
        return 1

    # Connect to Amoy
    print("\n🌐 Connecting to Polygon Amoy...")
    w3 = Web3(Web3.HTTPProvider(AMOY_RPC))

    if not w3.is_connected():
        print("❌ Failed to connect to Polygon Amoy")
        print("   Try alternative RPC:")
        print("   - https://polygon-amoy.drpc.org")
        print("   - https://polygon-amoy-bor-rpc.publicnode.com")
        return 1

    print(f"✅ Connected to Polygon Amoy")
    print(f"   Latest block: {w3.eth.block_number}")

    # Deploy
    contract_address = deploy_contract(w3, account, private_key, abi, bytecode)

    if contract_address:
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 60)
        print(f"Contract Address: {contract_address}")
        print()
        print("View on PolygonScan:")
        print(f"https://amoy.polygonscan.com/address/{contract_address}")
        print()
        print("Next steps:")
        print("1. Update src/zerotrustml/backends/ethereum_backend.py")
        print(f"   Set CONTRACT_ADDRESS = '{contract_address}'")
        print("2. Test the backend with: python test_phase10_coordinator.py")
        print("=" * 60)
        return 0
    else:
        print("\n❌ Deployment failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
