#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Compile ZeroTrustML Solidity Smart Contract
Generates ABI and bytecode for deployment
"""

import json
import sys
from pathlib import Path

try:
    from solcx import compile_source, install_solc, set_solc_version
    SOLCX_AVAILABLE = True
except ImportError:
    SOLCX_AVAILABLE = False
    print("⚠️  py-solc-x not available. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "py-solc-x"], check=True)
    from solcx import compile_source, install_solc, set_solc_version
    SOLCX_AVAILABLE = True

def compile_contract(contract_path: str, output_dir: str = "build"):
    """
    Compile Solidity contract and save ABI + bytecode

    Args:
        contract_path: Path to .sol file
        output_dir: Directory to save compiled artifacts
    """
    print(f"📝 Compiling contract: {contract_path}")
    print()

    # Install solc if not already installed
    try:
        from solcx import get_installed_solc_versions
        installed = get_installed_solc_versions()
        if not installed:
            print("📦 Installing solc compiler v0.8.20...")
            install_solc('0.8.20')
        else:
            print(f"✅ Solc already installed: {installed}")

        # Set version
        set_solc_version('0.8.20')

    except Exception as e:
        print(f"Error installing solc: {e}")
        return None

    # Read contract source
    with open(contract_path, 'r') as f:
        contract_source = f.read()

    # Compile
    print("\n🔨 Compiling...")
    try:
        compiled_sol = compile_source(
            contract_source,
            output_values=['abi', 'bin', 'bin-runtime', 'metadata'],
            solc_version='0.8.20'
        )
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return None

    # Extract contract interface
    contract_id = list(compiled_sol.keys())[0]
    contract_interface = compiled_sol[contract_id]

    print(f"✅ Compilation successful!")
    print(f"   Contract: {contract_id}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save ABI
    abi_file = output_path / "ZeroTrustMLGradientStorage.abi.json"
    with open(abi_file, 'w') as f:
        json.dump(contract_interface['abi'], f, indent=2)
    print(f"📄 ABI saved to: {abi_file}")

    # Save bytecode
    bin_file = output_path / "ZeroTrustMLGradientStorage.bin"
    with open(bin_file, 'w') as f:
        f.write(contract_interface['bin'])
    print(f"📄 Bytecode saved to: {bin_file}")

    # Save metadata
    metadata_file = output_path / "ZeroTrustMLGradientStorage.metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(json.loads(contract_interface['metadata']), f, indent=2)
    print(f"📄 Metadata saved to: {metadata_file}")

    # Print summary
    print()
    print("=" * 60)
    print("📊 COMPILATION SUMMARY")
    print("=" * 60)
    print(f"Contract Functions: {len([item for item in contract_interface['abi'] if item['type'] == 'function'])}")
    print(f"Contract Events: {len([item for item in contract_interface['abi'] if item['type'] == 'event'])}")
    print(f"Bytecode Size: {len(contract_interface['bin']) // 2} bytes")
    print()

    # Show function list
    print("Public Functions:")
    for item in contract_interface['abi']:
        if item['type'] == 'function':
            inputs = ', '.join(f"{inp['type']} {inp['name']}" for inp in item.get('inputs', []))
            outputs = ', '.join(out['type'] for out in item.get('outputs', []))
            visibility = item.get('stateMutability', 'nonpayable')
            print(f"  - {item['name']}({inputs}) → {outputs} [{visibility}]")

    print()
    print("Events:")
    for item in contract_interface['abi']:
        if item['type'] == 'event':
            inputs = ', '.join(f"{inp['type']} {inp['name']}" for inp in item.get('inputs', []))
            print(f"  - {item['name']}({inputs})")

    print()
    print("=" * 60)

    return contract_interface


def generate_python_abi_constant(abi_file: str):
    """Generate Python constant from ABI JSON"""
    with open(abi_file, 'r') as f:
        abi = json.load(f)

    print("\n📝 Python ABI Constant:")
    print("=" * 60)
    print("CONTRACT_ABI = " + json.dumps(abi, indent=2))
    print("=" * 60)

    # Save to Python file
    python_file = Path("build") / "contract_abi.py"
    with open(python_file, 'w') as f:
        f.write('"""Generated ABI for ZeroTrustMLGradientStorage contract"""\n\n')
        f.write(f"CONTRACT_ABI = {json.dumps(abi, indent=2)}\n")

    print(f"\n✅ Python ABI saved to: {python_file}")


if __name__ == "__main__":
    contract_path = "contracts/ZeroTrustMLGradientStorage.sol"

    if not Path(contract_path).exists():
        print(f"❌ Contract not found: {contract_path}")
        sys.exit(1)

    # Compile
    result = compile_contract(contract_path)

    if result:
        # Generate Python constant
        generate_python_abi_constant("build/ZeroTrustMLGradientStorage.abi.json")

        print("\n🎉 Contract compilation complete!")
        print("\nNext steps:")
        print("1. Review build/ZeroTrustMLGradientStorage.abi.json")
        print("2. Deploy to Polygon Mumbai testnet")
        print("3. Update EthereumBackend with the ABI and contract address")
    else:
        print("\n❌ Compilation failed")
        sys.exit(1)
