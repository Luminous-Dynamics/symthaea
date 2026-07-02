#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Command-Line Interface
Easy deployment for different use cases
"""

import argparse
import asyncio
import yaml
from pathlib import Path
from typing import Optional
# Import modular architecture
from zerotrustml.modular_architecture import (
    ZeroTrustMLCore,
    ZeroTrustMLFactory,
    UseCase,
    MemoryStorage,
    PostgreSQLStorage,
    HolochainStorage
)


class ZeroTrustMLCLI:
    """Command-line interface for ZeroTrustML"""

    def __init__(self):
        self.config: Optional[dict] = None
        self.node: Optional[ZeroTrustMLCore] = None

    def load_config(self, config_file: str, use_case: str) -> dict:
        """Load configuration from YAML file"""
        config_path = Path(config_file)

        if not config_path.exists():
            print(f"❌ Config file not found: {config_file}")
            sys.exit(1)

        with open(config_path) as f:
            all_configs = yaml.safe_load(f)

        if use_case not in all_configs:
            print(f"❌ Use case '{use_case}' not found in config")
            print(f"Available: {', '.join(all_configs.keys())}")
            sys.exit(1)

        return all_configs[use_case]

    def create_node(self, node_id: int, config: dict) -> ZeroTrustMLCore:
        """Create ZeroTrustML node from configuration"""
        use_case_str = config['use_case']
        use_case = UseCase(use_case_str)

        # Create storage backend
        storage_config = config['storage']
        backend = storage_config['backend']

        if backend == 'memory':
            storage = MemoryStorage()
        elif backend == 'postgresql':
            import os
            conn_string = storage_config.get('connection_string', 'postgresql://localhost/zerotrustml')
            # Expand environment variables
            if conn_string.startswith('${'):
                var_name = conn_string[2:-1]
                conn_string = os.getenv(var_name, 'postgresql://localhost/zerotrustml')
            storage = PostgreSQLStorage(conn_string)
        elif backend == 'holochain':
            import os
            conductor_url = storage_config.get('conductor_url', 'http://localhost:8888')
            if conductor_url.startswith('${'):
                var_name = conductor_url[2:-1]
                conductor_url = os.getenv(var_name, 'http://localhost:8888')
            storage = HolochainStorage(conductor_url)
        else:
            print(f"❌ Unknown storage backend: {backend}")
            sys.exit(1)

        # Create node
        node = ZeroTrustMLCore(
            node_id=node_id,
            use_case=use_case,
            storage_backend=storage,
            enable_async_checkpointing=storage_config.get('checkpoint_async', True)
        )

        return node

    async def start(self, args):
        """Start a ZeroTrustML node"""
        print("\n" + "="*60)
        print("🚀 STARTING ZEROTRUSTML NODE")
        print("="*60)

        # Load configuration
        config = self.load_config(args.config, args.use_case)
        self.config = config

        print(f"\n📋 Configuration:")
        print(f"   Use Case: {config['use_case']}")
        print(f"   Storage: {config['storage']['backend']}")
        print(f"   Node ID: {args.node_id}")

        # Create node
        self.node = self.create_node(args.node_id, config)

        print(f"\n✅ Node {args.node_id} ready!")
        print(f"\nPress Ctrl+C to stop...")

        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Shutting down...")
            await self.node.shutdown()
            print("✅ Shutdown complete")

    def test(self, args):
        """Run quick test"""
        print("\n" + "="*60)
        print("🧪 RUNNING QUICK TEST")
        print("="*60)

        config = self.load_config(args.config, args.use_case)

        async def run_test():
            node = self.create_node(999, config)

            print("\n📊 Testing gradient validation...")
            import numpy as np

            for i in range(5):
                gradient = np.random.randn(100)
                is_valid = await node.validate_gradient(
                    gradient,
                    peer_id=i,
                    round_num=1
                )
                print(f"   Gradient {i}: {'✅ Valid' if is_valid else '❌ Invalid'}")

            await node.shutdown()
            print("\n✅ Test complete!")

        asyncio.run(run_test())

    def info(self, args):
        """Show information about use cases"""
        print("\n" + "="*60)
        print("📚 ZEROTRUSTML USE CASES")
        print("="*60)

        use_cases = {
            'research': {
                'description': 'Lightweight, no persistence',
                'storage': 'Memory',
                'suited_for': 'Academic research, prototyping'
            },
            'warehouse': {
                'description': 'Traditional database',
                'storage': 'PostgreSQL',
                'suited_for': 'Warehouse automation, logistics'
            },
            'automotive': {
                'description': 'Immutable audit trail',
                'storage': 'Holochain DHT',
                'suited_for': 'Self-driving vehicles, safety-critical'
            },
            'medical': {
                'description': 'HIPAA compliant',
                'storage': 'Holochain DHT (encrypted)',
                'suited_for': 'Hospital collaboration, clinical trials'
            },
            'finance': {
                'description': 'SEC/FinCEN compliant',
                'storage': 'Holochain DHT',
                'suited_for': 'Fraud detection, risk modeling'
            },
            'drone_swarm': {
                'description': 'Lightweight checkpointing',
                'storage': 'Holochain DHT (sparse)',
                'suited_for': 'Drone coordination, edge devices'
            },
            'manufacturing': {
                'description': 'Multi-vendor coordination',
                'storage': 'PostgreSQL',
                'suited_for': 'Industry 4.0, supply chain'
            }
        }

        for name, info in use_cases.items():
            print(f"\n🔹 {name.upper()}")
            print(f"   Description: {info['description']}")
            print(f"   Storage: {info['storage']}")
            print(f"   Suited for: {info['suited_for']}")

    def compare(self, args):
        """Compare storage backends"""
        print("\n" + "="*60)
        print("⚖️  STORAGE BACKEND COMPARISON")
        print("="*60)

        print("\n┌─────────────┬──────────┬───────────┬────────────┬──────────┐")
        print("│ Backend     │ Speed    │ Audit     │ Compliance │ Cost     │")
        print("├─────────────┼──────────┼───────────┼────────────┼──────────┤")
        print("│ Memory      │ ⚡⚡⚡⚡  │ ❌        │ ❌         │ Free     │")
        print("│ PostgreSQL  │ ⚡⚡⚡    │ ✅        │ ⚠️         │ Low      │")
        print("│ Holochain   │ ⚡⚡      │ ✅✅✅    │ ✅✅       │ Medium   │")
        print("└─────────────┴──────────┴───────────┴────────────┴──────────┘")

        print("\n🔹 Memory Storage")
        print("   ✅ Fastest (no I/O)")
        print("   ✅ Zero cost")
        print("   ❌ No persistence")
        print("   ❌ No audit trail")
        print("   👉 Use for: Research, testing, prototyping")

        print("\n🔹 PostgreSQL Storage")
        print("   ✅ Fast (indexed queries)")
        print("   ✅ Reliable and mature")
        print("   ✅ Good for operational data")
        print("   ⚠️  Mutable (can be edited)")
        print("   ⚠️  Centralized (single point of failure)")
        print("   👉 Use for: Warehouses, manufacturing, internal use")

        print("\n🔹 Holochain Storage")
        print("   ✅ Immutable audit trail")
        print("   ✅ Tamper-evident")
        print("   ✅ Decentralized (no central authority)")
        print("   ✅ Regulatory compliance ready")
        print("   ⚠️  Slower writes (DHT consensus)")
        print("   ⚠️  More complex deployment")
        print("   👉 Use for: Automotive, medical, finance, drones")

        print("\n💡 Decision Guide:")
        print("   • Need audit trail for regulators? → Holochain")
        print("   • Multiple parties don't trust each other? → Holochain")
        print("   • Single organization, operational data? → PostgreSQL")
        print("   • Research or testing? → Memory")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='ZeroTrustML - Byzantine-resistant Federated Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a research node
  zerotrustml start --use-case research --node-id 1

  # Start warehouse robotics node
  zerotrustml start --use-case warehouse --node-id 42

  # Start autonomous vehicle node
  export HOLOCHAIN_CONDUCTOR=http://localhost:8888
  zerotrustml start --use-case automotive --node-id 1001

  # Run quick test
  zerotrustml test --use-case medical

  # Compare storage backends
  zerotrustml compare

  # Show all use cases
  zerotrustml info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start a ZeroTrustML node')
    start_parser.add_argument('--use-case', required=True, help='Use case (research, warehouse, automotive, etc.)')
    start_parser.add_argument('--node-id', type=int, required=True, help='Node ID')
    start_parser.add_argument('--config', default='zerotrustml.yaml', help='Config file path')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run quick test')
    test_parser.add_argument('--use-case', required=True, help='Use case to test')
    test_parser.add_argument('--config', default='zerotrustml.yaml', help='Config file path')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show use case information')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare storage backends')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cli = ZeroTrustMLCLI()

    if args.command == 'start':
        asyncio.run(cli.start(args))
    elif args.command == 'test':
        cli.test(args)
    elif args.command == 'info':
        cli.info(args)
    elif args.command == 'compare':
        cli.compare(args)


if __name__ == '__main__':
    main()
