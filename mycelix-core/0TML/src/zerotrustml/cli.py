# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML CLI - Command-line interface for ZeroTrustML Holochain

Provides commands for initializing, managing, and monitoring ZeroTrustML nodes.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from zerotrustml import __version__
from zerotrustml.core import Node, NodeConfig


def init_command(args):
    """Initialize a new ZeroTrustML node"""
    config_dir = Path.home() / ".zerotrustml"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "config.toml"

    if config_file.exists() and not args.force:
        print(f"❌ Configuration already exists at {config_file}")
        print("   Use --force to overwrite")
        return 1

    # Create configuration
    config_content = f"""# ZeroTrustML Node Configuration
node_id = "{args.node_id}"
data_path = "{args.data_path}"
model_type = "{args.model_type}"
holochain_url = "{args.holochain_url}"
aggregation = "{args.aggregation}"
batch_size = {args.batch_size}
learning_rate = {args.learning_rate}
"""

    config_file.write_text(config_content)

    # Create data directory if it doesn't exist
    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"✅ Initialized ZeroTrustML node: {args.node_id}")
    print(f"   Config: {config_file}")
    print(f"   Data:   {data_path}")
    print(f"\nNext steps:")
    print(f"  1. Review configuration: cat {config_file}")
    print(f"  2. Start node: zerotrustml start")

    return 0


def start_command(args):
    """Start the ZeroTrustML node"""
    config_dir = Path.home() / ".zerotrustml"
    config_file = config_dir / "config.toml"

    if not config_file.exists():
        print(f"❌ No configuration found at {config_file}")
        print("   Run 'zerotrustml init' first")
        return 1

    # Load configuration (simplified TOML parsing)
    config_data = {}
    for line in config_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"')
            config_data[key] = value

    # Create NodeConfig
    config = NodeConfig(
        node_id=config_data['node_id'],
        data_path=config_data['data_path'],
        model_type=config_data.get('model_type', 'resnet18'),
        holochain_url=config_data.get('holochain_url', 'ws://localhost:8888'),
        aggregation=config_data.get('aggregation', 'krum'),
        batch_size=int(config_data.get('batch_size', 32)),
        learning_rate=float(config_data.get('learning_rate', 0.01))
    )

    print(f"🚀 Starting ZeroTrustML node: {config.node_id}")
    print(f"   Holochain: {config.holochain_url}")
    print(f"   Aggregation: {config.aggregation}")

    # Create and start node
    node = Node(config)

    async def run_node():
        await node.start()
        print(f"\n✅ Node started successfully")
        print(f"   Accuracy: {node.accuracy:.2%}")
        print(f"   Credits:  {node.credits}")

        if args.rounds:
            print(f"\n🔄 Training for {args.rounds} rounds...")
            await node.train(rounds=args.rounds)
            print(f"\n✅ Training complete")
            print(f"   Final accuracy: {node.accuracy:.2%}")

    asyncio.run(run_node())
    return 0


def status_command(args):
    """Show node status"""
    config_dir = Path.home() / ".zerotrustml"
    config_file = config_dir / "config.toml"

    if not config_file.exists():
        print(f"❌ No configuration found")
        print("   Run 'zerotrustml init' first")
        return 1

    # Load config
    config_data = {}
    for line in config_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"')
            config_data[key] = value

    print(f"📊 ZeroTrustML Node Status")
    print(f"   Node ID:     {config_data['node_id']}")
    print(f"   Data Path:   {config_data['data_path']}")
    print(f"   Holochain:   {config_data.get('holochain_url', 'ws://localhost:8888')}")
    print(f"   Aggregation: {config_data.get('aggregation', 'krum')}")
    print(f"\n⚠️  Status check requires active node connection")
    print(f"   Run 'zerotrustml-node status' for live metrics")

    return 0


def credits_command(args):
    """Manage credits"""
    if args.action == 'balance':
        print(f"💰 Credits Balance")
        print(f"   Node:    {args.node_id or 'current'}")
        print(f"   Balance: ⚠️  Requires active node connection")
        print(f"\n   Run 'zerotrustml-node credits' for live balance")
        return 0

    print(f"❌ Unknown credits action: {args.action}")
    return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ZeroTrustML Holochain - Decentralized Federated Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zerotrustml init --node-id hospital-1 --data-path /data/medical-images
  zerotrustml start
  zerotrustml start --rounds 50
  zerotrustml status
  zerotrustml credits balance

For more information, visit: https://github.com/Luminous-Dynamics/0TML
"""
    )

    parser.add_argument('--version', action='version', version=f'ZeroTrustML {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize a new ZeroTrustML node')
    init_parser.add_argument('--node-id', required=True, help='Unique node identifier')
    init_parser.add_argument('--data-path', required=True, help='Path to training data')
    init_parser.add_argument('--model-type', default='resnet18', help='Model architecture')
    init_parser.add_argument('--holochain-url', default='ws://localhost:8888', help='Holochain conductor URL')
    init_parser.add_argument('--aggregation', default='krum', choices=['krum', 'fedavg', 'trimmed_mean', 'median'], help='Aggregation algorithm')
    init_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    init_parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    init_parser.add_argument('--force', action='store_true', help='Overwrite existing configuration')
    init_parser.set_defaults(func=init_command)

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the ZeroTrustML node')
    start_parser.add_argument('--rounds', type=int, help='Number of training rounds')
    start_parser.set_defaults(func=start_command)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show node status')
    status_parser.set_defaults(func=status_command)

    # Credits command
    credits_parser = subparsers.add_parser('credits', help='Manage credits')
    credits_parser.add_argument('action', choices=['balance'], help='Credits action')
    credits_parser.add_argument('--node-id', help='Node ID (default: current node)')
    credits_parser.set_defaults(func=credits_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if os.getenv('DEBUG'):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
