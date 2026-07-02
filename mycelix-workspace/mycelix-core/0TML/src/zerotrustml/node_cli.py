# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Node CLI - Advanced node management and monitoring

Provides detailed control and monitoring for ZeroTrustML nodes.
"""

import argparse
import asyncio
import sys

from zerotrustml import __version__
from zerotrustml.core import Node, NodeConfig


def start_node(args):
    """Start node with full control"""
    print(f"🚀 Starting ZeroTrustML Node")
    print(f"   Node ID: {args.node_id}")
    print(f"   Data:    {args.data_path}")
    print(f"   Model:   {args.model_type}")

    config = NodeConfig(
        node_id=args.node_id,
        data_path=args.data_path,
        model_type=args.model_type,
        holochain_url=args.holochain_url,
        aggregation=args.aggregation,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    node = Node(config)

    async def run():
        await node.start()
        print(f"\n✅ Node running")
        print(f"   Accuracy: {node.accuracy:.2%}")
        print(f"   Credits:  {node.credits}")

        # Keep running
        if not args.daemon:
            print(f"\n💡 Press Ctrl+C to stop")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Shutting down...")

    asyncio.run(run())
    return 0


def node_status(args):
    """Show detailed node status"""
    print(f"📊 ZeroTrustML Node Status")
    print(f"   Version: {__version__}")
    print(f"\n⚠️  Live status requires active connection")
    print(f"   This would show:")
    print(f"   - Current round")
    print(f"   - Model accuracy")
    print(f"   - Credit balance")
    print(f"   - Peer connections")
    print(f"   - Recent gradients")
    return 0


def node_train(args):
    """Train the node"""
    print(f"🔄 Training Node")
    print(f"   Rounds: {args.rounds}")
    print(f"   Model:  {args.model_type}")

    config = NodeConfig(
        node_id=args.node_id,
        data_path=args.data_path,
        model_type=args.model_type,
        holochain_url=args.holochain_url
    )

    node = Node(config)

    async def run():
        await node.start()
        print(f"\n🔄 Starting training...")
        await node.train(rounds=args.rounds)
        print(f"\n✅ Training complete")
        print(f"   Final accuracy: {node.accuracy:.2%}")

    asyncio.run(run())
    return 0


def node_credits(args):
    """Show credit balance and history"""
    print(f"💰 ZeroTrustML Credits")
    print(f"   Node: {args.node_id or 'current'}")
    print(f"\n⚠️  Live credits require active connection")
    print(f"   This would show:")
    print(f"   - Current balance")
    print(f"   - Recent transactions")
    print(f"   - Reputation level")
    print(f"   - Pending rewards")
    return 0


def node_peers(args):
    """Show connected peers"""
    print(f"🌐 Connected Peers")
    print(f"\n⚠️  Live peer list requires active connection")
    print(f"   This would show:")
    print(f"   - Peer node IDs")
    print(f"   - Reputation scores")
    print(f"   - Recent gradients")
    print(f"   - Network topology")
    return 0


def main():
    """Main entry point for zerotrustml-node CLI"""
    parser = argparse.ArgumentParser(
        description="ZeroTrustML Node Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zerotrustml-node start --node-id hospital-1 --data-path /data
  zerotrustml-node status
  zerotrustml-node train --rounds 50
  zerotrustml-node credits
  zerotrustml-node peers

For more information, visit: https://github.com/Luminous-Dynamics/0TML
"""
    )

    parser.add_argument('--version', action='version', version=f'ZeroTrustML Node {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the node')
    start_parser.add_argument('--node-id', required=True, help='Node identifier')
    start_parser.add_argument('--data-path', required=True, help='Path to training data')
    start_parser.add_argument('--model-type', default='resnet18', help='Model architecture')
    start_parser.add_argument('--holochain-url', default='ws://localhost:8888', help='Holochain URL')
    start_parser.add_argument('--aggregation', default='krum', help='Aggregation algorithm')
    start_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    start_parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    start_parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    start_parser.set_defaults(func=start_node)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show node status')
    status_parser.set_defaults(func=node_status)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the node')
    train_parser.add_argument('--node-id', required=True, help='Node identifier')
    train_parser.add_argument('--data-path', required=True, help='Path to training data')
    train_parser.add_argument('--model-type', default='resnet18', help='Model architecture')
    train_parser.add_argument('--holochain-url', default='ws://localhost:8888', help='Holochain URL')
    train_parser.add_argument('--rounds', type=int, required=True, help='Training rounds')
    train_parser.set_defaults(func=node_train)

    # Credits command
    credits_parser = subparsers.add_parser('credits', help='Show credits')
    credits_parser.add_argument('--node-id', help='Node identifier')
    credits_parser.set_defaults(func=node_credits)

    # Peers command
    peers_parser = subparsers.add_parser('peers', help='Show connected peers')
    peers_parser.set_defaults(func=node_peers)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import os
        if os.getenv('DEBUG'):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
