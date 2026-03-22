# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root"""
ZeroTrustML Monitoring CLI - Network monitoring and visualization

Provides dashboard and monitoring tools for ZeroTrustML networks.
"""

import argparse
import asyncio
import sys
from typing import List, Optional

from zerotrustml import __version__
from zerotrustml.monitoring import Dashboard, NetworkMonitor


def dashboard_command(args):
    """Start the monitoring dashboard"""
    print(f"📊 ZeroTrustML Dashboard")
    print(f"   Version: {__version__}")
    print(f"   Host:    {args.host}")
    print(f"   Port:    {args.port}")

    dashboard = Dashboard()

    async def run():
        await dashboard.start(host=args.host, port=args.port)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Dashboard stopped")

    return 0


def metrics_command(args):
    """Show network metrics"""
    print(f"📈 Network Metrics")
    print(f"\n⚠️  Live metrics require active network connection")
    print(f"\nThis would show:")
    print(f"  - Total nodes online")
    print(f"  - Average accuracy")
    print(f"  - Total gradients shared")
    print(f"  - Byzantine attacks detected")
    print(f"  - Credit distribution")
    print(f"  - Network throughput")

    return 0


def health_command(args):
    """Check network health"""
    print(f"🏥 Network Health Check")

    monitor = NetworkMonitor()
    health = monitor.get_network_health()

    print(f"\n   Status: {health['status']}")
    print(f"   Score:  {health['score']:.1%}")

    print(f"\n⚠️  Full health check requires active network")
    print(f"   Would check:")
    print(f"   - Node connectivity")
    print(f"   - DHT synchronization")
    print(f"   - Model convergence")
    print(f"   - Byzantine resistance")

    return 0


def topology_command(args):
    """Show network topology"""
    print(f"🌐 Network Topology")
    print(f"\n⚠️  Live topology requires active network connection")
    print(f"\nThis would show:")
    print(f"  - Node connections (graph)")
    print(f"  - Peer relationships")
    print(f"  - Geographic distribution")
    print(f"  - Communication patterns")

    return 0


def alerts_command(args):
    """Show network alerts"""
    print(f"🚨 Network Alerts")
    print(f"\n⚠️  Live alerts require active network connection")
    print(f"\nRecent alerts would include:")
    print(f"  - Byzantine attacks detected")
    print(f"  - Node failures")
    print(f"  - Model divergence")
    print(f"  - Performance degradation")

    return 0


def main():
    """Main entry point for zerotrustml-monitor CLI"""
    parser = argparse.ArgumentParser(
        description="ZeroTrustML Monitoring and Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  zerotrustml-monitor dashboard
  zerotrustml-monitor dashboard --host 127.0.0.1 --port 8000
  zerotrustml-monitor metrics
  zerotrustml-monitor health
  zerotrustml-monitor topology
  zerotrustml-monitor alerts

For more information, visit: https://github.com/Luminous-Dynamics/0TML
"""
    )

    parser.add_argument('--version', action='version', version=f'ZeroTrustML Monitor {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start monitoring dashboard')
    dashboard_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (use 0.0.0.0 for all interfaces)')
    dashboard_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    dashboard_parser.set_defaults(func=dashboard_command)

    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show network metrics')
    metrics_parser.set_defaults(func=metrics_command)

    # Health command
    health_parser = subparsers.add_parser('health', help='Check network health')
    health_parser.set_defaults(func=health_command)

    # Topology command
    topology_parser = subparsers.add_parser('topology', help='Show network topology')
    topology_parser.set_defaults(func=topology_command)

    # Alerts command
    alerts_parser = subparsers.add_parser('alerts', help='Show network alerts')
    alerts_parser.set_defaults(func=alerts_command)

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
