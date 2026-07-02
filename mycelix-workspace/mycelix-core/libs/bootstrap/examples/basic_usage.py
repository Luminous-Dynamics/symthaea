#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Hierarchical Bootstrap - Basic Usage Example
=============================================

This example demonstrates how to use the hierarchical bootstrap system
for resilient peer discovery.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from libs.bootstrap import (
    HierarchicalBootstrapClient,
    load_bootstrap_config,
    generate_conductor_config,
    write_conductor_config,
    PeerCache,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def basic_peer_discovery():
    """Basic example: discover peers in a network."""
    logger.info("=== Basic Peer Discovery ===")

    # Create client with default configuration
    async with HierarchicalBootstrapClient() as client:
        # Discover peers for a network space
        space_hash = "uhC0k_example_space_hash_here"

        result = await client.discover_peers(
            space_hash=space_hash,
            limit=10,
        )

        if result.success:
            logger.info(f"Found {len(result.peers)} peers from {result.source.value}")
            for peer in result.peers:
                logger.info(f"  - {peer.agent_pub_key[:20]}... ({len(peer.urls)} URLs)")
        else:
            logger.warning(f"Discovery failed: {result.error}")

        # Get statistics
        stats = client.get_stats()
        logger.info(f"Bootstrap stats: {stats['successful_discoveries']}/{stats['total_discoveries']} successful")


async def custom_config_example():
    """Example with custom bootstrap configuration."""
    logger.info("=== Custom Configuration ===")

    # Load configuration from custom path
    config = load_bootstrap_config(
        config_path=Path(__file__).parent.parent / "bootstrap_config.yaml"
    )

    logger.info(f"Loaded config version: {config.version}")
    logger.info(f"Primary servers: {len(config.primary)}")
    logger.info(f"Secondary servers: {len(config.secondary)}")
    logger.info(f"Peer cache enabled: {config.peer_cache.enabled}")

    # Create client with custom config
    client = HierarchicalBootstrapClient(config=config)

    try:
        await client.start()

        # Get best bootstrap URL
        best_url = await client.get_active_bootstrap_url()
        logger.info(f"Best bootstrap URL: {best_url}")

    finally:
        await client.stop()


async def peer_cache_example():
    """Example of using the peer cache directly."""
    logger.info("=== Peer Cache Example ===")

    # Create peer cache
    cache = PeerCache()

    # Add some peers
    cache.add_peer(
        agent_pub_key="uhCAk_example_agent_key_1",
        urls=["wss://example1.com:8888"],
        space_hash="uhC0k_example_space",
        region="us-west",
    )

    cache.add_peer(
        agent_pub_key="uhCAk_example_agent_key_2",
        urls=["wss://example2.com:8888"],
        space_hash="uhC0k_example_space",
        region="eu-west",
    )

    # Record success/failure
    cache.record_success("uhCAk_example_agent_key_1", "uhC0k_example_space")
    cache.record_failure("uhCAk_example_agent_key_2", "uhC0k_example_space")

    # Query cached peers
    peers = cache.get_peers("uhC0k_example_space", limit=10)
    logger.info(f"Cached peers: {len(peers)}")
    for peer in peers:
        logger.info(f"  - {peer.agent_pub_key[:30]}... (success rate: {peer.success_rate:.2f})")

    # Get stats
    stats = cache.stats()
    logger.info(f"Cache stats: {stats}")

    # Flush to disk
    cache.flush()


async def conductor_config_example():
    """Example of generating conductor configuration."""
    logger.info("=== Conductor Config Generation ===")

    # Generate conductor config with hierarchical bootstrap
    config = generate_conductor_config(
        environment_path=".hc-example",
        admin_port=39330,
        app_port=9999,
        use_test_keystore=True,
    )

    logger.info("Generated conductor config:")
    logger.info(f"  Bootstrap URL: {config['network']['bootstrap_url']}")
    logger.info(f"  Signal URL: {config['network']['transport_pool'][0]['signal_url']}")
    logger.info(f"  Admin port: {config['admin_interfaces'][0]['driver']['port']}")

    # Write to file (uncomment to actually write)
    # write_conductor_config(
    #     output_path="./conductor-config-generated.yaml",
    #     environment_path=".hc-example",
    #     admin_port=39330,
    #     app_port=9999,
    # )


async def health_check_example():
    """Example of monitoring bootstrap server health."""
    logger.info("=== Health Check Example ===")

    config = load_bootstrap_config()
    client = HierarchicalBootstrapClient(config=config)

    try:
        await client.start()

        # Wait for initial health checks
        await asyncio.sleep(2)

        # Get health status
        status = client.health_checker.get_status()

        logger.info(f"Total servers: {status['total_servers']}")
        logger.info(f"Healthy: {status['healthy']}")
        logger.info(f"Degraded: {status['degraded']}")
        logger.info(f"Unhealthy: {status['unhealthy']}")
        logger.info(f"Best server: {status['best_server']}")

        # Check individual servers
        for url, server_status in status['servers'].items():
            logger.info(f"  {url}: {server_status['status']} (circuit: {server_status['circuit_state']})")

    finally:
        await client.stop()


async def main():
    """Run all examples."""
    try:
        await basic_peer_discovery()
        print()

        await custom_config_example()
        print()

        await peer_cache_example()
        print()

        await conductor_config_example()
        print()

        await health_check_example()

    except Exception as e:
        logger.exception(f"Example failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
