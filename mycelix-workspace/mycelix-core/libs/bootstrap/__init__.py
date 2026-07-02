# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Hierarchical Bootstrap Module
=====================================

Provides resilient peer discovery through a hierarchical bootstrap architecture
that eliminates single points of failure (SPOF).

Architecture:
    1. Primary bootstrap servers (Holo + Mycelix-operated)
    2. Secondary bootstrap servers (community-operated)
    3. Local peer cache (previously discovered peers)
    4. DHT-based peer discovery (gossip fallback)

Quick Start:
    from libs.bootstrap import HierarchicalBootstrapClient

    # Create client with default config
    client = HierarchicalBootstrapClient()

    # Discover peers
    peers = await client.discover_peers(space_hash="my-network")

    # Get current best bootstrap URL for conductor config
    bootstrap_url = await client.get_active_bootstrap_url()

Usage with conductor:
    from libs.bootstrap import generate_conductor_network_config

    network_config = generate_conductor_network_config()
    # Returns dict suitable for conductor-config.yaml network section
"""

from .client import (
    HierarchicalBootstrapClient,
    BootstrapServer,
    BootstrapTier,
    BootstrapResult,
    BootstrapError,
    CircuitBreakerOpen,
)

from .peer_cache import (
    PeerCache,
    CachedPeer,
    PeerCacheConfig,
)

from .config_loader import (
    load_bootstrap_config,
    BootstrapConfig,
    generate_conductor_network_config,
)

from .health import (
    BootstrapHealthChecker,
    ServerHealth,
    HealthStatus,
)

from .conductor_template import (
    generate_conductor_config,
    write_conductor_config,
    generate_multi_node_configs,
    render_conductor_config_template,
)

__all__ = [
    # Main client
    "HierarchicalBootstrapClient",
    "BootstrapServer",
    "BootstrapTier",
    "BootstrapResult",
    "BootstrapError",
    "CircuitBreakerOpen",

    # Peer cache
    "PeerCache",
    "CachedPeer",
    "PeerCacheConfig",

    # Configuration
    "load_bootstrap_config",
    "BootstrapConfig",
    "generate_conductor_network_config",

    # Health checking
    "BootstrapHealthChecker",
    "ServerHealth",
    "HealthStatus",

    # Conductor config generation
    "generate_conductor_config",
    "write_conductor_config",
    "generate_multi_node_configs",
    "render_conductor_config_template",
]

__version__ = "1.0.0"
