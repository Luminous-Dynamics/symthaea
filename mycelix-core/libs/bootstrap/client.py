# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Hierarchical Bootstrap Client
=============================

Provides resilient peer discovery through a multi-tier bootstrap architecture.
Eliminates single points of failure by supporting multiple bootstrap sources
with intelligent fallback and caching.

Architecture:
    1. Primary servers (official Holo + Mycelix-operated)
    2. Secondary servers (community-operated)
    3. Tertiary servers (experimental/dev)
    4. Local peer cache (previously discovered peers)
    5. DHT-based discovery (gossip fallback)
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import json

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from .config_loader import (
    BootstrapConfig,
    ServerConfig,
    load_bootstrap_config,
    generate_conductor_network_config,
)
from .peer_cache import PeerCache, CachedPeer, PeerCacheConfig
from .health import BootstrapHealthChecker, HealthStatus, CircuitState

logger = logging.getLogger("mycelix.bootstrap.client")


class BootstrapTier(Enum):
    """Bootstrap server tier."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    LOCAL_CACHE = "local_cache"
    DHT_DISCOVERY = "dht_discovery"


class BootstrapError(Exception):
    """Base exception for bootstrap errors."""
    pass


class CircuitBreakerOpen(BootstrapError):
    """Raised when all circuit breakers are open."""
    pass


class AllServersExhausted(BootstrapError):
    """Raised when all bootstrap servers have been tried without success."""
    pass


@dataclass
class BootstrapServer:
    """Represents a bootstrap server with its state."""
    url: str
    tier: BootstrapTier
    weight: int = 100
    region: str = "global"
    timeout_ms: int = 5000
    healthy: bool = True
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    failure_count: int = 0


@dataclass
class DiscoveredPeer:
    """A peer discovered from bootstrap."""
    agent_pub_key: str
    urls: List[str]
    signed_at_ms: int
    expires_at_ms: int
    space_hash: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BootstrapResult:
    """Result of a bootstrap attempt."""
    success: bool
    peers: List[DiscoveredPeer]
    source: BootstrapTier
    source_url: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class HierarchicalBootstrapClient:
    """
    Multi-tier bootstrap client with intelligent fallback.

    Features:
    - Tries servers in order of tier and weight
    - Parallel requests within a tier for faster discovery
    - Circuit breaker pattern for failing servers
    - Local peer cache for offline operation
    - DHT gossip fallback for decentralized discovery
    """

    def __init__(
        self,
        config: Optional[BootstrapConfig] = None,
        peer_cache: Optional[PeerCache] = None,
        health_checker: Optional[BootstrapHealthChecker] = None,
    ):
        """
        Initialize the hierarchical bootstrap client.

        Args:
            config: Bootstrap configuration (loads default if not provided)
            peer_cache: Peer cache instance (creates new if not provided)
            health_checker: Health checker instance (creates new if not provided)
        """
        self.config = config or load_bootstrap_config()
        self.peer_cache = peer_cache or PeerCache(
            PeerCacheConfig(
                enabled=self.config.peer_cache.enabled,
                path=self.config.peer_cache.path,
                max_peers=self.config.peer_cache.max_peers,
                max_age_hours=self.config.peer_cache.max_age_hours,
            )
        )
        self.health_checker = health_checker or BootstrapHealthChecker(self.config)

        self._servers: Dict[BootstrapTier, List[BootstrapServer]] = {
            BootstrapTier.PRIMARY: [],
            BootstrapTier.SECONDARY: [],
            BootstrapTier.TERTIARY: [],
        }

        # Build server list from config
        self._init_servers()

        # Metrics
        self._total_discoveries = 0
        self._successful_discoveries = 0
        self._tier_success_counts: Dict[BootstrapTier, int] = {}
        self._active_bootstrap_url: Optional[str] = None

    def _init_servers(self) -> None:
        """Initialize server list from configuration."""
        for server_config in self.config.primary:
            self._servers[BootstrapTier.PRIMARY].append(BootstrapServer(
                url=server_config.url,
                tier=BootstrapTier.PRIMARY,
                weight=server_config.weight,
                region=server_config.region,
                timeout_ms=server_config.timeout_ms,
            ))

        for server_config in self.config.secondary:
            self._servers[BootstrapTier.SECONDARY].append(BootstrapServer(
                url=server_config.url,
                tier=BootstrapTier.SECONDARY,
                weight=server_config.weight,
                region=server_config.region,
                timeout_ms=server_config.timeout_ms,
            ))

        for server_config in self.config.tertiary:
            self._servers[BootstrapTier.TERTIARY].append(BootstrapServer(
                url=server_config.url,
                tier=BootstrapTier.TERTIARY,
                weight=server_config.weight,
                region=server_config.region,
                timeout_ms=server_config.timeout_ms,
            ))

        # Sort each tier by weight (descending)
        for tier in self._servers:
            self._servers[tier].sort(key=lambda s: s.weight, reverse=True)

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff delay with jitter."""
        retry = self.config.retry_strategy
        base_delay = retry.initial_backoff_ms * (retry.backoff_multiplier ** attempt)
        delay = min(base_delay, retry.max_backoff_ms)

        # Add jitter
        jitter = delay * retry.jitter_factor * (random.random() * 2 - 1)
        return (delay + jitter) / 1000  # Convert to seconds

    async def _fetch_peers_from_server(
        self,
        server: BootstrapServer,
        space_hash: str,
        limit: int = 16,
    ) -> Optional[List[DiscoveredPeer]]:
        """
        Fetch peers from a single bootstrap server.

        Args:
            server: Bootstrap server to query
            space_hash: Network/space hash
            limit: Maximum peers to request

        Returns:
            List of discovered peers, or None on failure
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available")
            return None

        # Check if server is available
        if not self.health_checker.is_server_available(server.url):
            logger.debug(f"Skipping unavailable server: {server.url}")
            return None

        timeout = aiohttp.ClientTimeout(total=server.timeout_ms / 1000)
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Construct the bootstrap request
                # Holo bootstrap API format
                request_url = f"{server.url.rstrip('/')}/bootstrap"
                payload = {
                    "space": space_hash,
                    "limit": limit,
                }

                async with session.post(request_url, json=payload) as response:
                    latency_ms = (time.time() - start_time) * 1000

                    if response.status != 200:
                        logger.warning(
                            f"Bootstrap server {server.url} returned {response.status}"
                        )
                        server.failure_count += 1
                        server.last_failure = time.time()
                        return None

                    data = await response.json()

                    # Parse response
                    peers = []
                    for peer_data in data.get("peers", []):
                        try:
                            peer = DiscoveredPeer(
                                agent_pub_key=peer_data["agent_pub_key"],
                                urls=peer_data.get("urls", []),
                                signed_at_ms=peer_data.get("signed_at_ms", 0),
                                expires_at_ms=peer_data.get("expires_at_ms", 0),
                                space_hash=space_hash,
                                meta=peer_data.get("meta", {}),
                            )
                            peers.append(peer)
                        except (KeyError, TypeError) as e:
                            logger.warning(f"Invalid peer data: {e}")

                    # Update server state
                    server.healthy = True
                    server.last_success = time.time()
                    server.failure_count = 0

                    logger.info(
                        f"Discovered {len(peers)} peers from {server.url} "
                        f"(latency: {latency_ms:.1f}ms)"
                    )

                    return peers

        except asyncio.TimeoutError:
            logger.warning(f"Timeout connecting to {server.url}")
            server.failure_count += 1
            server.last_failure = time.time()
            return None

        except aiohttp.ClientError as e:
            logger.warning(f"HTTP error from {server.url}: {e}")
            server.failure_count += 1
            server.last_failure = time.time()
            return None

        except Exception as e:
            logger.exception(f"Unexpected error from {server.url}")
            server.failure_count += 1
            server.last_failure = time.time()
            return None

    async def _discover_from_tier(
        self,
        tier: BootstrapTier,
        space_hash: str,
        limit: int = 16,
    ) -> BootstrapResult:
        """
        Attempt discovery from all servers in a tier.

        Uses parallel requests if configured.
        """
        servers = self._servers.get(tier, [])
        if not servers:
            return BootstrapResult(
                success=False,
                peers=[],
                source=tier,
                error="No servers configured for tier",
            )

        # Filter to healthy servers
        available_servers = [s for s in servers if s.healthy or s.failure_count < 5]
        if not available_servers:
            return BootstrapResult(
                success=False,
                peers=[],
                source=tier,
                error="All servers in tier are unhealthy",
            )

        start_time = time.time()

        if self.config.parallel.enabled:
            # Parallel discovery
            max_concurrent = min(
                self.config.parallel.max_concurrent,
                len(available_servers)
            )

            tasks = [
                self._fetch_peers_from_server(server, space_hash, limit)
                for server in available_servers[:max_concurrent]
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Find first successful result
            for i, result in enumerate(results):
                if isinstance(result, list) and result:
                    server = available_servers[i]
                    return BootstrapResult(
                        success=True,
                        peers=result,
                        source=tier,
                        source_url=server.url,
                        latency_ms=(time.time() - start_time) * 1000,
                    )

        else:
            # Sequential discovery with retry
            for server in available_servers:
                for attempt in range(self.config.retry_strategy.max_attempts_per_server):
                    if attempt > 0:
                        await asyncio.sleep(self._calculate_backoff(attempt))

                    peers = await self._fetch_peers_from_server(server, space_hash, limit)
                    if peers:
                        return BootstrapResult(
                            success=True,
                            peers=peers,
                            source=tier,
                            source_url=server.url,
                            latency_ms=(time.time() - start_time) * 1000,
                        )

        return BootstrapResult(
            success=False,
            peers=[],
            source=tier,
            latency_ms=(time.time() - start_time) * 1000,
            error=f"All servers in {tier.value} tier failed",
        )

    async def _discover_from_cache(self, space_hash: str, limit: int = 16) -> BootstrapResult:
        """Get peers from local cache."""
        cached_peers = self.peer_cache.get_peers(space_hash, limit=limit)

        if cached_peers:
            peers = [
                DiscoveredPeer(
                    agent_pub_key=p.agent_pub_key,
                    urls=p.urls,
                    signed_at_ms=int(p.last_seen * 1000),
                    expires_at_ms=int((p.last_seen + 86400) * 1000),  # +24h
                    space_hash=space_hash,
                )
                for p in cached_peers
            ]

            return BootstrapResult(
                success=True,
                peers=peers,
                source=BootstrapTier.LOCAL_CACHE,
            )

        return BootstrapResult(
            success=False,
            peers=[],
            source=BootstrapTier.LOCAL_CACHE,
            error="No cached peers available",
        )

    async def discover_peers(
        self,
        space_hash: str,
        limit: int = 16,
        include_cache: bool = True,
    ) -> BootstrapResult:
        """
        Discover peers using the hierarchical bootstrap strategy.

        Tries tiers in order:
        1. Primary servers
        2. Secondary servers
        3. Tertiary servers
        4. Local cache
        5. DHT discovery (if enabled)

        Args:
            space_hash: Network/space hash to find peers for
            limit: Maximum number of peers to return
            include_cache: Whether to try local cache as fallback

        Returns:
            BootstrapResult with discovered peers
        """
        self._total_discoveries += 1
        start_time = time.time()

        # Try tiers in order
        tiers = [BootstrapTier.PRIMARY, BootstrapTier.SECONDARY, BootstrapTier.TERTIARY]

        for tier in tiers:
            for tier_attempt in range(self.config.retry_strategy.max_tier_retries):
                if tier_attempt > 0:
                    await asyncio.sleep(self.config.retry_strategy.tier_backoff_ms / 1000)

                result = await self._discover_from_tier(tier, space_hash, limit)

                if result.success:
                    self._successful_discoveries += 1
                    self._tier_success_counts[tier] = self._tier_success_counts.get(tier, 0) + 1
                    self._active_bootstrap_url = result.source_url

                    # Update cache with discovered peers
                    if self.peer_cache.config.enabled:
                        for peer in result.peers:
                            self.peer_cache.add_peer(
                                agent_pub_key=peer.agent_pub_key,
                                urls=peer.urls,
                                space_hash=space_hash,
                            )

                    logger.info(
                        f"Bootstrap success from {tier.value} tier "
                        f"({len(result.peers)} peers in {result.latency_ms:.1f}ms)"
                    )
                    return result

        # All remote servers failed, try cache
        if include_cache and self.config.peer_cache.enabled:
            result = await self._discover_from_cache(space_hash, limit)
            if result.success:
                self._successful_discoveries += 1
                self._tier_success_counts[BootstrapTier.LOCAL_CACHE] = \
                    self._tier_success_counts.get(BootstrapTier.LOCAL_CACHE, 0) + 1

                logger.info(f"Using {len(result.peers)} cached peers (all remote servers failed)")
                return result

        # Complete failure
        total_time = (time.time() - start_time) * 1000
        logger.error(f"Bootstrap failed: all sources exhausted ({total_time:.1f}ms)")

        return BootstrapResult(
            success=False,
            peers=[],
            source=BootstrapTier.PRIMARY,
            latency_ms=total_time,
            error="All bootstrap sources exhausted",
        )

    async def get_active_bootstrap_url(self) -> str:
        """
        Get the currently active/best bootstrap URL.

        Useful for generating conductor configs.
        """
        # Return last successful URL if available
        if self._active_bootstrap_url:
            return self._active_bootstrap_url

        # Get best from health checker
        best = self.health_checker.get_best_server()
        if best:
            return best

        # Fall back to first configured primary
        if self.config.primary:
            return self.config.primary[0].url

        return "https://bootstrap-staging.holo.host"

    async def get_active_signal_url(self) -> str:
        """Get the currently active/best signal URL."""
        if self.config.signal.primary:
            return self.config.signal.primary[0].url
        return "wss://signal.holo.host"

    def get_conductor_network_config(self) -> Dict[str, Any]:
        """
        Generate conductor-config.yaml network section.

        Uses the currently best bootstrap and signal URLs.
        """
        return generate_conductor_network_config(
            config=self.config,
            active_bootstrap_url=self._active_bootstrap_url,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_discoveries": self._total_discoveries,
            "successful_discoveries": self._successful_discoveries,
            "success_rate": (
                self._successful_discoveries / self._total_discoveries
                if self._total_discoveries > 0 else 0
            ),
            "tier_success_counts": {
                tier.value: count
                for tier, count in self._tier_success_counts.items()
            },
            "active_bootstrap_url": self._active_bootstrap_url,
            "cache_stats": self.peer_cache.stats(),
            "health_status": self.health_checker.get_status(),
        }

    async def start(self) -> None:
        """Start background services (health checking)."""
        if self.config.health_check.enabled:
            await self.health_checker.start()

    async def stop(self) -> None:
        """Stop background services."""
        await self.health_checker.stop()
        self.peer_cache.flush()

    async def __aenter__(self) -> "HierarchicalBootstrapClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.stop()


# Convenience function for simple usage
async def discover_peers(
    space_hash: str,
    limit: int = 16,
    config_path: Optional[str] = None,
) -> List[DiscoveredPeer]:
    """
    Simple function to discover peers.

    Example:
        peers = await discover_peers("my-network-hash")
        for peer in peers:
            print(f"Found peer: {peer.agent_pub_key}")
    """
    config = load_bootstrap_config(config_path) if config_path else None
    client = HierarchicalBootstrapClient(config=config)

    try:
        result = await client.discover_peers(space_hash, limit)
        return result.peers
    finally:
        await client.stop()
