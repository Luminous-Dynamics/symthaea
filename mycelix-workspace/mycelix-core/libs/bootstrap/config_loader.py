# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Bootstrap Configuration Loader
==============================

Loads and validates the hierarchical bootstrap configuration.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

logger = logging.getLogger("mycelix.bootstrap.config")


@dataclass
class ServerConfig:
    """Configuration for a single bootstrap server."""
    url: str
    weight: int = 100
    region: str = "global"
    timeout_ms: int = 5000
    operator: str = "official"


@dataclass
class RetryConfig:
    """Retry strategy configuration."""
    max_attempts_per_server: int = 3
    initial_backoff_ms: int = 1000
    max_backoff_ms: int = 30000
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1
    max_tier_retries: int = 2
    tier_backoff_ms: int = 5000


@dataclass
class PeerCacheConfig:
    """Local peer cache configuration."""
    enabled: bool = True
    path: str = "~/.mycelix/peer_cache.json"
    max_peers: int = 500
    max_age_hours: int = 168  # 7 days
    min_peers_for_offline: int = 5
    refresh_interval_hours: int = 1


@dataclass
class DHTDiscoveryConfig:
    """DHT-based peer discovery configuration."""
    enabled: bool = True
    max_discovery_time_ms: int = 30000
    min_peers_before_ready: int = 3
    gossip_bootstrap_peers: int = 5


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    enabled: bool = True
    interval_seconds: int = 60
    timeout_ms: int = 3000
    failure_threshold: int = 3
    recovery_threshold: int = 2


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    enabled: bool = True
    failure_threshold: int = 5
    reset_timeout_seconds: int = 300
    half_open_requests: int = 3


@dataclass
class ParallelConfig:
    """Parallel discovery configuration."""
    enabled: bool = True
    max_concurrent: int = 3
    early_success_threshold: int = 1


@dataclass
class NetworkTuningConfig:
    """Network tuning parameters."""
    gossip_loop_iteration_delay_ms: int = 1000
    default_notify_remote_agent_count: int = 5
    default_notify_timeout_ms: int = 10000
    default_rpc_single_timeout_ms: int = 10000
    default_rpc_multi_remote_agent_count: int = 5
    default_rpc_multi_timeout_ms: int = 10000
    agent_info_expires_after_ms: int = 1200000
    tls_in_mem_session_storage: int = 512
    proxy_keepalive_ms: int = 120000
    proxy_to_expire_ms: int = 300000


@dataclass
class SignalConfig:
    """Signal server configuration."""
    primary: List[ServerConfig] = field(default_factory=list)
    secondary: List[ServerConfig] = field(default_factory=list)
    retry_max_attempts: int = 3
    retry_backoff_ms: int = 2000


@dataclass
class BootstrapConfig:
    """Complete bootstrap configuration."""
    version: str = "1.0"

    # Bootstrap servers by tier
    primary: List[ServerConfig] = field(default_factory=list)
    secondary: List[ServerConfig] = field(default_factory=list)
    tertiary: List[ServerConfig] = field(default_factory=list)

    # Sub-configurations
    peer_cache: PeerCacheConfig = field(default_factory=PeerCacheConfig)
    dht_discovery: DHTDiscoveryConfig = field(default_factory=DHTDiscoveryConfig)
    retry_strategy: RetryConfig = field(default_factory=RetryConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    network_tuning: NetworkTuningConfig = field(default_factory=NetworkTuningConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)

    # Metrics
    metrics_enabled: bool = True


def _parse_server_list(servers: List[Dict[str, Any]]) -> List[ServerConfig]:
    """Parse a list of server configurations."""
    result = []
    for server in servers:
        result.append(ServerConfig(
            url=server.get("url", ""),
            weight=server.get("weight", 100),
            region=server.get("region", "global"),
            timeout_ms=server.get("timeout_ms", 5000),
            operator=server.get("operator", "official"),
        ))
    return result


def load_bootstrap_config(
    config_path: Optional[Union[str, Path]] = None,
    env_prefix: str = "MYCELIX_BOOTSTRAP_"
) -> BootstrapConfig:
    """
    Load bootstrap configuration from file and environment.

    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Default values

    Args:
        config_path: Path to YAML/JSON config file
        env_prefix: Prefix for environment variables

    Returns:
        BootstrapConfig instance
    """
    config_dict: Dict[str, Any] = {}

    # Find default config path
    if config_path is None:
        # Look in standard locations
        search_paths = [
            Path(__file__).parent / "bootstrap_config.yaml",
            Path.home() / ".mycelix" / "bootstrap_config.yaml",
            Path("/etc/mycelix/bootstrap_config.yaml"),
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break

    # Load from file
    if config_path and Path(config_path).exists():
        config_path = Path(config_path)
        logger.info(f"Loading bootstrap config from: {config_path}")

        if config_path.suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                logger.warning("PyYAML not available, skipping YAML config")
            else:
                with open(config_path) as f:
                    config_dict = yaml.safe_load(f) or {}
        else:
            with open(config_path) as f:
                config_dict = json.load(f)

    # Parse bootstrap section
    bootstrap = config_dict.get("bootstrap", {})

    # Parse servers by tier
    primary = _parse_server_list(bootstrap.get("primary", []))
    secondary = _parse_server_list(bootstrap.get("secondary", []))
    tertiary = _parse_server_list(bootstrap.get("tertiary", []))

    # Add defaults if no servers configured
    if not primary:
        primary = [
            ServerConfig(url="https://bootstrap.mycelix.net", weight=100),
            ServerConfig(url="https://bootstrap-staging.holo.host", weight=90),
        ]

    # Parse peer cache config
    cache_dict = bootstrap.get("local_peer_cache", {})
    peer_cache = PeerCacheConfig(
        enabled=cache_dict.get("enabled", True),
        path=cache_dict.get("path", "~/.mycelix/peer_cache.json"),
        max_peers=cache_dict.get("max_peers", 500),
        max_age_hours=cache_dict.get("max_age_hours", 168),
        min_peers_for_offline=cache_dict.get("min_peers_for_offline", 5),
        refresh_interval_hours=cache_dict.get("refresh_interval_hours", 1),
    )

    # Parse DHT discovery config
    dht_dict = bootstrap.get("dht_discovery", {})
    dht_discovery = DHTDiscoveryConfig(
        enabled=dht_dict.get("enabled", True),
        max_discovery_time_ms=dht_dict.get("max_discovery_time_ms", 30000),
        min_peers_before_ready=dht_dict.get("min_peers_before_ready", 3),
        gossip_bootstrap_peers=dht_dict.get("gossip_bootstrap_peers", 5),
    )

    # Parse retry strategy
    retry_dict = bootstrap.get("retry_strategy", {})
    retry_strategy = RetryConfig(
        max_attempts_per_server=retry_dict.get("max_attempts_per_server", 3),
        initial_backoff_ms=retry_dict.get("initial_backoff_ms", 1000),
        max_backoff_ms=retry_dict.get("max_backoff_ms", 30000),
        backoff_multiplier=retry_dict.get("backoff_multiplier", 2.0),
        jitter_factor=retry_dict.get("jitter_factor", 0.1),
        max_tier_retries=retry_dict.get("max_tier_retries", 2),
        tier_backoff_ms=retry_dict.get("tier_backoff_ms", 5000),
    )

    # Parse health check config
    health_dict = bootstrap.get("health_check", {})
    health_check = HealthCheckConfig(
        enabled=health_dict.get("enabled", True),
        interval_seconds=health_dict.get("interval_seconds", 60),
        timeout_ms=health_dict.get("timeout_ms", 3000),
        failure_threshold=health_dict.get("failure_threshold", 3),
        recovery_threshold=health_dict.get("recovery_threshold", 2),
    )

    # Parse circuit breaker config
    cb_dict = bootstrap.get("circuit_breaker", {})
    circuit_breaker = CircuitBreakerConfig(
        enabled=cb_dict.get("enabled", True),
        failure_threshold=cb_dict.get("failure_threshold", 5),
        reset_timeout_seconds=cb_dict.get("reset_timeout_seconds", 300),
        half_open_requests=cb_dict.get("half_open_requests", 3),
    )

    # Parse parallel config
    parallel_dict = bootstrap.get("parallel", {})
    parallel = ParallelConfig(
        enabled=parallel_dict.get("enabled", True),
        max_concurrent=parallel_dict.get("max_concurrent", 3),
        early_success_threshold=parallel_dict.get("early_success_threshold", 1),
    )

    # Parse network tuning
    tuning_dict = config_dict.get("network_tuning", {})
    network_tuning = NetworkTuningConfig(
        gossip_loop_iteration_delay_ms=tuning_dict.get("gossip_loop_iteration_delay_ms", 1000),
        default_notify_remote_agent_count=tuning_dict.get("default_notify_remote_agent_count", 5),
        default_notify_timeout_ms=tuning_dict.get("default_notify_timeout_ms", 10000),
        default_rpc_single_timeout_ms=tuning_dict.get("default_rpc_single_timeout_ms", 10000),
        default_rpc_multi_remote_agent_count=tuning_dict.get("default_rpc_multi_remote_agent_count", 5),
        default_rpc_multi_timeout_ms=tuning_dict.get("default_rpc_multi_timeout_ms", 10000),
        agent_info_expires_after_ms=tuning_dict.get("agent_info_expires_after_ms", 1200000),
        tls_in_mem_session_storage=tuning_dict.get("tls_in_mem_session_storage", 512),
        proxy_keepalive_ms=tuning_dict.get("proxy_keepalive_ms", 120000),
        proxy_to_expire_ms=tuning_dict.get("proxy_to_expire_ms", 300000),
    )

    # Parse signal config
    signal_dict = config_dict.get("signal", {})
    signal_retry = signal_dict.get("retry_strategy", {})
    signal = SignalConfig(
        primary=_parse_server_list(signal_dict.get("primary", [])),
        secondary=_parse_server_list(signal_dict.get("secondary", [])),
        retry_max_attempts=signal_retry.get("max_attempts", 3),
        retry_backoff_ms=signal_retry.get("backoff_ms", 2000),
    )

    # Add default signal servers if none configured
    if not signal.primary:
        signal.primary = [
            ServerConfig(url="wss://signal.holo.host", weight=100),
        ]

    # Apply environment variable overrides
    _apply_env_overrides(
        primary, secondary, tertiary, peer_cache, dht_discovery,
        retry_strategy, health_check, circuit_breaker, env_prefix
    )

    return BootstrapConfig(
        version=config_dict.get("version", "1.0"),
        primary=primary,
        secondary=secondary,
        tertiary=tertiary,
        peer_cache=peer_cache,
        dht_discovery=dht_discovery,
        retry_strategy=retry_strategy,
        health_check=health_check,
        circuit_breaker=circuit_breaker,
        parallel=parallel,
        network_tuning=network_tuning,
        signal=signal,
        metrics_enabled=config_dict.get("metrics", {}).get("enabled", True),
    )


def _apply_env_overrides(
    primary: List[ServerConfig],
    secondary: List[ServerConfig],
    tertiary: List[ServerConfig],
    peer_cache: PeerCacheConfig,
    dht_discovery: DHTDiscoveryConfig,
    retry_strategy: RetryConfig,
    health_check: HealthCheckConfig,
    circuit_breaker: CircuitBreakerConfig,
    prefix: str
) -> None:
    """Apply environment variable overrides to configuration."""

    # Primary bootstrap URL override
    primary_url = os.environ.get(f"{prefix}PRIMARY_URL")
    if primary_url:
        # Insert at beginning of primary list
        primary.insert(0, ServerConfig(url=primary_url, weight=150))

    # Secondary bootstrap URL override
    secondary_url = os.environ.get(f"{prefix}SECONDARY_URL")
    if secondary_url:
        secondary.insert(0, ServerConfig(url=secondary_url, weight=80))

    # Peer cache path override
    cache_path = os.environ.get(f"{prefix}CACHE_PATH")
    if cache_path:
        peer_cache.path = cache_path

    # Retry attempts override
    max_attempts = os.environ.get(f"{prefix}MAX_ATTEMPTS")
    if max_attempts:
        try:
            retry_strategy.max_attempts_per_server = int(max_attempts)
        except ValueError:
            pass

    # DHT discovery enabled override
    dht_enabled = os.environ.get(f"{prefix}DHT_ENABLED")
    if dht_enabled:
        dht_discovery.enabled = dht_enabled.lower() in ("true", "1", "yes")

    # Health check enabled override
    hc_enabled = os.environ.get(f"{prefix}HEALTH_CHECK_ENABLED")
    if hc_enabled:
        health_check.enabled = hc_enabled.lower() in ("true", "1", "yes")


def generate_conductor_network_config(
    config: Optional[BootstrapConfig] = None,
    active_bootstrap_url: Optional[str] = None,
    active_signal_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a conductor-config.yaml compatible network section.

    Args:
        config: Bootstrap configuration (loads default if not provided)
        active_bootstrap_url: Override bootstrap URL (e.g., from health check)
        active_signal_url: Override signal URL

    Returns:
        Dictionary suitable for conductor-config.yaml network section
    """
    if config is None:
        config = load_bootstrap_config()

    # Select bootstrap URL
    if active_bootstrap_url:
        bootstrap_url = active_bootstrap_url
    elif config.primary:
        bootstrap_url = config.primary[0].url
    else:
        bootstrap_url = "https://bootstrap-staging.holo.host"

    # Select signal URL
    if active_signal_url:
        signal_url = active_signal_url
    elif config.signal.primary:
        signal_url = config.signal.primary[0].url
    else:
        signal_url = "wss://signal.holo.host"

    return {
        "transport_pool": [
            {
                "type": "webrtc",
                "signal_url": signal_url,
            }
        ],
        "default_network_type": "quic_bootstrap",
        "bootstrap_url": bootstrap_url,
        "tuning_params": {
            "gossip_loop_iteration_delay_ms": config.network_tuning.gossip_loop_iteration_delay_ms,
            "default_notify_remote_agent_count": config.network_tuning.default_notify_remote_agent_count,
            "default_notify_timeout_ms": config.network_tuning.default_notify_timeout_ms,
            "default_rpc_single_timeout_ms": config.network_tuning.default_rpc_single_timeout_ms,
            "default_rpc_multi_remote_agent_count": config.network_tuning.default_rpc_multi_remote_agent_count,
            "default_rpc_multi_timeout_ms": config.network_tuning.default_rpc_multi_timeout_ms,
            "agent_info_expires_after_ms": config.network_tuning.agent_info_expires_after_ms,
            "tls_in_mem_session_storage": config.network_tuning.tls_in_mem_session_storage,
            "proxy_keepalive_ms": config.network_tuning.proxy_keepalive_ms,
            "proxy_to_expire_ms": config.network_tuning.proxy_to_expire_ms,
        },
    }
