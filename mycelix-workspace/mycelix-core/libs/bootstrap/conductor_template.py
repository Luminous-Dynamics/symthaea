# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Conductor Configuration Template Generator
==========================================

Generates conductor-config.yaml files with hierarchical bootstrap support.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from .config_loader import BootstrapConfig, load_bootstrap_config

logger = logging.getLogger("mycelix.bootstrap.conductor_template")


def generate_conductor_config(
    config: Optional[BootstrapConfig] = None,
    environment_path: str = ".hc",
    admin_port: int = 39329,
    app_port: int = 9998,
    use_test_keystore: bool = True,
    active_bootstrap_url: Optional[str] = None,
    active_signal_url: Optional[str] = None,
    allowed_origins: str = "*",
    network_type: str = "quic_bootstrap",
    dpki_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Generate a complete conductor configuration with hierarchical bootstrap.

    Args:
        config: Bootstrap configuration (loads default if not provided)
        environment_path: Path to Holochain environment
        admin_port: Admin interface port
        app_port: Application interface port
        use_test_keystore: Use test keystore (dev only)
        active_bootstrap_url: Override bootstrap URL
        active_signal_url: Override signal URL
        allowed_origins: CORS allowed origins
        network_type: Network type (quic_bootstrap, quic_mdns, etc.)
        dpki_enabled: Enable DPKI

    Returns:
        Dictionary suitable for YAML serialization as conductor-config.yaml
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

    conductor_config = {
        "environment_path": environment_path,
        "use_dangerous_test_keystore": use_test_keystore,

        "admin_interfaces": [
            {
                "driver": {
                    "type": "websocket",
                    "port": admin_port,
                    "allowed_origins": allowed_origins,
                }
            }
        ],

        "app_interfaces": [
            {
                "driver": {
                    "type": "websocket",
                    "port": app_port,
                    "allowed_origins": allowed_origins,
                }
            }
        ],

        "network": {
            "transport_pool": [
                {
                    "type": "webrtc",
                    "signal_url": signal_url,
                }
            ],
            "default_network_type": network_type,
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
        },

        "dpki": {
            "no_dpki": not dpki_enabled,
        },

        # Mycelix extension: hierarchical bootstrap metadata
        "_mycelix_bootstrap": {
            "version": "1.0",
            "primary_servers": [s.url for s in config.primary],
            "secondary_servers": [s.url for s in config.secondary],
            "tertiary_servers": [s.url for s in config.tertiary],
            "peer_cache_enabled": config.peer_cache.enabled,
            "peer_cache_path": config.peer_cache.path,
            "health_check_enabled": config.health_check.enabled,
        },
    }

    return conductor_config


def write_conductor_config(
    output_path: Union[str, Path],
    config: Optional[BootstrapConfig] = None,
    **kwargs,
) -> None:
    """
    Generate and write conductor configuration to file.

    Args:
        output_path: Path to write conductor-config.yaml
        config: Bootstrap configuration
        **kwargs: Additional arguments for generate_conductor_config
    """
    conductor_config = generate_conductor_config(config=config, **kwargs)
    output_path = Path(output_path)

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise RuntimeError("PyYAML required for YAML output")
        with open(output_path, "w") as f:
            yaml.dump(conductor_config, f, default_flow_style=False, sort_keys=False)
    else:
        with open(output_path, "w") as f:
            json.dump(conductor_config, f, indent=2)

    logger.info(f"Wrote conductor config to {output_path}")


def generate_multi_node_configs(
    base_dir: Union[str, Path],
    num_nodes: int,
    config: Optional[BootstrapConfig] = None,
    base_admin_port: int = 39329,
    base_app_port: int = 9998,
    **kwargs,
) -> List[Path]:
    """
    Generate conductor configs for multiple nodes.

    Useful for local development with multiple Holochain nodes.

    Args:
        base_dir: Directory to write config files
        num_nodes: Number of node configurations to generate
        config: Bootstrap configuration
        base_admin_port: Starting admin port (increments for each node)
        base_app_port: Starting app port (increments for each node)
        **kwargs: Additional arguments for generate_conductor_config

    Returns:
        List of paths to generated config files
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    for i in range(num_nodes):
        node_config = generate_conductor_config(
            config=config,
            environment_path=f".hc-node-{i}",
            admin_port=base_admin_port + i,
            app_port=base_app_port + i,
            **kwargs,
        )

        output_path = base_dir / f"conductor-node-{i}.yaml"

        if YAML_AVAILABLE:
            with open(output_path, "w") as f:
                yaml.dump(node_config, f, default_flow_style=False, sort_keys=False)
        else:
            output_path = output_path.with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(node_config, f, indent=2)

        generated.append(output_path)

    logger.info(f"Generated {num_nodes} conductor configs in {base_dir}")
    return generated


# Template for conductor-config.yaml with inline comments
CONDUCTOR_CONFIG_TEMPLATE = """---
# Mycelix Conductor Configuration
# ================================
# Generated with hierarchical bootstrap support
# See: libs/bootstrap/bootstrap_config.yaml for full bootstrap configuration

environment_path: {environment_path}
use_dangerous_test_keystore: {use_test_keystore}

admin_interfaces:
  - driver:
      type: websocket
      port: {admin_port}
      allowed_origins: "{allowed_origins}"

app_interfaces:
  - driver:
      type: websocket
      port: {app_port}
      allowed_origins: "{allowed_origins}"

network:
  transport_pool:
    - type: webrtc
      signal_url: {signal_url}

  default_network_type: {network_type}

  # Primary bootstrap URL - automatically selected from hierarchical config
  # Fallback servers are configured in libs/bootstrap/bootstrap_config.yaml
  bootstrap_url: {bootstrap_url}

  tuning_params:
    gossip_loop_iteration_delay_ms: {gossip_loop_iteration_delay_ms}
    default_notify_remote_agent_count: {default_notify_remote_agent_count}
    default_notify_timeout_ms: {default_notify_timeout_ms}
    default_rpc_single_timeout_ms: {default_rpc_single_timeout_ms}
    default_rpc_multi_remote_agent_count: {default_rpc_multi_remote_agent_count}
    default_rpc_multi_timeout_ms: {default_rpc_multi_timeout_ms}
    agent_info_expires_after_ms: {agent_info_expires_after_ms}
    tls_in_mem_session_storage: {tls_in_mem_session_storage}
    proxy_keepalive_ms: {proxy_keepalive_ms}
    proxy_to_expire_ms: {proxy_to_expire_ms}

dpki:
  no_dpki: {no_dpki}

# Mycelix Hierarchical Bootstrap Configuration
# =============================================
# This section is used by the Mycelix bootstrap client for resilient peer discovery.
# The conductor itself only uses bootstrap_url above, but the Mycelix wrapper uses
# these fallback servers when the primary fails.
#
# Fallback order:
# 1. Primary servers (official Holo + Mycelix)
# 2. Secondary servers (community-operated)
# 3. Local peer cache (~/.mycelix/peer_cache.json)
# 4. DHT-based gossip discovery
#
# To run your own bootstrap server, see: docs/BOOTSTRAP_OPERATOR_GUIDE.md
"""


def render_conductor_config_template(
    config: Optional[BootstrapConfig] = None,
    **kwargs,
) -> str:
    """
    Render conductor config as a string with comments.

    Args:
        config: Bootstrap configuration
        **kwargs: Override values

    Returns:
        YAML string with comments
    """
    if config is None:
        config = load_bootstrap_config()

    defaults = {
        "environment_path": ".hc",
        "use_test_keystore": "true",
        "admin_port": 39329,
        "app_port": 9998,
        "allowed_origins": "*",
        "network_type": "quic_bootstrap",
        "no_dpki": "true",
    }

    # Get URLs
    defaults["bootstrap_url"] = (
        kwargs.get("active_bootstrap_url") or
        (config.primary[0].url if config.primary else "https://bootstrap-staging.holo.host")
    )
    defaults["signal_url"] = (
        kwargs.get("active_signal_url") or
        (config.signal.primary[0].url if config.signal.primary else "wss://signal.holo.host")
    )

    # Tuning params
    defaults["gossip_loop_iteration_delay_ms"] = config.network_tuning.gossip_loop_iteration_delay_ms
    defaults["default_notify_remote_agent_count"] = config.network_tuning.default_notify_remote_agent_count
    defaults["default_notify_timeout_ms"] = config.network_tuning.default_notify_timeout_ms
    defaults["default_rpc_single_timeout_ms"] = config.network_tuning.default_rpc_single_timeout_ms
    defaults["default_rpc_multi_remote_agent_count"] = config.network_tuning.default_rpc_multi_remote_agent_count
    defaults["default_rpc_multi_timeout_ms"] = config.network_tuning.default_rpc_multi_timeout_ms
    defaults["agent_info_expires_after_ms"] = config.network_tuning.agent_info_expires_after_ms
    defaults["tls_in_mem_session_storage"] = config.network_tuning.tls_in_mem_session_storage
    defaults["proxy_keepalive_ms"] = config.network_tuning.proxy_keepalive_ms
    defaults["proxy_to_expire_ms"] = config.network_tuning.proxy_to_expire_ms

    # Apply overrides
    defaults.update(kwargs)

    return CONDUCTOR_CONFIG_TEMPLATE.format(**defaults)
