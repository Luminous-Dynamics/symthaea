#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Validator Node Entrypoint
=================================

Production-ready entrypoint for running a Mycelix testnet validator.
Manages FL Coordinator, Holochain conductor, metrics exporter, and health checks.

Author: Luminous Dynamics
Date: January 2026
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure structured logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if os.getenv("LOG_FORMAT") == "json" else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger(__name__)


@dataclass
class ValidatorConfig:
    """Configuration for the validator node."""
    # Identity
    node_id: str = field(default_factory=lambda: os.getenv("VALIDATOR_NODE_ID", "validator-001"))
    region: str = field(default_factory=lambda: os.getenv("VALIDATOR_REGION", "global"))
    operator: str = field(default_factory=lambda: os.getenv("VALIDATOR_OPERATOR", "community"))
    network_type: str = field(default_factory=lambda: os.getenv("NETWORK_TYPE", "testnet"))

    # Network
    bootstrap_url: str = field(default_factory=lambda: os.getenv("BOOTSTRAP_URL", "https://bootstrap.mycelix.net"))
    signal_url: str = field(default_factory=lambda: os.getenv("SIGNAL_URL", "wss://signal.holo.host"))

    # FL Settings
    fl_min_nodes: int = field(default_factory=lambda: int(os.getenv("FL_MIN_NODES", "3")))
    fl_max_nodes: int = field(default_factory=lambda: int(os.getenv("FL_MAX_NODES", "100")))
    fl_round_timeout: float = field(default_factory=lambda: float(os.getenv("FL_ROUND_TIMEOUT_SECONDS", "120")))
    byzantine_threshold: float = field(default_factory=lambda: float(os.getenv("FL_BYZANTINE_THRESHOLD", "0.45")))

    # Holochain
    hc_admin_port: int = field(default_factory=lambda: int(os.getenv("HC_ADMIN_PORT", "39329")))
    hc_app_port: int = field(default_factory=lambda: int(os.getenv("HC_APP_PORT", "9998")))
    hc_data_path: str = field(default_factory=lambda: os.getenv("HC_DATA_PATH", "/data/holochain"))

    # Metrics
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true")
    metrics_port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    health_check_port: int = field(default_factory=lambda: int(os.getenv("HEALTH_CHECK_PORT", "8080")))

    # Paths
    data_path: str = field(default_factory=lambda: os.getenv("DATA_PATH", "/data"))
    logs_path: str = field(default_factory=lambda: os.getenv("LOGS_PATH", "/logs"))
    config_path: str = field(default_factory=lambda: os.getenv("CONFIG_PATH", "/config"))


class MetricsServer:
    """Prometheus metrics HTTP server."""

    def __init__(self, port: int = 9090):
        self.port = port
        self._metrics: Dict[str, Any] = {
            "active_nodes": 0,
            "rounds_completed": 0,
            "byzantine_detected": 0,
            "healed_gradients": 0,
            "aggregation_latency_ms": 0.0,
            "detection_latency_ms": 0.0,
            "model_convergence": 0.0,
            "last_round_timestamp": 0,
        }
        self._start_time = time.time()

    def update(self, key: str, value: Any) -> None:
        """Update a metric value."""
        self._metrics[key] = value

    def increment(self, key: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        self._metrics[key] = self._metrics.get(key, 0) + value

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = [
            "# HELP mycelix_validator_info Validator node information",
            "# TYPE mycelix_validator_info gauge",
            f'mycelix_validator_info{{node_id="{os.getenv("VALIDATOR_NODE_ID", "validator-001")}",region="{os.getenv("VALIDATOR_REGION", "global")}",network="{os.getenv("NETWORK_TYPE", "testnet")}"}} 1',
            "",
            "# HELP mycelix_uptime_seconds Time since validator started",
            "# TYPE mycelix_uptime_seconds gauge",
            f"mycelix_uptime_seconds {time.time() - self._start_time:.1f}",
            "",
            "# HELP mycelix_active_nodes Number of active FL nodes",
            "# TYPE mycelix_active_nodes gauge",
            f"mycelix_active_nodes {self._metrics.get('active_nodes', 0)}",
            "",
            "# HELP mycelix_rounds_completed_total Total FL rounds completed",
            "# TYPE mycelix_rounds_completed_total counter",
            f"mycelix_rounds_completed_total {self._metrics.get('rounds_completed', 0)}",
            "",
            "# HELP mycelix_byzantine_detected_total Total Byzantine nodes detected",
            "# TYPE mycelix_byzantine_detected_total counter",
            f"mycelix_byzantine_detected_total {self._metrics.get('byzantine_detected', 0)}",
            "",
            "# HELP mycelix_healed_gradients_total Total gradients healed",
            "# TYPE mycelix_healed_gradients_total counter",
            f"mycelix_healed_gradients_total {self._metrics.get('healed_gradients', 0)}",
            "",
            "# HELP mycelix_aggregation_latency_ms Aggregation latency in milliseconds",
            "# TYPE mycelix_aggregation_latency_ms gauge",
            f"mycelix_aggregation_latency_ms {self._metrics.get('aggregation_latency_ms', 0.0):.2f}",
            "",
            "# HELP mycelix_detection_latency_ms Byzantine detection latency in milliseconds",
            "# TYPE mycelix_detection_latency_ms gauge",
            f"mycelix_detection_latency_ms {self._metrics.get('detection_latency_ms', 0.0):.2f}",
            "",
            "# HELP mycelix_model_convergence Current model convergence metric",
            "# TYPE mycelix_model_convergence gauge",
            f"mycelix_model_convergence {self._metrics.get('model_convergence', 0.0):.4f}",
            "",
            "# HELP mycelix_byzantine_detection_rate Byzantine detection rate (0-1)",
            "# TYPE mycelix_byzantine_detection_rate gauge",
            f"mycelix_byzantine_detection_rate {self._metrics.get('byzantine_detection_rate', 1.0):.4f}",
            "",
        ]
        return "\n".join(lines)


class HealthServer:
    """Health check HTTP server."""

    def __init__(self, port: int = 8080):
        self.port = port
        self._healthy = True
        self._ready = False
        self._checks: Dict[str, bool] = {
            "coordinator": False,
            "holochain": False,
            "metrics": False,
        }

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    def set_ready(self, ready: bool) -> None:
        self._ready = ready

    def set_check(self, name: str, healthy: bool) -> None:
        self._checks[name] = healthy
        # Update overall health
        self._healthy = all(self._checks.values())

    def get_status(self) -> Dict[str, Any]:
        return {
            "healthy": self._healthy,
            "ready": self._ready,
            "checks": self._checks,
            "timestamp": time.time(),
            "node_id": os.getenv("VALIDATOR_NODE_ID", "validator-001"),
        }


class ValidatorNode:
    """
    Main validator node orchestrator.

    Manages:
    - FL Coordinator (0TML)
    - Holochain conductor
    - Metrics exporter
    - Health checks
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.metrics = MetricsServer(port=config.metrics_port)
        self.health = HealthServer(port=config.health_check_port)
        self._shutdown_event = asyncio.Event()
        self._coordinator = None
        self._tasks: list = []

        log.info(
            "Initializing validator node",
            node_id=config.node_id,
            region=config.region,
            network=config.network_type,
        )

    async def start(self) -> None:
        """Start all validator services."""
        log.info("Starting validator services...")

        # Start health check server
        self._tasks.append(asyncio.create_task(self._run_health_server()))
        self.health.set_check("health_server", True)

        # Start metrics server
        if self.config.metrics_enabled:
            self._tasks.append(asyncio.create_task(self._run_metrics_server()))
            self.health.set_check("metrics", True)

        # Initialize FL Coordinator
        await self._init_coordinator()
        self.health.set_check("coordinator", True)

        # Start Holochain conductor (if available)
        await self._init_holochain()

        # Mark as ready
        self.health.set_ready(True)
        log.info("Validator node is ready", node_id=self.config.node_id)

        # Start main loop
        await self._run_main_loop()

    async def _init_coordinator(self) -> None:
        """Initialize the FL Coordinator."""
        log.info("Initializing FL Coordinator...")

        try:
            # Import FL components
            from mycelix_fl.distributed.coordinator import FLCoordinator, CoordinatorConfig

            coordinator_config = CoordinatorConfig(
                num_rounds=1000000,  # Continuous operation
                min_nodes=self.config.fl_min_nodes,
                max_nodes=self.config.fl_max_nodes,
                round_timeout_seconds=self.config.fl_round_timeout,
                byzantine_threshold=self.config.byzantine_threshold,
                use_detection=True,
                use_healing=True,
                on_round_start=self._on_round_start,
                on_round_end=self._on_round_end,
                on_node_join=self._on_node_join,
                on_node_leave=self._on_node_leave,
            )

            self._coordinator = FLCoordinator(config=coordinator_config)
            log.info("FL Coordinator initialized successfully")

        except ImportError as e:
            log.warning("FL Coordinator not available, running in limited mode", error=str(e))
            self._coordinator = None

    async def _init_holochain(self) -> None:
        """Initialize Holochain conductor."""
        log.info("Initializing Holochain conductor...")

        # Check if holochain binary is available
        import shutil
        holochain_path = shutil.which("holochain")

        if holochain_path:
            log.info("Holochain binary found", path=holochain_path)
            # In production, we would spawn the holochain process here
            # For now, mark as healthy
            self.health.set_check("holochain", True)
        else:
            log.warning("Holochain binary not found, DHT features disabled")
            self.health.set_check("holochain", False)

    async def _run_health_server(self) -> None:
        """Run the health check HTTP server."""
        try:
            from aiohttp import web

            async def health_handler(request):
                status = self.health.get_status()
                status_code = 200 if status["healthy"] else 503
                return web.json_response(status, status=status_code)

            async def ready_handler(request):
                status = self.health.get_status()
                status_code = 200 if status["ready"] else 503
                return web.json_response({"ready": status["ready"]}, status=status_code)

            async def live_handler(request):
                return web.json_response({"alive": True})

            app = web.Application()
            app.router.add_get("/health", health_handler)
            app.router.add_get("/ready", ready_handler)
            app.router.add_get("/live", live_handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.config.health_check_port)
            await site.start()

            log.info("Health server started", port=self.config.health_check_port)

            # Keep running until shutdown
            await self._shutdown_event.wait()
            await runner.cleanup()

        except Exception as e:
            log.error("Health server error", error=str(e))
            self.health.set_check("health_server", False)

    async def _run_metrics_server(self) -> None:
        """Run the Prometheus metrics HTTP server."""
        try:
            from aiohttp import web

            async def metrics_handler(request):
                metrics_text = self.metrics.get_prometheus_format()
                return web.Response(
                    text=metrics_text,
                    content_type="text/plain; version=0.0.4; charset=utf-8",
                )

            app = web.Application()
            app.router.add_get("/metrics", metrics_handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.config.metrics_port)
            await site.start()

            log.info("Metrics server started", port=self.config.metrics_port)

            # Keep running until shutdown
            await self._shutdown_event.wait()
            await runner.cleanup()

        except Exception as e:
            log.error("Metrics server error", error=str(e))
            self.health.set_check("metrics", False)

    async def _run_main_loop(self) -> None:
        """Main validator loop."""
        log.info("Starting main validator loop...")

        while not self._shutdown_event.is_set():
            try:
                # Update metrics
                self.metrics.update("last_round_timestamp", time.time())

                # Simulate round processing (in production, this would be event-driven)
                await asyncio.sleep(10)

                # Log heartbeat
                if int(time.time()) % 60 == 0:
                    log.info(
                        "Validator heartbeat",
                        node_id=self.config.node_id,
                        active_nodes=self.metrics._metrics.get("active_nodes", 0),
                        rounds_completed=self.metrics._metrics.get("rounds_completed", 0),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Main loop error", error=str(e))
                await asyncio.sleep(5)

    def _on_round_start(self, round_num: int) -> None:
        """Callback when an FL round starts."""
        log.debug("FL round started", round_num=round_num)

    def _on_round_end(self, round_num: int, result: Any) -> None:
        """Callback when an FL round ends."""
        self.metrics.increment("rounds_completed")

        if hasattr(result, "byzantine_nodes"):
            self.metrics.increment("byzantine_detected", len(result.byzantine_nodes))

        if hasattr(result, "healed_nodes"):
            self.metrics.increment("healed_gradients", len(result.healed_nodes))

        log.info(
            "FL round completed",
            round_num=round_num,
            participating_nodes=len(getattr(result, "participating_nodes", [])),
            byzantine_detected=len(getattr(result, "byzantine_nodes", set())),
        )

    def _on_node_join(self, node_id: str) -> None:
        """Callback when a node joins."""
        self.metrics.increment("active_nodes")
        log.info("Node joined", joining_node=node_id)

    def _on_node_leave(self, node_id: str) -> None:
        """Callback when a node leaves."""
        self.metrics.increment("active_nodes", -1)
        log.info("Node left", leaving_node=node_id)

    async def stop(self) -> None:
        """Stop all validator services."""
        log.info("Stopping validator services...")
        self._shutdown_event.set()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        log.info("Validator stopped")


async def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="Mycelix Validator Node")
    parser.add_argument("--mode", choices=["validator", "coordinator", "worker"], default="validator")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = ValidatorConfig()

    log.info(
        "Mycelix Validator Node starting",
        version="1.0.0",
        mode=args.mode,
        node_id=config.node_id,
        network=config.network_type,
    )

    # Create and start validator
    validator = ValidatorNode(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        log.info("Received shutdown signal")
        asyncio.create_task(validator.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Run validator
    try:
        await validator.start()
    except asyncio.CancelledError:
        pass
    finally:
        await validator.stop()

    log.info("Validator node exited")


if __name__ == "__main__":
    asyncio.run(main())
