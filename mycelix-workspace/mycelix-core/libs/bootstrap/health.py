# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Bootstrap Server Health Checker
===============================

Monitors bootstrap server health and maintains availability status.
Implements circuit breaker pattern for failed servers.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from .config_loader import (
    BootstrapConfig,
    CircuitBreakerConfig,
    HealthCheckConfig,
    ServerConfig,
)

logger = logging.getLogger("mycelix.bootstrap.health")


class HealthStatus(Enum):
    """Health status of a server."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker state."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ServerHealth:
    """Health state of a single server."""
    url: str
    status: HealthStatus = HealthStatus.UNKNOWN
    circuit_state: CircuitState = CircuitState.CLOSED
    last_check: Optional[float] = None
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    latency_ms: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    error_message: Optional[str] = None
    circuit_opened_at: Optional[float] = None
    half_open_requests: int = 0


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    url: str
    healthy: bool
    latency_ms: float
    status_code: Optional[int] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class CircuitBreaker:
    """
    Circuit breaker for a single server.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Server is failing, requests are rejected immediately
    - HALF_OPEN: Testing if server has recovered
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_requests = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, potentially transitioning from OPEN to HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.reset_timeout_seconds

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_requests += 1
                if self._half_open_requests >= self.config.half_open_requests:
                    # Fully recovered
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    logger.info("Circuit breaker closed (server recovered)")
            else:
                self._failures = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker opened (recovery failed)")
            elif self._failures >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self._failures} failures")

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state  # May trigger OPEN -> HALF_OPEN transition
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                return self._half_open_requests < self.config.half_open_requests
        return False

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._last_failure_time = None
            self._half_open_requests = 0


class BootstrapHealthChecker:
    """
    Monitors health of all bootstrap servers.

    Features:
    - Periodic health checks
    - Circuit breaker per server
    - Latency tracking
    - Status aggregation
    """

    def __init__(
        self,
        config: BootstrapConfig,
        on_status_change: Optional[Callable[[str, HealthStatus], None]] = None,
    ):
        """
        Initialize the health checker.

        Args:
            config: Bootstrap configuration
            on_status_change: Callback when server status changes
        """
        self.config = config
        self.on_status_change = on_status_change

        self._servers: Dict[str, ServerHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

        # Initialize server tracking
        self._init_servers()

    def _init_servers(self) -> None:
        """Initialize tracking for all configured servers."""
        all_servers = (
            self.config.primary +
            self.config.secondary +
            self.config.tertiary
        )

        for server in all_servers:
            self._servers[server.url] = ServerHealth(url=server.url)
            if self.config.circuit_breaker.enabled:
                self._circuit_breakers[server.url] = CircuitBreaker(
                    self.config.circuit_breaker
                )

    async def check_server(self, url: str, timeout_ms: int = 5000) -> HealthCheckResult:
        """
        Perform a health check on a single server.

        Args:
            url: Server URL to check
            timeout_ms: Timeout in milliseconds

        Returns:
            HealthCheckResult with check details
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, skipping health check")
            return HealthCheckResult(
                url=url,
                healthy=True,  # Assume healthy if we can't check
                latency_ms=0,
                error="aiohttp not available",
            )

        start_time = time.time()
        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Use the /health or root endpoint
                check_url = url.rstrip("/")
                if not check_url.endswith("/health"):
                    check_url = f"{check_url}/health"

                async with session.get(check_url) as response:
                    latency_ms = (time.time() - start_time) * 1000

                    if response.status == 200:
                        return HealthCheckResult(
                            url=url,
                            healthy=True,
                            latency_ms=latency_ms,
                            status_code=response.status,
                        )
                    else:
                        return HealthCheckResult(
                            url=url,
                            healthy=False,
                            latency_ms=latency_ms,
                            status_code=response.status,
                            error=f"HTTP {response.status}",
                        )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                url=url,
                healthy=False,
                latency_ms=timeout_ms,
                error="Timeout",
            )
        except aiohttp.ClientError as e:
            return HealthCheckResult(
                url=url,
                healthy=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
        except Exception as e:
            logger.exception(f"Health check failed for {url}")
            return HealthCheckResult(
                url=url,
                healthy=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _update_server_health(self, result: HealthCheckResult) -> None:
        """Update server health state based on check result."""
        with self._lock:
            health = self._servers.get(result.url)
            if not health:
                return

            old_status = health.status
            health.last_check = result.timestamp
            health.latency_ms = result.latency_ms
            health.total_checks += 1

            if result.healthy:
                health.last_success = result.timestamp
                health.consecutive_successes += 1
                health.consecutive_failures = 0
                health.error_message = None

                # Update circuit breaker
                if result.url in self._circuit_breakers:
                    self._circuit_breakers[result.url].record_success()

                # Determine status
                if health.consecutive_successes >= self.config.health_check.recovery_threshold:
                    health.status = HealthStatus.HEALTHY
                else:
                    health.status = HealthStatus.DEGRADED

            else:
                health.last_failure = result.timestamp
                health.consecutive_failures += 1
                health.consecutive_successes = 0
                health.total_failures += 1
                health.error_message = result.error

                # Update circuit breaker
                if result.url in self._circuit_breakers:
                    self._circuit_breakers[result.url].record_failure()

                # Determine status
                if health.consecutive_failures >= self.config.health_check.failure_threshold:
                    health.status = HealthStatus.UNHEALTHY
                else:
                    health.status = HealthStatus.DEGRADED

            # Update circuit state
            if result.url in self._circuit_breakers:
                health.circuit_state = self._circuit_breakers[result.url].state

            # Notify on status change
            if old_status != health.status and self.on_status_change:
                try:
                    self.on_status_change(result.url, health.status)
                except Exception as e:
                    logger.warning(f"Status change callback failed: {e}")

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Check health of all servers.

        Returns:
            Dictionary mapping URL to HealthCheckResult
        """
        results = {}

        # Get all server URLs
        all_urls = list(self._servers.keys())

        # Check in parallel
        tasks = [
            self.check_server(url, self.config.health_check.timeout_ms)
            for url in all_urls
        ]

        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(all_urls, check_results):
            if isinstance(result, Exception):
                result = HealthCheckResult(
                    url=url,
                    healthy=False,
                    latency_ms=0,
                    error=str(result),
                )
            results[url] = result
            self._update_server_health(result)

        return results

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

            await asyncio.sleep(self.config.health_check.interval_seconds)

    async def start(self) -> None:
        """Start background health checking."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started bootstrap health checker")

    async def stop(self) -> None:
        """Stop background health checking."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped bootstrap health checker")

    def get_healthy_servers(
        self,
        tier: Optional[str] = None,
    ) -> List[Tuple[str, ServerHealth]]:
        """
        Get list of healthy servers, optionally filtered by tier.

        Args:
            tier: "primary", "secondary", or "tertiary"

        Returns:
            List of (url, health) tuples, sorted by quality
        """
        with self._lock:
            # Get server URLs for tier
            if tier == "primary":
                tier_urls = {s.url for s in self.config.primary}
            elif tier == "secondary":
                tier_urls = {s.url for s in self.config.secondary}
            elif tier == "tertiary":
                tier_urls = {s.url for s in self.config.tertiary}
            else:
                tier_urls = set(self._servers.keys())

            # Filter healthy
            healthy = [
                (url, health) for url, health in self._servers.items()
                if url in tier_urls and health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            ]

            # Sort by status (healthy first) then latency
            def sort_key(item: Tuple[str, ServerHealth]) -> Tuple[int, float]:
                _, health = item
                status_order = 0 if health.status == HealthStatus.HEALTHY else 1
                latency = health.latency_ms or 9999
                return (status_order, latency)

            return sorted(healthy, key=sort_key)

    def get_best_server(self) -> Optional[str]:
        """
        Get the best available server URL.

        Returns:
            URL of best server, or None if all unhealthy
        """
        # Try tiers in order
        for tier in ["primary", "secondary", "tertiary"]:
            healthy = self.get_healthy_servers(tier)
            if healthy:
                url, _ = healthy[0]
                # Check circuit breaker
                if url in self._circuit_breakers:
                    if self._circuit_breakers[url].allow_request():
                        return url
                else:
                    return url

        return None

    def is_server_available(self, url: str) -> bool:
        """Check if a specific server is available."""
        with self._lock:
            health = self._servers.get(url)
            if not health:
                return False

            if health.status == HealthStatus.UNHEALTHY:
                return False

            if url in self._circuit_breakers:
                return self._circuit_breakers[url].allow_request()

            return True

    def get_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        with self._lock:
            total = len(self._servers)
            healthy = sum(
                1 for h in self._servers.values()
                if h.status == HealthStatus.HEALTHY
            )
            degraded = sum(
                1 for h in self._servers.values()
                if h.status == HealthStatus.DEGRADED
            )
            unhealthy = sum(
                1 for h in self._servers.values()
                if h.status == HealthStatus.UNHEALTHY
            )

            return {
                "total_servers": total,
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "health_percentage": round(100 * healthy / total, 1) if total > 0 else 0,
                "best_server": self.get_best_server(),
                "servers": {
                    url: {
                        "status": h.status.value,
                        "circuit_state": h.circuit_state.value,
                        "latency_ms": h.latency_ms,
                        "consecutive_failures": h.consecutive_failures,
                        "error": h.error_message,
                    }
                    for url, h in self._servers.items()
                },
            }

    def reset_circuit_breaker(self, url: str) -> bool:
        """Manually reset a circuit breaker."""
        if url in self._circuit_breakers:
            self._circuit_breakers[url].reset()
            return True
        return False
