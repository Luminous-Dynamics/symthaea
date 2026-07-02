#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix CI-Ready Test Runner
============================

This script provides a CI-friendly way to run tests with proper
service detection and failure reporting.

Usage:
    # Run unit tests only (fast, no external deps)
    python scripts/run_tests.py --unit

    # Run integration tests (requires services)
    python scripts/run_tests.py --integration

    # Run all tests
    python scripts/run_tests.py --all

    # Run with specific services
    python scripts/run_tests.py --integration --with-holochain --with-redis

    # CI mode: fail loudly if expected services missing
    python scripts/run_tests.py --ci --require-holochain

    # Check service status only
    python scripts/run_tests.py --status

Exit Codes:
    0: All tests passed
    1: Some tests failed
    2: Missing required services (--ci mode only)
    3: Configuration error
"""

import argparse
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Colors for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.BOLD = cls.RESET = ""


@dataclass
class ServiceStatus:
    name: str
    available: bool
    check_details: str


class ServiceChecker:
    """Check availability of external services."""

    @staticmethod
    def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    @staticmethod
    def check_command(cmd: str) -> bool:
        try:
            result = subprocess.run(
                ["which", cmd],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def check_import(module: str) -> bool:
        try:
            __import__(module)
            return True
        except ImportError:
            return False

    def get_all_status(self) -> Dict[str, ServiceStatus]:
        """Get status of all services."""
        services = {}

        # Holochain
        holochain_env = os.environ.get("RUN_HOLOCHAIN_TESTS")
        holochain_port = self.check_port("127.0.0.1", 8888)
        services["holochain"] = ServiceStatus(
            name="Holochain Conductor",
            available=bool(holochain_env) or holochain_port,
            check_details="port 8888" if holochain_port else "RUN_HOLOCHAIN_TESTS env var" if holochain_env else "not running"
        )

        # Redis
        services["redis"] = ServiceStatus(
            name="Redis",
            available=self.check_port("127.0.0.1", 6379),
            check_details="port 6379"
        )

        # PostgreSQL
        services["postgres"] = ServiceStatus(
            name="PostgreSQL",
            available=self.check_port("127.0.0.1", 5432),
            check_details="port 5432"
        )

        # Anvil (Foundry)
        services["anvil"] = ServiceStatus(
            name="Anvil (Foundry)",
            available=self.check_command("anvil"),
            check_details="anvil binary"
        )

        # Ganache
        services["ganache"] = ServiceStatus(
            name="Ganache",
            available=self.check_command("ganache"),
            check_details="ganache binary"
        )

        # PyTorch
        services["pytorch"] = ServiceStatus(
            name="PyTorch",
            available=self.check_import("torch"),
            check_details="import torch"
        )

        # TensorFlow
        services["tensorflow"] = ServiceStatus(
            name="TensorFlow",
            available=self.check_import("tensorflow"),
            check_details="import tensorflow"
        )

        # Scipy
        services["scipy"] = ServiceStatus(
            name="SciPy",
            available=self.check_import("scipy"),
            check_details="import scipy"
        )

        # prometheus-client
        services["prometheus"] = ServiceStatus(
            name="Prometheus Client",
            available=self.check_import("prometheus_client"),
            check_details="import prometheus_client"
        )

        # py-solc-x
        services["solcx"] = ServiceStatus(
            name="py-solc-x",
            available=self.check_import("solcx"),
            check_details="import solcx"
        )

        return services


def print_status(services: Dict[str, ServiceStatus]):
    """Print service status in a nice format."""
    print(f"\n{Colors.BOLD}{'='*70}")
    print("SERVICE AVAILABILITY STATUS")
    print(f"{'='*70}{Colors.RESET}\n")

    external_services = ["holochain", "redis", "postgres", "anvil", "ganache"]
    python_deps = ["pytorch", "tensorflow", "scipy", "prometheus", "solcx"]

    print(f"{Colors.BOLD}External Services:{Colors.RESET}")
    for key in external_services:
        if key in services:
            s = services[key]
            if s.available:
                print(f"  {Colors.GREEN}[OK]{Colors.RESET} {s.name} ({s.check_details})")
            else:
                print(f"  {Colors.RED}[--]{Colors.RESET} {s.name} ({s.check_details})")

    print(f"\n{Colors.BOLD}Python Dependencies:{Colors.RESET}")
    for key in python_deps:
        if key in services:
            s = services[key]
            if s.available:
                print(f"  {Colors.GREEN}[OK]{Colors.RESET} {s.name}")
            else:
                print(f"  {Colors.YELLOW}[--]{Colors.RESET} {s.name}")

    print()


def run_tests(
    test_type: str,
    markers: List[str],
    extra_args: List[str],
    verbose: bool = False,
    coverage: bool = False,
    junit_xml: Optional[str] = None,
) -> int:
    """Run pytest with the specified configuration."""

    # Build pytest command
    cmd = ["pytest"]

    # Add markers
    if markers:
        marker_expr = " or ".join(markers)
        cmd.extend(["-m", marker_expr])

    # Add verbosity
    if verbose:
        cmd.append("-v")

    # Add coverage
    if coverage:
        cmd.extend(["--cov=0TML/src", "--cov-report=term-missing"])

    # Add JUnit XML output for CI
    if junit_xml:
        cmd.extend([f"--junit-xml={junit_xml}"])

    # Add any extra args
    cmd.extend(extra_args)

    print(f"\n{Colors.BOLD}Running {test_type} tests...{Colors.RESET}")
    print(f"Command: {' '.join(cmd)}\n")

    # Run pytest
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Mycelix CI-Ready Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--unit", action="store_true",
        help="Run unit tests only (no external deps)"
    )
    test_group.add_argument(
        "--integration", action="store_true",
        help="Run integration tests only"
    )
    test_group.add_argument(
        "--all", action="store_true",
        help="Run all tests"
    )
    test_group.add_argument(
        "--smoke", action="store_true",
        help="Run smoke tests only (quick CI check)"
    )
    test_group.add_argument(
        "--status", action="store_true",
        help="Check service status only, don't run tests"
    )

    # Service requirements (CI mode)
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: fail if required services are missing"
    )
    parser.add_argument(
        "--require-holochain", action="store_true",
        help="Require Holochain to be available"
    )
    parser.add_argument(
        "--require-redis", action="store_true",
        help="Require Redis to be available"
    )
    parser.add_argument(
        "--require-pytorch", action="store_true",
        help="Require PyTorch to be available"
    )
    parser.add_argument(
        "--require-ethereum", action="store_true",
        help="Require Ethereum test network (Anvil/Ganache)"
    )

    # Specific service tests
    parser.add_argument(
        "--with-holochain", action="store_true",
        help="Include Holochain tests"
    )
    parser.add_argument(
        "--with-rust", action="store_true",
        help="Include Rust bridge tests"
    )
    parser.add_argument(
        "--with-pytorch", action="store_true",
        help="Include PyTorch tests"
    )
    parser.add_argument(
        "--with-ethereum", action="store_true",
        help="Include Ethereum tests"
    )

    # Output options
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--junit-xml", type=str,
        help="Generate JUnit XML report"
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output"
    )

    # Extra pytest args
    parser.add_argument(
        "pytest_args", nargs="*",
        help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Check service status
    checker = ServiceChecker()
    services = checker.get_all_status()

    # Status only mode
    if args.status:
        print_status(services)
        return 0

    # Print status if verbose
    if args.verbose:
        print_status(services)

    # CI mode: check required services
    if args.ci:
        missing = []
        if args.require_holochain and not services["holochain"].available:
            missing.append("Holochain")
        if args.require_redis and not services["redis"].available:
            missing.append("Redis")
        if args.require_pytorch and not services["pytorch"].available:
            missing.append("PyTorch")
        if args.require_ethereum and not (services["anvil"].available or services["ganache"].available):
            missing.append("Ethereum (Anvil/Ganache)")

        if missing:
            print(f"\n{Colors.RED}{Colors.BOLD}ERROR: Required services not available:{Colors.RESET}")
            for service in missing:
                print(f"  - {service}")
            print("\nFailing because --ci mode requires these services.")
            print("Either start the services or remove the --require-* flags.")
            return 2

    # Build marker list
    markers = []

    if args.unit:
        markers = ["unit"]
        test_type = "unit"
    elif args.integration:
        markers = ["integration"]
        test_type = "integration"
    elif args.smoke:
        markers = ["smoke"]
        test_type = "smoke"
    elif args.all:
        markers = []  # No marker filter = all tests
        test_type = "all"
    else:
        # Default to unit tests
        markers = ["unit"]
        test_type = "unit"

    # Add specific service markers if requested
    if args.with_holochain:
        markers.append("holochain")
    if args.with_rust:
        markers.append("rust")
    if args.with_pytorch:
        markers.append("pytorch")
    if args.with_ethereum:
        markers.append("ethereum")

    # Run tests
    return_code = run_tests(
        test_type=test_type,
        markers=markers,
        extra_args=args.pytest_args or [],
        verbose=args.verbose,
        coverage=args.coverage,
        junit_xml=args.junit_xml,
    )

    # Print summary
    print(f"\n{Colors.BOLD}{'='*70}")
    if return_code == 0:
        print(f"{Colors.GREEN}ALL TESTS PASSED{Colors.RESET}")
    else:
        print(f"{Colors.RED}TESTS FAILED (exit code: {return_code}){Colors.RESET}")
    print(f"{'='*70}\n")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
