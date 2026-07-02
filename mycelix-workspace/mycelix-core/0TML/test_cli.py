# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test CLI Commands - Verify all CLI commands work correctly

Tests:
1. Import validation for all CLI modules
2. Help text generation
3. Basic command execution (dry-run mode)
"""

import sys
import os
import importlib
import subprocess

# Add src to path
SRC_PATH = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_PATH)

def test_imports():
    """Test that all CLI modules import correctly"""
    print("=" * 60)
    print("TEST 1: CLI Module Imports")
    print("=" * 60)

    modules = [
        "zerotrustml.cli",
        "zerotrustml.node_cli",
        "zerotrustml.monitoring.cli",
    ]

    for module in modules:
        importlib.import_module(module)
        print(f"✅ {module} imports successfully")


def test_cli_help():
    """Test that CLI help text works"""
    print("\n" + "=" * 60)
    print("TEST 2: CLI Help Text Generation")
    print("=" * 60)

    commands = [
        ("zerotrustml", ["python3", "-m", "zerotrustml.cli", "--help"]),
        ("zerotrustml-node", ["python3", "-m", "zerotrustml.node_cli", "--help"]),
        ("zerotrustml-monitor", ["python3", "-m", "zerotrustml.monitoring.cli", "--help"]),
    ]

    for name, cmd in commands:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "PYTHONPATH": SRC_PATH},
        )
        assert result.returncode == 0, f"{name} --help returned {result.returncode}"
        assert len(result.stdout) > 100, f"{name} help output too short"
        print(f"✅ {name} --help works ({len(result.stdout)} chars)")


def test_cli_commands():
    """Test basic CLI command structure"""
    print("\n" + "=" * 60)
    print("TEST 3: CLI Command Structure")
    print("=" * 60)

    from zerotrustml.cli import main as zerotrustml_main
    from zerotrustml.node_cli import main as node_main
    from zerotrustml.monitoring.cli import main as monitor_main

    assert callable(zerotrustml_main)
    assert callable(node_main)
    assert callable(monitor_main)
    print("✅ CLI entry points are callable")


def test_monitoring_classes():
    """Test monitoring module classes"""
    print("\n" + "=" * 60)
    print("TEST 4: Monitoring Classes")
    print("=" * 60)

    from zerotrustml.monitoring import Dashboard, NetworkMonitor

    dashboard = Dashboard()
    monitor = NetworkMonitor()
    health = monitor.get_network_health()

    assert isinstance(dashboard, Dashboard)
    assert isinstance(health, dict)
    assert "status" in health
    print(f"✅ Monitoring classes instantiated; status={health['status']}")


def main():
    """Run all CLI tests"""
    print("\n" + "=" * 70)
    print("ZeroTrustML CLI Test Suite")
    print("=" * 70)

    results = []

    # Test 1: Imports
    results.append(("Import Validation", test_imports()))

    # Test 2: Help text
    results.append(("Help Text Generation", test_cli_help()))

    # Test 3: Command structure
    results.append(("Command Structure", test_cli_commands()))

    # Test 4: Monitoring classes
    results.append(("Monitoring Classes", test_monitoring_classes()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if result:
            passed += 1

    print("=" * 70)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 70)

    return passed == len(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
