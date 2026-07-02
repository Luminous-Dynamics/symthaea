# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Production Monitoring System

Validates that production monitoring infrastructure works correctly.
"""

import asyncio
import json
from datetime import datetime
from zerotrustml.monitoring.production_monitor import (
    ProductionMonitor,
    MetricsCollector,
    AlertManager,
    SystemMetrics,
    Alert
)


async def test_metrics_collection():
    """Test metrics collector functionality"""
    print("\n" + "=" * 80)
    print("TEST 1: Metrics Collection")
    print("=" * 80)

    collector = MetricsCollector(window_size_minutes=60)

    # Simulate credit events
    print("\n📊 Simulating 50 credit issuance events...")
    for i in range(50):
        collector.record_credit_issued(
            node_id=f"node_{i % 5}",
            credits=100.0 + i,
            event_type="quality_gradient",
            response_time_ms=500 + i * 10
        )

    # Simulate Byzantine detections
    print("🔒 Simulating 5 Byzantine detections...")
    for i in range(5):
        collector.record_byzantine_detection(
            detector_id=f"validator_{i}",
            detected_id=f"malicious_{i}",
            evidence={"pogq": 0.15, "consecutive_failures": 20}
        )

    # Simulate some errors
    print("⚠️  Simulating 3 errors...")
    for i in range(3):
        collector.record_error(
            node_id=f"node_{i}",
            error_type="connection_timeout",
            message="Failed to connect to Holochain"
        )

    # Simulate rate limit hits
    print("🚫 Simulating 2 rate limit violations...")
    for i in range(2):
        collector.record_rate_limit_hit(
            node_id=f"spam_node_{i}",
            event_type="quality_gradient"
        )

    # Update node health
    print("❤️  Setting node health status...")
    for i in range(5):
        collector.record_node_health(f"node_{i}", is_healthy=True)

    # Get current metrics
    metrics = collector.get_current_metrics()

    print("\n📈 Current System Metrics:")
    print(f"  • Timestamp: {metrics.timestamp}")
    print(f"  • Events processed (last minute): {metrics.events_processed_last_minute}")
    print(f"  • Credits issued (last minute): {metrics.credits_issued_last_minute}")
    print(f"  • Average response time: {metrics.avg_response_time_ms:.2f}ms")
    print(f"  • P95 response time: {metrics.p95_response_time_ms:.2f}ms")
    print(f"  • P99 response time: {metrics.p99_response_time_ms:.2f}ms")
    print(f"  • Byzantine detections (last hour): {metrics.byzantine_detections_last_hour}")
    print(f"  • Active nodes: {metrics.active_nodes}")
    print(f"  • Healthy nodes: {metrics.healthy_nodes}")
    print(f"  • Error rate: {metrics.error_rate:.2%}")
    print(f"  • Rate limit violations: {metrics.rate_limit_violations}")

    # Validate
    assert metrics.events_processed_last_minute == 50, "Events count mismatch"
    assert metrics.byzantine_detections_last_hour == 5, "Detection count mismatch"
    assert metrics.active_nodes == 5, "Active nodes mismatch"
    assert metrics.healthy_nodes == 5, "Healthy nodes mismatch"
    assert metrics.rate_limit_violations == 2, "Rate limit violations mismatch"

    print("\n✅ TEST 1 PASSED: Metrics collection working correctly")


async def test_alerting():
    """Test alert manager functionality"""
    print("\n" + "=" * 80)
    print("TEST 2: Alerting System")
    print("=" * 80)

    alert_manager = AlertManager()

    # Test 1: Normal metrics (no alerts)
    print("\n📊 Test 2.1: Normal operation (should have no alerts)")
    normal_metrics = SystemMetrics(
        timestamp=datetime.now(),
        avg_response_time_ms=500,
        p99_response_time_ms=1500,
        error_rate=0.001,
        detection_rate=0.95,
        rate_limit_violations=5,
        active_nodes=100,
        healthy_nodes=98
    )

    alerts = alert_manager.check_metrics(normal_metrics)
    print(f"  Alerts triggered: {len(alerts)}")
    assert len(alerts) == 0, "Should have no alerts for normal metrics"
    print("  ✅ No alerts (correct)")

    # Test 2: High response time (warning)
    print("\n⚠️  Test 2.2: High response time (should trigger WARNING)")
    slow_metrics = SystemMetrics(
        timestamp=datetime.now(),
        avg_response_time_ms=6000,  # Above 5000ms threshold
        p99_response_time_ms=8000,
        error_rate=0.001,
        detection_rate=0.95,
        rate_limit_violations=5,
        active_nodes=100,
        healthy_nodes=98
    )

    alerts = alert_manager.check_metrics(slow_metrics)
    print(f"  Alerts triggered: {len(alerts)}")
    for alert in alerts:
        print(f"    • {alert.severity.upper()}: {alert.message}")
        print(f"      Resolution: {alert.resolution}")

    assert len(alerts) >= 1, "Should have at least 1 alert for high response time"
    assert any(a.severity == "warning" for a in alerts), "Should have warning alert"
    print("  ✅ Warning alert triggered (correct)")

    # Test 3: Critical issues
    print("\n🚨 Test 2.3: Critical issues (should trigger CRITICAL alerts)")
    critical_metrics = SystemMetrics(
        timestamp=datetime.now(),
        avg_response_time_ms=6000,
        p99_response_time_ms=15000,  # Above 10000ms threshold
        error_rate=0.08,  # Above 5% threshold
        detection_rate=0.80,  # Below 85% threshold
        rate_limit_violations=150,  # Above 100/hour threshold
        active_nodes=100,
        healthy_nodes=80  # Only 80% healthy (below 90% threshold)
    )

    alerts = alert_manager.check_metrics(critical_metrics)
    print(f"  Alerts triggered: {len(alerts)}")
    for alert in alerts:
        print(f"    • {alert.severity.upper()}: {alert.message}")

    critical_alerts = [a for a in alerts if a.severity == "critical"]
    print(f"\n  Critical alerts: {len(critical_alerts)}")
    assert len(critical_alerts) >= 2, "Should have multiple critical alerts"
    print("  ✅ Critical alerts triggered (correct)")

    # Test alert categories
    categories = set(a.category for a in alerts)
    print(f"\n  Alert categories: {categories}")
    expected_categories = {"performance", "availability", "security", "capacity"}
    assert len(categories & expected_categories) > 0, "Should have alerts from expected categories"

    print("\n✅ TEST 2 PASSED: Alerting system working correctly")


async def test_production_monitor():
    """Test full production monitor"""
    print("\n" + "=" * 80)
    print("TEST 3: Production Monitor Integration")
    print("=" * 80)

    monitor = ProductionMonitor(check_interval_seconds=1)

    # Simulate some activity
    print("\n📊 Simulating production activity...")
    for i in range(20):
        monitor.metrics_collector.record_credit_issued(
            node_id=f"node_{i % 3}",
            credits=100.0,
            event_type="quality_gradient",
            response_time_ms=500 + i * 20
        )

    for i in range(3):
        monitor.metrics_collector.record_byzantine_detection(
            detector_id=f"validator_{i}",
            detected_id=f"attacker_{i}",
            evidence={"pogq": 0.20}
        )

    # Get status report
    print("\n📋 Generating status report...")
    report = monitor.get_status_report()

    print(f"\n🔍 Production Monitor Status Report:")
    print(f"  • System Status: {report['system_status'].upper()}")
    print(f"  • Timestamp: {report['timestamp']}")
    print(f"\n  Throughput:")
    print(f"    - Events/minute: {report['metrics']['throughput']['events_per_minute']}")
    print(f"    - Credits/minute: {report['metrics']['throughput']['credits_per_minute']}")
    print(f"\n  Performance:")
    print(f"    - Avg response: {report['metrics']['performance']['avg_response_ms']}ms")
    print(f"    - P95 response: {report['metrics']['performance']['p95_response_ms']}ms")
    print(f"    - P99 response: {report['metrics']['performance']['p99_response_ms']}ms")
    print(f"\n  Security:")
    print(f"    - Byzantine detections: {report['metrics']['security']['byzantine_detections_last_hour']}")
    print(f"    - Detection rate: {report['metrics']['security']['detection_rate']:.2%}")
    print(f"\n  Health:")
    print(f"    - Active nodes: {report['metrics']['health']['active_nodes']}")
    print(f"    - Healthy nodes: {report['metrics']['health']['healthy_nodes']}")
    print(f"    - Error rate: {report['metrics']['health']['error_rate']:.2%}")
    print(f"\n  Alerts: {len(report['alerts'])}")
    for alert in report['alerts']:
        print(f"    • [{alert['severity'].upper()}] {alert['message']}")

    # Validate report structure
    assert 'system_status' in report, "Missing system_status"
    assert 'metrics' in report, "Missing metrics"
    assert 'alerts' in report, "Missing alerts"
    assert report['system_status'] in ['healthy', 'degraded', 'critical'], "Invalid system status"

    print("\n✅ TEST 3 PASSED: Production monitor integration working correctly")


async def test_monitoring_loop():
    """Test continuous monitoring loop"""
    print("\n" + "=" * 80)
    print("TEST 4: Continuous Monitoring Loop")
    print("=" * 80)

    monitor = ProductionMonitor(check_interval_seconds=1)

    # Start monitoring in background
    print("\n🔄 Starting monitoring loop...")
    monitor_task = asyncio.create_task(monitor.start())

    # Simulate activity while monitoring
    print("📊 Simulating activity for 3 seconds...")
    for i in range(3):
        monitor.metrics_collector.record_credit_issued(
            node_id=f"node_{i}",
            credits=100.0,
            event_type="quality_gradient",
            response_time_ms=500
        )
        await asyncio.sleep(1)

    # Stop monitoring
    print("🛑 Stopping monitoring loop...")
    monitor.stop()
    await asyncio.sleep(0.1)  # Let it shut down gracefully

    # Check that loop stopped
    assert not monitor.is_running, "Monitor should have stopped"

    print("\n✅ TEST 4 PASSED: Continuous monitoring loop working correctly")


async def run_all_tests():
    """Run all monitoring tests"""
    print("\n" + "=" * 80)
    print("PRODUCTION MONITORING SYSTEM TEST SUITE")
    print("=" * 80)

    start_time = datetime.now()

    try:
        await test_metrics_collection()
        await test_alerting()
        await test_production_monitor()
        await test_monitoring_loop()

        duration = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print(f"Total duration: {duration:.2f} seconds")
        print(f"Tests run: 4")
        print(f"Success rate: 100%")
        print("\n🎉 Production monitoring system is ready for deployment!")

        assert len(alert.history) == 1
        assert alert.history[0].severity == "critical"

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
