#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Experiment Monitor - Real-Time Health Checking
===============================================

Monitors running experiments to detect:
- Silent failures (no new artifacts)
- CUDA errors (check for assertion failures)
- Process crashes (check if PID still exists)
- Progress stalls (no new results in 30+ minutes)

Usage:
    python experiments/monitor_experiments.py --pid <PID>

Auto-detects experiment progress and alerts on problems.
"""

import argparse
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class ExperimentMonitor:
    """Monitor running experiments for health and progress"""

    def __init__(self, pid: int, results_dir: str = "results"):
        self.pid = pid
        self.results_dir = Path(results_dir)
        self.last_artifact_count = 0
        self.last_check_time = datetime.now()
        self.no_progress_threshold = timedelta(minutes=30)
        self.check_interval = 60  # seconds

    def is_process_running(self) -> bool:
        """Check if the experiment process is still running"""
        try:
            result = subprocess.run(
                ["ps", "-p", str(self.pid)],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"⚠️  Error checking process: {e}")
            return False

    def count_artifacts(self) -> int:
        """Count total artifact directories"""
        artifact_dirs = list(self.results_dir.glob("artifacts_*"))
        return len(artifact_dirs)

    def check_for_errors(self) -> List[str]:
        """Check recent results for errors"""
        errors = []

        # Check most recent 5 result files
        result_files = sorted(
            self.results_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:5]

        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)

                # Check if experiment completed
                if "metadata" in data and "end_time" in data["metadata"]:
                    # Experiment finished successfully
                    continue
                else:
                    # No end_time means it might have crashed
                    errors.append(f"Incomplete: {result_file.name}")

            except Exception as e:
                errors.append(f"Error reading {result_file.name}: {e}")

        return errors

    def get_latest_artifacts(self, n: int = 5) -> List[Path]:
        """Get n most recent artifact directories"""
        artifact_dirs = sorted(
            self.results_dir.glob("artifacts_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return artifact_dirs[:n]

    def check_artifact_contents(self) -> Dict[str, int]:
        """Check if recent artifacts contain actual data"""
        recent_artifacts = self.get_latest_artifacts(5)

        stats = {
            "total": len(recent_artifacts),
            "empty": 0,
            "valid": 0
        }

        for artifact_dir in recent_artifacts:
            files = list(artifact_dir.glob("*"))
            if len(files) == 0:
                stats["empty"] += 1
            else:
                # Check if detection_metrics.json exists
                if (artifact_dir / "detection_metrics.json").exists():
                    stats["valid"] += 1

        return stats

    def monitor_once(self) -> Dict:
        """Run a single monitoring check"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "process_running": self.is_process_running(),
            "artifact_count": self.count_artifacts(),
            "errors": self.check_for_errors(),
            "progress": False,
            "health": "UNKNOWN"
        }

        # Check for progress
        if status["artifact_count"] > self.last_artifact_count:
            status["progress"] = True
            self.last_artifact_count = status["artifact_count"]
            self.last_check_time = datetime.now()

        # Determine health status
        if not status["process_running"]:
            status["health"] = "DEAD"
        elif status["errors"]:
            status["health"] = "ERROR"
        elif status["progress"]:
            status["health"] = "HEALTHY"
        elif datetime.now() - self.last_check_time > self.no_progress_threshold:
            status["health"] = "STALLED"
        else:
            status["health"] = "RUNNING"

        # Check artifact quality
        artifact_stats = self.check_artifact_contents()
        status["artifact_stats"] = artifact_stats

        if artifact_stats["empty"] > 0 and artifact_stats["valid"] == 0:
            status["health"] = "FAILING_SILENTLY"

        return status

    def monitor_loop(self, duration_hours: Optional[float] = None):
        """Continuously monitor experiments"""
        print("=" * 70)
        print("🔍 EXPERIMENT MONITOR ACTIVE")
        print("=" * 70)
        print(f"Monitoring PID: {self.pid}")
        print(f"Results directory: {self.results_dir}")
        print(f"Check interval: {self.check_interval}s")
        print(f"No-progress threshold: {self.no_progress_threshold}")
        print()

        start_time = datetime.now()
        check_count = 0

        try:
            while True:
                check_count += 1
                status = self.monitor_once()

                # Print status
                print(f"\n[Check #{check_count}] {status['timestamp']}")
                print(f"  Health: {status['health']}")
                print(f"  Process: {'✓ Running' if status['process_running'] else '✗ DEAD'}")
                print(f"  Artifacts: {status['artifact_count']} total")
                print(f"  Progress: {'✓ YES' if status['progress'] else '- No new artifacts'}")

                if status["artifact_stats"]:
                    stats = status["artifact_stats"]
                    print(f"  Artifact Quality: {stats['valid']}/{stats['total']} valid, {stats['empty']} empty")

                if status["errors"]:
                    print(f"  Errors: {len(status['errors'])} detected")
                    for error in status["errors"][:3]:
                        print(f"    - {error}")

                # Alert on problems
                if status["health"] in ["DEAD", "ERROR", "FAILING_SILENTLY", "STALLED"]:
                    print("\n" + "="*70)
                    print(f"⚠️  ALERT: Experiment health is {status['health']}!")
                    print("="*70)

                    if status["health"] == "DEAD":
                        print("\n🚨 Process has terminated!")
                        print("   Check logs for errors.")
                        break

                    elif status["health"] == "FAILING_SILENTLY":
                        print("\n🚨 Experiments completing but producing NO RESULTS!")
                        print("   This is exactly what happened before.")
                        print("   Likely CUDA assertion error or configuration bug.")
                        print("   STOP THE PROCESS IMMEDIATELY!")
                        break

                    elif status["health"] == "STALLED":
                        print(f"\n⚠️  No progress in {self.no_progress_threshold}")
                        print("   Process may be hung or stuck.")

                # Check duration limit
                if duration_hours:
                    elapsed = (datetime.now() - start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        print(f"\n✓ Monitoring duration limit reached ({duration_hours}h)")
                        break

                # Wait for next check
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring interrupted by user")
            print(f"   Ran {check_count} checks")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor experiment health in real-time"
    )
    parser.add_argument(
        "--pid",
        type=int,
        required=True,
        help="Process ID of matrix_runner.py"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory (default: results)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N hours then exit (default: run forever)"
    )
    args = parser.parse_args()

    monitor = ExperimentMonitor(args.pid, args.results_dir)
    monitor.check_interval = args.interval
    monitor.monitor_loop(duration_hours=args.duration)


if __name__ == "__main__":
    main()
