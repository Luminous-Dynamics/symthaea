#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Progress Tracker: Resumable Experiment Execution
================================================

Tracks completed experiments in a manifest file to enable:
- Safe resumption after crashes or interruptions
- Skip already-completed experiments
- Progress monitoring across long runs

Usage:
    from scripts.progress_tracker import ProgressTracker

    tracker = ProgressTracker("progress_manifests/sanity_slice.json")

    for config in experiment_configs:
        key = tracker.make_key(config)

        if tracker.is_completed(key):
            print(f"Skipping {key} (already done)")
            continue

        # Run experiment
        run_experiment(config)

        # Mark complete
        tracker.mark_completed(key, artifact_dir="results/artifacts_...")

    # Summary
    print(tracker.summary())

Author: Luminous Dynamics
Date: November 8, 2025
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ProgressTracker:
    """
    Tracks experiment completion status for resumability.
    """

    def __init__(self, manifest_path: str):
        """
        Initialize tracker.

        Args:
            manifest_path: Path to JSON manifest file
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing manifest or create new
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'created': datetime.utcnow().isoformat() + "Z",
                'updated': datetime.utcnow().isoformat() + "Z",
                'completed': {},  # key -> {timestamp, artifact_dir, status}
                'failed': {},  # key -> {timestamp, error}
                'metadata': {}
            }

    def make_key(self, config: Dict) -> str:
        """
        Generate unique key for experiment config.

        Args:
            config: Experiment configuration dict

        Returns:
            Unique key string: "dataset_attack_defense_seed{seed}"
        """
        dataset = config.get('dataset', {}).get('name', 'unknown')
        attack = config.get('attack', {}).get('type', 'unknown')
        defense = config.get('baselines', ['unknown'])[0]
        seed = config.get('seed', 0)

        # Check if non-IID
        split_type = config.get('data_split', {}).get('type', 'iid')
        if split_type == 'dirichlet':
            alpha = config.get('data_split', {}).get('alpha', 0.5)
            dataset = f"{dataset}_noniid_{alpha}"

        return f"{dataset}_{attack}_{defense}_seed{seed}"

    def is_completed(self, key: str) -> bool:
        """Check if experiment is already completed."""
        return key in self.data['completed']

    def is_failed(self, key: str) -> bool:
        """Check if experiment previously failed."""
        return key in self.data['failed']

    def mark_completed(
        self,
        key: str,
        artifact_dir: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Mark experiment as completed.

        Args:
            key: Experiment key
            artifact_dir: Path to result artifacts
            metadata: Additional metadata (optional)
        """
        self.data['completed'][key] = {
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'artifact_dir': artifact_dir,
            'metadata': metadata or {}
        }

        # Remove from failed if present
        if key in self.data['failed']:
            del self.data['failed'][key]

        self._save()

    def mark_failed(self, key: str, error: str):
        """
        Mark experiment as failed.

        Args:
            key: Experiment key
            error: Error message or traceback
        """
        self.data['failed'][key] = {
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'error': error
        }

        self._save()

    def get_pending(self, all_keys: List[str]) -> List[str]:
        """
        Get list of pending experiments.

        Args:
            all_keys: List of all experiment keys

        Returns:
            List of keys that are not completed or failed
        """
        completed = set(self.data['completed'].keys())
        return [k for k in all_keys if k not in completed]

    def summary(self) -> Dict:
        """
        Get progress summary.

        Returns:
            Summary dict with counts and stats
        """
        total_completed = len(self.data['completed'])
        total_failed = len(self.data['failed'])

        return {
            'total_completed': total_completed,
            'total_failed': total_failed,
            'completion_rate': total_completed / (total_completed + total_failed)
                               if (total_completed + total_failed) > 0 else 0,
            'created': self.data['created'],
            'updated': self.data['updated']
        }

    def print_summary(self):
        """Print human-readable summary."""
        summary = self.summary()

        print("=" * 70)
        print("Progress Tracker Summary")
        print("=" * 70)
        print(f"Completed: {summary['total_completed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Completion rate: {summary['completion_rate'] * 100:.1f}%")
        print(f"Created: {summary['created']}")
        print(f"Updated: {summary['updated']}")
        print("=" * 70)

    def _save(self):
        """Save manifest to disk."""
        self.data['updated'] = datetime.utcnow().isoformat() + "Z"

        with open(self.manifest_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def reset(self):
        """Clear all progress (use with caution!)."""
        self.data['completed'] = {}
        self.data['failed'] = {}
        self._save()


def main():
    """CLI interface for progress tracker."""
    import argparse

    parser = argparse.ArgumentParser(description='Progress tracker CLI')
    parser.add_argument('manifest', help='Path to progress manifest')
    parser.add_argument('--summary', action='store_true', help='Show summary')
    parser.add_argument('--reset', action='store_true', help='Reset all progress')
    parser.add_argument('--list-completed', action='store_true', help='List completed')
    parser.add_argument('--list-failed', action='store_true', help='List failed')

    args = parser.parse_args()

    tracker = ProgressTracker(args.manifest)

    if args.reset:
        confirm = input("Reset all progress? (yes/no): ")
        if confirm.lower() == 'yes':
            tracker.reset()
            print("✅ Progress reset")
        else:
            print("Aborted")

    elif args.summary:
        tracker.print_summary()

    elif args.list_completed:
        print("\nCompleted experiments:")
        for key, info in tracker.data['completed'].items():
            print(f"  {key}")
            print(f"    Timestamp: {info['timestamp']}")
            if info.get('artifact_dir'):
                print(f"    Artifacts: {info['artifact_dir']}")

    elif args.list_failed:
        print("\nFailed experiments:")
        for key, info in tracker.data['failed'].items():
            print(f"  {key}")
            print(f"    Timestamp: {info['timestamp']}")
            print(f"    Error: {info['error'][:100]}...")

    else:
        tracker.print_summary()


if __name__ == '__main__':
    main()
