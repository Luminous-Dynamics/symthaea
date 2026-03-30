#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Emergency Diagnostic Script
Check experiment status and capture errors
"""

import sys
import subprocess
from pathlib import Path

print("=" * 70)
print("🔍 EXPERIMENT DIAGNOSTIC")
print("=" * 70)

# Check process
result = subprocess.run(
    ["ps", "aux"],
    capture_output=True,
    text=True
)
process_lines = [line for line in result.stdout.split('\n') if 'matrix_runner' in line and 'grep' not in line]

if process_lines:
    print("\n✅ Process IS running:")
    for line in process_lines:
        print(f"   {line}")
else:
    print("\n❌ Process NOT running!")

# Check artifacts
results_dir = Path("results")
artifact_dirs = sorted(results_dir.glob("artifacts_*"))
print(f"\n📊 Found {len(artifact_dirs)} artifact directories")

# Check which have files
valid_count = 0
empty_count = 0

for artifact_dir in artifact_dirs[-5:]:  # Check last 5
    files = list(artifact_dir.glob("*"))
    if len(files) > 0:
        print(f"   ✅ {artifact_dir.name}: {len(files)} files")
        valid_count += 1
    else:
        print(f"   ❌ {artifact_dir.name}: EMPTY!")
        empty_count += 1

print(f"\n Summary:")
print(f"   Valid artifacts: {valid_count}")
print(f"   Empty artifacts: {empty_count}")

# Try to run ONE experiment manually
print("\n🧪 Running manual test experiment...")
print("   (This will help diagnose what's failing)")

try:
    test_config = """
experiment_name: diagnostic_test
seed: 42
dataset:
  name: emnist
  data_dir: datasets
model:
  type: simple_cnn
  params:
    num_classes: 10
federated:
  num_clients: 10
  batch_size: 32
  learning_rate: 0.01
  local_epochs: 1
  fraction_clients: 1.0
attack:
  type: sign_flip
  byzantine_ratio: 0.20
baselines:
  - fedavg
training:
  num_rounds: 2
  eval_every: 2
output_dir: results
data_split:
  type: iid
  seed: 42
"""

    test_path = "/tmp/diagnostic_test.yaml"
    with open(test_path, 'w') as f:
        f.write(test_config)

    print(f"   Saved test config to {test_path}")
    print("   Run manually: python experiments/runner.py --config /tmp/diagnostic_test.yaml")
    print("   This will show actual error messages!")

except Exception as e:
    print(f"   ❌ Error creating test config: {e}")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("1. KILL current experiments (they're not saving results!)")
print("   kill 764241")
print("")
print("2. Run diagnostic test:")
print("   python experiments/runner.py --config /tmp/diagnostic_test.yaml")
print("")
print("3. Fix the error, then restart with reduced scope:")
print("   python experiments/matrix_runner.py --config configs/reduced_sanity.yaml")
print("=" * 70)
