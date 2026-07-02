#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Fix GPU device placement for all baseline Client classes.

Adds `self.model = self.model.to(self.device)` after device assignment in __init__.
"""

import re
from pathlib import Path

# Baselines to fix (fed avg already fixed)
BASELINES = [
    "baselines/fedprox.py",
    "baselines/scaffold.py",
    "baselines/krum.py",
    "baselines/multikrum.py",
    "baselines/bulyan.py",
    "baselines/median.py",
]

for baseline_file in BASELINES:
    path = Path(baseline_file)

    if not path.exists():
        print(f"⚠️  Skipping {baseline_file} (not found)")
        continue

    content = path.read_text()

    # Check if already fixed
    if "self.model = self.model.to(self.device)" in content:
        print(f"✓ {baseline_file} already fixed")
        continue

    # Pattern: After self.device = device, add the .to(device) line
    # Look for client __init__ methods
    pattern = r'(self\.device = device\s*\n)'
    replacement = r'\1\n        # Move model to device\n        self.model = self.model.to(self.device)\n'

    new_content, count = re.subn(pattern, replacement, content, count=1)

    if count > 0:
        path.write_text(new_content)
        print(f"✅ Fixed {baseline_file}")
    else:
        print(f"⚠️  Could not automatically fix {baseline_file} (pattern not found)")
        print(f"   Manual fix required")

print("\nDone! All baselines should now support GPU training.")
