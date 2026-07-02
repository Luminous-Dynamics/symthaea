#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Fix GPU device placement for all baseline Server classes."""

import re
from pathlib import Path

BASELINES = [
    "baselines/fedavg.py",
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
    if "self.device = device" in content and "def __init__(self, model: nn.Module, config:" in content and ", device: str = 'cpu')" in content:
        print(f"✓ {baseline_file} Server already has device")
        continue

    # Fix 1: Add device parameter to Server __init__ signature
    # Pattern: def __init__(self, model: nn.Module, config: XxxConfig):
    # Replace with: def __init__(self, model: nn.Module, config: XxxConfig, device: str = 'cpu'):
    pattern1 = r'(class \w+Server.*?def __init__\(self, model: nn\.Module, config: \w+Config)\):'
    replacement1 = r'\1, device: str = \'cpu\'):'
    content, count1 = re.subn(pattern1, replacement1, content, flags=re.DOTALL)

    # Fix 2: Add self.device = device and model.to(device) after self.config = config
    pattern2 = r'(self\.config = config\n)'
    replacement2 = r'\1        self.device = device\n        \n        # Move model to device\n        self.model = self.model.to(self.device)\n'
    content, count2 = re.subn(pattern2, replacement2, content, count=1)

    # Fix 3: Update create_*_experiment to pass device to Server
    # Pattern: server = XxxServer(global_model, config)
    # Replace with: server = XxxServer(global_model, config, device)
    pattern3 = r'(server = \w+Server\(global_model, config)\)'
    replacement3 = r'\1, device)'
    content, count3 = re.subn(pattern3, replacement3, content)

    if count1 > 0 or count2 > 0 or count3 > 0:
        path.write_text(content)
        print(f"✅ Fixed {baseline_file} (signature:{count1}, device:{count2}, create:{count3})")
    else:
        print(f"⚠️  Could not automatically fix {baseline_file}")

print("\nDone! All Server classes should now support GPU.")
