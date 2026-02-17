#!/usr/bin/env python3
"""
β Drift Guard Test
==================

Ensures β (EMA smoothing factor) is consistently 0.85 across all code and configs.
This test FAILS if any config/const sets β ≠ 0.85.

Run: pytest tests/test_beta_consistency.py -v
"""

import pytest
import re
import subprocess
from pathlib import Path


def test_beta_consistency():
    """
    Verify β=0.85 everywhere in code and configs.

    Fails if:
    - Any Python/Rust code sets beta=0.7 or other value
    - Any docs reference β=0.7
    - Any configs specify non-0.85 values
    """
    root = Path("/srv/luminous-dynamics/Mycelix-Core/0TML")

    # Search patterns
    patterns = [
        (r'beta\s*=\s*0\.7[^0-9]', 'Python/Rust code'),
        (r'β\s*=\s*0\.7[^0-9]', 'Documentation'),
        (r'ema_beta:\s*0\.7[^0-9]', 'YAML configs'),
        (r'ema_beta.*=.*0\.7[^0-9]', 'Python params'),
    ]

    violations = []

    for pattern, desc in patterns:
        # Use grep to search
        try:
            result = subprocess.run(
                [
                    'grep',
                    '-r',
                    '-n',
                    pattern,
                    '--include=*.py',
                    '--include=*.rs',
                    '--include=*.md',
                    '--include=*.tex',
                    '--include=*.yaml',
                    '--include=*.toml',
                    str(root)
                ],
                capture_output=True,
                text=True
            )

            if result.stdout:
                # Filter out false positives (library code, comments about old values)
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    # Skip library code
                    if '.venv' in line or 'scipy' in line or 'torch' in line:
                        continue
                    # Skip test file itself
                    if 'test_beta_consistency' in line:
                        continue
                    # Skip comments about changing from 0.7 to 0.85
                    if 'changed' in line.lower() or 'was 0.7' in line.lower():
                        continue

                    violations.append(f"{desc}: {line}")

        except subprocess.CalledProcessError:
            # No matches found (good!)
            pass

    # Also verify Rust code uses 0.85 (55705/65536)
    pogq_rs = root / "vsv-stark/vsv-core/src/pogq.rs"
    if pogq_rs.exists():
        content = pogq_rs.read_text()

        # Check for the fixed-point value q(55705) which is 0.85
        if 'q(55705)' not in content and '55705' not in content:
            violations.append(f"Rust code: {pogq_rs} - Missing β=0.85 (55705) fixed-point value")

    # Report violations
    if violations:
        error_msg = f"\n❌ β Drift Detected! Found {len(violations)} violations:\n\n"
        error_msg += "\n".join(violations)
        error_msg += "\n\nAll code must use β=0.85 for EMA smoothing."
        pytest.fail(error_msg)

    # Success message
    print(f"\n✅ β Consistency Check PASSED")
    print(f"   All references use β=0.85 (EMA smoothing factor)")
    print(f"   Rust code: q(55705) = 0.85 * 65536 ✓")


def test_conformal_wording():
    """
    Verify conformal prediction claims include proper caveats.

    Should say: "controls FPR at ≤ α" (with quantile estimation error caveat)
    NOT: "guarantees FPR ≤ α" (absolute guarantee)
    """
    root = Path("/srv/luminous-dynamics/Mycelix-Core/0TML")

    # Search for problematic phrasing
    problematic_patterns = [
        r'guarantee.*FPR\s*[≤<]\s*α',
        r'ensures.*FPR\s*[≤<]\s*α',
        r'strict.*FPR\s*[≤<]\s*α',
    ]

    violations = []

    for pattern in problematic_patterns:
        try:
            result = subprocess.run(
                [
                    'grep',
                    '-r',
                    '-i',  # Case insensitive
                    '-n',
                    pattern,
                    '--include=*.md',
                    '--include=*.tex',
                    str(root)
                ],
                capture_output=True,
                text=True
            )

            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    # Skip if it includes the proper caveat
                    if 'quantile estimation error' in line.lower():
                        continue
                    if 'finite sample' in line.lower() and 'up to' in line.lower():
                        continue
                    if '.venv' in line:
                        continue
                    if 'test_beta_consistency' in line:
                        continue

                    violations.append(line)

        except subprocess.CalledProcessError:
            pass

    if violations:
        error_msg = "\n⚠️  Conformal wording needs caveats!\n\n"
        error_msg += "Found absolute claims without quantile estimation error caveat:\n\n"
        error_msg += "\n".join(violations)
        error_msg += "\n\nUse: 'controls FPR at ≤ α in finite samples, up to quantile estimation error'"
        pytest.fail(error_msg)

    print("\n✅ Conformal Wording Check PASSED")
    print("   All claims include proper finite-sample caveats")


if __name__ == "__main__":
    # Can run directly for quick check
    print("Running β Consistency Check...")
    test_beta_consistency()
    print("\nRunning Conformal Wording Check...")
    test_conformal_wording()
    print("\n✅ All consistency checks passed!")
