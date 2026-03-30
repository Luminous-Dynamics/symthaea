#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Artifact Schema Validator
==========================

Validates all 6 JSON artifacts against schema.
Catches corruption, missing fields, type mismatches.

Usage:
    python scripts/validate_artifacts.py                    # Validate all
    python scripts/validate_artifacts.py results/artifacts_*  # Specific dir
    python scripts/validate_artifacts.py --strict           # Fail on warnings

Author: Luminous Dynamics
Date: November 8, 2025
"""

import json
import glob
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Try to import jsonschema; gracefully degrade if not available
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("⚠️  jsonschema not installed; falling back to basic validation")
    print("   Install: pip install jsonschema")
    print("")

# Expected files and their schema keys
EXPECTED_ARTIFACTS = {
    'detection_metrics.json': 'detection_metrics',
    'per_bucket_fpr.json': 'per_bucket_fpr',
    'ema_vs_raw.json': 'ema_vs_raw',
    'coord_median_diagnostics.json': 'coord_median_diagnostics',
    'bootstrap_ci.json': 'bootstrap_ci',
    'model_metrics.json': 'model_metrics'
}


def load_schema() -> Dict:
    """Load JSON schema from file."""
    schema_path = Path(__file__).parent / 'artifact_schema.json'

    with open(schema_path, 'r') as f:
        return json.load(f)


def basic_validate(data: Dict, artifact_type: str) -> List[str]:
    """
    Basic validation without jsonschema library.
    Checks for required top-level keys only.
    """
    errors = []

    # Define required keys per artifact type
    required_keys = {
        'detection_metrics': ['auroc', 'tpr_at_target_fpr', 'fpr_at_target_tpr', 'roc_curve'],
        'per_bucket_fpr': ['buckets', 'alpha'],
        'ema_vs_raw': ['experiments'],
        'coord_median_diagnostics': ['experiments'],
        'bootstrap_ci': ['metric', 'mean', 'ci_lower', 'ci_upper'],
        'model_metrics': ['test_accuracy', 'test_loss']
    }

    if artifact_type not in required_keys:
        errors.append(f"Unknown artifact type: {artifact_type}")
        return errors

    for key in required_keys[artifact_type]:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    # Check for NaN values in numeric fields
    if 'auroc' in data and (data['auroc'] == 'NaN' or data['auroc'] is None):
        errors.append("AUROC is NaN or null")

    if 'test_accuracy' in data and (data['test_accuracy'] == 'NaN' or data['test_accuracy'] is None):
        errors.append("test_accuracy is NaN or null")

    return errors


def schema_validate(data: Dict, artifact_type: str, schema: Dict) -> List[str]:
    """
    Full schema validation using jsonschema library.
    """
    if not HAS_JSONSCHEMA:
        return basic_validate(data, artifact_type)

    errors = []

    # Get the definition for this artifact type
    if artifact_type not in schema['definitions']:
        errors.append(f"No schema definition for: {artifact_type}")
        return errors

    artifact_schema = schema['definitions'][artifact_type]

    try:
        jsonschema.validate(instance=data, schema=artifact_schema)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation failed: {e.message}")
        if e.path:
            errors.append(f"  Path: {'.'.join(str(p) for p in e.path)}")
    except jsonschema.SchemaError as e:
        errors.append(f"Invalid schema: {e.message}")

    return errors


def validate_artifact_dir(artifact_dir: str, schema: Dict, strict: bool = False) -> Tuple[int, int]:
    """
    Validate all artifacts in a directory.

    Returns:
        (num_valid, num_invalid)
    """
    artifact_path = Path(artifact_dir)
    experiment_name = artifact_path.name

    all_valid = True
    warnings = []

    # Check for missing files
    for filename, artifact_type in EXPECTED_ARTIFACTS.items():
        file_path = artifact_path / filename

        if not file_path.exists():
            print(f"  ⚠️  Missing: {filename}")
            warnings.append(f"Missing file: {filename}")
            if strict:
                all_valid = False
            continue

        # Try to parse JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ❌ Malformed JSON: {filename}")
            print(f"     Error: {e}")
            all_valid = False
            continue

        # Validate against schema
        if HAS_JSONSCHEMA:
            errors = schema_validate(data, artifact_type, schema)
        else:
            errors = basic_validate(data, artifact_type)

        if errors:
            print(f"  ❌ Invalid: {filename}")
            for error in errors:
                print(f"     {error}")
            all_valid = False
        else:
            print(f"  ✅ {filename}")

    if warnings and not strict:
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    return (1 if all_valid else 0, 0 if all_valid else 1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate Zero-TrustML artifacts')
    parser.add_argument('directories', nargs='*', help='Artifact directories to validate')
    parser.add_argument('--strict', action='store_true', help='Fail on missing files')
    parser.add_argument('--all', action='store_true', help='Validate all artifacts in results/')

    args = parser.parse_args()

    print("=" * 70)
    print("Zero-TrustML Artifact Validator")
    print("=" * 70)
    print("")

    # Load schema
    try:
        schema = load_schema()
        print(f"✅ Loaded schema (version {schema.get('version', 'unknown')})")
    except FileNotFoundError:
        print("❌ Schema file not found: scripts/artifact_schema.json")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid schema JSON: {e}")
        sys.exit(1)

    print("")

    # Determine which directories to validate
    if args.all or not args.directories:
        artifact_dirs = sorted(glob.glob("results/artifacts_*"))
        if not artifact_dirs:
            print("ℹ️  No artifact directories found in results/")
            sys.exit(0)
    else:
        artifact_dirs = args.directories

    print(f"📊 Validating {len(artifact_dirs)} artifact directories...")
    print("")

    # Validate each directory
    total_valid = 0
    total_invalid = 0

    for artifact_dir in artifact_dirs:
        print(f"Validating: {Path(artifact_dir).name}")
        valid, invalid = validate_artifact_dir(artifact_dir, schema, args.strict)
        total_valid += valid
        total_invalid += invalid
        print("")

    # Summary
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print(f"Valid: {total_valid}")
    print(f"Invalid: {total_invalid}")

    if total_invalid > 0:
        print("")
        print("❌ Some artifacts failed validation")
        sys.exit(1)
    else:
        print("")
        print("✅ All artifacts passed validation")
        sys.exit(0)


if __name__ == '__main__':
    main()
