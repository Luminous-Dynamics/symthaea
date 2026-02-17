"""
Phase 4: Provenance & Reproducibility
======================================

Provides cryptographic provenance for:
1. Calibration blobs (PCA params, quantiles, profiles)
2. Run manifests (commit, config, seeds, artifact hashes)

Ensures:
- Detector calibration is tamper-evident
- Experiment results are reproducible
- Audit trail for compliance
"""

from .calibration_blob import CalibrationBlob, save_calibration, load_calibration
from .run_manifest import RunManifest, create_manifest, verify_manifest

__all__ = [
    'CalibrationBlob',
    'save_calibration',
    'load_calibration',
    'RunManifest',
    'create_manifest',
    'verify_manifest',
]
