import json
import yaml
from jsonschema import validate, ValidationError
from schema import MANIFEST_SCHEMA

# Status Constants
APPROVED_CC0 = "APPROVED_CC0"
APPROVED_ATTRIBUTION_REQUIRED = "APPROVED_ATTRIBUTION_REQUIRED"
QUARANTINE_REVIEW = "QUARANTINE_REVIEW"
REJECTED = "REJECTED"

# Approved licenses and their commercial use status
LICENSES = {
    "CC0-1.0": {"status": APPROVED_CC0, "attribution": False},
    "CC-BY-4.0": {"status": APPROVED_ATTRIBUTION_REQUIRED, "attribution": True},
    "CC-BY-NC-4.0": {"status": REJECTED, "reason": "NC license forbidden"},
    "CC-BY-ND-4.0": {"status": REJECTED, "reason": "ND license forbidden"},
}

def validate_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)

    # Schema validation
    try:
        validate(instance=manifest, schema=MANIFEST_SCHEMA)
    except ValidationError as e:
        return REJECTED, f"Manifest schema validation failed: {e.message}"

    # Check License
    license_id = manifest.get('license', {}).get('id')
    license_info = LICENSES.get(license_id)

    if not license_info:
        return QUARANTINE_REVIEW, f"License {license_id} requires human review."

    if license_info["status"] == REJECTED:
        return REJECTED, license_info["reason"]

    # Check Source URL (already checked by jsonschema, but logic preserved)
    if not manifest.get('source', {}).get('source_url'):
        return REJECTED, "Missing source URL."

    # Check Creator for attribution licenses
    if license_info.get("attribution") and not manifest.get('source', {}).get('creator'):
        return QUARANTINE_REVIEW, "Missing creator for attribution-required license."

    # Check AI provenance
    ai_state = manifest.get('ai', {}).get('provenance_state')
    if ai_state == 'unknown_ai_provenance':
        return QUARANTINE_REVIEW, "Unknown AI provenance requires human review."

    return license_info["status"], "Asset manifest is valid."
