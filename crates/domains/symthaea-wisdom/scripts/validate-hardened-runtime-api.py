#!/usr/bin/env python3
"""Validate that hardened runtime APIs win over legacy compatibility features."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []

cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
default_match = re.search(r'^default\s*=\s*\[(.*?)\]$', cargo, re.MULTILINE)
if not default_match or '"hardened-runtime-api"' not in default_match.group(1):
    errors.append("Cargo.toml: default features must include hardened-runtime-api")
for feature in ("hardened-runtime-api", "legacy-fail-stop-apis", "legacy-direct-state-mutation"):
    if not re.search(rf'^{re.escape(feature)}\s*=\s*\[\]$', cargo, re.MULTILINE):
        errors.append(f"Cargo.toml: missing empty feature declaration {feature}")

fail_stop_guard = (
    '#[cfg(any(test, all(feature = "legacy-fail-stop-apis", '
    'not(feature = "hardened-runtime-api"))))]'
)
direct_guard = (
    '#[cfg(any(test, all(feature = "legacy-direct-state-mutation", '
    'not(feature = "hardened-runtime-api"))))]'
)

required_fail_stop = {
    "src/evidence.rs": ["pub fn append("],
    "src/meta_cognition.rs": [
        "pub fn begin_prediction(",
        "pub fn update_self_model(",
        "pub fn update_self_model_with_complexity(",
    ],
    "src/lib.rs": [
        "pub fn set_ethics_policy_with_evidence(",
        "pub fn sweep_expired_action_permits_with_evidence(",
        "pub fn revoke_all_action_permits_with_evidence(",
        "pub fn issue_action_permit_with_evidence(",
        "pub fn consume_action_permit_with_evidence(",
        "pub fn update_from_experience(",
        "pub fn update_from_observation(",
        "pub fn update_from_observation_with_evidence(",
    ],
}
required_direct = {
    "src/lib.rs": [
        "pub fn set_ethics_policy(",
        "pub fn issue_action_permit(",
        "pub fn consume_action_permit(",
    ],
}

def require_guard(relative: str, signature: str, guard: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8")
    position = text.find(signature)
    if position < 0:
        errors.append(f"{relative}: missing compatibility method {signature!r}")
        return
    prefix = text[max(0, position - 500):position]
    if guard not in prefix:
        errors.append(f"{relative}: {signature!r} is not protected by hardened feature precedence")

for relative, signatures in required_fail_stop.items():
    for signature in signatures:
        require_guard(relative, signature, fail_stop_guard)
for relative, signatures in required_direct.items():
    for signature in signatures:
        require_guard(relative, signature, direct_guard)

lib = (ROOT / "src/lib.rs").read_text(encoding="utf-8")
if "pub(crate) fn set_ethics_policy_internal(" not in lib:
    errors.append("src/lib.rs: missing crate-private replay/bootstrap policy initializer")
for relative in ("src/deployment.rs", "src/service.rs"):
    text = (ROOT / relative).read_text(encoding="utf-8")
    if ".set_ethics_policy(" in text:
        errors.append(f"{relative}: production path calls public unjournaled policy mutation")

for path in sorted((ROOT / "tests").glob("*.rs")):
    text = path.read_text(encoding="utf-8")
    for method in (
        ".append(",
        ".begin_prediction(",
        ".update_self_model(",
        ".update_self_model_with_complexity(",
        ".set_ethics_policy_with_evidence(",
        ".issue_action_permit_with_evidence(",
        ".consume_action_permit_with_evidence(",
        ".update_from_experience(",
        ".update_from_observation(",
        ".update_from_observation_with_evidence(",
    ):
        if method in text:
            errors.append(
                f"{path.relative_to(ROOT)}: integration test depends on hidden compatibility method {method[:-1]}"
            )

if errors:
    print("hardened runtime API validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print("validated hardened runtime API feature precedence")
