#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Nixward CROSS-015: mechanical source-boundary regression fence.
#
# This is intentionally a source check, not a semantic proof. Governed
# authority modules must consume typed NixServiceObservedStateV1 rather than
# diagnostic ServiceStatus/UnitInfo/SystemdObserver representations.
#
# Keep this allowlist explicit. When a new authority/effect-binding module is
# introduced, add it here before it can participate in the governed path.
#
# systemd_transport.rs is deliberately excluded from the direct-systemctl
# check because it is the single observational transport waist. It is still
# protected from importing diagnostic semantic models.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AUTHORITY_FILES=(
  crates/core/nixward/src/action/approver_evidence.rs
  crates/core/nixward/src/action/authorization.rs
  crates/core/nixward/src/action/daemon_incarnation.rs
  crates/core/nixward/src/action/local_approval.rs
  crates/core/nixward/src/action/local_approval_ipc.rs
  crates/core/nixward/src/action/local_approval_projection.rs
  crates/core/nixward/src/action/local_approval_runtime.rs
  crates/core/nixward/src/action/local_approval_socket.rs
  crates/core/nixward/src/action/local_approval_store.rs
  crates/core/nixward/src/action/local_approval_submission.rs
  crates/core/nixward/src/action/service_domain.rs
  crates/core/nixward/src/action/service_state.rs
  crates/core/nixward/src/action/systemd_transport.rs
  crates/core/nixward/src/action/temporal.rs
)

DIAGNOSTIC_IDENTIFIERS='\b(ServiceStatus|UnitInfo|SystemdObserver)\b'
SYSTEMCTL_PATTERN='(?:Command|process::Command)::new\(\s*["\x27]systemctl["\x27]'
LEGACY_COMMAND_PATTERN='NixOSCommand::Custom'
LEGACY_SERVICE_CONSTRUCTOR_PATTERN='\bServiceManager::(start|stop|restart|reload|enable|disable)[[:space:]]*\('
LEGACY_SERVICE_RENDERER_PATTERN='\bServiceManager::render_legacy_command[[:space:]]*\('
REVERSE_SERVICE_CONVERSION_PATTERN='impl[[:space:]]+(TryFrom|From)<[^>]*NixOSCommand[^>]*>[[:space:]]+for[[:space:]]+NixServiceOperationV1'
IMPLICIT_SERVICE_RESTART_PATTERN='_[[:space:]]*=>[[:space:]]*NixOSCommand::Custom[[:space:]]*\{[[:space:]]*command:[[:space:]]*["\x27]systemctl["\x27]'
VALIDATED_OPERATION_DESERIALIZATION_PATTERN='(?s)#\[derive\([^]]*Deserialize[^]]*\)]\s*(?:pub[[:space:]]+)?(?:struct|enum)[[:space:]]+NixServiceOperationV1\b|impl[[:space:]]+[^\n{]*Deserialize[^\n{]*\bfor[[:space:]]+NixServiceOperationV1\b'
OBSERVED_STATE_DESERIALIZATION_PATTERN='(?s)#\[derive\([^]]*Deserialize[^]]*\)]\s*(?:pub[[:space:]]+)?(?:struct|enum)[[:space:]]+NixServiceObservedStateV1\b|impl[[:space:]]+[^\n{]*Deserialize[^\n{]*\bfor[[:space:]]+NixServiceObservedStateV1\b'
OBSERVED_STATE_PUBLIC_CONSTRUCTOR_PATTERN='(?s)impl[[:space:]]+NixServiceObservedStateV1[[:space:]]*\{.*?pub[[:space:]]+(?:async[[:space:]]+)?fn[[:space:]]+new[[:space:]]*\('
CAPABILITIES_DESERIALIZATION_PATTERN='(?s)#\[derive\([^]]*Deserialize[^]]*\)]\s*(?:pub[[:space:]]+)?(?:struct|enum)[[:space:]]+NixServiceOperationCapabilitiesV1\b|impl[[:space:]]+[^\n{]*Deserialize[^\n{]*\bfor[[:space:]]+NixServiceOperationCapabilitiesV1\b'
CAPABILITIES_PUBLIC_CONSTRUCTOR_PATTERN='(?s)impl[[:space:]]+NixServiceOperationCapabilitiesV1[[:space:]]*\{.*?pub[[:space:]]+(?:async[[:space:]]+)?fn[[:space:]]+new[[:space:]]*\('
OBSERVATION_CRATE_WIDE_FACTORY_PATTERN='\bpub\(crate\)[[:space:]]+(?:async[[:space:]]+)?fn[[:space:]]+(parse_systemd_properties|parse_systemd_observation|from_observed_state)[[:space:]]*\('
ENABLEMENT_EVIDENCE_DESERIALIZATION_PATTERN='(?s)#\[derive\([^]]*Deserialize[^]]*\)]\s*(?:pub[[:space:]]+)?(?:struct|enum)[[:space:]]+NixServiceEnablementEvidenceV1\b|impl[[:space:]]+[^\n{]*Deserialize[^\n{]*\bfor[[:space:]]+NixServiceEnablementEvidenceV1\b'
ENABLEMENT_EVIDENCE_PUBLIC_CONSTRUCTOR_PATTERN='(?s)impl[[:space:]]+NixServiceEnablementEvidenceV1[[:space:]]*\{.*?pub[[:space:]]+(?:async[[:space:]]+)?fn[[:space:]]+new[[:space:]]*\('
OBSERVATION_PUBLIC_FACTORY_PATTERN='\bpub[[:space:]]+(?:async[[:space:]]+)?fn[[:space:]]+(parse_systemd_properties|parse_systemd_observation|from_observed_state)[[:space:]]*\('
SYSTEMD_TRANSPORT_PUBLIC_API_PATTERN='\bpub[[:space:]]+(?:async[[:space:]]+)?fn[[:space:]]+(observe_service_properties|observe_service_state_properties)[[:space:]]*\('

scan_diagnostic_boundary() {
  local file="$1"
  rg -n --pcre2 "${DIAGNOSTIC_IDENTIFIERS}" "$file"
}

scan_direct_systemctl() {
  local file="$1"
  rg -n --pcre2 "${SYSTEMCTL_PATTERN}" "$file"
}

scan_legacy_custom_command() {
  local file="$1"
  rg -n --pcre2 "${LEGACY_COMMAND_PATTERN}" "$file"
}

scan_legacy_service_constructor() {
  local file="$1"
  rg -n --pcre2 "${LEGACY_SERVICE_CONSTRUCTOR_PATTERN}" "$file"
}

scan_public_observation_factory() {
  local file="$1"
  rg -n --pcre2 "${OBSERVATION_PUBLIC_FACTORY_PATTERN}" "$file"
}

scan_public_systemd_transport_api() {
  local file="$1"
  rg -n --pcre2 "${SYSTEMD_TRANSPORT_PUBLIC_API_PATTERN}" "$file"
}

scan_legacy_service_renderer() {
  local file="$1"
  rg -n --pcre2 "${LEGACY_SERVICE_RENDERER_PATTERN}" "$file"
}

scan_reverse_service_conversion() {
  local file="$1"
  rg -n --pcre2 "${REVERSE_SERVICE_CONVERSION_PATTERN}" "$file"
}

scan_implicit_service_restart() {
  local file="$1"
  rg -n --pcre2 "${IMPLICIT_SERVICE_RESTART_PATTERN}" "$file"
}

run_boundary_check() {
  local failed=0
  local file

  for file in "${AUTHORITY_FILES[@]}"; do
    if [[ ! -f "${ROOT}/${file}" ]]; then
      echo "ERROR: protected Nixward authority file is missing: ${file}" >&2
      failed=1
      continue
    fi

    if matches="$(scan_diagnostic_boundary "${ROOT}/${file}")"; then
      echo "ERROR: diagnostic systemd type crossed the governed boundary: ${file}" >&2
      echo "${matches}" >&2
      failed=1
    fi
  done

  # The transport waist is allowed to execute systemctl. Every other protected
  # authority module must remain free of direct host observation.
  # CROSS-022: governed authority/effect-binding modules must not consume the
  # legacy Custom command representation as semantic input.
  for file in "${AUTHORITY_FILES[@]}"; do
    [[ "${file}" == "crates/core/nixward/src/action/authorization.rs" ]] && continue
    if matches="$(scan_legacy_custom_command "${ROOT}/${file}")"; then
      echo "ERROR: governed authority module consumes legacy NixOSCommand::Custom: ${file}" >&2
      echo "${matches}" >&2
      failed=1
    fi
  done

  # CROSS-022: compatibility rendering is not an authority primitive.
  for file in "${AUTHORITY_FILES[@]}"; do
    [[ "${file}" == "crates/core/nixward/src/action/service_manager.rs" ]] && continue
    if matches="$(scan_legacy_service_renderer "${ROOT}/${file}")"; then
      echo "ERROR: governed authority module invokes the legacy service renderer: ${file}" >&2
      echo "${matches}" >&2
      failed=1
    fi
  done

  # CROSS-022: the service domain is one-way from semantic data toward the
  # compatibility renderer.
  if matches="$(scan_reverse_service_conversion "${ROOT}/crates/core/nixward/src")"; then
    echo "ERROR: reverse NixOSCommand -> NixServiceOperationV1 conversion is forbidden" >&2
    echo "${matches}" >&2
    failed=1
  fi

  if matches="$(rg -n '\bNixOSCommand\b' "${ROOT}/crates/core/nixward/src/action/service_domain.rs")"; then
    echo "ERROR: typed service domain must not depend on legacy NixOSCommand representation" >&2
    echo "${matches}" >&2
    failed=1
  fi

  # Validated service-domain and observed-state aggregates must be constructed
  # through their parsing/validation boundaries. Their closed leaf enums may
  # deserialize, but the aggregates must not: deserializing them would bypass
  # canonicalization, identity checks, or observation provenance.
  if matches="$(rg -U -n --pcre2 "${VALIDATED_OPERATION_DESERIALIZATION_PATTERN}" "${ROOT}/crates/core/nixward/src/action/service_domain.rs")"; then
    echo "ERROR: validated NixServiceOperationV1 must not deserialize around its constructor" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(rg -n --pcre2 "${OBSERVATION_CRATE_WIDE_FACTORY_PATTERN}" "${ROOT}/crates/core/nixward/src/action/service_state.rs")"; then
    echo "ERROR: observation/evidence factories must not widen to crate visibility" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(rg -U -n --pcre2 "${CAPABILITIES_DESERIALIZATION_PATTERN}" "${ROOT}/crates/core/nixward/src/action/service_state.rs")"; then
    echo "ERROR: NixServiceOperationCapabilitiesV1 must not deserialize around its observation boundary" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(rg -U -n --pcre2 "${ENABLEMENT_EVIDENCE_DESERIALIZATION_PATTERN}" "${ROOT}/crates/core/nixward/src/action/service_state.rs")"; then
    echo "ERROR: NixServiceEnablementEvidenceV1 must not deserialize around its observation boundary" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(rg -U -n --pcre2 "${OBSERVED_STATE_DESERIALIZATION_PATTERN}" "${ROOT}/crates/core/nixward/src/action/service_state.rs")"; then
    echo "ERROR: NixServiceObservedStateV1 must not deserialize around its observation boundary" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(rg -U -n --pcre2 "${CAPABILITIES_PUBLIC_CONSTRUCTOR_PATTERN}" "${ROOT}/crates/core/nixward/src/action/service_state.rs")"; then
    echo "ERROR: NixServiceOperationCapabilitiesV1 constructor must not be public" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(rg -U -n --pcre2 "${ENABLEMENT_EVIDENCE_PUBLIC_CONSTRUCTOR_PATTERN}" "${ROOT}/crates/core/nixward/src/action/service_state.rs")"; then
    echo "ERROR: NixServiceEnablementEvidenceV1 constructor must not be public" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(rg -U -n --pcre2 "${OBSERVED_STATE_PUBLIC_CONSTRUCTOR_PATTERN}" "${ROOT}/crates/core/nixward/src/action/service_state.rs")"; then
    echo "ERROR: NixServiceObservedStateV1 constructor must not be public" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(rg -n '^pub[[:space:]]+mod[[:space:]]+systemd_transport;' "${ROOT}/crates/core/nixward/src/action/mod.rs")"; then
    echo "ERROR: raw systemd transport module must remain crate-private" >&2
    echo "${matches}" >&2
    failed=1
  fi
  if matches="$(scan_public_systemd_transport_api "${ROOT}/crates/core/nixward/src/action/systemd_transport.rs")"; then
    echo "ERROR: raw systemd transport entry point must remain crate-private" >&2
    echo "${matches}" >&2
    failed=1
  fi

  if matches="$(scan_public_observation_factory "${ROOT}/crates/core/nixward/src/action/service_state.rs")"; then
    echo "ERROR: observed-state parsers/factories must not be public" >&2
    echo "${matches}" >&2
    failed=1
  fi
  for file in "${AUTHORITY_FILES[@]}"; do
    [[ "${file}" == "crates/core/nixward/src/action/systemd_transport.rs" ]] && continue
    if matches="$(scan_direct_systemctl "${ROOT}/${file}")"; then
      echo "ERROR: governed authority module directly invokes systemctl: ${file}" >&2
      echo "${matches}" >&2
      failed=1
    fi
  done

  local nixward_src="${ROOT}/crates/core/nixward/src"
  if matches="$(rg -n --pcre2 "${LEGACY_SERVICE_CONSTRUCTOR_PATTERN}" "${nixward_src}" --glob '*.rs' --glob '!**/action/service_manager.rs')"; then
    echo "ERROR: Nixward caller uses infallible legacy ServiceManager constructor" >&2
    echo "${matches}" >&2
    failed=1
  fi

  local daemon="${ROOT}/crates/core/nixward/src/bin/nixward_daemon.rs"
  if [[ ! -f "${daemon}" ]]; then
    echo "ERROR: Nixward daemon source is missing" >&2
    failed=1
  elif matches="$(scan_implicit_service_restart "${daemon}")"; then
    echo "ERROR: daemon wildcard action category may synthesize implicit systemctl restart" >&2
    echo "${matches}" >&2
    failed=1
  fi

  return "${failed}"
}

run_self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' RETURN

  printf '%s\n' 'let _ = ServiceStatus;' > "${tmp}/diagnostic.rs"
  if scan_diagnostic_boundary "${tmp}/diagnostic.rs"; then :; else
    echo "ERROR: CROSS-015 self-test failed to detect diagnostic type" >&2
    return 1
  fi

  printf '%s\n' 'Command::new("systemctl");' > "${tmp}/systemctl.rs"
  if scan_direct_systemctl "${tmp}/systemctl.rs"; then :; else
    echo "ERROR: CROSS-015 self-test failed to detect direct systemctl command use" >&2
    return 1
  fi

  printf '%s\n' '_ => NixOSCommand::Custom { command: "systemctl", args: vec!["restart"], };' > "${tmp}/implicit-restart.rs"
  if scan_implicit_service_restart "${tmp}/implicit-restart.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect implicit service restart fallback" >&2
    return 1
  fi

  printf '%s\n' 'NixServiceObservedStateV1::parse_systemd_properties(...);' > "${tmp}/typed.rs"
  if scan_diagnostic_boundary "${tmp}/typed.rs"; then
    echo "ERROR: CROSS-015 self-test falsely rejected typed evidence" >&2
    return 1
  fi

  printf '%s\n' 'NixOSCommand::Custom { .. };' > "${tmp}/legacy-domain.rs"
  if rg -n '\bNixOSCommand\b' "${tmp}/legacy-domain.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect legacy command dependency" >&2
    return 1
  fi

  printf '%s\n' 'ServiceManager::render_legacy_command(&typed);' > "${tmp}/legacy-renderer.rs"
  if scan_legacy_service_renderer "${tmp}/legacy-renderer.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect legacy renderer usage" >&2
    return 1
  fi

  printf '%s\n' 'impl TryFrom<&NixOSCommand> for NixServiceOperationV1 {}' > "${tmp}/reverse-service.rs"
  if scan_reverse_service_conversion "${tmp}/reverse-service.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect reverse service conversion" >&2
    return 1
  fi

  printf '%s\n' 'ServiceManager::restart("nginx");' > "${tmp}/legacy-service.rs"
  if scan_legacy_service_constructor "${tmp}/legacy-service.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect legacy ServiceManager constructor" >&2
    return 1
  fi

  printf '%s\n' 'let _ = NixOSCommand::Custom { .. };' > "${tmp}/legacy-custom.rs"
  if scan_legacy_custom_command "${tmp}/legacy-custom.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect legacy Custom command material" >&2
    return 1
  fi

  # The validated aggregates must be rejected, while the closed enum remains
  # permitted to deserialize. This prevents the fence itself from regressing
  # into an over-broad "no Deserialize in service_domain.rs" rule.
  printf '%s\n' $'#[derive(\n    Debug,\n    Deserialize,\n)]\npub struct NixServiceOperationV1;' > "${tmp}/aggregate-deserialize.rs"
  if rg -U -n --pcre2 "${VALIDATED_OPERATION_DESERIALIZATION_PATTERN}" "${tmp}/aggregate-deserialize.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect multiline aggregate deserialization" >&2
    return 1
  fi

  printf '%s\n' $'#[derive(\n    Deserialize,\n)]\nenum NixServiceOperationKindV1 { Start }' > "${tmp}/enum-deserialize.rs"
  if rg -U -n --pcre2 "${VALIDATED_OPERATION_DESERIALIZATION_PATTERN}" "${tmp}/enum-deserialize.rs"; then
    echo "ERROR: CROSS-022 self-test falsely rejected closed operation enum deserialization" >&2
    return 1
  fi

  printf '%s\n' $'#[derive(\n    Debug,\n    Deserialize,\n)]\npub struct NixServiceObservedStateV1;' > "${tmp}/observed-state-deserialize.rs"
  if rg -U -n --pcre2 "${OBSERVED_STATE_DESERIALIZATION_PATTERN}" "${tmp}/observed-state-deserialize.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect multiline observed-state deserialization" >&2
    return 1
  fi

  printf '%s\n' 'impl Deserialize for NixServiceOperationV1 { }' > "${tmp}/aggregate-custom-deserialize.rs"
  if rg -U -n --pcre2 "${VALIDATED_OPERATION_DESERIALIZATION_PATTERN}" "${tmp}/aggregate-custom-deserialize.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect custom aggregate deserialization" >&2
    return 1
  fi

  printf '%s\n' 'impl NixServiceObservedStateV1 {
    pub fn new(
        unit: String,
    ) {}' > "${tmp}/observed-state-public-constructor.rs"
  if rg -U -n --pcre2 "${OBSERVED_STATE_PUBLIC_CONSTRUCTOR_PATTERN}" "${tmp}/observed-state-public-constructor.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect public observed-state constructor" >&2
    return 1
  fi

  printf '%s\n' $'#[derive(\n    Deserialize,\n)]\npub struct NixServiceOperationCapabilitiesV1;' > "${tmp}/capabilities-deserialize.rs"
  if rg -U -n --pcre2 "${CAPABILITIES_DESERIALIZATION_PATTERN}" "${tmp}/capabilities-deserialize.rs"; then :; else
    echo "ERROR: CROSS-026 self-test failed to detect multiline capability-evidence deserialization" >&2
    return 1
  fi

  printf '%s\n' 'impl Deserialize for NixServiceOperationCapabilitiesV1 { }' > "${tmp}/capabilities-custom-deserialize.rs"
  if rg -U -n --pcre2 "${CAPABILITIES_DESERIALIZATION_PATTERN}" "${tmp}/capabilities-custom-deserialize.rs"; then :; else
    echo "ERROR: CROSS-026 self-test failed to detect custom capability-evidence deserialization" >&2
    return 1
  fi

  printf '%s\n' 'impl NixServiceOperationCapabilitiesV1 {
    pub async fn new(
        unit: String,
    ) {}' > "${tmp}/capabilities-public-async-constructor.rs"
  if rg -U -n --pcre2 "${CAPABILITIES_PUBLIC_CONSTRUCTOR_PATTERN}" "${tmp}/capabilities-public-async-constructor.rs"; then :; else
    echo "ERROR: CROSS-026 self-test failed to detect public async capability-evidence constructor" >&2
    return 1
  fi

  printf '%s\n' 'impl NixServiceOperationCapabilitiesV1 {
    pub fn new(
        unit: String,
    ) {}' > "${tmp}/capabilities-public-constructor.rs"
  if rg -U -n --pcre2 "${CAPABILITIES_PUBLIC_CONSTRUCTOR_PATTERN}" "${tmp}/capabilities-public-constructor.rs"; then :; else
    echo "ERROR: CROSS-026 self-test failed to detect public capability-evidence constructor" >&2
    return 1
  fi

  printf '%s\n' $'#[derive(\n    Deserialize,\n)]\npub struct NixServiceEnablementEvidenceV1;' > "${tmp}/enablement-evidence-deserialize.rs"
  if rg -U -n --pcre2 "${ENABLEMENT_EVIDENCE_DESERIALIZATION_PATTERN}" "${tmp}/enablement-evidence-deserialize.rs"; then :; else
    echo "ERROR: CROSS-025 self-test failed to detect multiline enablement-evidence deserialization" >&2
    return 1
  fi

  printf '%s\n' 'impl NixServiceEnablementEvidenceV1 {
    pub fn new(
        unit: String,
    ) {}' > "${tmp}/enablement-evidence-public-constructor.rs"
  if rg -U -n --pcre2 "${ENABLEMENT_EVIDENCE_PUBLIC_CONSTRUCTOR_PATTERN}" "${tmp}/enablement-evidence-public-constructor.rs"; then :; else
    echo "ERROR: CROSS-025 self-test failed to detect public enablement-evidence constructor" >&2
    return 1
  fi

  printf '%s\n' 'pub fn parse_systemd_properties(...) {}' > "${tmp}/observed-state-public-parser.rs"
  if scan_public_observation_factory "${tmp}/observed-state-public-parser.rs"; then :; else
    echo "ERROR: CROSS-023 self-test failed to detect public observed-state parser" >&2
    return 1
  fi

  printf '%s\n' 'pub fn from_observed_state(...) {}' > "${tmp}/observed-state-public-factory.rs"
  if scan_public_observation_factory "${tmp}/observed-state-public-factory.rs"; then :; else
    echo "ERROR: CROSS-023 self-test failed to detect public observed-state factory" >&2
    return 1
  fi

  printf '%s\n' 'pub(crate) fn parse_systemd_properties(...) {}' > "${tmp}/observed-state-crate-parser.rs"
  if scan_public_observation_factory "${tmp}/observed-state-crate-parser.rs"; then
    echo "ERROR: CROSS-023 self-test falsely rejected crate-private observation parser" >&2
    return 1
  fi

  printf '%s\n' 'pub(crate) fn parse_systemd_properties(...) {}' > "${tmp}/observed-state-crate-wide-parser.rs"
  if rg -n --pcre2 "${OBSERVATION_CRATE_WIDE_FACTORY_PATTERN}" "${tmp}/observed-state-crate-wide-parser.rs"; then :; else
    echo "ERROR: CROSS-027 self-test failed to detect crate-wide observation parser visibility" >&2
    return 1
  fi

  printf '%s\n' 'pub fn observe_service_properties(...) {}' > "${tmp}/systemd-transport-public.rs"
  if scan_public_systemd_transport_api "${tmp}/systemd-transport-public.rs"; then :; else
    echo "ERROR: CROSS-024 self-test failed to detect public systemd transport entry point" >&2
    return 1
  fi

  printf '%s\n' 'pub async fn observe_service_properties(...) {}' > "${tmp}/systemd-transport-public-async.rs"
  if scan_public_systemd_transport_api "${tmp}/systemd-transport-public-async.rs"; then :; else
    echo "ERROR: CROSS-024 self-test failed to detect public async systemd transport entry point" >&2
    return 1
  fi

  printf '%s\n' 'pub(crate) fn observe_service_properties(...) {}' > "${tmp}/systemd-transport-crate-private.rs"
  if scan_public_systemd_transport_api "${tmp}/systemd-transport-crate-private.rs"; then
    echo "ERROR: CROSS-024 self-test falsely rejected crate-private systemd transport entry point" >&2
    return 1
  fi

  printf '%s\n' 'pub mod systemd_transport;' > "${tmp}/systemd-transport-public-module.rs"
  if rg -n '^pub[[:space:]]+mod[[:space:]]+systemd_transport;' "${tmp}/systemd-transport-public-module.rs"; then :; else
    echo "ERROR: CROSS-024 self-test failed to detect public systemd transport module" >&2
    return 1
  fi

  printf '%s\n' 'pub(crate) mod systemd_transport;' > "${tmp}/systemd-transport-crate-private-module.rs"
  if rg -n '^pub[[:space:]]+mod[[:space:]]+systemd_transport;' "${tmp}/systemd-transport-crate-private-module.rs"; then
    echo "ERROR: CROSS-024 self-test falsely rejected crate-private systemd transport module" >&2
    return 1
  fi

  printf '%s\n' 'pub async fn parse_systemd_properties(...) {}' > "${tmp}/observed-state-public-async-parser.rs"
  if scan_public_observation_factory "${tmp}/observed-state-public-async-parser.rs"; then :; else
    echo "ERROR: CROSS-023 self-test failed to detect public async observed-state parser" >&2
    return 1
  fi

  printf '%s\n' 'impl NixServiceObservedStateV1 {
    pub async fn new(
        unit: String,
    ) {}' > "${tmp}/observed-state-public-async-constructor.rs"
  if rg -U -n --pcre2 "${OBSERVED_STATE_PUBLIC_CONSTRUCTOR_PATTERN}" "${tmp}/observed-state-public-async-constructor.rs"; then :; else
    echo "ERROR: CROSS-022 self-test failed to detect public async observed-state constructor" >&2
    return 1
  fi
}


run_self_test
run_boundary_check
echo "CROSS-015: Nixward governed observation boundary is clean."
