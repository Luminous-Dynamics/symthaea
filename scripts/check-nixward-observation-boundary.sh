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
  #
  # CROSS-022: governed authority/effect-binding modules must not consume the
  # legacy Custom command representation as semantic input. The legacy service
  # renderer is a one-way compatibility projection only.
  for file in "${AUTHORITY_FILES[@]}"; do
    # authorization.rs explicitly rejects Custom commands in the generic
    # intent constructor; these references are negative guards, not inputs.
    [[ "${file}" == "crates/core/nixward/src/action/authorization.rs" ]] && continue
    if matches="$(scan_legacy_custom_command "${ROOT}/${file}")"; then
      echo "ERROR: governed authority module consumes legacy NixOSCommand::Custom: ${file}" >&2
      echo "${matches}" >&2
      failed=1
    fi
  done

  # CROSS-022: compatibility rendering is not an authority primitive.
  # Authority modules must not invoke the legacy renderer directly, even if
  # they avoid mentioning NixOSCommand at the call site.
  for file in "${AUTHORITY_FILES[@]}"; do
    [[ "${file}" == "crates/core/nixward/src/action/service_manager.rs" ]] && continue
    if matches="$(scan_legacy_service_renderer "${ROOT}/${file}")"; then
      echo "ERROR: governed authority module invokes the legacy service renderer: ${file}" >&2
      echo "${matches}" >&2
      failed=1
    fi
  done

  # CROSS-022: the service domain is one-way from semantic data toward the
  # compatibility renderer. A reverse From/TryFrom implementation would make
  # legacy command bytes semantic input again.
  if matches="$(scan_reverse_service_conversion "${ROOT}/crates/core/nixward/src")"; then
    echo "ERROR: reverse NixOSCommand -> NixServiceOperationV1 conversion is forbidden" >&2
    echo "${matches}" >&2
    failed=1
  fi

  # CROSS-022: the typed service domain is intentionally upstream of the
  # legacy command representation. It must not mention NixOSCommand at all,
  # preventing accidental reverse conversion or semantic coupling.
  if matches="$(rg -n '\bNixOSCommand\b' "${ROOT}/crates/core/nixward/src/action/service_domain.rs")"; then
    echo "ERROR: typed service domain must not depend on legacy NixOSCommand representation" >&2
    echo "${matches}" >&2
    failed=1
  fi

  # Domain invariants must be established by NixServiceOperationV1::new().
  # The closed operation enum itself may deserialize because it has no free-form
  # data or normalization boundary. The aggregate NixServiceOperationV1 must not
  # derive/implement Deserialize, because that would bypass its validating
  # constructor and admit non-canonical unit strings.
  if matches="$(rg -n --pcre2 'Deserialize[^\\n]*for[[:space:]]+NixServiceOperationV1|NixServiceOperationV1[^\\n]*Deserialize' "+'"${ROOT}/crates/core/nixward/src/action/service_domain.rs"'+")"; then
    echo "ERROR: validated NixServiceOperationV1 must not deserialize around its constructor" >&2
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

  # CROSS-022: legacy ServiceManager lifecycle constructors remain
  # compatibility APIs, but governed callers must use the fallible typed
  # bridges. Keep the compatibility implementation and its regression tests
  # out of this scan.
  local nixward_src="${ROOT}/crates/core/nixward/src"
  if matches="$(rg -n --pcre2 "${LEGACY_SERVICE_CONSTRUCTOR_PATTERN}" "${nixward_src}" --glob '*.rs' --glob '!**/action/service_manager.rs')"; then
    echo "ERROR: Nixward caller uses infallible legacy ServiceManager constructor" >&2
    echo "${matches}" >&2
    failed=1
  fi

  # CROSS-022: no wildcard action-category fallback may synthesize a
  # systemctl command. New categories must be mapped explicitly or rejected.
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
  if scan_diagnostic_boundary "${tmp}/diagnostic.rs"; then
    :
  else
    echo "ERROR: CROSS-015 self-test failed to detect diagnostic type" >&2
    return 1
  fi

  printf '%s\n' 'Command::new("systemctl");' > "${tmp}/systemctl.rs"
  if scan_direct_systemctl "${tmp}/systemctl.rs"; then
    :
  else
    echo "ERROR: CROSS-015 self-test failed to detect direct systemctl command use" >&2
    return 1
  fi

  printf '%s\n' '_ => NixOSCommand::Custom { command: "systemctl", args: vec!["restart"], };' > "${tmp}/implicit-restart.rs"
  if scan_implicit_service_restart "${tmp}/implicit-restart.rs"; then
    :
  else
    echo "ERROR: CROSS-022 self-test failed to detect implicit service restart fallback" >&2
    return 1
  fi

  printf '%s\n' 'NixServiceObservedStateV1::parse_systemd_properties(...);' > "${tmp}/typed.rs"
  printf '%s\n' 'NixOSCommand::Custom { .. };' > "${tmp}/legacy-domain.rs"
  if rg -n '\bNixOSCommand\b' "${tmp}/legacy-domain.rs"; then
    :
  else
    echo "ERROR: CROSS-022 self-test failed to detect legacy command dependency" >&2
    return 1
  fi
  printf '%s\n' 'ServiceManager::render_legacy_command(&typed);' > "${tmp}/legacy-renderer.rs"
  if scan_legacy_service_renderer "${tmp}/legacy-renderer.rs"; then
    :
  else
    echo "ERROR: CROSS-022 self-test failed to detect legacy renderer usage" >&2
    return 1
  fi

  printf '%s\n' 'impl TryFrom<&NixOSCommand> for NixServiceOperationV1 {}' > "${tmp}/reverse-service.rs"
  if scan_reverse_service_conversion "${tmp}/reverse-service.rs"; then
    :
  else
    echo "ERROR: CROSS-022 self-test failed to detect reverse service conversion" >&2
    return 1
  fi

  printf '%s\n' 'ServiceManager::restart("nginx");' > "${tmp}/legacy-service.rs"
  if scan_legacy_service_constructor "${tmp}/legacy-service.rs"; then
    :
  else
    echo "ERROR: CROSS-022 self-test failed to detect legacy ServiceManager constructor" >&2
    return 1
  fi

  printf '%s\n' 'let _ = NixOSCommand::Custom { .. };' > "${tmp}/legacy-custom.rs"
  if scan_legacy_custom_command "${tmp}/legacy-custom.rs"; then
    :
  else
    echo "ERROR: CROSS-022 self-test failed to detect legacy Custom command material" >&2
    return 1
  fi

  if scan_diagnostic_boundary "${tmp}/typed.rs"; then
    echo "ERROR: CROSS-015 self-test falsely rejected typed evidence" >&2
    return 1
  fi
}

run_self_test
run_boundary_check
echo "CROSS-015: Nixward governed observation boundary is clean."
