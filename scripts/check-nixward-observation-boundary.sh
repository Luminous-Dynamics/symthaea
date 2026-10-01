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

  # Domain invariants must be established by NixServiceOperationV1::new().
  # A derived Deserialize implementation could bypass that constructor.
  if matches="$(rg -n '\\bDeserialize\\b' "${ROOT}/crates/core/nixward/src/action/service_domain.rs")"; then
    echo "ERROR: typed service domain must not deserialize around its validating constructor" >&2
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

  printf '%s\n' 'NixServiceObservedStateV1::parse_systemd_properties(...);' > "${tmp}/typed.rs"
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
