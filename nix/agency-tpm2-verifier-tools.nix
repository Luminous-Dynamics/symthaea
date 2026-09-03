# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Hermetic TPM2 verifier entry points for the Agency Kernel.
#
# The Rust platform-attestation crate pins the bytes of the executable it
# invokes. These wrappers make that executable identity meaningful by also
# freezing the tpm2-tools implementation, TCTI and PCR serialization semantics
# while removing caller-controlled loader/TCTI environment.
{ pkgs, tcti }:

let
  lib = pkgs.lib;
  tpm2Tools = pkgs.tpm2-tools;
  env = "${pkgs.coreutils}/bin/env";
  quoteTool = "${tpm2Tools}/bin/tpm2_quote";
  checkquoteTool = "${tpm2Tools}/bin/tpm2_checkquote";
  boundTcti = lib.escapeShellArg tcti;

  rejectQuoteOverrides = ''
    for arg in "$@"; do
      case "$arg" in
        -T|-T*|--tcti|--tcti=*|-F|-F*|--pcrs_format|--pcrs_format=*|--pcrs-format|--pcrs-format=*)
          echo "symthaea-tpm2-quote: caller may not override TCTI or PCR output format" >&2
          exit 64
          ;;
      esac
    done
  '';

  rejectCheckquoteOverrides = ''
    for arg in "$@"; do
      case "$arg" in
        -T|-T*|--tcti|--tcti=*)
          echo "symthaea-tpm2-checkquote: caller may not override TCTI" >&2
          exit 64
          ;;
      esac
    done
  '';

  quote = pkgs.writeShellScriptBin "symthaea-tpm2-quote" ''
    set -eu
    ${rejectQuoteOverrides}

    # The wrapper itself is the reviewed executable pinned by the Rust policy.
    # Clear ambient environment before entering tpm2-tools so LD_PRELOAD,
    # LD_LIBRARY_PATH and TPM2TOOLS_TCTI cannot change the reviewed verifier.
    exec ${env} -i \
      PATH=${lib.escapeShellArg "${tpm2Tools}/bin:${pkgs.coreutils}/bin"} \
      LANG=C \
      LC_ALL=C \
      ${quoteTool} \
        -T ${boundTcti} \
        -F serialized \
        "$@"
  '';

  checkquote = pkgs.writeShellScriptBin "symthaea-tpm2-checkquote" ''
    set -eu
    ${rejectCheckquoteOverrides}

    # checkquote is an offline verifier. Force the explicit off-TPM TCTI and
    # remove ambient loader/TCTI influence just as for the quote generator.
    exec ${env} -i \
      PATH=${lib.escapeShellArg "${tpm2Tools}/bin:${pkgs.coreutils}/bin"} \
      LANG=C \
      LC_ALL=C \
      ${checkquoteTool} \
        -T none \
        "$@"
  '';

in
pkgs.symlinkJoin {
  name = "symthaea-agency-tpm2-verifier-tools";
  paths = [ quote checkquote ];
  passthru = {
    inherit tpm2Tools tcti;
    quotePath = "${quote}/bin/symthaea-tpm2-quote";
    checkquotePath = "${checkquote}/bin/symthaea-tpm2-checkquote";
  };
}
