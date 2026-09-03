# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Hermetic TPM2 verifier entry points for the Agency Kernel.
#
# The Rust platform-attestation crate pins the bytes of the executable it
# invokes. A shell wrapper would still be vulnerable to LD_PRELOAD/BASH_ENV
# before the script could clear its environment, so these launchers are static
# native executables. They reject caller attempts to override protocol-critical
# options, clear the process environment, and only then execve() the exact
# tpm2-tools binary from the locked Nix closure.
{ pkgs, tcti }:

let
  lib = pkgs.lib;
  tpm2Tools = pkgs.tpm2-tools;
  quoteTool = "${tpm2Tools}/bin/tpm2_quote";
  checkquoteTool = "${tpm2Tools}/bin/tpm2_checkquote";

  # The bound TCTI is compiled into the static launcher. Keep the accepted
  # syntax deliberately narrow so it can be embedded in generated C source
  # without creating a second quoting language at this trust boundary.
  safeTcti = builtins.match "[A-Za-z0-9_./:=,-]+" tcti != null;

  mkLauncher = {
    name,
    realTool,
    boundTcti,
    quoteMode,
  }:
    let
      source = pkgs.writeText "${name}.c" ''
        #include <errno.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <unistd.h>

        #define REAL_TOOL "${realTool}"
        #define BOUND_TCTI "${boundTcti}"
        #define QUOTE_MODE ${if quoteMode then "1" else "0"}

        static int starts_with(const char *value, const char *prefix) {
            size_t prefix_len = strlen(prefix);
            return strncmp(value, prefix, prefix_len) == 0;
        }

        static int forbidden_override(const char *arg) {
            /* Reject every short-option cluster containing T. This also catches
             * forms such as -Tmssim or -QT... rather than assuming getopt will
             * receive -T as a separate argv element. */
            if (arg[0] == '-' && arg[1] != '-' && strchr(arg + 1, 'T') != NULL) {
                return 1;
            }
            if (strcmp(arg, "--tcti") == 0 || starts_with(arg, "--tcti=")) {
                return 1;
            }

        #if QUOTE_MODE
            /* PCR output serialization is part of the reviewed profile
             * protocol. Do not let a caller replace -F serialized. */
            if (arg[0] == '-' && arg[1] != '-' && strchr(arg + 1, 'F') != NULL) {
                return 1;
            }
            if (strcmp(arg, "--pcrs_format") == 0
                || starts_with(arg, "--pcrs_format=")
                || strcmp(arg, "--pcrs-format") == 0
                || starts_with(arg, "--pcrs-format=")) {
                return 1;
            }
        #endif
            return 0;
        }

        int main(int argc, char **argv) {
            if (argc < 1) {
                return 64;
            }
            for (int i = 1; i < argc; ++i) {
                if (forbidden_override(argv[i])) {
                    fputs("symthaea TPM2 launcher: protocol-critical option override rejected\n", stderr);
                    return 64;
                }
            }

            /* real-tool + -T + TCTI + optional (-F serialized) + caller args
             * + terminating NULL. */
            size_t capacity = (size_t)argc + 6U;
            char **child_argv = calloc(capacity, sizeof(*child_argv));
            if (child_argv == NULL) {
                return 70;
            }

            size_t n = 0;
            child_argv[n++] = (char *)REAL_TOOL;
            child_argv[n++] = (char *)"-T";
            child_argv[n++] = (char *)BOUND_TCTI;
        #if QUOTE_MODE
            child_argv[n++] = (char *)"-F";
            child_argv[n++] = (char *)"serialized";
        #endif
            for (int i = 1; i < argc; ++i) {
                child_argv[n++] = argv[i];
            }
            child_argv[n] = NULL;

            /* Nothing from the caller's environment crosses the launcher.
             * In particular: LD_PRELOAD, LD_LIBRARY_PATH, TPM2TOOLS_TCTI,
             * OPENSSL_CONF and locale-dependent parsing state are absent. */
            char *const child_env[] = {
                (char *)"LANG=C",
                (char *)"LC_ALL=C",
                (char *)"PATH=/no-such-path",
                NULL,
            };

            execve(REAL_TOOL, child_argv, child_env);
            int saved_errno = errno;
            free(child_argv);
            errno = saved_errno;
            perror("symthaea TPM2 launcher execve");
            return 127;
        }
      '';
    in
    pkgs.pkgsStatic.stdenv.mkDerivation {
      pname = name;
      version = "1";
      dontUnpack = true;

      buildPhase = ''
        runHook preBuild
        $CC -std=c11 -O2 -Wall -Wextra -Werror ${source} -o ${name}
        runHook postBuild
      '';

      installPhase = ''
        runHook preInstall
        mkdir -p "$out/bin"
        install -m 0555 ${name} "$out/bin/${name}"
        runHook postInstall
      '';
    };

  quote = mkLauncher {
    name = "symthaea-tpm2-quote";
    realTool = quoteTool;
    boundTcti = tcti;
    quoteMode = true;
  };

  checkquote = mkLauncher {
    name = "symthaea-tpm2-checkquote";
    realTool = checkquoteTool;
    # checkquote validates quote/signature/PCR data offline. Explicitly prevent
    # default TCTI discovery or ambient TPM access.
    boundTcti = "none";
    quoteMode = false;
  };

in
assert safeTcti;
pkgs.symlinkJoin {
  name = "symthaea-agency-tpm2-verifier-tools";
  paths = [ quote checkquote ];
  passthru = {
    inherit tpm2Tools tcti;
    quotePath = "${quote}/bin/symthaea-tpm2-quote";
    checkquotePath = "${checkquote}/bin/symthaea-tpm2-checkquote";
  };
}
