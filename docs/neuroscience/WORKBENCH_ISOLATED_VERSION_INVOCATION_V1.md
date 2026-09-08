# Workbench Isolated Version Invocation v1

Status: **candidate execution-observation theorem; local adversarial contracts pass, hosted real observation pending**

Schema:
- producer receipt: `symthaea-workbench-isolated-version-invocation-receipt-v1`
- independent verification: `symthaea-workbench-isolated-version-invocation-verification-v1`

## Purpose

The qualified Workbench chain now proves the exact realized root and the exact regular executable bytes at:

```text
<verified-root>/bin/wb_command
```

but program membership is not process execution.

This tranche closes only the next mechanical boundary:

```text
verified root/program membership
        +
qualified invocation-isolation profile
        +
exact absolute wb_command -version argv
        +
closed root-main-program execve environment
        +
fresh private process roots
        +
retained stdout/stderr
        +
independent fresh-runner verification
        ->
verified isolated Workbench version invocation
```

It does not execute a cortical transform and does not qualify scientific execution.

## Parent authority

The fresh verifier reruns the complete #924 root-NAR verification pipeline from retained raw parent evidence. It requires:

```text
closure_receipt_verified                     = true
root_nar_capture_verified                     = true
target_membership_verified                    = true
root_main_program_regular_executable_verified = true
```

and requires the parent to preserve:

```text
workbench_execution_qualified = false
transform_executed            = false
fmq010_established             = false
neural_alignment_established  = false
consciousness_evidence        = false
```

The execution producer cannot promote its own program identity.

## Exact command

The invocation is exactly:

```text
<independently-verified-root>/bin/wb_command -version
```

The executable is addressed by absolute verified-closure path. Host `PATH` lookup is forbidden.

The producer hashes the live executable immediately before and after the invocation. The fresh verifier requires both hashes to equal the independent #924 in-NAR content SHA-256.

This before/after check is not presented as a hostile-kernel or same-UID race theorem. It is an additional mutation detector around an already independently verified Nix-store program identity.

## Root-main-program execve environment

The process environment is exactly the #681 entry environment.

Fixed values:

```text
LANG=C
LC_ALL=C
TZ=UTC0
OMP_NUM_THREADS=1
OMP_DYNAMIC=FALSE
PATH=
```

Dynamic values bind only to fresh invocation roots:

```text
PWD              -> cwd
HOME             -> home
XDG_CONFIG_HOME  -> xdg_config_home
XDG_CACHE_HOME   -> xdg_cache_home
XDG_DATA_HOME    -> xdg_data_home
XDG_STATE_HOME   -> xdg_state_home
XDG_RUNTIME_DIR  -> xdg_runtime_dir
XDG_CONFIG_DIRS  -> xdg_config_dirs
XDG_DATA_DIRS    -> xdg_data_dirs
TMPDIR           -> tmpdir
TMP              -> tmpdir
TEMP             -> tmpdir
```

All ten root roles are:
- newly created for the invocation;
- mutually distinct;
- absolute;
- mode `0700`;
- empty before execution;
- non-reused;
- removed after execution with cleanup confirmation.

`TMPDIR`, `TMP`, and `TEMP` intentionally refer to the same fresh `tmpdir` role.

The subprocess contract additionally fixes:
- `cwd` to the isolated `cwd` root;
- `PWD` to the same path;
- stdin to `/dev/null`;
- exact stdout/stderr byte retention;
- `close_fds = true`;
- empty `pass_fds`;
- process umask `0077`;
- `shell = false`;
- no host environment inheritance.

## Entry environment is not descendant environment

This PR preserves #681's distinction:

```text
RunnerEntryEnvironment
    + VerifiedProgramBytes
        -> ProgramDefinedEnvironmentTransition
        -> DescendantEnvironment
```

The Nix/Qt root-main-program wrapper may transform environment variables before executing an internal Workbench binary.

Therefore successful version invocation does not establish post-entry environment equivalence.

## Version-output identity

The existing Lineage-B verifier defines Workbench version identity as:

```text
SHA256(stdout || stderr)
```

for:

```text
wb_command -version
```

where `||` means stdout bytes followed by stderr bytes.

This tranche preserves exactly that historical commitment rule under the explicit label:

```text
stdout-then-stderr-v1
```

The producer retains both streams independently. The fresh verifier recomputes the aggregate digest from retained bytes and independently requires exactly one semantic line:

```text
Version: 2.1.0
```

No new version-hash convention is introduced.

## Diagnostic machine context

The producer records the #681 diagnostic context:

```text
cpu_vendor
cpu_family
cpu_model
cpu_stepping
cpu_flags_digest
kernel_release
```

`cpu_flags_digest` is SHA-256 over sorted unique CPU flags with newline framing.

These fields are diagnostic observations. They are not automatically promoted into scientific identity.

```text
DiagnosticField
    -- measured causal relevance -->
NormativeScientificField
```

remains the promotion rule.

## Producer authority

The Nix-bearing execution producer is observation-only.

Every producer authority field is false, including:

```text
program_membership_verified
invocation_profile_verified
invocation_executed
version_output_bound
same_host_repeatability_established
path_equivalence_established
cross_cpu_equivalence_established
workbench_execution_qualified
scientific_execution_qualified
transform_executed
atlas_correctness_established
fmq010_established
neural_alignment_established
consciousness_evidence
```

This is deliberate. An execution producer cannot certify its own receipt.

## Fresh-verifier authority

Only the fresh no-Nix verifier may establish:

```text
program_membership_verified = true
invocation_profile_verified = true
invocation_executed         = true
version_output_bound        = true
```

It must preserve:

```text
same_host_repeatability_established = false
path_equivalence_established        = false
cross_cpu_equivalence_established   = false
workbench_execution_qualified       = false
scientific_execution_qualified      = false
transform_executed                  = false
atlas_correctness_established       = false
fmq010_established                  = false
neural_alignment_established        = false
consciousness_evidence              = false
```

Core invariant:

```text
VerifiedIsolatedVersionInvocation != QualifiedScientificExecution
```

## Why `workbench_execution_qualified` stays false

A successful `-version` process proves that the independently identified Workbench entry program can be invoked under the qualified root-main-program environment contract.

It does not prove:
- same-host numerical repeatability;
- path-independent transform results;
- CPU-dispatch equivalence;
- cortical-label byte equivalence;
- semantic transform correctness;
- atlas correctness.

#681 explicitly requires those stronger gates before transfer of scientific execution authority.

The narrow promotion in this tranche is therefore `invocation_executed`, not general Workbench scientific execution qualification.

## Retained evidence

The producer receipt retains:
- parent closure capture digest;
- verified-root-derived executable path;
- executable SHA-256 before and after execution;
- producer and profile implementation digests;
- exact argv and exit status;
- exact entry environment and its canonical digest;
- fresh-root lifecycle observations;
- CPU/kernel diagnostics;
- raw stdout and stderr;
- each raw stream digest and byte length;
- `SHA256(stdout || stderr)`;
- parsed Workbench version;
- a canonical capture digest.

The capture directory is closed-world:

```text
receipt.json
raw/wb-version.stdout
raw/wb-version.stderr
```

Extra files and symlinks fail verification.

## Hosted topology

The exact-head workflow has three stages:

```text
static contracts
    ↓
Nix-bearing producer
    #638 closure capture
    #924 root-NAR capture
    isolated wb_command -version observation
    ↓
immutable PR-head-keyed raw artifact
    ↓
fresh no-Nix verifier
    rerun #924 from raw parent evidence
    reverify #681 profile
    verify version receipt + raw bytes
```

The fresh verifier requires both `nix` and `nix-store` to be absent from `PATH`.

## Adversarial contracts

The focused suite covers:
- exact version-line parsing;
- wrong/duplicate/non-UTF8 version output;
- host-free exact entry environment;
- private/empty/distinct root creation;
- exact `-version` subprocess configuration;
- root substitution;
- executable-path substitution;
- executable pre/post hash substitution;
- argv substitution;
- nonzero exit;
- host `PATH` reintroduction;
- `PWD`/cwd mismatch;
- temporary-alias mismatch;
- root reuse;
- missing cleanup;
- duplicate root directories;
- raw stdout tamper;
- version digest tamper;
- parsed-version tamper;
- producer authority escalation;
- capture-digest tamper;
- extra inventory;
- profile substitution;
- environment-digest substitution;
- malformed diagnostic digest;
- unknown top-level receipt fields.

## Next boundary

If the real hosted observation qualifies, the next scientifically meaningful step is **not** another version-query wrapper.

The next execution promotion should pair:
1. the #576 immutable scientific-input snapshot;
2. integration of that snapshot into `derive()`;
3. this qualified execution capsule/program/version identity;
4. repeated same-host cortical transforms under fresh invocation roots;
5. exact output-byte and semantic-label comparison.

That integration necessarily creates a new Lineage-B generator implementation root because `derive()` will begin consuming snapshot logic and qualified execution evidence.

No FMQ-010 comparison should occur before that new scientific commitment is independently reconstructed by the archival verifier.
