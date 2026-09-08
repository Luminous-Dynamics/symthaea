# Workbench Invocation Isolation Profile v1

Status: **candidate profile only; no Workbench invocation has been executed or qualified by this profile**

Schema: `symthaea-workbench-invocation-isolation-profile-v1`

## Purpose

The Workbench execution-capsule profile (#624) closes the intended platform and major process-environment boundary. The closure stack (#629, #638, #667) identifies and independently verifies the realized Nix runtime closure.

The next scientific risk is narrower: even with the same closure and scientific input bytes, one invocation can inherit mutable process state or leave state that changes a later invocation.

This profile therefore defines a **per-invocation isolation contract** before any real cortical transform is admitted.

```text
verified Workbench closure
    + absolute closure-root main program
    + zero inherited host environment
    + exact runner-controlled execve environment
    + fresh private cwd/XDG/HOME/TMP roots per invocation
    + closed file-descriptor/stdin surface
        -> candidate isolated Workbench invocation
```

This still does not establish numerical reproducibility or scientific correctness.

## Program binding

The root program must be derived only from an independently verified Workbench closure root:

```text
<verified-root>/bin/wb_command
```

The execution path must be absolute. Host `PATH` lookup is forbidden.

The selected nixpkgs package uses `qt6.wrapQtAppsHook`. At the pinned nixpkgs revision, that hook wraps executable Qt binaries with Nix-generated program bytes that can prefix closure-owned Qt/XDG paths before executing the hidden program. Therefore the runner-controlled environment and the final descendant environment are deliberately **not treated as identical concepts**.

The real root NAR still decides the realized node shape. This profile does not pre-claim whether `bin/wb_command` is a regular wrapper or symlink; #690 preserves that observation boundary.

## Environment stage: root-main-program execve

`process_environment` describes exactly the environment supplied by the scientific runner at the `execve` boundary for the independently verified root main program.

It does **not** claim to be a byte-for-byte description of every environment seen by descendant processes after the verified program begins executing.

The stage is therefore explicitly:

```text
root-main-program-execve
```

The runner constructs this entry environment from an empty mapping.

### Fixed entry values

```text
LANG=C
LC_ALL=C
TZ=UTC0
OMP_NUM_THREADS=1
OMP_DYNAMIC=FALSE
PATH=
```

`PATH` is explicitly empty rather than merely omitted. The verified root program is invoked by absolute path. If the closure-delivered program requires an unbound host command search path, execution must fail visibly rather than silently discovering `/usr/bin` or another host path.

The pinned Nix shell-wrapper implementation itself uses an absolute Nix shell shebang and ultimately executes the wrapped program by absolute path. Qt wrapper arguments may also inject closure-owned runtime paths. Those transformations are behavior of verified closure bytes, not ambient host state.

### Dynamic per-invocation entry bindings

The entry environment contains exact bindings to fresh directories created for that invocation:

```text
PWD              -> invocation.cwd
HOME             -> invocation.home
XDG_CONFIG_HOME  -> invocation.xdg_config_home
XDG_CACHE_HOME   -> invocation.xdg_cache_home
XDG_DATA_HOME    -> invocation.xdg_data_home
XDG_STATE_HOME   -> invocation.xdg_state_home
XDG_RUNTIME_DIR  -> invocation.xdg_runtime_dir
XDG_CONFIG_DIRS  -> invocation.xdg_config_dirs
XDG_DATA_DIRS    -> invocation.xdg_data_dirs
TMPDIR           -> invocation.tmpdir
TMP              -> invocation.tmpdir
TEMP             -> invocation.tmpdir
```

`PWD` is bound to the same fresh directory used as the process working directory. `TMPDIR`, `TMP`, and `TEMP` all bind to the same fresh temporary root so a dependency cannot escape the isolation contract merely by consulting a different conventional temporary-directory variable.

These are not optional descriptive paths. A later execution receipt must prove that the actual **runner-supplied execve values** equal the actual fresh roots created for that invocation.

All dynamic environment paths must be absolute.

## Entry environment is not automatically descendant environment

A verified root program may itself transform environment variables before executing its underlying binary. The selected Nix Qt machinery is an example: it can prefix closure-owned `QT_PLUGIN_PATH`, `XDG_DATA_DIRS`, and `XDG_CONFIG_DIRS` based on the realized closure.

Therefore v1 explicitly requires:

```text
entry_environment_is_final_environment       false
verified_program_may_transform_environment   true
host_injection_after_execve_allowed          false
post_entry_environment_equivalence_assumed   false
```

The key distinction is:

```text
RunnerEntryEnvironment
    + VerifiedProgramBytes
        -> ProgramDefinedEnvironmentTransition
        -> DescendantEnvironment
```

The runner is authorized to control the entry environment. It is **not** authorized to invent extra host values after `execve`.

The verified program's own deterministic transformations are part of program semantics and closure identity. A later theorem may reconstruct those transformations if an exact final-environment equality claim becomes necessary. Until then, empirical output reproducibility remains required and no final-environment equivalence is assumed.

This avoids two opposite errors:

- pretending the runner-controlled entry environment is automatically the final Workbench environment;
- manually reproducing Nix/Qt wrapper behavior outside the verified closure and thereby creating a second, unqualified runtime implementation.

## Per-invocation isolation roots

Every Workbench command receives independent fresh private roots for:

```text
cwd
HOME
XDG_CONFIG_HOME
XDG_CACHE_HOME
XDG_DATA_HOME
XDG_STATE_HOME
XDG_RUNTIME_DIR
XDG_CONFIG_DIRS
XDG_DATA_DIRS
TMPDIR/TMP/TEMP
```

The important word is **individual**.

```text
fresh once per pipeline != fresh once per invocation
```

Otherwise command 1 could leave cache/config/temp state that command 2 consumes.

Every root must:

- begin absent;
- be created privately with mode `0700`;
- have an absolute path;
- be bound to the declared entry variable(s) where applicable;
- never be reused;
- fail closed if a stale destination already exists;
- be cleaned after the invocation;
- have cleanup confirmed before the runner reports the isolation lifecycle complete.

v1 also requires process umask `0077`.

Machine-local dynamic paths are retained as execution evidence when required for verification but should not be echoed into ordinary success logs.

## Why host-environment inheritance remains forbidden

The runner must not copy the parent environment and overwrite a selected set of keys.

Therefore values such as these cannot silently enter the controlled execve boundary:

```text
host PATH
LD_LIBRARY_PATH
PYTHONPATH
DISPLAY / WAYLAND_DISPLAY
DBUS_SESSION_BUS_ADDRESS
proxy variables
operator Qt overrides
shell/user customization
arbitrary process variables
```

Any closure-defined environment transition after execve must come from the verified program itself, not from a host-side repair step.

## Standard streams and descriptors

Each invocation uses:

```text
stdin  -> /dev/null
stdout -> retained exact bytes
stderr -> retained exact bytes
```

Unrelated inherited file descriptors are closed and `pass_fds` is empty.

Thus the program cannot accidentally depend on inherited terminal input or an operator-open file descriptor under the qualified contract.

## Scientific path humility

The #576 scientific-input snapshot returns a path-independent digest over `(role -> committed SHA-256)` while placing role bytes in a fresh snapshot root.

That is correct for **input identity**. It does not prove:

```text
same input bytes at path A
    ==
same output bytes at path B
```

Some programs retain filenames, provenance strings, working-directory paths, or path-derived metadata. Therefore v1 neither classifies snapshot-root/scratch-root/cwd strings as scientific identity nor assumes they are irrelevant.

Before any path-equivalence claim, the same closure and same scientific inputs must be executed under perturbations of at least:

```text
snapshot_root
scratch_root
working_directory
```

The experiment must compare both raw output bytes and semantic/scientific content.

The existing NeuroBridge mapping stack already supplies the final discrete scientific equality mechanism: canonical semantic surfaces compile through #490, and #491 provides exact 20,484-vertex disagreement evidence. No arbitrary floating-point epsilon is needed for the final Glasser assignment reproducibility gate.

Possible outcomes remain distinct:

```text
raw bytes equal + semantic map equal
    -> no observed representation or scientific drift

raw bytes different + semantic map equal
    -> representation/metadata drift without observed cortical-label drift

semantic map different
    -> reproducibility failure requiring explanation
```

No result is preselected.

## Numerical execution class

`x86_64-linux` is not treated as a complete floating-point execution identity.

The exact upstream Workbench v2.1.0 source contains runtime CPU dispatch in `src/kloewe/dot/src/dot.c`. Under `DOT_AUTO` on x86_64 it can select among:

```text
AVX512FMA
AVX512
AVX
SSE2
NAIVE
```

based on CPU feature detection.

The pinned upstream v2.1.0 annotated tag resolves to:

```text
99bf66b28572005f6419330d3f58b6262637412d
```

This motivates, but does not itself establish, numerical divergence.

The profile therefore hard-rejects both shortcuts:

```text
same x86_64 architecture -> equivalent result
same Nix closure         -> cross-CPU equivalent result
```

Neither implication is granted.

## Promotion order for numerical reproducibility

The initial study should proceed in increasing scope:

```text
1. same scientific commitment + same host, repeated fresh invocations
2. same host, perturbed snapshot/scratch/cwd paths
3. same CPU execution class on another machine
4. different x86_64 CPU dispatch classes
5. only later: another architecture such as aarch64-linux
```

This separates ordinary nondeterminism from path sensitivity and CPU-dispatch sensitivity.

## Diagnostic context is not automatically scientific identity

v1 requires future execution receipts to retain:

```text
cpu_vendor
cpu_family
cpu_model
cpu_stepping
cpu_flags_digest
kernel_release
```

These begin as diagnostic fields, not normative scientific fields.

```text
diagnostic field
    + measured output sensitivity
        -> candidate normative execution field
```

This avoids both under-specifying the runtime and over-specifying scientific identity with incidental machine properties before evidence shows relevance.

## Authority state

The profile establishes only:

```text
invocation_contract_defined         true
```

All stronger fields remain false:

```text
invocation_executed                 false
path_equivalence_established        false
same_host_repeatability_established false
cross_cpu_equivalence_established   false
workbench_execution_qualified       false
transform_executed                  false
atlas_correctness_established       false
fmq010_established                  false
neural_alignment_established        false
consciousness_evidence              false
```

In particular:

```text
IsolatedInvocationProfile != ReproducibleExecution
ReproducibleExecution != AtlasCorrectness
```

## Qualification contracts

The dependency-free verifier binds this profile to #624's exact parent platform/environment/main-program choices. A separate stdlib-only consistency checker does not reuse the verifier and independently maps every dynamic execve binding to its corresponding fresh-root policy.

The adversarial suite contains **46 authored contracts**, covering:

- exact current profile validation;
- closed-world/top-level/platform/parent drift rejection;
- absolute verified-root main-program binding;
- host PATH lookup and non-empty PATH rejection;
- bool/int laundering at program, isolation, descendant, and authority boundaries;
- exact `root-main-program-execve` stage;
- zero host-environment inheritance;
- fixed locale/timezone/OpenMP drift;
- missing/retargeted dynamic binding rejection;
- `PWD -> cwd` equality;
- `TMPDIR/TMP/TEMP -> tmpdir` alias equality;
- XDG config/cache/data/state/runtime/system-dir isolation;
- rejection of entry-environment == final-environment laundering;
- preservation of verified-program environment-transition possibility;
- prohibition on post-execve host injection;
- prohibition on post-entry environment-equivalence assumptions;
- per-invocation reuse/stale-root/absolute-path/root-equality constraints;
- inherited descriptor/stdout-policy downgrade rejection;
- path-equivalence promotion rejection;
- missing perturbation axes rejection;
- x86_64 and same-closure cross-CPU shortcuts;
- runtime dispatch erasure/mode drift;
- cross-CPU gate and diagnostic-context drift;
- authority escalation rejection;
- duplicate JSON-key rejection;
- CLI round-trip validation.

The dedicated workflow deliberately executes neither Nix nor Workbench.

These contracts are authored, not called hosted-green until the exact focused workflow executes successfully.

## Intended next boundary

Only after the parent capture/verifier stack is independently qualified should an execution-observation producer consume this profile.

That producer should bind:

```text
independently verified closure receipt
+ exact invocation-isolation profile
+ exact root-NAR program-membership evidence
+ exact wb_command version observation
+ exact runner-controlled root-main-program execve environment
+ exact fresh-root lifecycle
+ CPU/kernel diagnostics
```

It should **not** claim the root entry environment is automatically the final Workbench descendant environment.

A separate hostile verifier should reconstruct that receipt before any Lineage-B scientific transform is allowed to treat the execution capsule as qualified.
