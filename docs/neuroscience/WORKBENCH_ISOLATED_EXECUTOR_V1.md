# Workbench Isolated Executor v1

Status: **candidate reusable executor mechanism only; no cortical transform authority**

## Purpose

This tranche extracts the process-isolation behavior already exercised by the hosted-qualified isolated Workbench version observation into a small reusable execution primitive.

It deliberately does **not** integrate that primitive into Lineage-B `derive()`, consume a scientific-input snapshot, execute HCP-MMP1 transform commands, or change any scientific evidence schema.

The intended boundary is:

```text
qualified #681 invocation contract
        ↓
minimal reusable process executor
        ↓
future generator integration
```

not:

```text
executor exists
        ↓
scientific execution qualified
```

## Parent contracts

The executor consumes the exact existing profiles:

```text
data/neuroscience/workbench_execution_capsule_profile_v1.json
data/neuroscience/workbench_invocation_isolation_profile_v1.json
```

and validates them through:

```text
scripts/verify_workbench_invocation_isolation_profile.py
```

The executor does not define a second environment schema.

## Inputs

`observe_isolated_execution(...)` accepts:

- an **absolute** executable path;
- a caller-supplied argument list, excluding `argv[0]`;
- an expected canonical `sha256:<64 lowercase hex>` executable root;
- the #681 invocation profile;
- the #624 parent execution-capsule profile;
- an existing scratch parent directory;
- a canonical temporary-root stem.

The function constructs:

```text
argv = [absolute_executable, *args]
```

It never performs host `PATH` lookup.

## Executable boundary

Before process creation, the executor requires:

```text
AbsolutePath = true
DirectSymlink = false
RegularFile = true
ExecutablePermission = true
SHA256(program) = ExpectedProgramSHA256
```

The executable is hashed again after the child terminates and must retain exactly the same expected digest.

Therefore a successful observation requires:

```text
PreExecutableSHA
    = ExpectedExecutableSHA
    = PostExecutableSHA
```

This is a before/after stable-byte theorem. It is not a claim of protection against an adversary able to replace the pathname, execute alternate bytes, and restore the original bytes entirely inside that observation window. Exact Nix-root membership and executable provenance remain separate upstream theorems (#924/#976/#991).

## Entry environment

The executor uses the exact #681 root-main-program entry environment.

Fixed values:

```text
LANG=C
LC_ALL=C
TZ=UTC0
OMP_NUM_THREADS=1
OMP_DYNAMIC=FALSE
PATH=
```

Dynamic variables bind only to fresh roots created for that invocation:

```text
PWD
HOME
XDG_CONFIG_HOME
XDG_CACHE_HOME
XDG_DATA_HOME
XDG_STATE_HOME
XDG_RUNTIME_DIR
XDG_CONFIG_DIRS
XDG_DATA_DIRS
TMPDIR
TMP
TEMP
```

The executor explicitly requires:

```text
PWD = invocation.cwd
TMPDIR = TMP = TEMP
```

and passes only the fixed plus dynamic profile keys to the child.

Host-environment inheritance is therefore forbidden at the process-entry boundary.

As in #681, this entry environment is **not** claimed to be the final descendant environment. Verified Nix/Qt wrapper bytes may transform environment variables before executing the hidden Workbench binary.

## Per-invocation roots

Every invocation receives ten distinct fresh roots:

```text
cwd
home
xdg_config_home
xdg_cache_home
xdg_data_home
xdg_state_home
xdg_runtime_dir
xdg_config_dirs
xdg_data_dirs
tmpdir
```

Each root must be:

```text
mode = 0700
empty_before = true
reuse_allowed = false
```

Construction failures require partial-root cleanup before the error escapes.

After the child terminates, the complete isolation base is removed and cleanup must be confirmed before a successful observation returns.

## Process contract

The subprocess is created with:

```text
shell = false
cwd = fresh invocation.cwd
env = exact #681 entry environment
stdin = /dev/null
stdout = retained exact bytes
stderr = retained exact bytes
close_fds = true
pass_fds = ()
umask = 0077
check = false
```

The primitive intentionally observes nonzero exit status rather than converting it into success.

`require_success(...)` is the separate fail-closed helper that rejects nonzero exit status, executable-byte mismatch, or unconfirmed cleanup.

This distinction prevents:

```text
ProcessObserved
    -> ProcessSucceeded
```

from becoming an implicit conversion.

## File-descriptor theorem

The behavioral suite opens an inheritable sentinel descriptor in the parent process and proves the child cannot observe that underlying file through `/proc/self/fd`.

This tests the actual `close_fds + empty pass_fds` behavior instead of merely checking source text.

## Stream identity

The observation retains raw `stdout` and `stderr` bytes in memory and separately binds:

```text
stdout_sha256 = SHA256(stdout)
stderr_sha256 = SHA256(stderr)
```

No text decoding is required by this primitive.

Scientific callers remain responsible for interpreting command-specific output semantics.

## Umask and concurrency boundary

v1 preserves the mechanism already used by the qualified #976 observation: the parent process temporarily sets umask `0077` around process creation and restores it immediately afterward.

`os.umask()` is process-global.

Therefore this v1 primitive is qualified only for **single-threaded scientific orchestration**. It does not establish safe concurrent invocation from an arbitrary multithreaded host process.

A future concurrency generalization must qualify a different launch mechanism or a stronger process-level isolation theorem rather than silently broadening v1.

```text
SingleThreadedExecutorQualification
    !=
ConcurrentHostSafety
```

## Observation schema

A completed process observation has schema:

```text
symthaea-workbench-isolated-execution-observation-v1
```

and retains:

- executable path and pre/expected/post hashes;
- exact argv and exit status;
- cwd/stdin/fd/umask policy;
- exact entry-environment map and digest;
- per-root lifecycle evidence;
- exact stdout/stderr bytes and hashes;
- an all-false authority object.

The object is a local execution observation, not a retained scientific evidence record.

## Runtime authority is closed

The executor returns:

```text
executor_mechanism_qualified = false
real_workbench_invocation_verified = false
same_host_repeatability_established = false
path_equivalence_established = false
cross_cpu_equivalence_established = false
workbench_execution_qualified = false
scientific_execution_qualified = false
transform_executed = false
atlas_correctness_established = false
fmq010_established = false
neural_alignment_established = false
consciousness_evidence = false
```

Even after hosted qualification, runtime code does not self-assert that its own source theorem passed.

Qualification belongs to the exact Git source/workflow evidence.

## Adversarial contracts

The dependency-free behavioral suite contains 21 contracts covering:

1. child-observed cwd/PWD/stdin/umask/process policy;
2. host environment non-inheritance;
3. exact fixed/dynamic environment key set;
4. ten distinct private roots and confirmed cleanup;
5. inherited parent file descriptor closure observed by the child;
6. exact stdout/stderr digest binding;
7. closed all-false runtime authority;
8. nonzero exit retained without laundering;
9. zero-exit `require_success` path;
10. pre-execution executable digest mismatch before process creation;
11. post-execution executable mutation failure with cleanup;
12. subprocess launch failure cleanup;
13. isolation construction failure cleanup;
14. invalid temporary-root stem rejection;
15. NUL argument rejection;
16. non-string argument rejection;
17. relative executable rejection;
18. direct executable symlink rejection;
19. non-executable regular-file rejection;
20. PWD/cwd divergence rejection;
21. temp-alias divergence rejection.

The focused workflow additionally validates the real #681/#624 profiles and executes one smoke process through those actual profile files.

## Generator-root consequence

This PR does **not** modify `derive()` and therefore does not yet alter Lineage-B `GeneratorImplementationRoot`.

When a later integration makes `derive()` depend on this executor, the executor implementation becomes output-affecting code and must enter the scientific generator commitment.

The expected future generator surface is at least:

```text
common
gifti
derive
snapshot
executor
```

assuming those are exactly the modules that can affect scientific output bytes at that integration point.

Admission/capture/verifier-only code that cannot alter output bytes should remain in the qualification envelope rather than being indiscriminately mixed into the generator root.

## Deliberate non-claims

A successful qualification of this PR does **not** establish:

```text
real HCP/BALSA input custody
run_manifest_captured
real Workbench transform invocation
same-host transform repeatability
snapshot integration
path equivalence
cross-CPU equivalence
scientific execution qualification
atlas correctness
FMQ-010
neural alignment
consciousness evidence
```

Core invariant:

```text
QualifiedExecutorMechanism != QualifiedScientificExecution
```

## Next boundary

After this primitive qualifies, the next generator-affecting tranche may combine:

```text
#576 immutable scientific-input snapshot
+
#991 verified run admission
+
this isolated executor
```

inside Lineage-B `derive()`.

That integration must:

- remove ambient `verify_inputs()` Workbench execution from the scientific path;
- consume snapshot paths for all fourteen scientific inputs;
- bind the exact admitted Workbench bytes/version roots;
- use this executor for every Workbench command;
- expand the generator implementation root coherently;
- migrate scientific/evidence commitments;
- replay the qualified #525 archival verifier against the new generator world;
- establish same-host repeatability separately before path or CPU transfer claims;
- defer FMQ-010 until all of those boundaries are green.
