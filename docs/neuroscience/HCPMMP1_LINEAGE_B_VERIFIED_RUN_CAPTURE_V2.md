# HCP-MMP1 Lineage-B Verified Run Capture v2

Status: **candidate qualification mechanism; no real Lineage-B run manifest captured by this profile**

## Purpose

PR #540 qualified a useful operator-side run-manifest capture theorem: close the fourteen-role scientific-input set, hash the selected bytes, bind the WN56/RVVG source pair, and publish a private candidate run manifest without running a cortical transform.

Its Workbench identity boundary was intentionally simpler: the capturer hashed an operator-selected executable and invoked that local executable with `-version`.

The later Workbench chain is materially stronger:

```text
#624 selected Workbench profile
    -> realized closure identity / raw observation
    -> #924 independent root-NAR program membership
    -> #681 isolated invocation contract
    -> #976 retained isolated real version observation
```

This v2 profile connects that qualified Workbench observation to Lineage-B run admission without allowing the run capturer to mint a new Workbench identity itself.

Core theorem:

```text
QualifiedOperatorByteCapture
    !=
QualifiedWorkbenchExecutionAdmission
```

and the successor construction is:

```text
externally qualified Workbench verification package
        +
operator-selected fourteen scientific input files
        +
local exact program-byte match
        ->
candidate Lineage-B run manifest
```

## Exact qualified Workbench trust package

The operator-facing v2 wrapper is pair-specific. It trust-anchors exactly the hosted-qualified #976 observation.

### Qualification source

```text
PR                         = 976
head                       = 5da8729fbe2d6017739c859c486a58f4d457c852
focused workflow run       = 34265087269
platform                   = x86_64-linux
```

### Retained verification file

The independently reconstructed JSON from #976 is retained in-repository at:

```text
data/neuroscience/evidence/workbench_isolated_version_verification_5da8729f.json
```

Its exact file root is:

```text
sha256:bd79a0280f0a9e5ad2308d6824ccda2f28d0b5cffa3bb4f89b3afa7089afe4d9
```

The original GitHub artifact roots remain provenance/transport evidence:

```text
independent verification archive
sha256:047060be35a90bdb0b6cf463b0b739fff506efefd4317bf04fc97b4a107506d7

raw evidence archive
sha256:b7d30bd90c0943135bb3163502a8a9169aeb8d086a55f6f3d98d81c0947308a7
```

The retained JSON is canonical JSON plus one final newline. The v2 validator rejects a semantically equivalent but byte-different representation.

### Verified Workbench identity

```text
root
/nix/store/g4bn2ilmgr5xpsazz6jw8bn8n0398lna-connectome-workbench-2.1.0

relative main program
bin/wb_command

program content
sha256:ad461ffeef56a0d807617e41ca65e38a2c25f1ec47abaea560a0bf843613db88

version
2.1.0

version output
sha256:d4e353408c9d76bc7c4ffe476e4161e0dc443add050ad499baa2cb93c84630c5

closure identity
sha256:4b9820c088e3ab1481833c1659c449b0acc4b168f172d8f43abf383fae6b8a6a
```

## Qualified capture profile root

The machine-readable profile is retained at:

```text
data/neuroscience/hcpmmp1_lineage_b_verified_run_capture_profile_v2.json
```

It is also exact canonical JSON plus one final newline. Its qualified file root is:

```text
sha256:975c7727ecb9faf7e3d46a039aa4e6601b7e2498b36443bf9012c51b4f2e9499
```

This root commits to the exact #976 PR/head/run, retained verification-file root, archive roots, Workbench program/version/closure identity, platform, and narrow verification authority state.

## Why the trust anchor is not caller-selectable

A parser that merely accepts:

```text
profile says verification root = X
verification bytes hash to X
```

has only proven mutual consistency. A caller could fabricate a new profile and matching verification object unless some authority outside that pair fixes the expected root.

Therefore the **qualified operator-facing wrapper** hard-codes both exact trust anchors:

```text
QualifiedProfileFileRoot
    = sha256:975c7727ecb9faf7e3d46a039aa4e6601b7e2498b36443bf9012c51b4f2e9499

QualifiedVerificationFileRoot
    = sha256:bd79a0280f0a9e5ad2308d6824ccda2f28d0b5cffa3bb4f89b3afa7089afe4d9
```

and fixes both repository paths. Its CLI exposes no `--verified-run-profile` or `--workbench-verification` substitution flags.

The reusable parser remains useful for closed-world validation and adversarial testing, but parser success by itself is not permission to claim the #976 qualification theorem.

Core invariant:

```text
MutuallyConsistentMetadata != QualifiedExternalTrustRoot
```

## Single-read evidence binding

The reusable parser reads the selected profile bytes once and the retained Workbench verification bytes once inside the capture window.

For each object:

```text
RawBytes
    -> SHA256(RawBytes)
    -> parse those same RawBytes
    -> canonical-byte check
```

There is no sequence of:

```text
hash pathname
    -> reopen pathname
    -> parse different bytes
```

for the profile or retained verification object.

The qualified wrapper then checks the exact digests returned from that capture window against its built-in trust anchors **before publication**.

## No Workbench execution during capture

Neither the reusable v2 capture library nor the qualified wrapper imports `subprocess` or invokes Workbench.

In particular the v2 capture path does not execute:

```text
wb_command -version
```

and does not execute any scientific Workbench transform.

Workbench `program_content_sha256` and `version_output_sha256` come only from the retained qualified #976 verification package.

The local program pathname is derived as:

```text
verified root / bin/wb_command
```

The exact local file must exist as the expected regular object and its bytes are hashed before and after the operator-input capture window. Both hashes must equal the independently retained #976 program-content root.

This establishes a local byte match to the previously verified program object. It does **not** reverify the entire local Nix closure.

Therefore:

```text
LocalProgramBytesMatch = true
LocalClosureReverified = false
```

## Scientific input capture

The candidate run still requires exactly the fourteen v1 Lineage-B scientific input roles.

Each operator-supplied input:

- must be present in the exact closed role set;
- must not be supplied through a direct symlink;
- is resolved to a regular file;
- is opened with `O_NOFOLLOW` where supported;
- is checked as an actually opened regular file with `fstat()`;
- is SHA-256 hashed through the opened descriptor;
- is re-hashed before capture completes;
- fails the capture if its bytes change during the window.

The method and completed candidate run are revalidated through the existing Lineage-B closed-world manifest boundary.

This v2 step still does not snapshot the scientific bytes. #576 remains the qualified immutable-snapshot primitive, and integration into scientific execution remains a later theorem.

## Candidate run schema

v2 intentionally emits the existing Lineage-B run schema rather than inventing a parallel scientific schema.

The Workbench fields are bound to the qualified observation:

```text
workbench.path
    = <verified-root>/bin/wb_command

workbench.sha256
    = qualified program-content SHA-256

workbench.version_output_sha256
    = qualified #976 version-output SHA-256
```

The fourteen input entries retain the operator-selected resolved paths plus the captured SHA-256 byte roots.

Execution ID and authorization-reference text remain execution/provenance metadata, not scientific-byte identity and not legal-entitlement or independence oracles.

## Output and disclosure boundary

The candidate manifest is written as a new private `0600` file and never deliberately overwrites an existing output path.

Successful CLI output is a digest-only receipt. It may expose qualification roots and closed authority state but does not print:

- operator scientific input paths;
- full input records;
- `authorization_reference`;
- the complete run manifest.

## Authority state

The retained #976 verification package establishes exactly:

```text
program_membership_verified = true
invocation_profile_verified = true
invocation_executed = true
version_output_bound = true
```

while retaining false values for stronger claims including repeatability, path equivalence, cross-CPU equivalence, Workbench scientific execution, transform execution, atlas correctness, FMQ-010, neural alignment, and consciousness evidence.

A successful **real** v2 candidate capture may additionally report:

```text
qualified_workbench_verification_bound = true
local_program_bytes_match = true
operator_input_bytes_captured = true
```

while preserving:

```text
local_closure_reverified = false
workbench_execution_qualified = false
scientific_execution_qualified = false
transform_executed = false
atlas_correctness_established = false
fmq010_established = false
neural_alignment_established = false
consciousness_evidence = false
```

Qualifying this software/profile in CI does **not** itself set `run_manifest_captured = true`. That state requires an actual operator capture with the real authorized scientific inputs.

## Provenance-root classification

This tranche changes capture/qualification code and retains a Workbench qualification object. It does not modify the Lineage-B transform or semantic generator.

Therefore its implementation belongs to:

```text
QualificationEnvelopeRoot
```

not:

```text
GeneratorImplementationRoot
```

No new scientific result root is created merely by qualifying this run-admission mechanism.

## Adversarial qualification

The combined dependency-free test surface covers:

- exact fourteen-role closure;
- duplicate/missing-role rejection;
- duplicate JSON-key rejection;
- canonical profile and verification bytes;
- retained verification-root substitution;
- profile and verification authority escalation;
- Python bool/int laundering rejection;
- Workbench root mismatch;
- local program byte mismatch;
- local program drift during capture;
- scientific input drift during capture;
- direct input-symlink rejection;
- restrictive no-overwrite output;
- single-read retained-verification semantics;
- no `subprocess` / ambient Workbench probe surface;
- fixed qualified wrapper paths;
- exact profile and verification trust anchors;
- pre-publication rejection of every trust-anchor mismatch;
- digest-only qualified CLI receipt;
- exact real retained #976 qualification-package reconstruction.

The dedicated workflow also statically rejects network/process-execution surfaces and asserts the exact file roots of the committed qualification package.

## Non-claims

This v2 profile does not establish:

- authorized HCP/BALSA acquisition;
- an actually captured real run manifest;
- immutable scientific-input snapshot consumption;
- full local Nix closure re-verification;
- a scientific Workbench transform;
- same-host repeatability;
- path equivalence;
- cross-CPU equivalence;
- external execution independence;
- atlas correctness;
- FMQ-010;
- empirical neural alignment;
- consciousness evidence;
- benchmark de-quarantine authority.

## Next boundary

After the parent custody/replay chain and this admission mechanism qualify on their exact heads, the next generator-affecting tranche should be narrower than a full scientific run:

```text
qualified v2 candidate run
    -> #576 immutable scientific snapshot
    -> minimal #681-conformant isolated Workbench executor
    -> derive() consumes only snapshot paths + isolated executor
    -> new GeneratorImplementationRoot
    -> migrated retained evidence commitment
    -> explicit #525 archival replay against the new generator world
```

Same-host repeated real cortical execution should follow before path-perturbation or FMQ-010 work.
