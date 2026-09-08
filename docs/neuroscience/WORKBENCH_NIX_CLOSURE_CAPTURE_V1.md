# Workbench Nix Closure Capture v1

Status: **candidate raw-observation producer; no closure-capture or scientific qualification is claimed**

Schema: `symthaea-workbench-nix-closure-capture-receipt-v1`

## Purpose

This layer connects the static Workbench execution-capsule selection profile to an actually observed Nix realization without allowing the capture tool to silently become scientific authority.

The v1 observation theorem is:

```text
qualified-shape #624 profile
+ exact root flake.lock selection
        ↓
derive immutable nixpkgs installable
        ↓
raw nix build observation
        ↓
exactly one realized /nix/store root
        ↓
raw nix path-info --json-format 2 --recursive observation
        ↓
#629 canonical closure compiler
        ↓
normalized closure identity
```

The output remains an **unqualified observation receipt**. It does not establish Workbench execution correctness, atlas correctness, FMQ-010, neural alignment, or consciousness evidence.

## Root substitution is forbidden

The CLI does not accept a store root or arbitrary installable from the caller.

It reads:

- `data/neuroscience/workbench_execution_capsule_profile_v1.json`;
- the exact current `flake.lock`.

The capture derives:

```text
github:<locked owner>/<locked repo>/<profile rev>#<profile package attribute>
```

For the current v1 profile this is:

```text
github:NixOS/nixpkgs/9ae611a455b90cf061d8f332b977e387bda8e1ca#connectome-workbench
```

The profile revision/NAR root must equal the selected `flake.lock` node, and the root input must actually select that node. A caller cannot replace the realized root after this binding.

## Exact command contract

The producer records exact argv arrays and byte-retains stdout/stderr for:

```text
nix --version

nix eval --raw --impure --expr builtins.currentSystem

nix build --no-link --print-out-paths --no-write-lock-file \
  github:NixOS/nixpkgs/<exact-rev>#connectome-workbench

nix path-info --json --json-format 2 --recursive <realized-root>
```

Commands are executed without a shell. No command string is reparsed by `/bin/sh`.

`nix build` must report exactly one canonical `/nix/store/...` path. Multiple roots, malformed output, or a nonzero exit fail closed.

The path-info command is constructed only from that observed realization root.

## Why JSON format 2 is explicit

`nix path-info --json` is an experimental interface whose JSON representation changed in Nix 2.33. v1 therefore freezes:

```text
Nix 2.33.6
--json-format 2
```

The hosted workflow also freezes the Nix installer URL and pins the Nix installer GitHub Action by commit SHA.

The raw JSON bytes are retained. The parser does not pretend the normalized projection is equivalent to the raw Nix response.

## Raw observation versus normalized identity

The receipt keeps two intentionally different identities:

```text
RawObservationDigest
    = H(exact command records + retained byte digests)

NormalizedClosureDigest
    = #629 closure identity over
      {store path, canonical SHA-256 NAR identity, references}
```

Therefore:

```text
RawObservationRoot != NormalizedClosureRoot
```

Nix may expose metadata that is not intrinsic to the scientific runtime closure, such as registration-related fields. Those bytes remain covered by the raw observation while the #629 scientific closure projection ignores them.

A change in unconsumed Nix metadata may therefore change the raw observation digest without changing the normalized closure digest. This is intentional.

## Hash translation boundary

Nix JSON format 2 emits NAR SHA-256 values in SRI/base64 form. The producer accepts only canonical SHA-256 SRI values and converts the decoded 32 bytes to:

```text
sha256:<64 lowercase hex>
```

before calling #629.

The translation is part of this capture implementation and is covered by adversarial tests. It is not hidden inside the scientific closure compiler.

## Platform observation

The producer observes Nix's current system token separately and requires exact equality with the #624 v1 profile:

```text
x86_64-linux
```

A successful aarch64 realization cannot be laundered into the x86_64 profile merely because the package name and version match.

This still does not establish cross-machine floating-point equivalence; Workbench itself is not executed by this PR.

## Failure evidence

A failed realization is still an observation.

The producer retains raw command outputs and emits:

```text
status = observation-incomplete-unqualified
```

when the observation cannot reach a canonical closure identity.

It does not reinterpret infrastructure failure as scientific failure, and it does not manufacture a partial closure identity from incomplete evidence.

If realization stdout does not identify exactly one canonical store path, path-info is not executed.

## No overwrite

Receipt destinations are create-only. Capture is first assembled in a private temporary sibling directory and atomically renamed into place.

An existing destination is rejected. A second run therefore cannot silently rewrite evidence produced by the first run.

## Receipt structure

A successful receipt contains:

```text
receipt.json
raw/
  nix-version.stdout
  nix-version.stderr
  platform.stdout
  platform.stderr
  realization.stdout
  realization.stderr
  path-info.stdout
  path-info.stderr
normalized/
  closure_identity.json
```

`receipt.json` binds:

- exact profile and flake-lock SHA-256 digests;
- exact selected nixpkgs owner/repository/revision/NAR root;
- exact package attribute and derived installable;
- capture and #629 normalizer implementation SHA-256 digests;
- each argv, exit code, sidecar byte length, and sidecar SHA-256;
- raw observation digest;
- normalized closure file SHA-256 and closure digest when available;
- a self-contained capture digest.

## Facts are not authority

The receipt may truthfully report mechanical facts such as:

```text
selection_revalidated
nix_version_observed
platform_observed
realization_command_observed
realized_root_observed
path_info_observed
canonical_closure_identity_compiled
```

These are deliberately separate from authority.

The authority object must keep all stronger claims false:

```text
closure_capture_qualified             false
workbench_execution_qualified         false
transform_executed                    false
atlas_correctness_established         false
fmq010_established                    false
neural_alignment_established          false
consciousness_evidence                false
```

A receipt being internally well formed is not proof that the producer is correct.

## Qualification boundary

This PR qualifies only the **producer contract and its hosted observation attempt**.

Its focused static suite covers selection drift, root substitution, multi-output ambiguity, platform drift, JSON duplication/schema drift, SRI normalization, orphan closure entries, raw-versus-normalized identity separation, failure retention, and no-overwrite behavior.

The hosted lane additionally performs a real Nix realization and publishes the resulting receipt artifact.

The receipt is not called independently verified by this PR.

## Required next layer

A separate stacked verifier must treat the receipt directory as hostile input and independently reconstruct:

1. profile/lock selection identity;
2. exact argv contracts;
3. retained sidecar byte hashes and lengths;
4. command/result consistency;
5. Nix version and platform observations;
6. realized root identity from raw build stdout;
7. path-info SRI translation;
8. #629 normalized closure identity;
9. raw observation digest;
10. capture digest;
11. authority non-escalation.

Only after that verifier independently qualifies should the observed realization be eligible to become the realization component of the #624 execution capsule.

## Scientific non-claims

Even a hosted-green capture means only:

```text
this exact selected Workbench package was realized on this x86_64-linux observation lane
and this exact Nix closure was observed and canonically normalized
```

It does **not** mean:

```text
Workbench has executed a cortical transform
HCP/BALSA inputs are correct or authorized
Lineage B is scientifically qualified
FMQ-010 is established
human neural geometry aligns with Symthaea geometry
any consciousness claim is supported
```

Those remain later experiments.
