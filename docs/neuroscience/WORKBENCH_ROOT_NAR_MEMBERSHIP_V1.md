# Workbench Root NAR Membership v1

Status: **candidate pure interpretation theorem; no real Workbench root NAR has been captured or verified by this PR**

Schema: `symthaea-workbench-root-nar-membership-v1`

## Purpose

The closure stack identifies a realized Workbench root and its canonical NAR SHA-256. The invocation profile then requires the eventual scientific runner to execute:

```text
<independently-verified-root>/bin/wb_command
```

A remaining provenance gap is subtle but important:

```text
verified root NAR hash
    + producer-reported SHA256(bin/wb_command)
```

does not let a fresh verifier independently prove that the reported program bytes are actually a member of that root.

A producer could otherwise hash an unrelated file and label it `bin/wb_command`.

The stronger theorem is:

```text
verified root NAR SHA-256
    + exact retained root NAR bytes
        -> independent NAR hash check
        -> canonical filesystem parse
        -> exact target-path membership
        -> target node type / executable bit / contents SHA-256
```

No live Nix store is required by the verifier.

## Why the NAR is the correct witness

Nix defines the Nix Archive (NAR) as the canonical serialization of the filesystem object tree. A NAR contains only regular files, directories, and symlinks; regular-file execute state is represented explicitly; directory entries are canonically ordered.

The NAR hash stored for a Nix store object is the cryptographic hash of that canonical serialization.

Therefore, once the independently verified closure says:

```text
root path R
nar_sha256 H
```

retained bytes `N` can prove membership without consulting `/nix/store` when:

```text
SHA256(N) = H
```

and an independent NAR parser reconstructs the target node from `N`.

This gives a cryptographic bridge:

```text
verified closure root entry
        ↓
verified root NAR bytes
        ↓
verified bin/wb_command membership
```

## Hash-before-parse boundary

The verifier first streams the entire candidate NAR and requires its SHA-256 to equal the already verified root NAR hash.

Only matching bytes proceed to structural parsing.

Conceptually:

```text
untrusted candidate NAR
    -> SHA256 mismatch -> reject

cryptographically committed root NAR
    -> parse canonical filesystem tree
```

This prevents arbitrary producer bytes from being treated as a trusted filesystem witness merely because they are syntactically NAR-shaped.

## v1 NAR grammar

The parser implements the canonical `nix-archive-1` structure:

```text
regular
    optional executable marker
    contents

symlink
    target

directory
    strictly ordered entries
```

All NAR framing strings use 64-bit little-endian lengths and zero padding to an eight-byte boundary.

v1 verifies:

- exact `nix-archive-1` magic;
- exact node-open/node-close grammar;
- only `regular`, `directory`, and `symlink` node types;
- zero padding;
- strictly increasing directory-entry names;
- no duplicate names;
- no empty, `.`, `..`, slash-containing, or NUL-containing entry names;
- no trailing archive bytes;
- bounded directory recursion;
- bounded structural strings;
- streaming regular-file contents with no whole-file materialization.

The 16 MiB structural-string bound is a verifier safety profile, not a claim that the abstract NAR format itself has that filename/symlink limit. Ordinary Workbench package metadata is expected to remain far below it; a real root NAR exceeding this v1 bound would require an explicit profile revision rather than silent parser widening.

## Target path

The default target is:

```text
bin/wb_command
```

The supplied target string must itself be exact canonical relative POSIX spelling. The parser rejects absolute paths, `.` / `..`, duplicate separators, normalized-away components, slash-containing components, and NULs.

This prevents:

```text
bin//wb_command
bin/./wb_command
bin/../wb_command
```

from becoming alternate textual identities for the same target.

## Target result

For a regular file, the verifier returns:

```text
node_type = regular
executable = true | false
content_length
content_sha256
```

The contents SHA-256 is streamed directly from the NAR payload.

For a symlink it returns:

```text
node_type = symlink
symlink_target_hex
```

and deliberately does **not** manufacture a regular-file contents hash.

That distinction matters because the selected package may use wrapper/symlink mechanics. Until a real root NAR is observed, this PR does not assume whether `bin/wb_command` is a regular wrapper or symlink.

If the real target is a symlink, a later theorem must resolve that symlink within the same verified NAR under explicit cycle/traversal rules before claiming exact executable bytes.

## Execute bit is independent evidence

Two regular files can contain identical bytes while differing in executable state.

NAR serialization commits the execute marker separately from contents, so v1 returns both:

```text
content_sha256
executable
```

and never infers one from the other.

Thus:

```text
same file bytes != same NAR filesystem object
```

when execute state differs.

## Authority boundary

Successful verification means only:

```text
nar_bytes_match_verified_root = true
target_membership_verified = true
```

It does not establish:

```text
workbench_execution_qualified
transform_executed
atlas_correctness
fmq010
neural_alignment
consciousness_evidence
```

The theorem is:

```text
ProgramMembership != ProgramExecution
```

and also:

```text
ProgramExecution != ScientificCorrectness
```

## Adversarial qualification

The synthetic-NAR suite covers 23 authored contracts including:

- valid executable regular target;
- non-executable regular target retained as non-executable;
- symlink target retained without fake file hash;
- NAR-hash mismatch rejection;
- noncanonical expected-hash rejection;
- wrong magic rejection even under a matching test hash;
- non-zero padding rejection;
- unsorted entry rejection;
- duplicate entry rejection;
- invalid directory-name rejection;
- absolute and traversal target rejection;
- missing target rejection;
- trailing-byte rejection;
- unknown node-type rejection;
- malformed regular-field rejection;
- truncated contents rejection;
- oversized structural-string rejection before large allocation;
- file-content mutation changing both NAR and file identities;
- execute-bit mutation changing NAR identity without changing file-content SHA;
- empty-directory handling;
- CLI round trip.

These contracts are authored by this PR but are not called hosted-qualified until the exact focused workflow executes successfully.

## Future observation producer

After the parent closure/verifier and invocation-profile stack are qualified, a separate producer can retain the exact root NAR with a pinned command such as the Nix 2.33 legacy dump interface:

```text
nix-store --dump <verified-root>
```

The producer should preserve raw stderr, exit status, command identity, Nix version, and the raw NAR bytes as an artifact.

A fresh no-Nix verifier can then:

```text
1. obtain root nar_sha256 from independently verified closure evidence
2. hash retained root.nar
3. require equality
4. parse root.nar with this qualified interpreter
5. prove bin/wb_command membership
6. obtain exact file-content SHA and execute state without live-store consultation
```

Only after this bridge is independently verified should #624's required actual `wb_command` SHA-256 be admitted into an execution-capsule receipt.
