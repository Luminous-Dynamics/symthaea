# Trusted Research Qualifier Admission v2

## Status

`SCI-INFRA-001B2` defines the GitHub admission and evidence-sealing layer above the portable bootstrap source-qualification theorem in `SCI-INFRA-001A2`.

Its core separations are:

```text
trusted policy / parser / subject guard / harness / sealer
!=
candidate source / qualifier manifest

candidate execution evidence
!=
authenticated final binding

manifest-declared scope
!=
canonical program scope
```

A successful final binding remains narrow:

```text
authority=adapter-binding-only
scientific_claim=NONE
```

It cannot upgrade the bootstrap harness into scientific, replication, independence, canonical-program-scope, or final locked-build authority.

## Why v2

The earlier SCI-INFRA-001B draft used a trusted `pull_request_target` workflow but eventually executed parser/harness bytes from the candidate checkout. A manifest-only child only proved those helper bytes matched its PR base; a stacked base is not necessarily trusted default-branch policy.

B2 removes that ambiguity with separate trusted and candidate checkouts, a trusted Git-object subject guard, scrubbed candidate subprocesses, fresh-runner authenticated sealing, and an explicit final evidence seal.

## Trusted policy set

```text
trusted/
  .github/workflows/research-bootstrap-qualification.yml
  scripts/run-research-qualification-manifest.py
  scripts/validate-research-qualification-subject.py
  scripts/qualify-research-crate.sh
  scripts/seal-research-qualification-binding.py

candidate/
  exact qualifier head
  exact sole source parent
  one data-only qualifier manifest
  frozen research source
```

The workflow, adapter, subject guard, harness, and sealer execute only from the trusted checkout. Candidate or stacked-base copies of those helpers are never invoked as policy authority.

The trusted checkout is pinned to the exact default-branch policy SHA supplied by the scheduler. The final binding records the trusted policy commit plus exact workflow/adapter/harness/sealer identities. The policy commit transitively commits the subject-guard bytes as well.

## Three-runner trust lifecycle

PR-triggered qualification is intentionally split across fresh GitHub-hosted runners:

```text
ADMIT runner
  authenticated live-PR preflight
  trusted subject/object guard
  freeze canonical manifest digest
          |
          v
EXECUTE runner
  re-derive exact subject
  trusted subject/object guard
  candidate subprocess environment scrubbed
  portable Cargo/test/Clippy theorem
  emit explicitly UNSEALED artifact
          |
          v
SEAL runner (fresh runner)
  fresh trusted + candidate checkouts
  download unsealed artifact
  re-derive exact subject
  trusted subject/object guard
  authenticated live-PR postflight
  validate every staged evidence byte
  mint final binding LAST
  upload SEALED artifact
```

The key security property is deliberately narrower than “credential-free runner”:

> Candidate-controlled Cargo/build-script subprocesses receive a scrubbed environment, and no repository-token-bearing shell step is executed on that same runner after candidate code has run.

GitHub runner infrastructure still has its own service context, so B2 does **not** claim hostile-code sandboxing. The fresh seal runner prevents candidate execution from persisting same-runner state that is later combined with the repository token used for authenticated postflight.

## Candidate subject theorem

The candidate checkout must satisfy:

```text
HEAD == expected_head
parents(HEAD) == [expected_base]
manifest.source_parent == expected_base
diff --no-renames --no-ext-diff(expected_base, expected_head) == [manifest_path]
```

Every declared source path must also have identical Git object identity in the source parent and qualifier head.

Thus a qualifier may target a frozen source/feature branch without treating that branch's CI helpers as trusted policy.

## Git object and symlink boundary

Canonical path strings are not sufficient by themselves. The trusted subject guard additionally requires:

- the qualifier manifest is a non-executable `100644` Git blob;
- the checked-out manifest is a regular non-symlink file contained by the candidate checkout;
- declared source roots are only regular blobs or Git trees;
- source roots may not be symlinks or gitlinks/submodules;
- recursively declared source trees may not contain symlink or gitlink descendants;
- source working-tree paths resolve inside the candidate checkout;
- parent/head source object mode, type, and object identity are identical;
- rename heuristics and external diff helpers do not participate in qualification scope.

This closes a class of cases where a syntactically canonical repository path could otherwise resolve through a different filesystem or Git-object type than the theorem intended.

## Closed data-only manifest

The manifest contains only:

- `schema`;
- `program`;
- `source_parent`;
- `expected_rust`;
- `packages`;
- `source_paths`.

Unknown fields, duplicate JSON keys, path traversal, non-canonical paths, duplicate package/source lists, and filename/program disagreement fail closed.

Canonical manifest identity is SHA-256 of compact sorted validated JSON.

### Important scope ceiling

The current manifest still declares `program`, `expected_rust`, `packages`, and `source_paths`. B2 proves that the exact declared scope was executed and binds that scope through the canonical manifest digest. It does **not** prove that the candidate-selected scope is the canonical or complete scope for the named research program.

Therefore:

```text
program=SCI-X + adapter PASS
!=
canonical SCI-X qualification
```

Consumers must key authority to the exact manifest digest and declared scope, not to the program label alone.

A successor theorem should introduce a trusted or source-anchored qualification profile whose digest fixes the canonical package/path/toolchain scope independently of the qualifier manifest. Until that profile exists, B2 grants no canonical-program-scope authority.

## Authenticated admission

For `pull_request_target`, the admission runner authenticates current PR state before any candidate Cargo/build-script execution and requires:

- state remains open;
- draft remains false;
- head SHA remains the frozen qualifier head;
- base SHA remains the frozen source parent;
- head and base repositories equal the workflow repository.

Admission then validates the candidate as data only and emits bounded job outputs: canonical manifest path, expected Rust version, and canonical manifest digest.

Manual `workflow_dispatch` is supported separately from the default branch. It requires explicit exact head, base, and manifest inputs; no PR-currentness claim is made for manual dispatch.

## Candidate execution boundary

The execute runner independently checks out the exact trusted policy and exact candidate subject, re-runs the subject guard and manifest resolver, and requires its manifest digest/toolchain result to equal the admission runner outputs.

The actual qualification adapter is then launched through `env -i` with only the minimal Rust/Cargo process environment. It does not receive `GITHUB_TOKEN`, `GH_TOKEN`, or GitHub OIDC token variables. The trusted adapter independently rejects `run` if those credential-bearing variables are present.

After candidate code completes, the execute runner performs no authenticated PR shell operation. Its only evidence transition is an explicitly named unsealed artifact.

## Unsealed inter-job artifact

Successful execution produces:

```text
research-bootstrap-unsealed-<run>-<attempt>
  receipt.txt
  Cargo.lock.generated
  Cargo.lock.patch
  manifest-binding.json   # provisional adapter output only
```

This artifact is deliberately **not final authority**. Its retention is short and its name says `unsealed`.

It exists solely to transfer candidate execution evidence across the runner boundary. The fresh seal runner treats every byte as untrusted input and revalidates it before a final binding can exist.

A successful execution artifact therefore means only:

```text
candidate execution produced a candidate evidence set
```

not:

```text
GitHub currentness verified after execution
or
adapter binding sealed
```

## Fresh-runner authenticated final seal

The final sealer runs on a fresh runner after successful execution.

Before creating the final evidence directory it:

1. checks out and verifies the exact trusted policy commit again;
2. checks out the exact candidate head again;
3. re-runs the trusted Git-object subject guard;
4. re-runs manifest resolution and requires the same canonical manifest digest/toolchain identity observed during admission and execution;
5. verifies the sealer is itself executing from the trusted checkout and binds its exact SHA-256;
6. re-verifies candidate head, sole parent, manifest, and frozen source topology;
7. requires the downloaded staging directory to contain exactly four expected regular, non-symlink files;
8. strictly re-validates the harness receipt using the closed receipt vocabulary;
9. recomputes SHA-256 of `Cargo.lock.generated` and `Cargo.lock.patch` and requires exact equality with the receipt;
10. re-derives every `source_object` directly from the frozen source-parent Git objects;
11. requires the provisional adapter binding to equal the exact trusted adapter contract with no missing, extra, or changed fields;
12. performs authenticated live-PR postflight for PR-triggered admission;
13. recomputes trusted policy identities immediately before minting the final seal.

Only then may the final evidence directory be created.

## Final evidence set

The sealed upload contains only:

```text
receipt.txt
Cargo.lock.generated
Cargo.lock.patch
manifest-binding.json
```

The first three files are copied from the fully verified unsealed set. The final `manifest-binding.json` is written **last** and with exclusive creation semantics.

Its schema is:

`symthaea.research-qualifier-binding.v2`

and it binds at least:

- `seal_profile = authenticated-postflight-v1`;
- `authority = adapter-binding-only`;
- `scientific_claim = NONE`;
- canonical manifest SHA-256;
- source parent and qualifier head;
- trusted policy commit;
- trusted workflow SHA-256;
- trusted adapter SHA-256;
- trusted harness SHA-256;
- trusted sealer SHA-256;
- harness receipt SHA-256;
- generated lock SHA-256;
- lock patch SHA-256;
- admission mode;
- live PR postflight status;
- PR number where applicable;
- GitHub run ID and run attempt.

A final sealed artifact can exist only after the fresh runner's authenticated postflight succeeds.

## Strict receipt consumption

A zero exit status from the portable harness is insufficient.

The trusted adapter and final sealer parse `receipt.txt` using a closed vocabulary. Scalar keys must occur exactly once. Only these fields may repeat:

- `package`;
- `source_path`;
- `source_object`;
- `qualifier_path`.

Every gate must equal `PASS`. Unknown fields fail closed, including authority-like fields.

The final sealer additionally closes the gap between declared hashes and actual evidence bytes by recomputing the generated-lock and lock-patch digests itself.

## Non-circular activation boundary

B2 cannot honestly establish its own hosted-runner trust while it is introducing the workflow that provides that trust.

The trusted GitHub events used here depend on the workflow being present in trusted default-branch policy. Therefore the bootstrap sequence is intentionally:

```text
1. source-review / source-validate SCI-INFRA-001A2 + SCI-INFRA-001B2
2. merge the trusted policy onto the default branch without claiming hosted B2 PASS
3. create a separate exact qualifier/canary subject that changes only an admitted manifest
4. let the now-default-branch B2 workflow execute that subject
5. treat only that later sealed run as hosted admission/sealing execution evidence
```

A PR that introduces or changes the trusted workflow cannot cite a run of its own candidate workflow bytes as trusted-policy qualification evidence.

## Scheduler and repository-policy boundary

Runner-backed execution is admitted only for:

- a same-repository, open, non-draft `pull_request_target` subject; or
- explicit `workflow_dispatch` from the repository default branch.

Draft, converted-to-draft, closed, and fork-origin PR events do not enter candidate execution.

Same-PR concurrency uses `cancel-in-progress: true` as scheduler behavior. No cancellation or hosted-runner-health claim is made until observed.

Repository/organization Actions policy must permit the selected trusted event. Platform permission to schedule `pull_request_target` is an operational prerequisite, not qualification evidence. If GitHub policy changes or blocks the event, the qualification theorem fails closed rather than treating another trigger as equivalent automatically.

## Platform boundary

`pull_request_target` is a privileged GitHub event. Its usefulness here is that trusted policy comes from the default-branch context, not from the candidate merge context.

B2 still treats platform admission as a separate assumption. It keeps permissions read-only, admits only same-repository subjects, persists no checkout credentials, gives candidate subprocesses a scrubbed environment, and moves authenticated postflight to a fresh runner.

This remains **not** a hostile-code sandbox. Truly adversarial candidate code requires a stronger isolated execution substrate than a general GitHub-hosted build runner.

## Source-level validation

The policy sources have runner-independent checks:

```bash
bash -n scripts/qualify-research-crate.sh
scripts/qualify-research-crate.sh --self-test
python3 -m py_compile scripts/run-research-qualification-manifest.py
python3 scripts/run-research-qualification-manifest.py --self-test
python3 -m py_compile scripts/validate-research-qualification-subject.py
python3 scripts/validate-research-qualification-subject.py --self-test
python3 -m py_compile scripts/seal-research-qualification-binding.py
python3 scripts/seal-research-qualification-binding.py --self-test
```

The workflow source must additionally parse as YAML and later pass repository action/workflow syntax qualification.

These checks establish source-contract behavior only. They do not establish hosted-runner execution or a real research-crate qualification.

## Nonclaims

SCI-INFRA-001B2 does not establish:

- hostile-code sandboxing;
- canonical program scope for a candidate-declared manifest;
- hosted-runner or scheduler availability;
- final `Cargo.lock` acceptance;
- reproducible binaries;
- merge enforcement;
- scientific correctness, novelty, causality, replication, or independence.

Those require separate evidence lines.