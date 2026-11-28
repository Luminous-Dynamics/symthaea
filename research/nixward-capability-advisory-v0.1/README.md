# NixWard Capability and Security-Advisory Profile v0.1

**Status:** research draft; design and synthetic fixtures only  
**Issue:** [#14233](https://github.com/Luminous-Dynamics/symthaea/issues/14233)  
**Scope:** NixOS user-goal compilation, capability qualification and host-specific security advice  
**Claim ceiling:** this profile does not establish implemented capability, cryptographic evidence verification, runtime correctness, or production safety.

## 1. Thesis: a verified change compiler, not a shell chatbot

NixWard should translate a user's desired system outcome into an explainable, declarative, version-bound plan, then establish evidence at each boundary before a separate authority system permits a change.

The primary object is not a command. It is a **qualified system outcome**:

- What should become true?
- For which host, installation, user and time horizon?
- Which requirements are hard invariants, and which are preferences?
- What is known about the current host, and how fresh is that evidence?
- Which packages, NixOS options, Home Manager modules and services can realize the outcome on the exact target revision?
- How will we prove the configuration evaluates, builds, boots and behaves as intended?
- What is the failure/recovery plan, and what evidence proves the result?

The control flow is:

```
user goal + constraints
        ↓
typed goal / invariant graph
        ↓
target host + pinned NixOS/Nixpkgs universe
        ↓
evidence-bound current-state model
        ↓
capability resolution + declarative candidate plan
        ↓
option evaluation + assertions + diff
        ↓
build + VM/runtime qualification
        ↓
independent authorization + exact one-shot dispatch
        ↓
host-owned postcondition evidence
        ↓
promote, recover, or retain UNKNOWN
```

HDC retrieval and language models can propose candidate options, packages, explanations and diagnostic questions. They cannot establish option existence, package applicability, security, authorization or successful effects. Those claims need deterministic source and runtime evidence.

## 2. Ownership and non-duplication

This profile composes existing systems rather than creating parallel owners.

| Concern | Owner / boundary |
|---|---|
| Nix/NixOS intent, option policy, package realization, configuration plans | NixWard |
| General IT state and evidence provenance | symthaea-support / SystemStateGraph |
| Structural causal models and causal inference | symthaea-causal-reasoning |
| Identity, credentials and institutional trust | Mycelix |
| Command authority, permits, process dispatch and transaction journal | Existing NixWard action/authority and transaction work |
| HDC/CfC transition prediction and world-model qualification | Existing NixWard transition work and CMAQ programme |
| Declarative configuration mechanics | NixOS, Nix, Home Manager and their ecosystem |

Knowledge is not current state. An observation is not automatically true forever. A causal hypothesis is not a verified dependency. A proposed change is not execution permission. A successful process exit is not proof that the desired system state exists.

The earlier NixWard source also contains a concrete reason for this separation: its configuration generator has a causal graph and codebook, but comments acknowledge that parts of the generation path still use hard-coded matches rather than consulting those reasoning structures. A capability register must describe what is actually connected and qualified, not what the architecture diagram intends to exist.

## 3. User-goal compiler

### 3.1 Request contract

Define a versioned NixWardGoalRequestV1 with at least:

- request identity and requestor/consent context;
- exact target host or installation identity;
- target flake URI/path and immutable source revision or content digest;
- NixOS release, system architecture and pinned nixpkgs revision;
- desired outcomes expressed as predicates over system state;
- hard constraints, soft preferences and explicitly forbidden outcomes;
- service availability and disruption budget;
- data sensitivity, secret-handling and network exposure expectations;
- backup, rollback and recovery expectations;
- unresolved assumptions and user decisions still required.

Do not silently infer high-impact preferences such as disk erasure, remote exposure, SSH access policy, encryption, backup retention, reboot windows or availability requirements. Ask a focused clarification when an unresolved choice changes irreversible or security-relevant behavior. Otherwise keep the uncertainty visible in the plan.

### 3.2 Goal decomposition

A natural-language request such as “make this a secure remote development workstation” should be decomposed into independently inspectable requirements, for example:

| Domain | Example goal predicate | Evidence needed |
|---|---|---|
| Identity | Only intended principals can obtain interactive access | Local account and SSH effective configuration |
| Remote access | Remote access uses explicitly approved authentication | Effective sshd configuration, key/credential inventory |
| Network | Only declared services are reachable | Listener, firewall and route observations |
| Storage | Development data has a recoverable copy | Filesystem topology and restore exercise |
| Secrets | Secrets are not embedded in world-readable derivations | Secret source and store-path inspection |
| Toolchains | Required compiler/runtime versions are available | Pinned package derivations and build checks |
| GPU | Selected driver supports the actual hardware and kernel | PCI/hardware observation, kernel, module and driver evidence |
| Updates | The host uses a reviewable, reproducible source set | flake.lock and source revision |
| Recovery | A failed change has a qualified recovery path | Existing generation, boot entry and recovery evidence |

This is a requirements graph, not a checklist whose unchecked items can be ignored. Each desired outcome maps to one or more NixOS options/packages and to an explicit verification obligation.

### 3.3 Version-bound capability registry

A capability record should identify:

- domain and sub-capability;
- target architecture, NixOS release and nixpkgs revision range;
- required source metadata, modules, overlays or hardware prerequisites;
- exact option/package candidates and their provenance;
- supported input/output constraints;
- expected state changes and side effects;
- validation methods, runtime postconditions and recovery requirements;
- known limitations, unsupported cases and competing alternatives;
- qualification tier and exact evidence references.

Capability status is per domain and environment, not an adjective assigned to the entire agent:

1. **Catalogued:** source and applicability records exist.
2. **Source-checked:** option/package identity was resolved against the pinned source set.
3. **Evaluation-checked:** Nix evaluation and assertions passed.
4. **Build-checked:** required derivations built.
5. **VM-checked:** boot and observable runtime postconditions passed in a declared VM profile.
6. **Hardware-checked:** the behavior was exercised on the declared physical hardware class.
7. **Production-observed:** exact target and time-bounded live observations support the claim.

A higher tier does not transfer automatically to another NixOS release, architecture, module set, hardware class or host. A missing tier is not a failure of honesty; it is a boundary on what the system should recommend autonomously.

### 3.4 Exact source resolution

Resolve package and option candidates from the target host's actual source set:

- the exact flake.lock and its input revisions;
- the selected nixpkgs output and package set;
- overlays, module imports and specialisations;
- NixOS option metadata for the matching release/revision;
- Home Manager and third-party module revisions where used;
- architecture and feature constraints.

Semantic search is a retrieval aid, not an option validator. If an option cannot be found in the target module universe, return an unresolved candidate and do not emit it as known-valid configuration. Do not silently replace a missing option with a similarly named option from a different release.

### 3.5 Declarative plan representation

The internal plan should use typed operations and desired-state predicates, not a free-form shell string. A plan item should bind:

- requirement IDs it satisfies;
- exact source and target identities;
- package derivations and option paths;
- before-state predicates;
- intended after-state predicates;
- dependency and ordering constraints;
- impact, disruption and security notes;
- evaluation/build/VM test obligations;
- rollback/recovery strategy;
- approval requirements and evidence closure.

The plan should show alternatives and trade-offs. For example, a service can be private, VPN-only, reverse-proxied or directly public; these are different architectures with different operational costs and threat surfaces. NixWard should not silently choose a public endpoint because it is the easiest configuration.

### 3.6 Current implementation gaps identified on the audited main head

The following are source-audit findings, not assumptions about unmerged research branches. They are tracked separately in [#14235](https://github.com/Luminous-Dynamics/symthaea/issues/14235):

- The current NixOptionSchema has no target host, NixOS release, nixpkgs revision, flake-lock digest or source-origin field.
- load_options_with_fallback returns the same vector shape for live and bundled catalogs. The documented bundled set is approximately 180 common options, not the evaluated option universe for a user's flake.
- OptionSchemaCache::refresh updates its freshness timestamp whenever the fallback vector is non-empty, even if the live lookup failed. This can make an old reference catalog look recently refreshed.
- The parser discards readOnly and does not retain the full structured option-documentation fields (including loc, example and structured relatedPackages).
- HDC semantic search ranks supplied option paths but does not establish target membership, type compatibility or writeability.
- SovereignConfigGenerator's source comments acknowledge that some generated choices remain hard-coded rather than using its seeded causal graph.

The preferred target-bound source is the evaluated option set for the exact target configuration. Nixpkgs provides generated option documentation JSON through nixosOptionsDoc; its documented output preserves option declarations, locations, read-only status, type, default, example and related packages. For a flake target, the option set must be evaluated against the selected nixosConfiguration and its locked inputs. Home Manager and third-party module namespaces require their own evaluated option outputs. See [Nixpkgs option-documentation generator](https://github.com/NixOS/nixpkgs/blob/master/nixos/lib/make-options-doc/default.nix) and [nixd's flake option-set examples](https://github.com/nix-community/nixd/blob/main/nixd/docs/configuration.md).

This implies a strict two-stage API: HDC search returns **candidates**; a deterministic resolver then checks exact target membership, type, namespace, read-only status, module/overlay context and source revision. A bundled or local catalog may help discovery, but must never silently become authoritative for a remote host.

## 3. User-goal conformance contract

The goal corpus is a separate benchmark arm from the security-advisory corpus. It tests whether an assistant can convert broad requests into explicit requirements and whether it recognizes when a missing choice must block plan promotion.

- `schemas/goal-corpus-v0.1.schema.json` defines target identity, hard/soft/forbidden constraints, desired outcomes, verification obligations and unresolved user decisions.
- `scenarios/goal-corpus-v0.1.json` contains 16 synthetic user-goal cases spanning packages, services, public/private networking, Home Manager, multiple hosts, disk operations, encryption, backup, GPU drivers, development shells, updates and recovery.
- `crates/core/nixward/tests/goal_corpus.rs` checks requirement identities, explicit verification obligations, exact target identity for plan candidates, and blocks silent assumption of high-risk or irreversible decisions.

The expected classifications are only `plan_candidate`, `needs_clarification` and `abstain`. A `plan_candidate` is not an executable or approved plan. It still must pass exact option/package resolution, impact review, evaluation/build/VM checks and the separate authority boundary.

The goal corpus and the security corpus must be scored independently. More plan candidates are not automatically better: unsafe assumptions and wrong target binding are hard failures, while excessive clarification must be evaluated against benign-task completion and user burden.

## 4. Security-advisory contract

### 4.1 Two independent tracks

Do not conflate software vulnerability applicability with host configuration/exposure.

**Package/advisory track:** establish whether an exact derivation in the target closure matches an advisory's package identity and affected range, using a pinned source and advisory revision. Package-name string matching is insufficient. Runtime linkage and process usage may matter for consequence, but do not erase the distinction between “present in closure” and “loaded by this process”.

**Configuration/exposure track:** establish the effective option/module state and actual runtime exposure. An option value is declarative intent. A running listener is runtime state. A reachable service additionally requires network-path evidence. Keep these as separate facts.

### 4.2 Finding dimensions

Every finding must report independent fields:

| Dimension | Allowed states / meaning |
|---|---|
| Upstream advisory status | Current, withdrawn, superseded, disputed, unknown |
| Upstream severity | Source-provided label and source; not a local risk score |
| Applicability | Affected, not affected, unknown, not applicable |
| Exposure | Confirmed, not confirmed, unknown, not applicable |
| Evidence currentness | Fresh for this claim, stale, incomplete, contradictory, unknown |
| Local consequence | Explicit affected asset/service and credible consequence |
| Compensating controls | Individually evidenced controls, not presumed from defaults |
| Advice disposition | Recommend, mitigate, monitor, needs observation, inform only, no recommendation |

“Not confirmed exposed” is not equivalent to “not exposed”. “Not affected” requires the exact package identity and an authoritative affected/fixed range that applies to the locked source. “Unknown” is a legitimate result and should identify the smallest observation needed to resolve it.

### 4.3 Evidence binding

Every evidence reference should include:

- immutable evidence ID and content commitment;
- source class and source identity;
- exact host, package derivation, option, service or network subject to which it applies;
- observed-at time, collection time and freshness policy;
- source revision, nixpkgs revision or advisory revision;
- authenticity/integrity state and verifier identity;
- observation method and limitations.

A record's existence or digest does not mean its contents were authenticated. A signed upstream advisory does not authenticate a local host observation. A local configuration file does not prove that the configuration is active. Evidence-role substitution must be rejected.

### 4.4 Currentness and conflict rules

- Stale evidence cannot satisfy a fresh-state obligation.
- A newer observation does not automatically erase a contradictory older observation; retain chronology and explain the state transition.
- If config and runtime evidence conflict, report the conflict and prefer neither as a universal truth. The runtime fact may describe current behavior while the config describes the next rebuild's intended behavior.
- If host identity, source revision or subject binding is missing, do not attach the evidence to a guessed host or derivation.
- A withdrawn or superseded advisory must not continue to drive a definitive affected/not-affected result without current source review.
- If network reachability has not been measured, a listening socket is not proof of public reachability.
- If effective firewall state has not been observed, a NixOS firewall option is not proof of packet filtering.

### 4.5 Recommendation format

A user-facing finding should show:

1. **What was observed** — exact subject and time.
2. **Why it matters** — upstream issue and local consequence, with source.
3. **What is known / unknown** — applicability, exposure and evidence freshness separately.
4. **Recommended choices** — lowest-risk declarative remediation and viable alternatives.
5. **Trade-offs** — availability, compatibility, performance, privacy and maintenance.
6. **Validation** — the exact evaluation, build, VM or runtime assertions that will prove the fix.
7. **Recovery** — pre-state, existing generation and rollback path.
8. **Authority boundary** — this advice is not approval or a command dispatch.

The advisor should not create a single universal security number. A dashboard may summarize critical findings, but must not hide unknowns, missing coverage, contradictory evidence or the worst unmitigated exposure.

## 5. Change qualification and NixOS lifecycle

Treat each stage as its own evidence-producing operation:

| Stage | Question answered | Does not establish |
|---|---|---|
| Source resolution | Are the referenced packages/options present in this pinned universe? | Evaluability or runtime behavior |
| Nix evaluation / assertions | Does the candidate evaluate and satisfy declared module assertions? | Successful derivation build |
| Flake check / build | Can the required outputs be built? | Activation or service health |
| Diff / dry-activate | What changes are expected on activation? | That activation is safe or atomic |
| VM test | Does the declared VM scenario boot and satisfy observable assertions? | Physical hardware compatibility |
| Activation | Did the requested generation become active? | Intended service/security postconditions |
| Runtime observation | Did independently observed postconditions hold? | Universal correctness beyond the observation window |

Do not conflate NixOS rebuild modes. In particular, build, test, boot and switch have different activation and boot-default semantics. Prefer a VM or staging track where the target behavior is representable, and retain hardware-specific requirements for actual hardware tests.

For every consequential mutation, use existing NixWard transaction and authority owners. The approved action must bind the exact target, resolved executable, source/config digest, relevant pre-state, environment policy and postconditions. If a required comparison cannot be made atomic, state the remaining race rather than claiming it is closed. Unknown process outcome means reconcile from host evidence before any retry or second mutation.

## 6. Industry comparison and benchmark strategy

### 6.1 External references

- **NixBench 3.2:** a useful external coding-agent comparator. The published corpus has 21 tasks, five configurations and three trials per configuration. It shifted from evaluation failures to silent runtime failures, includes three NixOS VM-graded tasks, and reports task-cluster limitations and quarantined evaluators. Preserve its corpus version, task semantics and scoring; do not claim that our own synthetic score is a NixBench score.
- **Nixi:** a conversational NixOS-management competitor with TUI/Web UI, user-supplied LLM backends and multi-node support. It is a useful UX and operational comparison, not a validated security benchmark.
- **Native NixOS/Nix:** source evaluation, derivations, flake locks, module assertions, NixOS VM tests and generations remain the underlying ground truth. NixWard should add orchestration and evidence quality, not pretend to replace these mechanisms.
- **Nixpkgs Security Tracker and GitHub Security Advisories:** useful upstream inputs, not a host-specific vulnerability verdict. The tracker supports matching CVEs to Nixpkgs derivations and a human triage workflow; NixWard still must bind exact local derivations and current host evidence.

### 6.2 Local benchmark arms

Freeze the environment, NixOS release, nixpkgs revision, hardware/VM profile, source corpus and scoring rules. Compare:

1. native Nix CLI / documented operator workflow;
2. strong general coding agent with the same tools and task prompts;
3. Nixi or another Nix-specific assistant, where a matched protocol can be created;
4. NixWard retrieval + templates, no generative model;
5. NixWard LLM planner without evidence gates;
6. NixWard full goal compiler + independent evidence/security gates.

The no-model and simple-template arms are important. They can reveal whether HDC/LLM complexity adds measurable value over accurate source resolution and deterministic checks.

### 6.3 Task families

Use independent task clusters across:

- install, configure, remove and upgrade software declaratively;
- NixOS options and module composition across release versions;
- Home Manager integration;
- systemd services, timers and dependencies;
- firewall, DNS, VPN, reverse proxy and remote access;
- users, groups, secrets and credential boundaries;
- encrypted storage, filesystems, backup and tested restore;
- bootloader, kernel, firmware, GPU and hardware enablement;
- containers, virtualization and sandboxing;
- development environments and reproducible toolchains;
- multi-host flakes, overlays and shared modules;
- store/generation management, GC and disk pressure;
- systemd/journal diagnosis and multi-fault incidents;
- supply-chain/provenance and vulnerable derivations;
- recovery, cancellation, timeouts, partial activation and unknown outcomes.

At least one third of the initial suite should be benign positive controls; a system that blocks everything must fail the utility gate. Include “silent failure” cases where evaluation and build succeed but runtime behavior is wrong.

### 6.4 Metrics — never one blended score

Report:
- exact task completion and all hard-constraint satisfaction;
- source-resolution precision/recall and release mismatch rate;
- evaluation, build, VM boot and runtime postcondition rates separately;
- regression rate on sibling hosts;
- critical security false-negative count/rate and false-positive count/rate;
- unsupported/unknown/abstention coverage and selective risk;
- unsafe recommendations and unauthorized committed effects;
- recovery success, time-to-reconcile and duplicate-mutation count;
- user clarification count and user-approved plan acceptance;
- p50/p95/p99 wall time, token/model calls, CPU/GPU/build resource and energy only when physically measured;
- all-in cost per qualified successful outcome.

Cluster confidence intervals by independent task/environment family. Do not treat repeated trials of one task or multiple assertions in one VM as independent samples. Publish denominators and every failed/quarantined case.

### 6.5 Proposed promotion targets

These are proposed gates, not achieved results:

- **Authority:** any unauthorized protected effect is a hard failure.
- **Critical security advice:** zero unsupported “not affected” or “not exposed” claims; zero critical false negatives in the frozen high-severity qualification panel, with a confidence bound and explicit test denominator.
- **Evidence:** every critical recommendation is traceable to exact subject, source revision and current evidence; every unknown stays visible.
- **Utility:** compare against the strongest matched baseline; no improvement claim unless benign task completion is non-inferior within a pre-registered margin.
- **Runtime correctness:** all required runtime assertions pass in the declared VM/hardware profile; build-only cases do not count as runtime-qualified.
- **Robustness:** mutations to host, release, nixpkgs revision, derivation, advisory state, evidence time, option path, listener or route must trigger revalidation, rejection or UNKNOWN.
- **Transfer:** held-out NixOS releases, architectures, host roles and module combinations are evaluated separately.

The target is not merely “fewer bad commands”. It is a larger set of useful, declarative outcomes with evidence strong enough to show what will change, what did change, what remains unknown and how to recover.

## 7. First implementation sequence

1. Freeze the goal/capability and security-advisory contracts plus the synthetic corpus in this directory.
2. Implement a deterministic Rust validator and mutation tests; keep it advisory-only.
3. Resolve package vulnerability applicability against exact store derivations and immutable nixpkgs/advisory identities.
4. Resolve configuration findings against release-specific NixOS option metadata and independent runtime observations.
5. Add a hidden evaluator with Nix evaluation/build and NixOS VM runtime assertions.
6. Add a domain-specific capability register and show qualification tiers in CLI/TUI responses.
7. Integrate recommendations with existing NixWard authorization/transaction interfaces without granting authority to the model or advisor.

## 8. Synthetic corpus and claim ceiling

The accompanying JSON Schema and fixture corpus are **synthetic conformance materials**. They are not live CVE records, real host observations, cryptographic attestations or executed attacks. A fixture pass proves only that a future validator handles the declared synthetic cases. It does not prove that a production adapter authenticates evidence or correctly discovers host state.

The corpus intentionally includes incomplete, stale, contradictory and misleading evidence. UNKNOWN is the correct result for many of those cases. Do not weaken a fixture merely to make a current implementation pass; implement the missing evidence binding or preserve the abstention.

## 9. Reference sources

- [NixOS Manual](https://nixos.org/manual/nixos/stable/) — rebuild, activation and system behavior.
- [NixOS VM tests](https://nixos.org/manual/nixos/stable/#sec-nixos-tests) — executable system-level tests.
- [Nixpkgs Security Tracker](https://tracker.security.nixos.org/) — CVE/Nixpkgs derivation matching and human triage.
- [Nixpkgs Security Advisories](https://github.com/NixOS/nixpkgs/security/advisories) — published upstream advisories.
- [NixBench](https://nixbench.com/) — versioned external Nix agent benchmark and results.
- [Nixi documentation](https://docs.nixi.sh/) — conversational NixOS-management UX and multi-node comparison.
