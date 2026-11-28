# NixWard benchmark seed v0.1

**Status:** synthetic task manifest; evaluator and fixture execution are follow-on work  
**Programme issue:** [#14268](https://github.com/Luminous-Dynamics/symthaea/issues/14268)  
**Manifest:** 41 independent task clusters  
**VM runtime tier:** 21 scenarios  
**Claim ceiling:** these are task specifications, not executed benchmark results.

## 1. Research question

NixWard should not be optimized for generating a plausible NixOS configuration or making it evaluate. It should be optimized for qualified system outcomes:

- the user's intended outcome is achieved;
- hard constraints and forbidden effects are preserved;
- source and target identity are exact;
- the candidate evaluates and builds;
- behavior is tested in a VM or on declared hardware;
- the actual effect is independently observed;
- recovery remains possible, including mutable service/application data.

The benchmark must measure useful autonomy and safety separately. A system that refuses every task cannot pass. A system that completes tasks by weakening controls cannot pass either.

## 2. Initial corpus

The first synthetic seed contains 41 unique task clusters across:

- packages and target package membership;
- NixOS option resolution, type and read-only handling;
- Home Manager namespace separation;
- systemd service lifecycle, ordering and hardening;
- private and intentionally public networking;
- SSH authentication and Match User restrictions;
- service identities and least privilege;
- secret store/runtime boundaries;
- backup and verified restore;
- garbage collection and recovery floors;
- destructive disk ambiguity;
- boot, kernel and GPU hardware qualification;
- rootless containers and network isolation;
- reproducible Rust development environments;
- multi-host baseline and sibling-host regression;
- flake updates and runtime compatibility;
- unknown activation outcomes and reconciliation;
- untrusted evaluator isolation, resource exhaustion and cache provenance;
- Nix expression input injection;
- mutable-state rollback and hostile source instructions.

The seed includes 21 NixOS VM scenarios, including benign positive cases, silent runtime defects, network negative controls and recovery under failure. The benchmark programme target remains at least 80 independent clusters; the current seed is the first tranche, not the complete benchmark.

### Files

- `schemas/benchmark-manifest-v0.1.schema.json` — strict Draft 2020-12 task-manifest schema.
- `scenarios/benchmark-seed-v0.1.json` — 41 synthetic task definitions with unique case/cluster identities, target profile, prompt, state, constraints, forbidden effects, allowed tools, observations, plan requirements, postconditions, runtime tier, evaluator class, hard gates and failure injections.

The manifest has no claimed pinned lock digests yet. Those are assigned when each executable target fixture is frozen; every actual run receipt must record the exact NixOS release, architecture, nixpkgs revision, flake.lock digest, module-set digest, Nix version and evaluator image.

## 3. External baseline: NixBench 3.2.0

The current published NixBench 3.2.0 calibration is the closest direct agent comparator:

- 21 tasks;
- five agent configurations;
- three trials per configuration;
- 315 task-trial observations;
- full NixOS systems against pinned nixpkgs;
- silent runtime failures and three VM-graded tasks.

The 2026-10-04 calibration reported GPT-6 Astra and GPT-6.1 Sol at 52/63 (82.5%), Claude Opus 5.5 at 51/63 (81.0%), Sonnet 5.5 at 47/63 (74.6%), and GPT-6 Luna at 20/63 (31.7%). These figures belong only to that exact corpus and protocol. The top configurations' uncertainty intervals overlap; repeated trials of the same task are not independent. The report also identifies three quarantined evaluator disputes and zero timeouts.

**NixWard policy:** run the public corpus unchanged as an external comparison. Never relabel our 41/80+ task corpus as NixBench, mix corpus versions or claim superiority from a raw pass-rate difference. Audit evaluator correctness and record disagreements before drawing conclusions.

Sources:
- [NixBench 3.2.0 results](https://nixbench.com/)
- [2026-10-04 NixBench 3.2 calibration](https://nixbench.com/docs/runs/2026-10-04-nixbench-3.2-calibration.html)

## 4. Industry comparison

| System | What it contributes | NixWard strategy |
|---|---|---|
| Native Nix/NixOS | Declarative module system, pinned inputs, evaluation, derivations, generations and VM tests | Treat as ground truth; do not reinvent Nix semantics |
| Nixi | Conversational administration, persistent memory, local/SSH execution, managed and read-only monitored nodes, TUI/Web UI | Compete on source identity, evidence closure, safe planning and qualified effects; borrow good operator UX |
| Colmena | Parallel multi-host evaluation/build/deployment | Reuse deployment mechanics after independent per-host planning/qualification |
| deploy-rs | Deployment profiles and post-activation probe/rollback behavior | Require actual service/security postconditions, not only host reachability |
| nixos-anywhere + Disko | Remote installation and declarative disk layout | Integrate only behind exact device identity, backup/restore and destructive-operation approval gates |

Nixi's documentation describes its own project as early development (v0.1.0); there is no matched independent benchmark result in the material used for this seed. Compare capabilities and UX, not invented head-to-head scores.

Sources:
- [Nixi overview](https://docs.nixi.sh/)
- [Nixi architecture](https://docs.nixi.sh/architecture/)
- [Nixi multi-node management](https://docs.nixi.sh/multi-node/)
- [Colmena](https://colmena.cli.rs/)
- [deploy-rs](https://github.com/serokell/deploy-rs)
- [nixos-anywhere](https://github.com/nix-community/nixos-anywhere)
- [Disko](https://github.com/nix-community/disko)

## 5. Benchmark arms

Every baseline receives the same task prompt, source fixture, target observations, tool budget and time/resource limit.

1. Native operator / documented Nix CLI workflow.
2. Strong general coding agent with matched tools.
3. Nixi where protocol/tool mapping is comparable.
4. NixWard templates and deterministic retrieval without a generative planner.
5. NixWard LLM planner without target evidence and runtime gates.
6. Full NixWard: typed goal compiler, exact target resolver, security-advisory evidence, isolated evaluation, build, VM/runtime postconditions and recovery.

Ablations are mandatory. They show whether HDC/CfC/active inference actually improve completion, clarification quality, evidence selection and cost beyond deterministic Nix resolution and tests.

## 6. Evaluation tiers

| Tier | Evidence | Passing means |
|---|---|---|
| Static contract | Independent manifest/oracle checks | The requested plan and constraints are represented correctly |
| Source resolution | Exact option/package source data | Every authoritative option/package resolves in the target universe |
| Evaluation | Pinned target flake evaluation | Configuration evaluates with recorded warnings/errors |
| Build | Required derivations build | Candidate closure is buildable |
| VM runtime | Booted NixOS VM assertions | Required system/service/network postconditions hold |
| External runtime | Independent probe from named vantage point | Declared real network/service behavior is observed |
| Hardware | Declared physical device and driver | Hardware-dependent behavior passes on that device class |
| Recovery | Failure-injected transaction and mutable data | Recovery works without duplicate/unauthorized mutations |

Do not silently downgrade a VM task to a build task because the runner cannot start KVM. Quarantine it with an explicit reason and keep its denominator visible.

## 7. Scorecards

Keep metrics independent rather than collapsing them into one “agent safety” number:

- end-to-end task success, hard-constraint satisfaction and forbidden-effect count;
- option/package resolution precision, recall and unsupported emission;
- syntax, evaluation, build, VM, external runtime and hardware success separately;
- pass@1 and pass^k by independent task cluster;
- benign completion, clarification count and user-approved plan rate;
- critical security false negatives and false positives by evidence class;
- overconfident UNKNOWN handling and unsupported "not affected/not exposed" claims;
- unauthorized protected effects (hard failure);
- target drift, sibling-host regressions and source mismatch;
- duplicate mutations, reconciliation time and recovery correctness;
- p50/p95 duration, CPU/memory/build resource, cache use, energy only if measured, and all-in cost per qualified outcome.

Use task-clustered confidence intervals or a pre-registered cluster bootstrap. Do not count three trials of one task or multiple VM assertions as independent task samples. Publish denominators, per-task failures, confidence intervals, evaluator disputes and quarantines.

## 8. Promotion targets (proposed, not achieved)

- Zero unauthorized protected effects.
- Zero critical false-safe claims without required exact and fresh evidence.
- Zero accepted expression-injection cases in the frozen adversarial panel.
- At least 99% target option/package resolution precision on the held-out resolvable set; abstention for genuinely unknown cases.
- At least 95% benign completion on the frozen in-scope panel, non-inferior to the strongest matched baseline.
- At least 90% held-out NixWard pass@1 after evaluator-validity review.
- At least 95% pass^3 on repeatable low-risk cases; hardware and recovery tiers reported separately.
- Every VM/runtime case passes all its declared postconditions.

The targets must not be achieved by weakening or removing difficult tests. A disputed evaluator is recorded, reviewed and quarantined, never silently erased.

## 9. Paradigm shifts to test

1. **Outcome rather than command:** score the desired state and side effects, not whether a shell command returned zero.
2. **Target-bound knowledge:** semantic retrieval proposes options; exact evaluated target catalogs establish whether they exist and are writable.
3. **Evidence as part of the plan:** every requirement maps to the observation or test that can prove it.
4. **Runtime rather than evaluation:** silent-failure tasks require booted VM or hardware assertions.
5. **Recovery as a first-class outcome:** NixOS generation rollback does not revert mutable application state, so the benchmark tests service/database recovery, backup restore and acknowledged writes.
6. **Unknown-outcome discipline:** a disconnected runner or SSH session is not proof of failure or success; reconcile the target before retrying.
7. **Untrusted evaluation:** evaluating an arbitrary flake must not expose controller credentials or host files. Nix build sandboxing and evaluator restrictions are distinct controls.
8. **Useful autonomy:** task completion and security are reported independently so a deny-everything system cannot win.

## 10. Next execution steps

1. Materialize pinned target flakes for the 41 seed cases.
2. Add an independent task oracle and run receipts with exact environment and corpus digests.
3. Stand up at least 21 NixOS VM scenarios; include runtime silent-failure assertions.
4. Extend to at least 80 task clusters, including held-out releases, architectures, module sets and overlays.
5. Run all six benchmark arms under frozen budgets.
6. Audit evaluator disagreement and reproduce the strongest baseline results before any SOTA claim.
7. Publish the scorecard with all-in costs, confidence intervals and per-case evidence.

Related:
- [#14233](https://github.com/Luminous-Dynamics/symthaea/issues/14233) goal compiler and security-advisory assurance.
- [#14235](https://github.com/Luminous-Dynamics/symthaea/issues/14235) target-bound option catalog.
- [#14241](https://github.com/Luminous-Dynamics/symthaea/issues/14241) typed generator integrity.
- [#14268](https://github.com/Luminous-Dynamics/symthaea/issues/14268) benchmark programme.
- [#14270](https://github.com/Luminous-Dynamics/symthaea/issues/14270) untrusted evaluator isolation.
