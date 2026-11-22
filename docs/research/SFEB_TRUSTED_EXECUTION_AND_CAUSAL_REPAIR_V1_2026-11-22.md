# SFEB v1.1 — trusted execution, causal repair and frontier evaluation

**Date:** 2026-11-22  
**Status:** architecture and schema proposal; executor and measurements pending  
**Parent programme:** PR #13162, Evidence-gated frontend engineering

## Executive thesis

SFEB should not compete as another screenshot-to-code score. It should become a benchmark and engineering substrate for **evidence-qualified frontend change**: every claimed improvement is bound to an exact intervention, reproducible browser observations, independently derived oracles and preserved requirements.

The next step is not to add more agent prompts. It is to make the evidence chain resistant to self-report, replay, environment drift, hidden network dependencies and benchmark gaming.

## 1. The three-plane architecture

Separate authority, execution and verification.

### Control plane

Owns immutable campaign/task revisions, source and fixture identities, environment manifests, resource budgets, run nonces, network policy and allowed workflow identities. It creates paired baseline/candidate jobs but cannot modify their evidence after execution.

### Execution plane

Runs the existing `symthaea-browser` CDP capture substrate inside a disposable isolated worker. It emits an append-only, run-local event trace and immutable artifacts. It does not decide whether the task passes and cannot mint its own trust root.

### Verification plane

Runs separately from the worker. It validates signed run statements, source/fixture/environment identity, trace continuity and artifact bytes; then replays the declared oracle rules from raw observations. The verifier, not the agent or worker, derives oracle outcomes and accepted/rejected disposition.

This avoids making the agent, browser worker and evaluator one combined trust domain.

## 2. Cryptographic evidence model

Use the in-toto Statement v1 envelope with an SFEB-specific predicate type. Sign the statement through a workload identity and verify its Sigstore bundle against an explicit out-of-band trust policy. Use SLSA provenance separately for the runner/controller/evaluator build identities.

The run predicate binds:
- campaign, report, case, task and baseline/candidate role;
- a unique run ID and one-time challenge nonce;
- source commit, fixture digest and environment-manifest digest;
- runner revision, executor image, browser binary and controller binary digests;
- isolation-profile and network-policy digests;
- trace digest, event count and terminal event hash;
- screenshot, accessibility, layout and relevant network/console artifacts.

Do **not** put runner-authored oracle IDs or pass/fail disposition in the trusted predicate. The signed statement attests which immutable evidence bundle the workflow produced. The independent evaluator derives the verdict.

The workflow identity policy must pin the OIDC issuer, exact repository, workflow path, immutable workflow commit and permitted ref. A JSON field saying `repository=Luminous-Dynamics/symthaea` is not authentication. Verify certificate identity and claims from the signed bundle. Reject wildcard identity matching and untrusted fork workflows.

### Hash-chain contract

Trace events are serialized using RFC 8785 JSON Canonicalization Scheme (JCS). Each event hash covers the complete canonical event except its own `event_sha256` field. The first event references an all-zero predecessor; later events reference the preceding event hash. The signed statement binds the final trace digest and event count.

Hash chains detect modification, reorder and truncation relative to the signed root. They do not establish that the observer was honest or complete. The independent execution platform, sandbox boundary and separately controlled observer remain part of the trusted computing base.

### Isolation is a hard gate

The worker must run in a disposable microVM or a reviewed equivalent, non-root, with a read-only fixture/source mount, bounded CPU/memory/disk/time, no host network, no home mount, no Docker socket and no host browser debugging endpoint. Network is default-deny; only explicit fixture-local services are reachable. All attempted egress is recorded and hashed.

CDP must remain inside the sandbox. Use a private pipe or isolated controller channel; never bind the debugging port on a host interface. A container alone is not automatically an adequate boundary: document the actual kernel, namespace, capability, mount and egress controls.

## 3. Independent oracle derivation

The current report checker accepts an asserted oracle ID when checking a claimed kill. The next verifier must instead derive each oracle from trace and observation data:

`raw trace + immutable artifacts + frozen oracle definitions -> derived oracle outcomes`

The report is a result index, not an authority. Its oracle IDs are checked against verifier output from the same trace digest and evaluator revision. A worker's self-reported disposition is diagnostic only.

Required negative controls include:
- valid artifact bytes paired with a false oracle ID;
- a receipt copied from another task, fixture or source commit;
- omitted final trace events and reordered events;
- duplicate event sequence numbers and cross-run event replay;
- a correctly signed statement from the wrong repository/workflow/ref;
- a valid statement whose subject digest does not match the report or bundle;
- unapproved network egress and a browser endpoint exposed outside the sandbox.

A positive control must also test legitimate, semantics-preserving edits and equivalent run traces so the verifier does not merely reject everything.

## 4. Causal repair as a first-class benchmark capability

A repair agent should produce an evidence graph rather than a plausible paragraph:

`requirement -> contract -> failed oracle -> browser event -> semantic region -> source anchor -> root-cause hypothesis -> patch -> post-repair counterfactual -> invariants`

For each repair, record ranked hypotheses, supporting event hashes, semantic region/source anchor, calibrated confidence, patch digest, before/after execution statement digests, a placebo run and invariant results.

A causal claim requires a reproduced pre-repair failure, an intervention that changes the target outcome under the same pinned fixture/environment, and evidence that unrelated invariants remain intact. The placebo must not produce the same improvement. Replicated support requires a second independent intervention/run block or randomized holdout, not merely a second screenshot.

Measure:
- defect localization top-1/top-3 and MRR;
- target defect repair rate and time-to-repair;
- regression rate across unrelated assertions;
- placebo false-credit rate;
- patch size and repair attempts;
- confidence calibration (Brier score and ECE);
- cost per accepted repair.

This makes causal diagnosis, rather than just edit generation, a measurable capability.

## 5. Responsive layout as a constraint field

Five fixed viewports are useful smoke points, but they invite breakpoint overfitting. Evaluate the interval, not only the points.

For each responsive contract:
1. preserve the named reference widths;
2. sweep a deterministic coarse grid over the supported width range;
3. detect geometry/visibility violations and adaptively sample around the first failing interval or CSS transition;
4. add frozen-seed randomized widths as a holdout;
5. test zoom/device-scale-factor and reduced-motion states separately.

Report the width interval of each failure, not simply "mobile passed". Maintain semantic invariants (no required content lost, no hidden action without equivalent access, no overlap/clipping, no unexpected horizontal overflow) alongside region-aware visual quality.

## 6. Accessibility as journey capability

Static accessibility-tree presence is insufficient. Require actual keyboard journey completion, focus order/visibility, accessible-name and label integrity, state announcements, no keyboard traps and role/permission behavior. Treat WCAG 2.2 AA as the baseline conformance target; separately report high-bar AAA criteria rather than silently making them mandatory. Use independent assistive-technology review on a stratified subset and all disputed critical results.

W3C's WCAG 2.2 additions make focus appearance, focus not obscured and minimum target sizing useful explicit test dimensions. Automated checks should be paired with keyboard and human review, not treated as proof of usability.

## 7. Industry comparison without score laundering

As of 2026-11-22:

- **Vision2Web** is the closest broad website-development comparator: 193 tasks, 16 categories, 918 prototype images and 1,255 test cases across static, interactive multi-page and full-stack levels. Its verification combines workflow-driven GUI testing and VLM judging. The public board displays historical 2026 runs and a season selector; distinguish published historical results from a current-season SOTA result.
- **Design2Code** remains a visual reconstruction comparison over 484 real-world webpages. It is a useful isolated visual lane, not an end-to-end functional/security/maintenance substitute.
- **WebdriverIO Browser Agent Benchmark** fixes model, harness and prompt while swapping browser tools. Its page, updated October 9, 2026, lists seven tools, 100 live-web tasks, 2,511 agent runs and an observed leading success figure of 62% (103/165 on the displayed row), alongside cost/time/token data. The benchmark itself warns that overlapping 95% intervals do not establish a reliable ranking. It measures browser-tool choice, not website code-generation quality.
- **SWE-Bench Pro audit**: OpenAI reported 200 of 731 public tasks flagged by an automated pipeline and 249 identified in human annotation, estimating roughly 30% task-quality problems. The defects included over-strict tests, underspecified prompts, low-coverage tests and misleading prompts. This supports an oracle-qualification layer; it does not imply the same defect rate in SFEB.

Never pool raw results from these benchmarks. Reproduce Vision2Web, Design2Code and the browser-agent methodology independently where authorized. Report model, tools, versions, prompts, task split, evaluator, environment, cost and confidence intervals for each.

## 8. SFEB benchmark scorecard

Do not reduce the benchmark to a single weighted score before hard gates pass.

### Qualification hard gates

- 100% rejection of critical, security and evidence-integrity mutants;
- at least 95% high-severity and 90% overall mutation kill rates;
- zero positive-control false rejections;
- all metamorphic relations hold;
- zero artifact/receipt/trace/identity binding failures;
- zero unapproved external egress;
- no accepted run with missing or inconclusive evidence.

These are acceptance targets, not achieved results. Include numerators, denominators and Wilson intervals. A 30-case qualification set is an engineering gate, not a precise estimate of population reliability.

### Agent outcomes

Report the existing A0-A4 matched ablations separately by task level:
- requirement/contract correctness;
- visual region quality on desktop/tablet/mobile and random widths;
- journey completion and state-machine correctness;
- keyboard/accessibility journey completion;
- security and network-boundary behavior;
- defect localization and repair efficacy;
- unrelated regression rate;
- reproducibility across repeated trials;
- time, model/tool/runtime/human cost per accepted task.

Hard gates remain independent of weighted quality. Use a Pareto frontier for quality vs cost vs latency vs repair risk. An agent should not compensate for a critical accessibility or security failure with a better screenshot.

### SOTA decision

Keep the existing pre-registered rule: either a positive paired 95% interval for A4-A0 with all hard gates passing, or non-inferiority within two percentage points plus at least 30% lower cost per accepted task. Add a separately reported causal-repair result once that lane has enough seeded/hidden tasks and qualified oracles.

## 9. Work sequence

1. **This change:** publish execution-predicate, trace-event and causal-repair schemas plus trust/causal protocol.
2. Wire RFC 8785 hashing, event-chain verification and schema validation into the standalone Rust checker; add adversarial tests.
3. Implement in-toto/Sigstore verification with an exact workload identity policy and negative identity tests.
4. Build the isolated Chromium executor around the current CDP substrate; capture trace and artifacts.
5. Build independent trace-to-oracle replay; remove trust in worker-reported oracle IDs.
6. Execute the 30-case oracle campaign and report full denominators/intervals.
7. Run the 30 public tasks, then 150 matched tasks with controller-only and hidden partitions; expand to 500+ for publication.
8. Submit compatible official external evaluations separately.

## 10. Evidence status

This document and its schemas are design work. They have not been compiled, integrated into the Rust checker, cryptographically exercised or run in Chromium. No executor attestation, trusted-runner claim, causal repair result, SFEB score or SOTA superiority is claimed.

## References

- in-toto Attestation Framework v1: https://github.com/in-toto/attestation/tree/main/spec/v1
- SLSA provenance v1.2: https://slsa.dev/spec/v1.2/build-provenance
- Sigstore Cosign: https://docs.sigstore.dev/quickstart/quickstart-cosign/
- RFC 8785 JSON Canonicalization Scheme: https://www.rfc-editor.org/rfc/rfc8785
- Vision2Web: https://proceedings.mlr.press/v306/he26n.html and https://vision2web-bench.github.io/
- Design2Code: https://aclanthology.org/2025.naacl-long.199/
- WebdriverIO Browser Agent Benchmark: https://benchmark.webdriver.io/
- OpenAI, "Separating signal from noise in coding evaluations" (2026-07-08): https://openai.com/index/separating-signal-from-noise-coding-evaluations/
- WCAG 2.2: https://www.w3.org/TR/wcag/
