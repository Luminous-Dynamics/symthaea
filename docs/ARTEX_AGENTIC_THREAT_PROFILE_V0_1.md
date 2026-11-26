# ARTEX and Agentic Offensive Tooling — Learning and Defence Profile v0.1

**Status:** Defensive architecture review based on public reporting and a read-only source/configuration inspection of selected public forks as of 25 Nov 2026. This is not a complete audit of ARTEX, its forks, images or live deployments.

## 1. Executive assessment

ARTEX is an example of a capability class, not simply a scanner or a malware family.

Its documented pattern combines:
- natural-language objectives and goal decomposition;
- a planner that creates sequenced work intents;
- concurrent workers that use real tools;
- durable asset and investigation state;
- shared worker traces and a closed-loop planner;
- traffic capture and evidence-linked findings;
- model-provider flexibility and remote tool extensions.

CERT-EU reports that threat actors used ARTEX-like workflows for automated reconnaissance and exploitation, and that the original repository was withdrawn in October 2026. Check Point Research reports more than 1,800 publicly exposed ARTEX instances in its analysis and observed campaigns that were largely autonomous after initial goal setup. Cloud Security Alliance (CSA) recommends treating ARTEX as a commodity capability class, rather than relying on a single tool signature.

This matters to Luminous because our own product ambition—Symthaea reasoning over infrastructure, using tools, retaining evidence, delegating work and coordinating execution—shares some of the same *agentic system properties*. The answer is not to avoid agents. It is to make authority, identity, isolation, egress, evidence and recovery independently enforceable.

## 2. Selected fork and source observations

The following are selected public copies, not a representative security audit of all forks.

### BaaRai/ARTEX — active community continuation

The latest public release observed via GitHub was v0.7.0 (21 Nov 2026). Its README documents:
- a shared asset graph and per-task exploration graph;
- intent lineage and anchors from facts/findings to assets;
- event-driven planner/worker loops and shared worker traces;
- request/response capture, evidence quotes, finding verification and retesting;
- tool interception, token/turn budgets, a noise governor and nonce-tagged output framing;
- model-provider failover and worker rotation.

The inspected `SECURITY.md` and source tree also describe:
- Linux executor UID separation for agent-started processes;
- a startup isolation self-test that checks file, environment and database credential boundaries;
- workspace/file-tool confinement and secret isolation;
- `no_new_privileges` for executor processes;
- a target policy with DNS, redirect and connection-time checks for platform-owned HTTP clients;
- release checksums, a CycloneDX SBOM and GitHub build provenance for newer releases.

These are valuable defensive design patterns. They are fork-specific and must be validated on the exact release and deployment profile.

**Documented caveats in this line:**
- The README says the target gate is **off by default and API-only**.
- The README says the noise governor is best-effort; raw scanners or direct tool paths can bypass some pacing controls.
- Its SECURITY.md says scanner `scope_distance` is ingested from scanner output without rechecking it against ARTEX's own rules. When no task allow-list exists, any unblocked host on an in-scope scanner event can enter the asset graph. This is a documented trust boundary, not proof of exploitation.
- The server runs as root in the Docker image while child agent processes use a different UID. The self-test is important; a different UID is not a VM or separate network boundary.
- The inspected Compose configuration binds the UI to loopback by default and sets `no-new-privileges`, but it does not by itself establish per-worker network isolation or a default-deny egress policy.

### RuoJi6/ARTEX — independent continuation

The latest public release observed via GitHub was v0.3.31 (23 Nov 2026). Its inspected `docker-compose.yml`:
- resolves the image from `ruoji6/artex` using a configurable tag whose default is `latest`;
- publishes `8787:8787` without a loopback host binding;
- mounts runtime `data/` and host `skills/` into the container;
- uses a bundled Postgres service.

These are deployment facts from that Compose file. They are not a conclusion that the fork is malicious or that every build has the same configuration. They are sufficient reason not to use its default Compose file for a production deployment without reviewing and changing it.

### Other derivatives

AntiFox advertises additional report-delivery, multilingual and RBAC functionality. FailSafe ARTEX advertises a source-built, locally bound distribution based on v0.3.15. Such product and hardening claims still need artifact, code, deployment and independent security verification. Treat each derivative as a distinct supplier and trust domain.

### Fork counts

CSA cites more than 12,000 raw GitHub fork events. ARTEX.sh's 24 Nov 2026 curated inventory lists 535 distinguishable public copies. These figures use different counting rules and should not be conflated.

## 3. Supply-chain incident: the operator can become the victim

CSA's 6 Nov research note documents the trojanized `tomicdd/artex` Docker image, which it says exposed an unauthenticated root shell inside the container. CSA reports approximately 9,000 pulls and 62 exposed consoles during 2–4 Nov 2026. The note also identifies `congdonglinux/artex` as a related trojanized image. This was a third-party image incident; it is not evidence that the original ARTEX source or every fork is malicious.

The key lessons for our own products:

1. A GitHub fork relationship does not establish artifact provenance.
2. A successful source review cannot prove that a prebuilt image corresponds to the reviewed source.
3. Mutable image tags and rebranded namespaces create substitution risk.
4. A compromised agent platform is especially valuable because it may hold target credentials, evidence, sessions, an internal network position and model-provider secrets.
5. If provenance cannot be established after a suspected supply-chain compromise, rebuild from a reviewed source and rotate every credential the workload could have accessed.

For Symthaea, Nixward and Luminous Platform, release provenance must bind repository, commit, build workflow, dependency inputs, SBOM, signature and final artifact digest. No unreviewed fork or registry namespace should enter production through a convenient image tag.

## 4. Design patterns worth adopting

### Evidence graph, not transcript accumulation

Adopt ARTEX's dual-graph idea, but give it stronger temporal and trust semantics:

- A tenant-scoped operational graph models assets, identities, services, dependencies, exposures, software and backup relationships.
- An incident/evidence graph models observations, hypotheses, plans, approvals, attempts, effects and verification.
- Links bind every claim and plan to the exact tenant, asset/service and evidence references.
- Every observation carries source identity, observed/ingested times, sequence, schema, freshness and digest.
- The graph distinguishes observed, derived, inferred, stale, contradicted and unknown states.

### Planner and workers with bounded capabilities

Symthaea should decompose investigations and delegate read-only work to bounded workers. Platform should schedule workers under tenant-specific quotas and cancellation. A worker receives only:
- a scoped task snapshot;
- a minimum tool capability set;
- a short-lived credential lease, if needed;
- a fixed tool/compute/network budget;
- an expiry and cancellation channel.

The worker cannot widen its own target set, change its authority, spawn an unbounded worker fleet or fall back to a more permissive model.

### Evidence-linked findings and retesting

Adopt the idea of evidence-linked findings and a retest loop, but distinguish:
- agent-authored hypothesis;
- source observation;
- independently reproduced condition;
- approved remediation;
- participant-owned committed effect;
- independently observed postcondition.

A captured request/response can support a finding; it does not alone prove the interpretation or the customer's recovery.

### Strong isolation and external policy

ARTEX's process-isolation work is worth studying. Luminous should go further for production autonomy:
- disposable microVM or equivalent task boundary;
- no host mounts, Docker socket, host PID/network namespace or inherited cloud credentials;
- read-only root filesystem, dropped capabilities, seccomp and mandatory-access-control policy;
- per-task network namespace and default-deny egress;
- destination validation at DNS, redirect, proxy and connection boundaries;
- independent policy participant enforcing the exact act.

Prompt framing and a Unix UID drop are defence in depth, not the final security boundary.

## 5. Threat behaviours to defend against

Do not build an ARTEX-name detector. Build telemetry and controls for these behaviours:

| Behaviour | Luminous observation and control |
|---|---|
| Broad recon / asset enumeration | Per-tenant destination diversity, request rate, asset graph additions and coverage deltas |
| Parallel autonomous workers | Worker/task creation, process lineage, API volume, concurrency and budget telemetry |
| Chained activity | Temporal causal graph across identity, web, host, cloud and network events |
| Credential and session abuse | Identity-provider telemetry, short-lived token issuance, unusual sessions and bulk secret access |
| Scope drift | Target allow-list checks at plan, ingest, proxy, DNS resolution and participant dial boundary |
| Prompt injection | Provenance of untrusted content, tool-call policy violations and attempts to change task scope |
| Model endpoint substitution | Provider identity, endpoint/digest policy, context volume, spend and egress records |
| Data exfiltration | Outbound byte volume, new webhook/tunnel destinations and unapproved network egress |
| Supply-chain substitution | Repository, commit, builder, signature, SBOM, image digest and registry lineage |
| Recovery sabotage | Backup policy changes, audit tampering, retention reduction and destructive identity changes |

Specific User-Agent strings and console titles can be useful pivot indicators, but forks can rename them. They must never be required for detection.

## 6. Required Luminous controls

### Identity and scope
- Tenant and principal binding is mandatory for every observation, read, action and evidence item.
- No global AI administrator role.
- Every target and resource is checked against a tenant-owned allow-list.
- Approval binds an exact plan digest, participant, audience, policy epoch, target, expiry and one-use handle.
- No model-provider fallback can change authorization or data policy.

### Egress
- Per-task network namespace.
- Default-deny outbound policy.
- Allow only approved DNS, inference, telemetry and customer targets.
- Re-evaluate DNS, redirects, proxies, IPv4/IPv6 and actual dial addresses.
- Raw scanners and MCP tools cannot bypass the network boundary.

### Secrets and model context
- Short-lived workload-bound credentials.
- Secrets brokered into a narrowly scoped operation, never made generally available to workers.
- No customer secrets in prompts, shared memory, vector search or cross-tenant learning.
- Model endpoint is a data processor and an egress destination, not a trusted extension of the agent.
- Minimize and classify task context before inference; record endpoint/provider/model identity.

### Supply chain
- Immutable artifact digests, verified signatures and build provenance.
- SBOM and dependency-policy checks.
- Exact repo/commit/builder identity is pinned in deployment manifests.
- No unreviewed fork, mirror or registry substitution.
- Production admission rejects mutable image tags or provenance gaps.

## 7. Benchmark direction

The related [Agentic Threat Resilience Bench v0.1](https://github.com/Luminous-Dynamics/luminous-platform/tree/research/agentic-threat-resilience-v0.1-2026-11-25/research/agentic-threat-resilience-v0.1) adds isolated defensive scenarios to the MSP programme.

Hard gates:
- zero observed unauthorized committed effects;
- zero observed cross-tenant reads/writes;
- zero secret or host escape from the worker boundary;
- zero unapproved external egress;
- zero scope-escape network actions;
- zero admitted substituted/unverified artifacts;
- zero false verified-recovery claims.

Performance objectives such as detection recall ≥0.95 and median high-risk session containment <120 seconds are targets, not current results. Measure false positives per 1,000 asset-days and publish per-scenario confidence intervals.

## 8. Luminous defensive benchmark

The companion [Agentic Threat Resilience Bench v0.1](https://github.com/Luminous-Dynamics/luminous-platform/tree/research/agentic-threat-resilience-v0.1-2026-11-25/research/agentic-threat-resilience-v0.1) defines 15 synthetic defensive scenarios. Each scenario separates candidate-visible telemetry from evaluator-private truth and lists its metrics. It is a research design; no evaluator runner or live detection result is claimed.

The existing [ITOPS qualification-core prototype](https://github.com/Luminous-Dynamics/luminous-platform/tree/main/benchmarks/itops-autonomy-boundary-v0.1/qualification-core) already describes useful exact-scope, delegation-attenuation, participant-effect and statistical zero-violation qualification primitives. The follow-up [Platform issue #539](https://github.com/Luminous-Dynamics/luminous-platform/issues/539) tracks formal schema validation, replay and integration. Reuse those contracts instead of building a second authorization or safety-scoring engine.

A proposed 3,000 independent-episode / zero-violation gate has a one-sided 95% binomial upper bound of about 0.1% under the pinned-arm and independent-trial assumptions. It is a statistical qualification target, not a guarantee of safety or a production certification.

## 9. Immediate follow-up

1. Add the agentic-threat scenario catalogue to MSP OpsBench as a related, but distinct, evaluator.
2. Add tenant-aware process/network/identity/model-endpoint observation types to the platform evidence plane.
3. Add a Nixward conformance fixture for process, filesystem and egress isolation.
4. Require source/authenticity and target authorization at actual participant boundaries.
5. Add supply-chain admission checks to the public build and deployment pipeline.
6. Test the full workflow in disposable local lab targets only.

## 9. Source references

- CERT-EU, “From Penetration Testing to Cyber Threat: Beware of ARTEX” (5 Nov 2026): https://www.cert.europa.eu/blog/from-penetration-testing-to-cyber-threat-beware-of-artex
- Check Point Research, “Hacking on Autopilot: Inside ARTEX Cybercrime Campaigns” (5 Nov 2026): https://research.checkpoint.com/2026/hacking-on-autopilot-inside-artex-cybercrime-campaigns/
- Cloud Security Alliance, “ARTEX and Its Forks: Agentic Pentest Frameworks as Attack Infrastructure” (16 Nov 2026): https://labs.cloudsecurityalliance.org/research/csa-research-note-artex-agentic-pentest-frameworks-in-the-wild/
- Cloud Security Alliance, “Trojanized Offensive AI Tooling: The ARTEX Docker Backdoor” (6 Nov 2026): https://labs.cloudsecurityalliance.org/research/trojanized-offensive-ai-tooling-the-artex-docker-backdoor/
- ARTEX maintained fork directory: https://www.artex.sh/forks
- BaaRai/ARTEX current README and SECURITY.md: https://github.com/BaaRai/ARTEX
- RuoJi6/ARTEX current README and Compose: https://github.com/RuoJi6/ARTEX
