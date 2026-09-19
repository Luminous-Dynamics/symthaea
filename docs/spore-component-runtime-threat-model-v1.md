# Spore Component Runtime v1 — Threat Model

Status: design-only companion to `spore-component-runtime-plan-v1.md`. No runtime qualification claim.

## Assets to protect

Spore Components must not be able to compromise or manufacture:

- physical boot identity;
- recovery selection;
- Last Known Good state;
- generation promotion;
- boot-attempt accounting;
- trusted qualification evidence;
- Nixward execution permits;
- Xenia/owner/organization authority;
- secrets or credentials not explicitly granted;
- arbitrary host filesystem/network/device access;
- availability of the native Spore/recovery core.

## Adversaries

Assume components may be:

- accidentally buggy;
- malicious by design;
- supply-chain substituted;
- built with a compromised compiler;
- authored by an untrusted third party;
- induced by Symthaea or another agent from untrusted inputs;
- deliberately crafted to exhaust CPU, memory, host calls, output buffers, or host-side resources;
- crafted to exploit runtime/host vulnerabilities;
- crafted to confuse capability attribution or component identity;
- crafted to abuse one legitimate capability to obtain a stronger unintended effect.

Publisher identity is provenance, not evidence of benign behavior.

## Trust boundaries

```text
untrusted component bytes
        |
        v
validation / exact identity
        |
        v
WIT/world compatibility
        |
        v
manifest request parsing
        |
        v
admission policy + current grant
        |
        v
opaque EffectiveComponentCapability
        |
        v
Wasmtime Component instance
        |
        v
host capability adapters
        |
        +----> read/observe/compute
        |
        `----> proposal only

---------------- HARD AUTHORITY BOUNDARY ----------------

Nixward privileged execution
Xenia / owner authority
recovery / LKG / promotion
```

## Threat: ambient authority

### Risk

A generic CLI/POSIX-shaped environment can accidentally expose filesystem, environment, sockets, clocks, entropy, stdio, or process-like behavior beyond what a tool requires.

### Required control

Default Component world exposes no ambient WASI. Add only exact custom WIT interfaces and explicitly selected standard WASI interfaces.

`wasi:cli/command` is not the default Spore plugin world.

## Threat: confused deputy

### Risk

A component with one broad host capability may use the host as a deputy to access resources beyond the user's intended grant.

### Required control

Prefer opaque host resource handles and policy-bound endpoint IDs over raw paths/URLs/IPs.

Examples:

```text
project-handle-7 + relative/path
weather-provider + typed request
```

instead of:

```text
/home/user/project
0.0.0.0/0
arbitrary URL
```

Host must revalidate every host call against the exact effective capability, not only at component startup.

## Threat: resource exhaustion

### Risk

Guest consumes unbounded CPU, memory, tables, instances, I/O, host calls or output capacity.

### Required control

Per-invocation limits:

- deterministic fuel budget;
- memory/table/instance/memory-count limits;
- host-call count;
- input/output byte ceilings;
- wall timeout as defense-in-depth;
- future async concurrency/stream quotas.

A trap/limit failure must terminate the invocation without poisoning the host or authoritative Spore state.

## Threat: native artifact substitution

### Risk

Wasmtime serialized native code is mistaken for portable/publisher-authored component identity or loaded through unsafe deserialization without sufficient provenance.

### Required control

Portable identity is the canonical component + WIT + manifest subject. Native compiled artifacts are disposable caches.

If native serialized caches are used, their identity must additionally bind runtime/engine configuration and target, and unsafe deserialization must never receive untrusted bytes lacking the runtime's exact precompile provenance requirements.

## Threat: component identity confusion

### Risk

Same name/tag/version is treated as same component after bytes or contract change.

### Required control

Use independent exact digests for:

- component binary;
- WIT contract;
- manifest;
- composite subject.

Registry tags/locations and semantic versions are discovery metadata only.

## Threat: manifest self-authorization

### Risk

A component requests a capability and that request is treated as authority.

### Required control

```text
request != grant != effective capability
```

Effective capabilities are non-Serde/opaque runtime objects produced only after policy/currentness evaluation.

## Threat: interface smuggling

### Risk

Component imports a host interface not represented by the reviewed manifest, or manifest requests authority absent from the exact component world.

### Required control

Admission cross-checks exact Component Model imports against the manifest request set and host policy. Unexpected/missing imports fail before instantiation.

## Threat: authority laundering through proposals

### Risk

A component output is interpreted as authorization to change the system.

### Required control

All mutation outputs are typed proposals only.

```text
component output
    != execution permit
    != qualified candidate
    != owner approval
```

Nixward/Xenia/Spore independently validate and authorize consequential changes.

## Threat: composition amplification

### Risk

Component A with capability X dynamically loads component B with Y and effectively gains X+Y.

### Required control

No dynamic authority-expanding composition in v1. Statically composed artifacts are admitted as new immutable subjects with their full imports/manifests re-evaluated.

## Threat: recovery coupling

### Risk

Component runtime crash, runtime upgrade, component failure, or registry outage prevents boot/recovery or corrupts machine-health evidence.

### Required control

Spore recovery core has no dependency on Wasmtime/WASI/Component execution. Component runtime is optional/expendable userspace.

Component state may contribute presentation/tool evidence only under explicit verifier policy; it cannot mint local machine-health truth.

## Threat: runtime sandbox vulnerabilities

### Risk

Wasmtime/WASI contains implementation defects.

### Required control

Defense in depth:

- minimize imports even if runtime sandbox is correct;
- keep native host process unprivileged where feasible;
- systemd/Linux sandbox the component host where feasible;
- separate privileged execution into another boundary;
- pin exact runtime through Nix/Cargo lock;
- scan advisories and qualify runtime upgrades;
- maintain a hostile-component corpus across upgrades.

## Threat: nondeterminism hidden as deterministic computation

### Risk

Clock/random/network access changes behavior but the component is qualified as reproducible/pure.

### Required control

Manifest declares a determinism profile and admission verifies imports are compatible with it.

Pure profiles cannot import clock/random/network/environment interfaces.

## Threat: large host-returned objects

### Risk

Even bounded guest memory is defeated by oversized host responses or repeated host resource creation.

### Required control

Host-side response limits and resource-handle quotas are independent of guest linear-memory limits.

## Threat: secret exfiltration

### Risk

A component legitimately gains network output and finds a way to read unrelated host secret state.

### Required control

No ambient filesystem/env access. Secrets are separate typed host resources with minimum scope, and a component only receives a secret capability if its exact world/manifest/policy explicitly permits it. Initial Spore component profiles expose no secret interface.

## Qualification philosophy

For each capability profile, test both positive behavior and explicit negative controls.

A green functional test is insufficient if the component was never challenged to cross its boundary.

Initial hostile corpus should include:

- infinite loop;
- memory growth bomb;
- table/instance growth;
- malformed component;
- unexpected import;
- forbidden filesystem/network/environment import;
- oversized request/response;
- repeated host-call exhaustion;
- identity/manifest substitution;
- WIT-version substitution;
- grant narrowing/revocation after admission;
- component trap during host call;
- component runtime termination while native recovery remains unaffected.

## Nonclaim

Passing this threat-model corpus will establish only the exact Component Runtime profile under test. It does not prove arbitrary third-party components safe, nor does it grant recovery/deployment authority to the runtime.
