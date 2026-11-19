# Symthaea Browser network security boundary

Status: architecture/security clarification. This document does not add a Chromium network interceptor, DNS proxy, cgroup policy or privileged firewall writer.

## 1. The three separate authorities

Symthaea Browser has a valuable application safety layer, but it is not the browser's network sandbox.

1. **Navigation intent / action authority.** `BrowserCapability::Navigate`, explicit grants, user interaction and Phi-gated actions answer whether the agent may request a navigation or interaction.
2. **URL preflight.** `BrowserSafetyPolicy::evaluate_url` checks URL parsing, scheme, credentials, hostname policy and IP-literal/local-network policy before CDP dispatch.
3. **Network enforcement.** Chromium performs DNS, sockets, TLS, redirects, subresource loading, service-worker fetches and other networking inside the browser process. That traffic needs an operating-system/workload boundary independent of the URL preflight.

A successful `PolicyDecision::Allow` proves only that the preflight policy accepted the supplied URL/action. It does not prove that Chromium connected to an approved IP, that every redirect/subresource was authorized, or that all browser network protocols were constrained.

Phi is caution evidence, not network authority. A browser action capability is not an egress firewall rule.

## 2. Why Prism's Reqwest client is not the browser boundary

Prism's `prism-egress` is a server-side HTTP(S) transport. It is appropriate for Symthaea/Prism services that explicitly submit a request through that client. It does not automatically mediate Chromium's independent network stack.

Routing only the top-level `Navigate` URL through a Reqwest client would not constrain Chromium's:
- redirect follow-up requests;
- images, scripts, stylesheets, frames and other subresources;
- workers and service-worker fetches;
- WebSocket and WebTransport sessions;
- WebRTC/STUN/TURN and UDP traffic;
- QUIC/HTTP/3 and browser-managed connection reuse;
- DNS prefetch, preconnect and browser-managed resolver behavior;
- downloads, extension/network activity or native browser helpers.

CDP `Network` events can improve observation and policy feedback, but an event observer is not a kernel enforcement boundary. It can miss or mischaracterize flows outside its instrumentation.

Do not claim that `BrowserSafetyPolicy` or `BrowserExecutor` provides DNS-rebinding protection by itself. The code comment in `safety.rs` already correctly says that a production network boundary must pin DNS and re-check redirects; this document makes the workload-level requirement explicit.

## 3. Recommended deployment boundary

Run each browser session or isolation cohort as a distinct Nixward workload identity, preferably an unprivileged guest or dedicated systemd service with its own cgroup v2 binding.

The target policy should:
- deny egress by default for the browser workload;
- allow DNS only to a trusted resolver/proxy, not arbitrary UDP/TCP 53;
- block unapproved DoH/alternate-resolver paths and direct IP-literal escapes;
- learn/authorize DNS answer sets through an authenticated resolver covenant;
- apply exact IPv4 and IPv6 address/port policy to the browser cgroup;
- include TCP and, only when explicitly required, UDP/QUIC/WebRTC allowances;
- bind rules to the exact browser workload, policy generation and short validity period;
- deny stale, replayed or unbound rules and preserve the host firewall baseline;
- observe metadata only by default, never browser payloads/cookies/credentials.

Nixward's current contracts are the intended foundation:
- `network_covenant.rs`: schema-v2 workload identities, owner-authorized capabilities, and explicit address scopes for hostname destinations (public-unicast-only or exact approved CIDRs);
- `network_enforcement.rs`: cgroup-BPF rule compilation and target proof;
- `network_runtime_lab.rs`: isolated runtime acceptance;
- `network_telemetry.rs`: privacy-preserving metadata evidence.

The current Nixward network code explicitly leaves hostname rules unresolved until a DNS/resolution covenant exists. The proposed bridge is tracked in [Nixward PR #162](https://github.com/Luminous-Dynamics/nixward/pull/162). Its parent capability schema now binds an explicit `AddressScopePolicy`, and the resolution-lease model checks both the parent scope digest and every resolved address. This closes the policy-representation gap, not the runtime broker gap: the PR still does not implement a trusted resolver, signed lease issuer, BPF map writer or automatic activation. The proposal is initially scoped to Prism's exact HTTP hop. Browser integration needs a resolver-proxy path that supplies Chromium's actual DNS answers to the same enforcement authority; a separate Nixward resolution that Chromium never consumes would not establish DNS/socket consistency.

## 4. LAN access is a separate, explicit profile

The default browser policy correctly denies private/local destinations. Keep that default.

Legitimate local navigation (router admin, local development server, NAS, home automation) should use a distinct profile with:
- explicit user navigation intent or gesture;
- a destination scope limited to the required LAN CIDRs and ports;
- a separate workload/session identity or narrowly scoped temporary policy;
- the same DNS answer-set validation and rebinding protection;
- short expiry, revocation and audit;
- no inheritance by public-web fetch or background content.

Do not set `allow_private_networks = true` globally to restore local browsing. A blanket private-network exception turns any compromised or hostile page into a possible pivot through the browser's host network.

## 5. Threat model and test corpus

The isolated browser-network test environment should expose local fake services and deterministic DNS responses. No production network should be modified by these tests.

| ID | Scenario | Required outcome |
|---|---|---|
| BROW-EGR-001 | Top-level URL points to loopback/private IP literal | Preflight denies under the default profile |
| BROW-EGR-002 | Public hostname resolves to private IPv4 or ULA IPv6 | Workload policy denies; no socket reaches the private listener |
| BROW-EGR-003 | DNS answer changes between preflight and Chromium connect | Resolver/enforcement path binds actual Chromium DNS answer and denies unapproved target |
| BROW-EGR-004 | Public page redirects to localhost/metadata/private host | Destination is denied at workload boundary; CDP observation is supplementary |
| BROW-EGR-005 | Public page embeds private IP image/script/iframe | Subresource socket is denied even though no new top-level Navigate action occurs |
| BROW-EGR-006 | Service worker initiates fetch after navigation | Same workload policy applies; no inherited broad authority |
| BROW-EGR-007 | Page opens WebSocket/WebTransport to unapproved host | TCP/UDP destination is denied by network policy |
| BROW-EGR-008 | Page uses WebRTC/STUN/UDP or direct IP | UDP path is denied unless separately authorized |
| BROW-EGR-009 | Browser attempts DoH or a custom resolver | Only configured resolver/proxy is reachable |
| BROW-EGR-010 | User explicitly opens approved LAN host | Temporary LAN profile permits only approved CIDRs/ports and expires |
| BROW-EGR-011 | Browser process moves cgroups or a second process shares the cgroup | Identity mismatch/rebinding fails closed; policy does not silently follow |
| BROW-EGR-012 | Nixward broker, resolver proxy or BPF attach restarts/fails | No fail-open route; reconcile unknown map state before any new grant |

Every case needs a negative control proving that the security boundary is actually exercised. A parent hostname capability must never derive its address scope from the resolver result or from Symthaea's recommendation; the owner-approved Nixward scope is the upper bound. Record both application navigation events and independently observed kernel socket-admission events. A kernel connect hook records an attempted/admitted socket operation; it does not prove completed TCP or TLS. If the test needs connection completion, use the fake listener as a separate oracle.

## 6. Integration sequence

1. **Clarify the code contract:** keep `evaluate_url` explicitly labelled as URL preflight, not SSRF/network enforcement.
2. **Create isolated workload identity:** map each browser service/guest to a Nixward `WorkloadIdentity` and cgroup binding.
3. **Add trusted DNS path:** force Chromium DNS through the configured resolver/proxy; reject alternate resolver channels; deliver the actual observed A/AAAA set to the Nixward enforcement broker.
4. **Enforce at the cgroup:** atomically stage exact IP/port sets under the owner-bound parent address scope, with monotonic expiry; v1 TCP first, UDP/QUIC only through a separate explicit profile.
5. **Correlate evidence:** action receipt, resolver snapshot, kernel socket metadata and browser network event share an opaque session/navigation ID. Do not store page payloads by default.
6. **Qualify in a disposable NixOS VM:** execute BROW-EGR-001..012 with IPv4/IPv6, redirects, subresources, service workers, WebSockets, UDP and DNS-rebinding fixtures.
7. **Keep profiles separate:** public browser, explicit LAN browser, and server-side Prism public-web each have different authority and resolver policy.

## 7. Explicit non-claims

This document does not claim:
- that CDP currently intercepts or authorizes every Chromium request;
- that the current BrowserSafetyPolicy prevents DNS rebinding;
- that the browser is currently isolated in a Nixward cgroup with BPF enforcement;
- that Prism Reqwest pins Chromium's DNS results;
- that Nixward PR #162 already implements a resolver proxy or exact-IP browser enforcement;
- that the 12 scenarios above have been executed;
- that any new security boundary is production-qualified.

Related work:
- Nixward workload-bound resolution lease proposal: https://github.com/Luminous-Dynamics/nixward/pull/162
- Prism evidence-enforced egress and PEAB: https://github.com/Luminous-Dynamics/mycelix/pull/6299
- Nixward network architecture: https://github.com/Luminous-Dynamics/nixward/pull/67
