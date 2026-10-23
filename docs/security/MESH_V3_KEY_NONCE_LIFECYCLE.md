# Mesh v3 key and nonce lifecycle design

**Status:** proposal for security review. Not a wire-format specification and not implementation evidence.

**Related work:** [PR #8376](https://github.com/Luminous-Dynamics/symthaea/pull/8376) (nonce audit), [PR #8471](https://github.com/Luminous-Dynamics/symthaea/pull/8471) (packet/fragment nonce model), [PR #8491](https://github.com/Luminous-Dynamics/symthaea/pull/8491) (handshake admission model), [PR #8508](https://github.com/Luminous-Dynamics/symthaea/pull/8508) (LoRa fragment-AEAD containment), [PR #8519](https://github.com/Luminous-Dynamics/symthaea/pull/8519) (replay/effect crash consistency), [PR #8533](https://github.com/Luminous-Dynamics/symthaea/pull/8533) (cloned-sender model), and [issue #8534](https://github.com/Luminous-Dynamics/symthaea/issues/8534) (runtime implementation).

## 1. Decision in one sentence

Make nonce uniqueness a consequence of **authenticated session-key separation plus a non-wrapping per-key sequence allocation, with an explicit clone/rollback boundary**—not a consequence of short epochs, truncated source IDs, startup randomness alone, or a counter that exists only in memory.

No production wire change should be made until this proposal is reviewed, modeled, measured against transport constraints, and assigned a versioned compatibility/migration plan.

## 2. The design fact we must not hand-wave

For AEAD, the required property is uniqueness of the pair `(effective key, nonce)` for every encryption invocation under that key. Different nonce bytes under a different key are fine; the same nonce under different keys is not the problem. Conversely, a counter is not sufficient if two processes share the same effective key and cloned counter state.

There is also a boundary no clever local algorithm can remove:

> If two offline process instances begin with identical cryptographic state, have no fresh independent entropy, no non-clonable state, and no external observable that distinguishes them, a deterministic algorithm cannot guarantee different next nonces for both.

The instances have identical inputs and state, so their deterministic next outputs are identical. This is an indistinguishability boundary, not a missing random-number trick. A UUID helps only to the extent its entropy source is genuinely fresh and not cloned/rolled back; it is probabilistic. A software fencing token helps only if stale copies cannot bypass the authority and reach the encryption/send side effect.

Therefore v3 must choose an explicit assurance profile:

- **Profile A — online/reachable peers:** start a new authenticated session after restart/recovery; derive new directional traffic keys from the authenticated session secret and full context. A cloned old context must not remain authorized to send.
- **Profile B — offline/long-lived sender:** use a provably non-clonable/rollback-resistant key-and-counter authority, such as an adequately reviewed hardware-backed key service or hardware monotonic allocator, and enforce its exclusive use at the actual AEAD/send boundary.
- **Profile C — no unique-state authority available:** stop encrypted transmission and require re-enrolment/rekey when state freshness cannot be established. Do not silently resume a potentially reused key/counter namespace.

A software-only implementation may offer a documented probabilistic clone-risk profile, but it must not label that profile as an absolute nonce-uniqueness guarantee.

## 3. Cryptographic construction constraints

Prefer standard reviewed components. Do not design a new AEAD, home-grown KDF, or an ad hoc hybrid handshake.

### 3.1 Session and key namespace

1. Authenticate the peer identities and bind them to the same canonical handshake transcript before accepting session keys.
2. Establish a fresh session secret through the reviewed, versioned key-agreement profile already approved for the protocol. Use a standard KDF such as HKDF (RFC 5869) with explicit, length-delimited context.
3. Derive **directional and purpose-separated** keys. Context must bind at least protocol/version, negotiated suite, canonical sender and receiver identities, session identifier/generation, direction, and purpose (packet vs any separately encrypted fragment/control channel).
4. A claimed source ID, wire epoch, or key-version byte is not proof of possession and must not, by itself, select authority. It must be covered by authenticated negotiation or authenticated packet context.
5. Do not treat HPKE as an interchangeable drop-in mutual handshake: RFC 9180 specifies a particular public-key encryption construction and key schedule. Use it only where its sender/receiver and authentication semantics fit the system's protocol. Otherwise use the separately reviewed handshake and a standard KDF.

References: [RFC 5869 — HKDF](https://www.rfc-editor.org/rfc/rfc5869.html), [RFC 9180 — HPKE](https://www.rfc-editor.org/rfc/rfc9180.html), [RFC 8439 — ChaCha20 and Poly1305](https://www.rfc-editor.org/rfc/rfc8439.html), [RFC 5116 — AEAD interface requirements](https://www.rfc-editor.org/rfc/rfc5116.html).

### 3.1.1 Global nonce namespace across APIs

Nonce uniqueness is an invariant across **all invocations using the same effective key**, not separately within each function or transport. The existing packet and fragment helpers use different layouts, but those layouts are not automatically disjoint. A deterministic runtime witness in [PR #8558](https://github.com/Luminous-Dynamics/symthaea/pull/8558) constructs identical 12-byte nonces from the packet and fragment APIs with one shared key and valid packet type. This demonstrates a conditional code-path collision, not that every deployment shares those keys.

V3 must therefore enforce one of the following and test the actual effective keys emitted/selected by the runtime:

- derive distinct, reviewed purpose-specific keys for packet AEAD, fragment AEAD, and other encrypted channels using explicit KDF domain separation; or
- specify and prove globally disjoint nonce allocation across every use of a shared key.

A different function name, packet type, route, or nonce field layout is not proof of separation. The test oracle must observe all ciphertext-producing entry points and compare the full effective-key/nonce pair. The shared key must not be assumed separate by convention alone.

### 3.2 Nonce and sequence

- Freeze an exact, byte-level nonce layout for the new protocol version and test it against golden vectors. Never alter the existing v2 layout in place.
- Under each directional session key, use a simple injective sequence allocation—for example, a protocol-reviewed 96-bit counter representation, or a reviewed fixed-prefix plus sufficiently wide counter. Field size, byte order, reset semantics, and range are part of the spec.
- Counter exhaustion is a state transition to **rekey or stop**. Wrap must be impossible at the API boundary; tests must set a tiny reduced counter range and hit exhaustion.
- If keys persist across a crash, reserve counter ranges atomically and durably before use. A crash may burn unused counter values; it must never restore a value already used. Range reservation alone does **not** defend against a full snapshot clone that copies both the key and the reserved range.
- If the protocol instead uses random nonces, state the per-key collision bound, message-volume limit, CSPRNG and fork/snapshot assumptions, and recovery policy. Randomness is not a mathematical guarantee.

A session identifier helps bind contexts and route messages; it is not a nonce uniqueness mechanism unless it actually separates effective keys or forms part of a correctly analyzed nonce namespace.

### 3.2.1 Defense-in-depth candidate: nonce-misuse-resistant AEAD

The v3 comparison should include a reviewed **nonce-misuse-resistant AEAD** as a separate experimental profile, not only variants of the current ChaCha20-Poly1305 discipline. RFC 8452 specifies AES-GCM-SIV and explicitly targets the catastrophic consequences of accidental nonce reuse; it still leaks equality when the same nonce encrypts identical plaintext, so reuse must remain a failed correctness/security gate rather than an accepted operating mode. RFC 8452 recommends random nonces even for this construction and is an IRTF Informational RFC, not an Internet Standards Track specification ([RFC 8452](https://www.rfc-editor.org/rfc/rfc8452.html)).

The RustCrypto `aes-gcm-siv` crate documents RFC 8452 support and a no-alloc in-place path, but its current crate-level security warning says the crate itself has not been audited; some dependencies were audited in the context of `aes-gcm`, which is not equivalent to an audit of the complete crate/construction/integration ([crate documentation](https://docs.rs/aes-gcm-siv/latest/aes_gcm_siv/)). This candidate therefore requires threat-model review, dependency audit, target-side constant-time suitability review, and independent cryptographic assessment before use.

Benchmark at least these arms under the same fault corpus:

1. **Baseline:** current ChaCha20-Poly1305 with strict per-effective-key nonce uniqueness and fail-closed rollback recovery.
2. **Misuse-resistant candidate:** AES-256-GCM-SIV (or an approved equivalent), with independently generated nonces and the same key-domain separation, replay, and recovery controls.
3. **Broken-control arm:** intentionally reuse a nonce under each candidate to verify that the oracle detects the repeated pair and that the measured confidentiality impact matches each primitive's documented guarantees. This is a negative control only; never enable deliberate reuse in deployment.

Measure p50/p95/p99 encryption/decryption latency, energy per packet, peak memory and code size on each supported CPU class, cold-start cost, ciphertext bytes including nonce/tag/framing, adapter MTU fit, and fault-injection behavior on restart/snapshot/rollback. RFC 8452 reports slower encryption due to its two-pass construction and near-AES-GCM decryption performance for multikilobyte messages; those figures are not a substitute for measurements on the small frames and target CPUs used by this mesh.

Do **not** assume misuse resistance is a free fix. It reduces catastrophic consequences of some nonce mistakes; it does not remove identity/authentication bugs, replay risk, key compromise, equality leakage, or the need for an explicit protocol version and tested key lifecycle. The existing hard target remains zero duplicate `(effective key, nonce)` pairs for supported strict profiles.

### 3.3 Clone, snapshot and rollback

Treat these as different threat models, not one generic "restart" test:

1. **Crash/restart with preserved durable state:** reserved counter blocks do not repeat.
2. **Counter rollback or restored backup:** sender refuses old key state or obtains a new authenticated session/authority generation.
3. **Two live clones:** either only one clone can access the cryptographic send authority, or each clone obtains fresh independent keying material and peer authorization before sending.
4. **Offline cloned clone with no external distinction:** absolute uniqueness is unavailable for deterministic software state alone; enforce the chosen hardware/entropy assumption or fail closed.
5. **Clone after send but before recording:** durable allocator semantics prevent re-use or force a new session/key.
6. **Key rotation and grace period:** old keys are receive-only / retired according to an authenticated policy; generation claims cannot resurrect a retired key.

A fencing value only helps when stale holders are technically prevented from reaching encryption and transmission. A local boolean check or a token copied in the same snapshot is not fencing.

## 4. LoRa and fragmentation decision

The currently documented path has a 222-byte `LORA_MTU`; serializing a maximal fragment to 222 bytes and then wrapping it in a 12-byte nonce plus a 16-byte tag produces 250 bytes before any lower-layer overhead. That cannot be called MTU-safe under the current contract.

**Default v3 candidate:** encrypt/authenticate the complete compressed packet envelope once, then fragment the resulting ciphertext. Fragment headers can carry reassembly fields and a non-cryptographic error-detection code, but the receiver must not release the reconstructed application packet until the full packet AEAD authentication succeeds. Reassembly code must bound per-peer memory, fragments, timeouts, duplicates, and incomplete-message counts.

Only retain per-fragment AEAD if a measured threat requires independent fragment authentication. If retained, define a new frame profile with reduced plaintext capacity and an explicit version. Calculate actual serialized length including radio, fragmentation, nonce, tag, FEC and adapter-specific overhead **before the first frame is sent**; reject atomically if it will not fit. Do not send a partly delivered envelope and then discover the later fragment is over MTU.

This is a protocol choice requiring benchmark and interoperability evidence, not a silent code tweak.

## 5. Replay and application effects are separate

AEAD authenticates ciphertext; it does not make delivery exactly once. Replay admission and the side effect need a crash-consistent design:

- Persist message identity with the application inbox/effect transaction where possible.
- Otherwise use an idempotency key plus an outbox/recovery protocol and define how external actuators or APIs reconcile partial completion.
- Test both crash windows: effect before replay commit (duplicate effect risk) and replay commit before effect (lost effect risk).
- Expiring an in-memory replay cache or restarting the receiver is not durable replay resistance.

See [PR #8519](https://github.com/Luminous-Dynamics/symthaea/pull/8519) for the bounded crash-window model, not a claim of runtime exactly-once semantics.

## 6. Required model, test and benchmark contract

All outcomes must be tied to an exact source SHA, protocol vector hash, model/config hash, tool digest, raw output and claim scope. A parser error, timeout, TLC deadlock, absent receipt, or missing backend is **not** an expected mutant counterexample.

### Safety gates (hard failures)

| Scenario family | Required result |
|---|---|
| Repeated encryptions, same key/counter after restart | No repeated (effective-key, nonce) pair; otherwise fail closed |
| Two source IDs collide in a truncated namespace under a shared key | Configuration/key design prevents collision or is rejected |
| Clone/fork/snapshot/rollback with two active senders | Zero duplicate pairs; chosen hardware/entropy assumptions tested explicitly |
| Counter wrap and exhaustion | Rekey/stop before reuse; no wrap acceptance |
| Key rotation and direction/type domain separation | No reused pair; wrong-context decrypt rejected |
| Missing key/backend, unknown epoch or failed session | No encryption/send on unsupported profile; no plaintext fallback |
| LoRa and all adapters | Every frame within the *actual* configured MTU; zero partial send on preflight failure |
| Crash around replay/effect commit | No duplicate effect and no lost committed effect within stated transaction/recovery model |
| Positive controls | Valid roundtrip, reorder policy, fragmentation, FEC and interoperable vectors remain correct |

### Comparative benchmark

Use a fixed corpus and equal resource/time budgets. Compare:

1. Conventional source review plus deterministic/property-based tests.
2. TLC/model-based review alone.
3. AI-assisted review alone.
4. Combined pipeline: adversarial scenario generation → formal model → runtime tests → independent evidence validator → human review.

Report per scenario class: true-positive and false-positive counts, mutant kill rate, false acceptance rate, positive-control retention, time/cost to an independently accepted finding, human review time, explored state counts, p50/p95 runtime, peak memory, bytes per frame, fragment/retry overhead, and evidence reproducibility across clean builds. Keep held-out seeded mutants out of prompt/context used to develop detectors.

**Targets, not measured results:** 100% of registered critical mutants rejected for the intended reason; zero critical false acceptances; zero nonce pair repeats in the exhaustive/fault-injected scope; all positive controls retained; 100% receipts validated and reproducible; no benchmark improvement purchased by weakening a safety condition.

A claim of outperforming industry/SOTA requires an independently reviewed, held-out, matched-budget comparison with confidence intervals—not raw counts of generated findings or success of a single bounded model.

## 6.1 Industry qualification baseline and the gap this benchmark fills

Use established primitive and implementation test suites as baselines; do not rebrand a home-grown nonce harness as a replacement for them.

- [RFC 8439 §2.8](https://www.rfc-editor.org/rfc/rfc8439.html#section-2.8) specifies a 96-bit ChaCha20-Poly1305 nonce that must differ for each invocation under the same key. [RFC 5116 §3.1](https://www.rfc-editor.org/rfc/rfc5116.html#section-3.1) explicitly discusses nonce generation and multi-device coordination when multiple devices use one key.
- [C2SP/Wycheproof](https://github.com/C2SP/wycheproof) provides implementation-agnostic vectors for known attacks, specification inconsistencies and implementation bugs, including ChaCha20-Poly1305. Integrate applicable upstream vectors for primitive behavior and malformed/invalid input handling.
- [NIST CAVP](https://www.nist.gov/programs-projects/cryptographic-algorithm-validation-program) provides formal algorithm/component validation for approved and recommended algorithms. It is a distinct assurance layer and does not, by itself, certify this application's session lifecycle, key-sharing configuration, snapshot rollback policy or the coupling of packet and fragment APIs.

The gap for Symthaea is **composition and lifecycle**: library vectors usually give a concrete key/nonce/input and expected primitive result; they do not infer whether every production call site across APIs reuses an effective key or whether a cloned process can reuse state. The runtime nonce-pair oracle therefore complements, rather than replaces, upstream vectors.

| Assurance layer | Baseline | Additional Symthaea target |
|---|---|---|
| Primitive correctness | RFC vectors and applicable Wycheproof cases | Keep upstream tests pinned/traceable; no weakening of tag, parse or wrong-key rejection |
| Protocol/model properties | Bounded TLC strict models + intended legacy counterexamples | Record exact state counts, raw traces, model/config/tool hashes and claim ceiling |
| Runtime key/nonce lifecycle | Actual helper outputs and fault-injected sender/recovery states | Observe all encryption entry points; zero repeated (effective key, nonce) pairs in strict scope |
| Integration/transport | Interoperability and adapter MTU checks | Zero partial sends on MTU preflight failure, zero cleartext fallback, valid controls retained |
| Comparative claim | Independent review / held-out tests | Equal-budget baselines, held-out mutants, false-acceptance rates, review cost and latency/airtime percentiles |

A credible outperform-SOTA claim requires results from all relevant layers, measured against a frozen test corpus and an independent reviewer. A larger number of generated findings, more TLC states, or more passing vectors is not on its own a better security result.

## 7. Migration plan (must be versioned)

1. Keep v2 decoding and its limitations explicit; never reinterpret its nonce bytes under new semantics.
2. Define v3 packet/key/nonce and transport framing in a separate frozen specification with golden wire vectors.
3. Add a decoder and sender that explicitly negotiate/advertise v3 support under authenticated transcript binding. Unknown versions and failed negotiation are fail-closed for encrypted traffic.
4. Decide whether mixed-version relays can safely forward opaque packets; never downgrade encryption silently.
5. Run two independently implemented golden-vector/interoperability tests plus constrained-device MTU/airtime/energy tests.
6. Make release promotion depend on implementation evidence and independent review, not merely passing model checks.

## 8. Current evidence boundary

- The dedicated clone-lifecycle TLC reference and intended mutant check passed at the recorded head in [PR #8533](https://github.com/Luminous-Dynamics/symthaea/pull/8533).
- The handshake lifecycle bounded model and five intended mutants passed their dedicated TLC job in [PR #8491](https://github.com/Luminous-Dynamics/symthaea/pull/8491).
- The replay/effect model and its two crash-window mutants passed their dedicated TLC job in [PR #8519](https://github.com/Luminous-Dynamics/symthaea/pull/8519).
- The packet/fragment nonce lifecycle model in [PR #8471](https://github.com/Luminous-Dynamics/symthaea/pull/8471) is a separate workstream and must not be promoted until its corrected exact-head model job passes.
- These are bounded model results, not runtime nonce-safety proofs, cryptographic proofs, or proof that hardware fencing is implemented.

## 9. Release decision rule

Do not release v3 encrypted mode until (a) the construction and migration are reviewed, (b) every supported platform has an explicit clone/restart assurance profile, (c) required models and negative controls pass, (d) test-only instrumentation observes zero repeated pairs under the declared fault corpus, (e) interoperable golden vectors pass independently, and (f) actual transport and recovery behavior conforms. Where the system cannot prove its configured uniqueness assumption, the correct result is a safe refusal to send encrypted traffic.
