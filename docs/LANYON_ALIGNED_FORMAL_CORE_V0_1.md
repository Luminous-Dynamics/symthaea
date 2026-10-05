# Lanyon-Aligned Formal Cognitive Core

Status: Research tranche / design specification
Version: 0.1
Date: 2026-10-05
Branch: research/lanyon-aligned-formal-core-v1

## Purpose

This document defines a concrete path for making a small, independently checkable part of Symthaea follow a strong specification -> implementation -> proof discipline.

The goal is not to imitate Lanyon's internal implementation, and it is not to claim that Symthaea is formally verified as a whole. The goal is narrower:

Make one cognitive transition have a canonical machine-readable specification, an executable implementation, and independently checkable evidence that the implementation and proof refer to the same semantics.

This is useful whether or not any relationship with Lanyon AI ever occurs.

## Why this is compatible with Lanyon

Lanyon publicly describes its workflow as Specification -> Implementation + Proof, with the same formal DSL specification feeding deterministic code generation and proof generation. Its public benchmarks explicitly evaluate whether the proof models the same scheme, branch structure, and code actually executed, and distinguish faithful, partial, misformalized, and disconnected results.

Symthaea already contains several complementary pieces:

- exact binary HDC operations over BinaryHV
- human-readable HDC CompositionAlgebra
- a symbolic EML intermediate representation with compile/evaluate/verify stages
- a Z3 bridge and independent SMT-LIB witness generation
- Binius circuits for binary-field HDC operations
- cycle commitments and zero-knowledge proof infrastructure
- extensive evidence and provenance hardening

What is missing is one small canonical semantic object tying these layers together.

## Design principle

The first formal core should be deliberately small.

Do not begin by trying to formalize the full consciousness stack, IIT/Phi itself, every multimodal encoder, the entire liquid network, or all crates.

Start with a primitive where exact semantics are cheap and unambiguous:

BinaryHV state transition

Input:
- left: 16,384-bit vector
- right: 16,384-bit vector

Operation:
- bind = coordinate-wise XOR

Output:
- result = left XOR right

Contract:
- exact dimension = 16,384 bits
- exact storage = 2,048 bytes
- explicit bit ordering
- deterministic operation

This gives Symthaea a miniature version of the pattern Lanyon has demonstrated on scientific kernels.

## Proposed layers

### 1. Canonical specification

Introduce a small typed representation, name TBD, for formal cognitive kernels.

Conceptually:

KernelSpec
- id
- version
- domains
- inputs
- outputs
- operations
- semantics
- invariants
- serialization

The specification must identify semantic domains, not just function names.

For example:

BinaryHV:
- domain = GF(2)^16384
- storage = 2048 bytes
- bit order = explicitly defined

bind:
- semantics = coordinate-wise XOR
- aliases = bind, xor, GF2_add

The specification must not silently inherit semantics from comments or prose.

### 2. Canonical serialization

The specification needs a deterministic byte representation.

That serialization should be used for:
- specification digest
- evidence manifests
- proof statements
- runtime receipts
- reproducibility
- future ZK public inputs

Suggested digest:

spec_digest = BLAKE3(domain_tag || canonical_spec_version || canonical_serialized_spec)

The digest must identify the exact semantic contract, not merely a filename.

### 3. Reference interpreter

Every supported KernelSpec should have a small reference interpreter.

For BinaryHV bind:

reference_bind(a,b) = bytewise XOR

The reference interpreter is intentionally boring. Its role is to provide an independent semantic oracle against optimized runtime implementations.

Optimization must not become the definition.

### 4. Optimized implementation

The production implementation may use AVX-512, AVX2/FMA, SSE, NEON, or a scalar fallback, but it must identify the KernelSpec it claims to implement.

A successful execution should produce a receipt containing at least:
- spec_digest
- implementation_id
- input_digest(s)
- output_digest
- runtime_target
- feature_set
- operation_count or step_count

The receipt is evidence of what was claimed, not proof that it was correct.

### 5. Independent equivalence checker

An independent checker should compare:

reference(spec, input) == optimized_implementation(spec, input)

For BinaryHV bind this can be exact equality over all 2,048 bytes.

The checker should be capable of consuming a receipt without importing the production implementation's semantic helper code.

The key property is to avoid creating a checker that simply repeats the same implementation bug.

### 6. Proof backend

The same KernelSpec should be able to generate proof obligations.

For BinaryHV bind, initial obligations are simple algebraic identities:

bind(a,b) = bind(b,a)
bind(bind(a,b),c) = bind(a,bind(b,c))
bind(a,a) = zero
bind(a,zero) = a

These should be proven over the same semantic representation used by the checker.

Later backends can map the same specification into Lean, SMT-LIB2/Z3, Binius, or other proof systems.

The proof backend is downstream from the semantic specification. It must not become the source of truth.

## Critical semantic boundary: floating point

A central lesson from Lanyon's public verification methodology is that the mathematical expression and the actual machine computation are not automatically the same object.

This matters directly to Symthaea.

The unified HDC-LTC path contains optimized floating-point kernels and a fast tanh approximation. Therefore a future formal proof must distinguish:

1. the ideal continuous-time equation;
2. the discrete update actually evaluated;
3. the approximation contract for nonlinear functions;
4. the hardware arithmetic semantics.

For example:

ideal:
tanh(x)

implementation:
fast_tanh(x)

formal claim:
|fast_tanh(x) - tanh(x)| <= epsilon over an explicitly specified interval

runtime claim:
the selected implementation evaluates fast_tanh with the specified arithmetic.

The first formal-core version should not pretend these are interchangeable.

## Current semantic-gap register

The following items are identified from the current public Symthaea tree and should remain explicit until closed.

### Gap A - cycle integrity proof object can be structurally valid without a proof

CycleIntegrityProof includes proof_bytes as optional, while the current structural verifier checks the commitment chain, Phi range, verdict vocabulary, and domain tag.

The structural verifier does not itself verify a cryptographic proof, and generate_cycle_proof currently constructs an object with empty proof_bytes.

That is acceptable for a commitment-structure API but too strong for an API documented as proving that the cognitive loop executed correctly.

Required future separation:

verify_commitment_structure(...)
verify_cryptographic_proof(...)
verify_full(...)

The intended invariant is:

verify_full = structure_check AND proof_check

Empty proof bytes must not satisfy the full verification path.

### Gap B - current CfC Binius benchmark witnesses sigma

The current CfC temporal benchmark creates sigma as a witness and proves the recurrence conditioned on that witness.

That is a valid algebraic sub-proof, but it is not yet an end-to-end proof of the activation function itself.

Terminology should therefore distinguish:

CfC recurrence proof

from:

end-to-end CfC implementation proof

Closing this gap requires constraining the sigma/activation semantics or explicitly adopting an approximation contract.

### Gap C - hand-authored formal physics definitions are not yet runtime-derived

The physics formal verification path currently contains carefully constructed problem definitions and emits SMT-LIB obligations for conservation laws.

That is useful evidence.

The next step is to make the relationship between runtime equation -> canonical symbolic representation -> formal obligation machine-checkable, so that a proof cannot silently drift from the actual executable equation.

### Gap D - ContinuousHV inverse is approximate

The HDC binding audit already documents that ContinuousHV inverse is not an exact algebraic inverse in the way BinaryHV XOR binding is.

Therefore the formal core should prioritize:
1. exact BinaryHV operations;
2. explicit approximation contracts for ContinuousHV operations;
3. no generic theorem that silently treats the two algebras identically.

## Proposed first vertical slice

KernelSpec(BinaryHV.bind)
    |
    +--> reference interpreter
    |
    +--> optimized Rust implementation
    |
    +--> exact equality checker
    |
    +--> algebraic property proofs
    |
    +--> Binius circuit
    |
    +--> canonical evidence receipt

A successful acceptance test should answer independently:

1. What was the semantic specification?
2. What exact implementation claimed to realize it?
3. What input was supplied?
4. What output was produced?
5. Did the independent interpreter produce the same output?
6. Which formal properties were checked?
7. Which proof backend checked them?
8. Which exact spec version did every artifact refer to?

## Proposed second vertical slice

After BinaryHV bind:

BinaryHV bundle
    |
    +--> exact/tie policy specification
    +--> optimized implementation
    +--> reference interpreter
    +--> property suite
    +--> Binius proof

Bundling requires a more explicit contract because tie handling, weighting, vector count, and bit semantics matter.

## Proposed third vertical slice

Only after the discrete algebra is stable:

HDC-LTC discrete step
    |
    +--> exact f32 arithmetic semantics
    +--> explicit operation ordering
    +--> fast_tanh approximation contract
    +--> reference interpreter
    +--> runtime-equivalence checker
    +--> bounded-error proof

This is where the project can meaningfully engage with Lanyon-style questions about whether the formal object is actually the executed numerical method.

## Evidence levels

The formal core should use an explicit vocabulary.

Level 0 - stated
The specification states a property. No runtime evidence.

Level 1 - tested
The implementation passes executable tests.

Level 2 - independently replayed
An independent checker reproduces the result.

Level 3 - formally verified
A machine-checkable proof establishes the stated property for the stated domain.

Level 4 - cryptographically attested
A valid cryptographic proof binds the execution/evidence to the relevant public statement.

No higher-level label should be emitted when only a lower level has been demonstrated.

## Why this could matter to Lanyon

The compelling collaboration story is not:

Symthaea is a conscious AI.

It is:

Symthaea is building a different cognitive substrate, and we are trying to give its computational primitives the same semantic discipline that formally verified scientific computing gives numerical kernels.

That creates concrete technical questions:

- Can a cognitive primitive have a compact formal DSL?
- Can one semantic object generate both optimized code and proof obligations?
- Can HDC algebra be treated as a formally composable computational substrate?
- How should approximate nonlinear functions be specified and certified?
- Can cognitive state transitions receive bounded-error or exact contracts?
- Can ZK attestations be bound to the same semantic specification used for formal proofs?

## What not to do

Do not create a large Lanyon integration abstraction before one primitive is closed.

Do not add a second competing DSL merely because Lanyon uses a DSL.

Do not claim that a proof of an HDC primitive proves cognition, intelligence, or consciousness.

Do not make the proof artifact authoritative while leaving the implementation relationship implicit.

Do not use numerical agreement at a handful of samples as a substitute for a universal theorem.

Do not equate a cryptographic commitment with proof of execution.

## Acceptance gate for this research tranche

A future implementation should produce one evidence packet containing:

- canonical_spec
- spec_digest
- reference_result
- runtime_result
- equivalence_result
- property_results
- formal_proof_metadata
- runtime_environment
- implementation_commit

An independent checker must reject any packet where:
- the spec digest is wrong;
- the implementation is missing;
- reference and runtime outputs differ;
- the proof refers to a different spec;
- a claimed formal property has no valid proof result;
- required cryptographic proof material is absent.

## Relationship to Lanyon

This document does not imply any partnership, endorsement, or communication with Lanyon AI.

The comparison is architectural.

Lanyon's public work demonstrates that compact scientific specifications can drive implementations and proofs together, and its public evaluation methodology places unusual emphasis on semantic faithfulness between proofs and executed code.

The objective here is to carry that discipline into a small, explicit portion of Symthaea without overclaiming what has been verified.

## Immediate engineering order

1. Freeze the BinaryHV bind semantic contract.
2. Introduce a minimal typed KernelSpec representation.
3. Implement canonical serialization and spec hashing.
4. Build the independent reference interpreter.
5. Connect the existing BinaryHV implementation to a spec identifier.
6. Add an equivalence receipt.
7. Bind the existing Binius bind proof to the same spec digest.
8. Add negative tests for spec/proof/implementation mismatch.
9. Only then generalize the pattern to bundle and CfC.

## Source pointers

Symthaea repository:
https://github.com/Luminous-Dynamics/symthaea

Lanyon AI:
https://lanyon.ai/

Lanyon formal-verification rubric:
https://lanyon.ai/research/linear-benchmarking/

Lanyon advection-diffusion research:
https://lanyon.ai/research/advection-diffusion/

Lanyon public generated solver repository:
https://github.com/lanyonai/AdvectionDiffusion
