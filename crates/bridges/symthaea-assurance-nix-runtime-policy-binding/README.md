# symthaea-assurance-nix-runtime-policy-binding

Binds the terminal fd-pinned Linux TPM+IMA provider identity and one verified-at-assessment Nix runtime closure to the exact executable/dependency expectations of a reviewed `VerifierRuntimeContinuityPolicy`.

Qualification requires the provider's direct Nix store root to equal the closure root, the provider executable digest to equal the runtime policy's expected executable digest, the verified closure digest to equal the runtime policy's expected dependency-closure digest, and the exact reviewed runtime verifier role/policy to be pinned by the binding policy.

The successful opaque `NixBoundRuntimePolicy` means the reviewed runtime policy is semantically bound to this qualified provider executable and this verified closure. It deliberately does **not** claim that the closure was re-observed at the provider's exact use-time, that the loader dependency graph was atomically pinned, that continuous verifier execution has already been established, that the Nix database is current/non-equivocating, that root cannot mutate the store, that wall-clock time is trusted, or that physical authority is granted.
