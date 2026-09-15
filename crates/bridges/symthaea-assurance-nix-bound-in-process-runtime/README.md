# Nix-bound in-process runtime policy

Binds the strongest current in-process Linux TPM+IMA provider capability to one verified-at-assessment Nix runtime closure and one exact verifier-runtime continuity policy.

The bridge requires the provider host store root to equal the closure root, the provider host executable BLAKE3 to equal the runtime policy's expected executable digest, the verified closure digest to equal the runtime policy's expected dependency-closure digest, and the runtime verifier role to match an exact reviewed role pin.

The result means that the runtime policy's static executable/dependency expectations are backed by the exact authority-current in-process verifier host and exact qualified Nix closure. It does not establish temporal co-observation, mapped-memory identity, atomic loader dependency pinning, Nix database currentness/non-equivocation, trusted time, or physical authority.
