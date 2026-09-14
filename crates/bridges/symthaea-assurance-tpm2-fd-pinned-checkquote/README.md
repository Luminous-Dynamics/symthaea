# FD-pinned TPM2 checkquote assurance

This bridge strengthens the raw TPM2 quote adapter by ensuring that the exact file opened and hashed is also the file executed. On Linux it launches the already-opened ELF through `/proc/self/fd/0`, using that same open file description as child stdin, so replacement of the original pathname after hashing cannot substitute a different executable.

The production policy requires a canonical direct `/nix/store/<hash>-…` path whose store root and executable are root-owned and non-writable, and whose bytes begin with the ELF magic. It also binds the exact inner `Tpm2CheckquotePolicy` digest.

The claim is deliberately bounded: this closes executable-path TOCTOU for the verifier binary and proves a reviewed Nix-store file identity. It does not atomically pin or independently verify the full dynamic dependency closure, defeat a compromised root user, establish trusted time, or grant physical authority.
