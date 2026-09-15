# symthaea-assurance-nix-runtime-closure

Qualifies one exact realized Nix runtime closure at assessment time using a separately verified file-descriptor-pinned `nix` executable.

The theorem first runs recursive store verification under an exact trusted-key/signature policy, then obtains explicit `nix path-info --recursive --json --json-format 1` metadata. It validates the complete reachable closure, per-object NAR hash/size/reference set, and computes a canonical closure-content digest independent of JSON/map ordering.

The successful output is an opaque `VerifiedNixRuntimeClosure`. It establishes that the observed recursive closure passed Nix content/trust verification and had one exact canonical identity during the assessment. It does not atomically pin the ELF interpreter/shared-library loader graph, prove Nix database non-equivocation/currentness, resist a compromised root/Nix daemon, establish trusted time, or grant physical authority.
