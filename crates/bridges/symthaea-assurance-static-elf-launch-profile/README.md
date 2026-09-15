# Static ELF launch profile

Qualifies the exact Nix-bound verifier executable bytes for a loader-free Linux ELF launch profile: ELF64 little-endian, reviewed machine/type, no `PT_INTERP`, no `DT_NEEDED`, no writable+executable `PT_LOAD`, and one explicit non-executable `PT_GNU_STACK`.

This is an assessment theorem only. It does not perform `execveat`, prove fd-pinned launch, exclude post-launch `dlopen`/manual executable mappings, establish mapping continuity, trusted time, root-resistant immutability, or physical authority.
