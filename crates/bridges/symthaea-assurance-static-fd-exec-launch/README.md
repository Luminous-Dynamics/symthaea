# Static fd exec launch

Prepares a retained-file-descriptor launch for an exact qualified static ELF verifier and exposes the irreversible Linux `execveat(fd, "", ..., AT_EMPTY_PATH)` boundary without handwritten unsafe FFI.

Preparation rebinds the retained fd to the exact static-profile/Nix-bound executable identity and commits an explicit argv/environment/launch-nonce plan. A successful `execveat` never returns; therefore this crate does **not** manufacture a success capability. Post-exec success, first-instruction control, mapping continuity, syscall confinement, trusted freshness/time, and physical authority require independent evidence.
