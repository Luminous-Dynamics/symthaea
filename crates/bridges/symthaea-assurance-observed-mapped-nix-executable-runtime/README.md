# symthaea-assurance-observed-mapped-nix-executable-runtime

Linux-only bounded observation of the verifier's actual executable mappings. The bridge rebinds the exact qualified Nix closure manifest, requires a stable `/proc/self/maps` snapshot, rejects W+X/unapproved anonymous/deleted/non-Nix executable mappings, binds file-backed executable mappings to their current device/inode identities and qualified closure store objects, and compares executable mapped bytes through `/proc/self/mem` with the corresponding backing-file bytes.

This establishes executable mapping/backing-file equality only at the bounded observation. It does not establish mapping continuity since `execve`, atomic loader pinning, current re-verification of the Nix NAR closure, resistance to privileged in-place mutation, trusted time, or physical authority.
