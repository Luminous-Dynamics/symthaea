# Bootstrap executable mapping continuity

This Linux x86_64 GNU/musl bridge consumes the exact #3455 stopped-exec confirmation and the exact #3514 confined-bootstrap transcript. It rebinds both public evidence vectors to their opaque parent digests, independently re-observes the still-stopped tracee at the ready gate, and requires the complete file-backed executable mapping set, ranges, inode identity, and mapped/backing bytes to equal the launch evidence exactly.

Combined with #3514's syscall-entry mediation and an independent transcript check excluding `exec*`, executable `mmap/mprotect`, `mremap`, `remap_file_pages`, `pkey_mprotect`, `shmat`, and thread/fork escape, this establishes tracee-action file-backed executable-mapping continuity across the bootstrap interval. It does not establish kernel pseudo-map identity continuity, protection against another privileged process/root mutating the tracee, trusted time, or physical authority.
