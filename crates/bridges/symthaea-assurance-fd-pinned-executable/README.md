# symthaea-assurance-fd-pinned-executable

Reusable Linux assurance primitive for external tools whose executable identity must not be re-resolved by pathname after verification.

The production profile accepts only an already-canonical direct `/nix/store/<hash>-.../...` ELF under a root-owned non-writable store object. It opens the executable once, hashes and records that exact open inode, retains the file description across invocations, launches through `/proc/self/fd/0`, and re-checks the same inode and bytes after every invocation.

This closes pathname-replacement TOCTOU for the executable image and gives downstream assurance code one shared execution authority instead of cloning tool-specific launch logic.

The claim is deliberately bounded: this does not atomically pin the ELF interpreter/shared-library closure, prove Nix database currentness, exclude privileged in-place mutation that is fully restored during an invocation, establish trusted time, or grant physical authority.
