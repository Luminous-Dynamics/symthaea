# CROSS-068A — NixOS definition-path symlink policy

Systemd can report a unit source path that is a symlink into the Nix store. Current NixOS materializes unit links to `/nix/store/*-unit-...` paths in common configurations.

Therefore CROSS-068A does not equate `O_NOFOLLOW` with a universal 'no symlinks' policy.

The capture policy is:

1. capture the exact systemd-reported path as source identity through CROSS-067;
2. resolve the reported path explicitly;
3. when resolution changes the path, require the resolved target to be under `/nix/store/`;
4. open the resolved target with `O_NOFOLLOW`;
5. record the resolved path alongside the reported path;
6. include both resolution metadata and content digest in the sealed commitment;
7. reject any resolved target outside `/nix/store/` rather than silently following it.

This leverages the NixOS store's immutability model for the resolved target while retaining a separate source-identity commitment. It does not claim that arbitrary mutable Unix/systemd paths are safe to follow.

Future hardening can add an explicit systemd-root/filesystem-profile policy for non-Nix-store definitions and a stronger kernel path-resolution primitive where appropriate.