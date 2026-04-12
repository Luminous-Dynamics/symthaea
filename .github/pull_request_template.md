## Summary

- describe the user-visible or operator-visible change
- call out any control-plane, auth, protocol, or deployment impact

## Validation

- [ ] `Hardened Lib Regressions` is green if lib/control-plane code changed
- [ ] `Hardened Daemon Regressions` is green if daemon/protocol code changed
- [ ] `Hardened API Regressions` is green if API/auth/privacy code changed
- [ ] `Hardened Nix Regressions` is green if Nix/module/deployment code changed
- [ ] any new or changed protocol surface is covered by focused tests

## Control-Plane Checklist

- [ ] schema/runtime/docs stay aligned for any protocol change
- [ ] no new auth shape was introduced
- [ ] remote execution capability was not widened without explicit review
- [ ] any reserved-but-unimplemented verb is intentional and documented
