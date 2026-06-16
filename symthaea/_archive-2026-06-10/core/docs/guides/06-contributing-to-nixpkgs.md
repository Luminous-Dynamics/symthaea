# Contributing to nixpkgs

*nixpkgs is the largest package repository in any Linux distribution. Here's how to add to it.*

---

## Why Contribute?

nixpkgs has 100,000+ packages, but the one you need might be missing, outdated, or broken. Contributing fixes this — for you and everyone else.

Contributing to nixpkgs is also the fastest way to deeply understand Nix. You learn by doing.

---

## Your First Package

Let's package a simple Rust CLI tool.

### 1. Fork and clone nixpkgs

```bash
# Fork on GitHub, then:
git clone --depth 1 https://github.com/YOUR-USER/nixpkgs.git
cd nixpkgs
git remote add upstream https://github.com/NixOS/nixpkgs.git
git checkout -b add-my-tool
```

### 2. Create the package

```nix
# pkgs/by-name/my/my-tool/package.nix
{
  lib,
  rustPlatform,
  fetchFromGitHub,
}:

rustPlatform.buildRustPackage rec {
  pname = "my-tool";
  version = "1.2.3";

  src = fetchFromGitHub {
    owner = "author";
    repo = "my-tool";
    rev = "v${version}";
    hash = "sha256-AAAA...";  # nix-prefetch-github author my-tool --rev v1.2.3
  };

  cargoHash = "sha256-BBBB...";  # build once, copy hash from error

  meta = with lib; {
    description = "A tool that does something useful";
    homepage = "https://github.com/author/my-tool";
    license = licenses.mit;
    maintainers = with maintainers; [ your-github-username ];
    mainProgram = "my-tool";
  };
}
```

### 3. Test it

```bash
nix-build -A my-tool
./result/bin/my-tool --version

# Run the tests
nix-build -A my-tool.tests  # if the package defines tests
```

### 4. Submit a PR

```bash
git add pkgs/by-name/my/my-tool/
git commit -m "my-tool: init at 1.2.3"
git push origin add-my-tool
# Open PR on GitHub against NixOS/nixpkgs
```

**PR title format:** `my-tool: init at 1.2.3` (for new packages) or `my-tool: 1.2.3 -> 1.3.0` (for updates)

---

## Updating an Existing Package

The most common contribution. A package is outdated and you want the latest version.

```bash
# Find the package
fd package.nix pkgs/by-name/fi/firefox/

# Update version and hashes
# Edit: change `version`, clear `hash` and `cargoHash`
# Build: it will fail with the correct hash — copy it in
nix-build -A firefox

# Or use the update script if one exists
nix-shell maintainers/scripts/update.nix --argstr package firefox
```

---

## NixOS Module Contributions

If you want to add a new NixOS service:

```nix
# nixos/modules/services/misc/my-service.nix
{ config, lib, pkgs, ... }:

let
  cfg = config.services.myService;
in {
  options.services.myService = {
    enable = lib.mkEnableOption "my awesome service";
    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
      description = "Port to listen on";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.my-service = {
      description = "My Awesome Service";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        ExecStart = "${pkgs.my-tool}/bin/my-tool --port ${toString cfg.port}";
        DynamicUser = true;
        Restart = "on-failure";
      };
    };
  };
}
```

Register it in `nixos/modules/module-list.nix` and submit a PR.

---

## PR Checklist

Before submitting your PR:

- [ ] Package builds: `nix-build -A package-name`
- [ ] Tests pass (if any): `nix-build -A package-name.tests`
- [ ] `meta.maintainers` includes your GitHub username
- [ ] `meta.license` is correct
- [ ] Description is clear and concise
- [ ] PR title follows convention: `package: action at version`
- [ ] You've added yourself to `maintainers/maintainer-list.nix` (first contribution only)

---

## The Review Process

1. **Ofborg** (CI bot) builds your package on x86_64-linux and aarch64-linux
2. A nixpkgs committer reviews the code
3. They may request changes — this is normal and helpful
4. Once approved, it merges into `nixos-unstable`
5. After testing, it flows to the stable channel (`nixos-24.11`, etc.)

Typical turnaround: 1-7 days for simple packages, longer for modules.

---

## Resources

- [nixpkgs contributing guide](https://github.com/NixOS/nixpkgs/blob/master/CONTRIBUTING.md)
- [Nix manual: writing packages](https://nixos.org/manual/nixpkgs/stable/#chap-quick-start)
- [by-name convention](https://github.com/NixOS/nixpkgs/tree/master/pkgs/by-name) — where new packages go
- `#nixpkgs` on Matrix — ask for help with packaging

---

*Every package you add helps someone else. That's the power of open source — and the NixOS community is one of the most welcoming in Linux.*
