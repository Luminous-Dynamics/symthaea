# Mastering Nix

*You can edit configuration.nix and rebuild. Now let's understand the machinery.*

---

## The Nix Language in 5 Minutes

Nix is a pure, lazy, functional language. You only need to know a few things:

### Attribute Sets (like JSON objects)
```nix
{
  name = "Symthaea";
  version = "0.6";
  features = [ "consciousness" "HDC" "IIT" ];
}
```

### Functions (always one argument)
```nix
# A function that takes `name` and returns a greeting
greet = name: "Hello, ${name}!";

# A function that takes an attribute set (pattern matching)
mkService = { name, port, enable ? true }: {
  services.${name} = {
    inherit enable;
    listenPort = port;
  };
};
```

### Let Bindings (local variables)
```nix
let
  hostname = "guardian";
  domain = "local";
in {
  networking.hostName = hostname;
  networking.domain = domain;
  networking.fqdn = "${hostname}.${domain}";
}
```

### Imports and With
```nix
# Import another file
{ imports = [ ./hardware-configuration.nix ./services.nix ]; }

# `with` brings names into scope
environment.systemPackages = with pkgs; [ vim git firefox ];
# Same as: [ pkgs.vim pkgs.git pkgs.firefox ]
```

### Conditionals
```nix
services.xserver.videoDrivers =
  if config.hardware.nvidia.modesetting.enable
  then [ "nvidia" ]
  else [ "modesetting" ];

# mkIf — conditional module option (preferred in NixOS)
services.openssh.enable = lib.mkIf config.networking.firewall.enable true;
```

That's the whole language. Everything else is libraries.

---

## Writing Your First Module

A NixOS module is a function that takes `{ config, lib, pkgs, ... }` and returns options and config.

```nix
# modules/development.nix
{ config, lib, pkgs, ... }:

let
  cfg = config.mySystem.development;
in {
  # Declare options
  options.mySystem.development = {
    enable = lib.mkEnableOption "development tools";

    languages = lib.mkOption {
      type = lib.types.listOf (lib.types.enum [ "rust" "python" "node" "go" ]);
      default = [ "rust" ];
      description = "Programming languages to install";
    };

    editors = lib.mkOption {
      type = lib.types.listOf (lib.types.enum [ "neovim" "vscode" "helix" ]);
      default = [ "neovim" ];
    };
  };

  # Define config based on options
  config = lib.mkIf cfg.enable {
    environment.systemPackages = with pkgs;
      # Always
      [ git curl jq fd ripgrep ]
      # Per language
      ++ lib.optionals (builtins.elem "rust" cfg.languages) [ rustc cargo rust-analyzer ]
      ++ lib.optionals (builtins.elem "python" cfg.languages) [ python3 python3Packages.pip ]
      ++ lib.optionals (builtins.elem "node" cfg.languages) [ nodejs_22 ]
      ++ lib.optionals (builtins.elem "go" cfg.languages) [ go gopls ]
      # Editors
      ++ lib.optionals (builtins.elem "neovim" cfg.editors) [ neovim ]
      ++ lib.optionals (builtins.elem "vscode" cfg.editors) [ vscode ];

    # Enable direnv for all dev users
    programs.direnv = {
      enable = true;
      nix-direnv.enable = true;
    };
  };
}
```

**Use it:**
```nix
# configuration.nix
{
  imports = [ ./modules/development.nix ];

  mySystem.development = {
    enable = true;
    languages = [ "rust" "python" "node" ];
    editors = [ "neovim" "vscode" ];
  };
}
```

You just created a reusable, configurable module. Share it with friends or publish it as a flake.

---

## Overlays: Modifying Packages

An overlay lets you modify or add packages without forking nixpkgs.

```nix
# overlays/default.nix
final: prev: {
  # Override an existing package
  htop = prev.htop.overrideAttrs (old: {
    patches = (old.patches or []) ++ [ ./htop-custom-colors.patch ];
  });

  # Add a new package
  my-tool = final.callPackage ./pkgs/my-tool.nix { };

  # Pin a package to a specific version
  discord = prev.discord.overrideAttrs (old: rec {
    version = "0.0.45";
    src = prev.fetchurl {
      url = "https://dl.discordapp.net/apps/linux/${version}/discord-${version}.tar.gz";
      sha256 = "sha256-AAAA...";
    };
  });
}
```

**Apply it in your flake:**
```nix
# flake.nix
nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
  modules = [
    ./configuration.nix
    { nixpkgs.overlays = [ (import ./overlays) ]; }
  ];
};
```

---

## Flake Patterns

### Multi-Machine Config
```nix
# flake.nix — managing desktop + laptop + server
{
  outputs = { nixpkgs, ... }@inputs: {
    nixosConfigurations = {
      desktop = nixpkgs.lib.nixosSystem {
        modules = [ ./hosts/common ./hosts/desktop ];
      };
      laptop = nixpkgs.lib.nixosSystem {
        modules = [ ./hosts/common ./hosts/laptop ];
      };
      server = nixpkgs.lib.nixosSystem {
        modules = [ ./hosts/common ./hosts/server ];
      };
    };
  };
}
```

### Sharing Modules as a Flake
```nix
# Your published module flake
{
  outputs = { ... }: {
    nixosModules.myModule = import ./modules/my-module.nix;
    nixosModules.default = self.nixosModules.myModule;
  };
}

# Someone else uses it:
{
  inputs.cool-module.url = "github:you/cool-module";
  outputs = { nixpkgs, cool-module, ... }: {
    nixosConfigurations.host = nixpkgs.lib.nixosSystem {
      modules = [ cool-module.nixosModules.default ];
    };
  };
}
```

### Dev Shell with Multiple Commands
```nix
devShells.default = pkgs.mkShell {
  packages = with pkgs; [ just cargo-watch sqlx-cli ];
  shellHook = ''
    echo "Available commands:"
    echo "  just build    — build the project"
    echo "  just test     — run tests"
    echo "  just db       — start database"
  '';
  DATABASE_URL = "postgres://localhost/myapp";
};
```

---

## Debugging Tips

```bash
# Why is this package so big?
nix why-depends /run/current-system nixpkgs#firefox

# What does this option do?
nixos-option services.xserver.enable

# What changed between generations?
nix store diff-closures /nix/var/nix/profiles/system-41-link /nix/var/nix/profiles/system-42-link

# Evaluate an expression
nix eval --expr '1 + 1'
nix eval --file ./configuration.nix --json

# Build without switching (dry run)
nixos-rebuild dry-build

# Show the dependency graph
nix-store -q --graph /run/current-system | dot -Tsvg > system-graph.svg
```

---

*Next: [Secrets & Security](05-secrets-and-security.md) — Managing passwords, API keys, and certificates declaratively.*
