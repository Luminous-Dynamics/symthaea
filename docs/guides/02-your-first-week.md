# Your First Week on NixOS

*You survived Day 1. Now let's make this system truly yours.*

---

## Home Manager: Declaring Your User Space

Your system config (`configuration.nix`) manages the OS. But what about your dotfiles, your shell theme, your git config, your editor setup?

**Home Manager** does for your user what NixOS does for the system: makes it declarative.

If your flake already includes Home Manager (Sovereign Inoculation sets this up), you have a `home.nix` file. Here's what goes in it:

```nix
# home.nix
{ config, pkgs, ... }:
{
  home.username = "yourname";
  home.homeDirectory = "/home/yourname";

  # Git — no more .gitconfig
  programs.git = {
    enable = true;
    userName = "Your Name";
    userEmail = "you@example.com";
    extraConfig = {
      pull.rebase = true;
      init.defaultBranch = "main";
    };
    aliases = {
      co = "checkout";
      st = "status";
      lg = "log --oneline --graph --all";
    };
  };

  # Shell — no more .bashrc/.zshrc
  programs.zsh = {
    enable = true;
    autosuggestion.enable = true;
    syntaxHighlighting.enable = true;
    shellAliases = {
      ll = "ls -la";
      rebuild = "sudo nixos-rebuild switch --flake /etc/nixos";
      update = "cd /etc/nixos && sudo nix flake update && sudo nixos-rebuild switch --flake .";
    };
  };

  # Starship prompt
  programs.starship = {
    enable = true;
    settings = {
      add_newline = false;
      character.success_symbol = "[>](green)";
    };
  };

  # Firefox — declarative extensions
  programs.firefox = {
    enable = true;
    profiles.default = {
      extensions = with pkgs.nur.repos.rycee.firefox-addons; [
        ublock-origin
        bitwarden
      ];
    };
  };

  home.stateVersion = "24.11";
}
```

Apply with:
```bash
home-manager switch --flake /etc/nixos
```

**The power move:** Your entire user environment — shell, editor, git, browser — is now in a file you can version control, share, and reproduce on any machine.

---

## Development Environments with `nix develop`

This is where NixOS shines for developers. Instead of installing tools globally, create per-project environments.

**Create a `flake.nix` in your project:**

```nix
# myproject/flake.nix
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          rustc cargo rust-analyzer
          pkg-config openssl
          nodejs_22
        ];

        shellHook = ''
          echo "Rust $(rustc --version)"
          echo "Node $(node --version)"
        '';
      };
    };
}
```

**Enter the environment:**
```bash
cd myproject
nix develop
# You now have Rust, Node, and OpenSSL — only in this shell
# Leave the shell and they're gone
# No version managers, no containers, no conflicts
```

**With direnv (automatic):**
```nix
# Add to home.nix:
programs.direnv = {
  enable = true;
  nix-direnv.enable = true;
};
```

Then add `use flake` to `.envrc`:
```bash
echo "use flake" > myproject/.envrc
direnv allow
# Now every time you `cd myproject`, the environment activates automatically
```

This replaces: nvm, pyenv, rbenv, rustup (partially), Docker dev containers, and virtualenv. All at once.

---

## Customizing Your Desktop

### GNOME
```nix
# In configuration.nix:
programs.dconf.enable = true;

# In home.nix:
dconf.settings = {
  "org/gnome/desktop/interface" = {
    color-scheme = "prefer-dark";
    font-name = "Inter 11";
    monospace-font-name = "JetBrains Mono 10";
  };
  "org/gnome/desktop/wm/preferences" = {
    button-layout = "appmenu:minimize,maximize,close";
  };
};
```

### Hyprland
```nix
# In home.nix:
wayland.windowManager.hyprland = {
  enable = true;
  settings = {
    monitor = ",preferred,auto,1";
    "$mod" = "SUPER";
    bind = [
      "$mod, Return, exec, kitty"
      "$mod, Q, killactive"
      "$mod, Space, exec, wofi --show drun"
      "$mod, 1, workspace, 1"
      "$mod, 2, workspace, 2"
    ];
    general = {
      gaps_in = 5;
      gaps_out = 10;
      border_size = 2;
    };
  };
};
```

---

## Understanding the Nix Store

You might notice that packages aren't at `/usr/bin/firefox`. They're at places like:

```
/nix/store/abc123-firefox-128.0/bin/firefox
```

Each package is in its own directory, identified by a hash of all its inputs. This means:
- Two versions of the same package can coexist
- Packages can't interfere with each other
- Rollback is instant (just point symlinks to the old paths)
- Garbage collection removes packages that no generation references

```bash
# See what's in the store
nix path-info -rsSh /run/current-system | sort -k2 -h | tail -20

# Clean up old generations (keeps last 5)
sudo nix-collect-garbage --delete-older-than 7d
```

---

## Five Things To Do This Week

1. **Set up Home Manager** and move your git config into it
2. **Create a `flake.nix`** for one of your projects with `nix develop`
3. **Enable direnv** so project environments activate automatically
4. **Push your NixOS config to GitHub/Codeberg** — it's already a git repo
5. **Read one NixOS option** on search.nixos.org that you didn't know existed

---

## When To Use What

| Want to... | Do this |
|-----------|---------|
| Install an app permanently | Add to `configuration.nix`, rebuild |
| Try an app temporarily | `nix-shell -p appname` |
| Configure your shell/editor/git | Add to `home.nix` (Home Manager) |
| Set up a project environment | Create `flake.nix` in the project |
| Enable a system service | Add to `configuration.nix`, rebuild |
| Update all packages | `nix flake update` then rebuild |
| Roll back a bad change | `nixos-rebuild switch --rollback` or boot previous generation |

---

*Next: [Mastering Nix](03-mastering-nix.md) — Writing modules, overlays, and contributing to nixpkgs.*
