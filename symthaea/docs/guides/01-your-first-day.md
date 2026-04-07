# Your First Day on NixOS

*A gentle guide for people who just installed NixOS and aren't sure what to do next.*

---

## The Most Important Thing

Your system is defined by a single file: `/etc/nixos/configuration.nix` (or a flake, if Sovereign Inoculation set you up with one).

Every change to your system — installing software, enabling services, configuring networking — happens by editing this file and running one command:

```bash
sudo nixos-rebuild switch
```

That's it. That's the whole operating system in one sentence. Edit. Rebuild. Done.

---

## Your Safety Net

Before you do anything else, know this: **you cannot break your system permanently.**

Every time you run `nixos-rebuild switch`, NixOS creates a new "generation." Your boot menu lists every previous generation. If something goes wrong:

1. Reboot
2. Select the previous generation from the boot menu
3. You're back to exactly where you were

This is like having automatic backups of your entire operating system, built into the OS itself. No other system does this.

---

## Installing Software

On other Linux distros, you'd run `apt install` or `pacman -S`. On NixOS, you add the package to your configuration:

```nix
# Open your config
sudo nano /etc/nixos/configuration.nix

# Find the systemPackages line and add your package:
environment.systemPackages = with pkgs; [
  vim
  git
  firefox    # ← add packages here
  spotify
  discord
];

# Save, then apply:
sudo nixos-rebuild switch
```

**"But that's slower than apt install!"** — Yes, by about 30 seconds. In exchange, you get:
- A list of every package on your system, in one place
- The ability to reproduce your exact setup on another machine
- The ability to roll back if a package breaks something

You can also use `nix-shell` to try a package without installing it permanently:

```bash
nix-shell -p cowsay
cowsay "I'm trying NixOS!"
# cowsay is available in this shell only
# exit the shell and it's gone — no cleanup needed
```

---

## Enabling Services

Services (like SSH, a web server, or Bluetooth) are enabled the same way — in your config file:

```nix
# Enable Bluetooth
hardware.bluetooth.enable = true;
hardware.bluetooth.powerOnBoot = true;

# Enable printing
services.printing.enable = true;

# Enable Tailscale VPN
services.tailscale.enable = true;
```

Then `sudo nixos-rebuild switch`. The service starts automatically.

**Finding options:** Every NixOS option is documented at [search.nixos.org/options](https://search.nixos.org/options). Search for what you want — there's probably an option for it.

---

## The Flake (If You Have One)

If Sovereign Inoculation set you up with a flake (a `flake.nix` file), your config is version-controlled and reproducible. Here's what each file does:

```
/etc/nixos/
  flake.nix              # The "recipe" — what inputs (nixpkgs, home-manager) to use
  flake.lock             # Pinned versions — ensures reproducibility
  configuration.nix      # Your system configuration
  hardware-configuration.nix  # Auto-detected hardware (don't edit manually)
```

To update your system (get new packages from nixpkgs):

```bash
cd /etc/nixos
sudo nix flake update    # updates flake.lock with latest versions
sudo nixos-rebuild switch --flake .
```

---

## Five Things To Do Today

1. **Add your favorite packages** to `configuration.nix` and rebuild
2. **Try `nix-shell -p`** with a package you're curious about
3. **Search [search.nixos.org](https://search.nixos.org)** for a package or option you need
4. **Look at your boot menu** (reboot and watch) — see the generations listed
5. **Edit one thing** and rebuild — feel the confidence of knowing you can always roll back

---

## When Something Goes Wrong

**"nixos-rebuild failed with an error"**
- Read the error message. It usually tells you exactly what's wrong.
- Most common: a typo in configuration.nix, a missing semicolon, or a package name that doesn't exist.
- Your system is unchanged — the failed rebuild didn't apply anything.

**"I installed a package but it's not in my PATH"**
- Did you log out and back in? Some packages need a new shell session.
- For GUI apps, check your application menu.

**"Something broke after a rebuild"**
- Reboot, select the previous generation. You're safe.
- Then figure out what you changed and fix it.

**"I want to go back to how things were"**
```bash
sudo nixos-rebuild switch --rollback
```

---

## The Philosophy

NixOS is different because it treats your system as **a document, not a state.**

Other OSes accumulate changes over time — install this, remove that, tweak a config file, run a script. Eventually, nobody knows exactly what's installed or why. The system becomes a snowflake.

NixOS keeps your entire system in one file. You can read it, share it, reproduce it, and roll it back. Your system is a **document** that describes a machine, not a machine that accumulated changes.

This takes some getting used to. But once it clicks, you'll never want to go back.

---

*Next: [Your First Week](02-your-first-week.md) — Home Manager, development environments, and making NixOS truly yours.*
