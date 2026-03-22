#!/usr/bin/env bash

# Enter nix-shell with all required libraries and set LD_LIBRARY_PATH
nix-shell -p \
  xorg.libX11 \
  xorg.libxcb \
  xorg.libXcomposite \
  xorg.libXcursor \
  xorg.libXdamage \
  xorg.libXext \
  xorg.libXfixes \
  xorg.libXi \
  xorg.libXrandr \
  xorg.libXrender \
  gtk3 \
  pango \
  atk \
  cairo \
  gdk-pixbuf \
  glib \
  alsa-lib \
  freetype \
  fontconfig \
  dbus \
  --run '
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$(nix-build --no-out-link "<nixpkgs>" -A xorg.libX11)/lib"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(nix-build --no-out-link "<nixpkgs>" -A gtk3)/lib"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(nix-build --no-out-link "<nixpkgs>" -A glib)/lib"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(nix-build --no-out-link "<nixpkgs>" -A xorg.libxcb)/lib"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(nix-build --no-out-link "<nixpkgs>" -A pango)/lib"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(nix-build --no-out-link "<nixpkgs>" -A cairo)/lib"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(nix-build --no-out-link "<nixpkgs>" -A atk)/lib"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(nix-build --no-out-link "<nixpkgs>" -A alsa-lib)/lib"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(nix-build --no-out-link "<nixpkgs>" -A dbus)/lib"

    echo "LD_LIBRARY_PATH set, launching Firefox..."
    node screenshot-firefox.js
  '