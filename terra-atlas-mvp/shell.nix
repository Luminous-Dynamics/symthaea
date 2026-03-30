# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    nodejs_20
    nodePackages.npm
    nodePackages.typescript

    # PostgreSQL with PostGIS
    postgresql_15
    postgresql15Packages.postgis

    # Database management tools
    pgcli

    # Prisma engines for NixOS
    prisma-engines
    nodePackages.prisma

    # Python for scripts
    python3

    # Git for version control
    git

    # Environment tools
    direnv

    # Process manager
    overmind
    tmux

    # Playwright browser dependencies
    glib
    nss
    nspr
    atk
    at-spi2-atk
    cups
    dbus
    expat
    libdrm
    xorg.libX11
    xorg.libXcomposite
    xorg.libXdamage
    xorg.libXext
    xorg.libXfixes
    xorg.libXrandr
    xorg.libxcb
    mesa
    libxkbcommon
    pango
    cairo
    alsa-lib
    systemd
  ];

  shellHook = ''
    echo "🌍 Terra Atlas Development Environment"
    echo "=================================="
    echo "PostgreSQL: $(postgres --version)"
    echo "Node.js: $(node --version)"
    echo "npm: $(npm --version)"
    echo ""

    # Set up LD_LIBRARY_PATH for Playwright browsers
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.glib
      pkgs.nss
      pkgs.nspr
      pkgs.atk
      pkgs.at-spi2-atk
      pkgs.cups
      pkgs.dbus
      pkgs.expat
      pkgs.libdrm
      pkgs.xorg.libX11
      pkgs.xorg.libXcomposite
      pkgs.xorg.libXdamage
      pkgs.xorg.libXext
      pkgs.xorg.libXfixes
      pkgs.xorg.libXrandr
      pkgs.xorg.libxcb
      pkgs.mesa
      pkgs.libxkbcommon
      pkgs.pango
      pkgs.cairo
      pkgs.alsa-lib
      pkgs.systemd
    ]}:$LD_LIBRARY_PATH"
    echo "✅ Playwright browser dependencies loaded"

    echo ""
    echo "📊 Database Commands:"
    echo "  npm run db:setup              # Set up database"
    echo ""
    echo "🎭 Visual Testing:"
    echo "  npx playwright install chromium  # Install browser"
    echo "  npx playwright test              # Run tests"
    echo ""
    echo "🚀 Development:"
    echo "  npm run dev                   # Start development server"
    echo ""
    
    # Set up PostgreSQL data directory if it doesn't exist
    export PGDATA="$PWD/postgres-data"
    export PGHOST="$PWD/postgres-data"
    export PGDATABASE="terra_atlas"
    export PGUSER="$USER"
    
    # Fix Prisma on NixOS by providing engine paths
    export PRISMA_QUERY_ENGINE_LIBRARY="${pkgs.prisma-engines}/lib/libquery_engine.node"
    export PRISMA_QUERY_ENGINE_BINARY="${pkgs.prisma-engines}/bin/query-engine"
    export PRISMA_SCHEMA_ENGINE_BINARY="${pkgs.prisma-engines}/bin/schema-engine"
    export PRISMA_MIGRATION_ENGINE_BINARY="${pkgs.prisma-engines}/bin/migration-engine"
    export PRISMA_INTROSPECTION_ENGINE_BINARY="${pkgs.prisma-engines}/bin/introspection-engine"
    export PRISMA_FMT_BINARY="${pkgs.prisma-engines}/bin/prisma-fmt"
    
    if [ ! -d "$PGDATA" ]; then
      echo "📦 Initializing PostgreSQL data directory..."
      initdb -D "$PGDATA" --auth=trust --encoding=UTF8
      
      # Start PostgreSQL
      pg_ctl -D "$PGDATA" -l "$PWD/postgres.log" start
      
      # Wait for PostgreSQL to start
      sleep 2
      
      # Create database
      createdb terra_atlas
      
      echo "✅ PostgreSQL initialized and started!"
    else
      # Check if PostgreSQL is running
      if ! pg_ctl -D "$PGDATA" status > /dev/null 2>&1; then
        echo "🔄 Starting PostgreSQL..."
        pg_ctl -D "$PGDATA" -l "$PWD/postgres.log" start
        sleep 2
      fi
      echo "✅ PostgreSQL is ready!"
    fi
    
    # Export DATABASE_URL for Prisma
    export DATABASE_URL="postgresql://$USER@localhost/terra_atlas?host=$PGDATA"
    echo "📝 DATABASE_URL set for local development"
  '';
}