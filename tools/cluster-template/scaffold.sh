#!/usr/bin/env bash
# scaffold.sh — Generate a new Mycelix cluster frontend from template
#
# Usage:
#   ./scaffold.sh --cluster climate --display "Climate" --port 8103 --accent "#22c55e" --icon "🌍"
#
# Creates:
#   mycelix-<cluster>/apps/leptos/     — Leptos CSR frontend
#   mycelix-<cluster>/apps/tauri/      — Tauri v2 desktop wrapper
#   mycelix-<cluster>/crates/<cluster>-leptos-types/ — WASM-safe types

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/template"

# --- Parse arguments ---
CLUSTER=""
DISPLAY=""
PORT=""
ACCENT="#0ed4a8"
ICON="🌿"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)  CLUSTER="$2";  shift 2 ;;
        --display)  DISPLAY="$2";  shift 2 ;;
        --port)     PORT="$2";     shift 2 ;;
        --accent)   ACCENT="$2";   shift 2 ;;
        --icon)     ICON="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CLUSTER" || -z "$DISPLAY" || -z "$PORT" ]]; then
    echo "Usage: $0 --cluster <name> --display <name> --port <81XX> [--accent <color>] [--icon <emoji>]"
    echo ""
    echo "Example:"
    echo "  $0 --cluster climate --display \"Climate\" --port 8103 --accent \"#22c55e\" --icon \"🌍\""
    exit 1
fi

# Derived names
PKG="mycelix-${CLUSTER}"             # mycelix-climate
PASCAL="$(echo "$CLUSTER" | sed -r 's/(^|_)([a-z])/\U\2/g')"  # Climate
TARGET_DIR="$REPO_ROOT/$PKG"

echo "=== Scaffolding $DISPLAY cluster frontend ==="
echo "  Cluster:  $CLUSTER"
echo "  Package:  $PKG"
echo "  Pascal:   $PASCAL"
echo "  Port:     $PORT"
echo "  Accent:   $ACCENT"
echo "  Icon:     $ICON"
echo "  Target:   $TARGET_DIR"
echo ""

# --- Check target doesn't already have a Leptos app ---
if [[ -d "$TARGET_DIR/apps/leptos/src" ]]; then
    echo "ERROR: $TARGET_DIR/apps/leptos/src already exists!"
    echo "Remove it first or choose a different cluster name."
    exit 1
fi

# --- Copy and process Leptos frontend ---
echo "Creating Leptos frontend..."
mkdir -p "$TARGET_DIR/apps/leptos/src/pages"
mkdir -p "$TARGET_DIR/apps/leptos/style"
mkdir -p "$TARGET_DIR/apps/leptos/static"

for tmpl in $(find "$TEMPLATE_DIR/apps/leptos" -name "*.tmpl" -type f); do
    # Compute relative path within template
    rel="${tmpl#$TEMPLATE_DIR/apps/leptos/}"
    dest="$TARGET_DIR/apps/leptos/${rel%.tmpl}"
    mkdir -p "$(dirname "$dest")"
    sed \
        -e "s|{{cluster_name}}|${CLUSTER}|g" \
        -e "s|{{cluster_display}}|${DISPLAY}|g" \
        -e "s|{{cluster_pkg}}|${PKG}|g" \
        -e "s|{{cluster_pascal}}|${PASCAL}|g" \
        -e "s|{{cluster_icon}}|${ICON}|g" \
        -e "s|{{trunk_port}}|${PORT}|g" \
        -e "s|{{accent_color}}|${ACCENT}|g" \
        "$tmpl" > "$dest"
done

# Copy non-template static files (icons, etc.)
find "$TEMPLATE_DIR/apps/leptos/static" -type f ! -name "*.tmpl" -exec cp {} "$TARGET_DIR/apps/leptos/static/" \; 2>/dev/null

# --- Copy and process Tauri desktop ---
echo "Creating Tauri desktop wrapper..."
mkdir -p "$TARGET_DIR/apps/tauri/src-tauri/src"
mkdir -p "$TARGET_DIR/apps/tauri/src-tauri/capabilities"

for tmpl in $(find "$TEMPLATE_DIR/apps/tauri" -name "*.tmpl" -type f); do
    rel="${tmpl#$TEMPLATE_DIR/apps/tauri/}"
    dest="$TARGET_DIR/apps/tauri/${rel%.tmpl}"
    mkdir -p "$(dirname "$dest")"
    sed \
        -e "s|{{cluster_name}}|${CLUSTER}|g" \
        -e "s|{{cluster_display}}|${DISPLAY}|g" \
        -e "s|{{cluster_pkg}}|${PKG}|g" \
        -e "s|{{cluster_pascal}}|${PASCAL}|g" \
        -e "s|{{cluster_icon}}|${ICON}|g" \
        -e "s|{{trunk_port}}|${PORT}|g" \
        -e "s|{{accent_color}}|${ACCENT}|g" \
        "$tmpl" > "$dest"
done

# Copy non-template files
cp "$TEMPLATE_DIR/apps/tauri/src-tauri/capabilities/default.json" \
   "$TARGET_DIR/apps/tauri/src-tauri/capabilities/default.json"

# --- Copy and process types crate (skip if exists) ---
TYPES_DIR="$TARGET_DIR/crates/${CLUSTER}-leptos-types"
if [[ -f "$TYPES_DIR/src/lib.rs" ]]; then
    echo "Types crate already exists at $TYPES_DIR — skipping."
else
    echo "Creating types crate..."
mkdir -p "$TYPES_DIR/src"

for tmpl in $(find "$TEMPLATE_DIR/crates/CLUSTER-leptos-types" -name "*.tmpl" -type f); do
    rel="${tmpl#$TEMPLATE_DIR/crates/CLUSTER-leptos-types/}"
    dest="$TYPES_DIR/${rel%.tmpl}"
    mkdir -p "$(dirname "$dest")"
    sed \
        -e "s|{{cluster_name}}|${CLUSTER}|g" \
        -e "s|{{cluster_display}}|${DISPLAY}|g" \
        -e "s|{{cluster_pkg}}|${PKG}|g" \
        -e "s|{{cluster_pascal}}|${PASCAL}|g" \
        "$tmpl" > "$dest"
done
fi

# --- Done ---
echo ""
echo "=== Scaffold complete! ==="
echo ""
echo "Next steps:"
echo "  1. cd $TARGET_DIR/apps/leptos"
echo "  2. trunk serve                    # Dev server on port $PORT"
echo "  3. trunk build --release          # Production WASM build"
echo ""
echo "  For desktop:"
echo "  1. cd $TARGET_DIR/apps/tauri"
echo "  2. cargo tauri dev                # Desktop dev mode"
echo ""
echo "  Add domain types in:"
echo "  $TYPES_DIR/src/lib.rs"
echo ""
echo "  Add pages in:"
echo "  $TARGET_DIR/apps/leptos/src/pages/"
