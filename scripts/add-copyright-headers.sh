#!/usr/bin/env bash
# add-copyright-headers.sh — Insert AGPL-3.0 copyright header into all source files
#
# Usage: ./scripts/add-copyright-headers.sh [--dry-run]
#
# Inserts a copyright header at the top of .rs, .ts, .js, .py, .svelte, .nix files
# that don't already have one. Skips vendored, generated, and third-party files.

set -uo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No files will be modified."
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Copyright headers by language
read -r -d '' RUST_HEADER << 'HEOF' || true
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

HEOF

read -r -d '' PY_HEADER << 'HEOF' || true
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

HEOF

read -r -d '' SVELTE_HEADER << 'HEOF' || true
<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->

HEOF

count=0
skipped=0

add_header() {
    local file="$1"
    local header="$2"

    # Skip if file already has SPDX identifier
    if head -5 "$file" 2>/dev/null | grep -q "SPDX-License-Identifier"; then
        skipped=$((skipped + 1))
        return 0
    fi

    # Skip if file already has our copyright
    if head -5 "$file" 2>/dev/null | grep -q "Luminous Dynamics"; then
        skipped=$((skipped + 1))
        return 0
    fi

    # Skip empty files
    if [[ ! -s "$file" ]]; then
        return 0
    fi

    # Skip binary files (by extension — the find patterns already filter to source files)

    if $DRY_RUN; then
        count=$((count + 1))
        return 0
    fi

    # Handle shebang lines — preserve them at top
    if head -1 "$file" | grep -q '^#!'; then
        local tmpfile
        tmpfile=$(mktemp)
        head -1 "$file" > "$tmpfile"
        echo "" >> "$tmpfile"
        printf '%s\n' "$header" >> "$tmpfile"
        tail -n +2 "$file" >> "$tmpfile"
        mv "$tmpfile" "$file"
    else
        local tmpfile
        tmpfile=$(mktemp)
        printf '%s\n' "$header" > "$tmpfile"
        cat "$file" >> "$tmpfile"
        mv "$tmpfile" "$file"
    fi

    count=$((count + 1))
}

echo "Scanning for source files..."

# Process files by type
process_files() {
    local pattern="$1"
    local header="$2"
    local label="$3"

    count=0
    skipped=0

    while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        [[ ! -w "$file" ]] && continue
        add_header "$file" "$header"
    done < <(find "$REPO_ROOT" -name "$pattern" \
        -not -path "*/target/*" \
        -not -path "*/.git/*" \
        -not -path "*/.cargo/*" \
        -not -path "*/node_modules/*" \
        -not -path "*/.claude/*" \
        -not -path "*/.venv/*" \
        -not -path "*/.next/*" \
        -not -path "*/vendor/*" \
        -not -path "*/third_party/*" \
        -not -path "*/lib/openzeppelin*" \
        -not -path "*/lib/murky/*" \
        -not -path "*/.gradle/*" \
        -not -path "*/dist/*" \
        -not -path "*/build/*" \
        -not -path "*/pkg/*" \
        -not -path "*stale*" \
        -not -path "*/phi-lab/*" \
        -not -path "*site-packages*" \
        -not -path "*/venv/*" \
        -not -path "*__pycache__*" \
        -type f 2>/dev/null)

    echo "  $label: $count updated, $skipped already had header"
}

# Rust
process_files "*.rs" "$RUST_HEADER" "Rust (.rs)"
rs_count=$count

# TypeScript
process_files "*.ts" "$RUST_HEADER" "TypeScript (.ts)"
ts_count=$count

# TSX
process_files "*.tsx" "$RUST_HEADER" "TSX (.tsx)"
tsx_count=$count

# JavaScript
process_files "*.js" "$RUST_HEADER" "JavaScript (.js)"
js_count=$count

# JSX
process_files "*.jsx" "$RUST_HEADER" "JSX (.jsx)"
jsx_count=$count

# Python
process_files "*.py" "$PY_HEADER" "Python (.py)"
py_count=$count

# Svelte
process_files "*.svelte" "$SVELTE_HEADER" "Svelte (.svelte)"
svelte_count=$count

# Nix
process_files "*.nix" "$PY_HEADER" "Nix (.nix)"
nix_count=$count

total=$((rs_count + ts_count + tsx_count + js_count + jsx_count + py_count + svelte_count + nix_count))
echo ""
echo "Total files updated: $total"
if $DRY_RUN; then
    echo "[DRY RUN] No files were actually modified."
fi
