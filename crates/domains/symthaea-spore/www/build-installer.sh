#!/usr/bin/env bash
# Build installer-only portal (no consciousness demo tabs)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read the installer HTML shell
SHELL_HTML="$SCRIPT_DIR/installer-shell.html"
if [ ! -f "$SHELL_HTML" ]; then
  echo "ERROR: installer-shell.html not found" >&2
  exit 1
fi

# Read installer CSS
CSS_FILE="$SCRIPT_DIR/css/installer.css"
CSS_CONTENT=$(cat "$CSS_FILE")

# Read installer JS files only (no consciousness demo)
JS_CONTENT=""
for f in tab-inoculate.js constellation.js system-card.js ceremony.js; do
  JS_FILE="$SCRIPT_DIR/js/$f"
  if [ ! -f "$JS_FILE" ]; then
    echo "WARNING: js/$f not found, skipping" >&2
    continue
  fi
  JS_CONTENT+="$(cat "$JS_FILE")"
  JS_CONTENT+=$'\n'
done

# Substitute placeholders
CSS_TMP=$(mktemp)
JS_TMP=$(mktemp)
trap "rm -f '$CSS_TMP' '$JS_TMP'" EXIT

echo "$CSS_CONTENT" > "$CSS_TMP"
echo "$JS_CONTENT" > "$JS_TMP"

awk -v css_file="$CSS_TMP" -v js_file="$JS_TMP" '
  /__INSTALLER_CSS_PLACEHOLDER__/ {
    while ((getline line < css_file) > 0) print line
    close(css_file)
    next
  }
  /__INSTALLER_JS_PLACEHOLDER__/ {
    while ((getline line < js_file) > 0) print line
    close(js_file)
    next
  }
  { print }
' "$SHELL_HTML" > "$SCRIPT_DIR/installer.html"

BYTES=$(wc -c < "$SCRIPT_DIR/installer.html")
echo "Built installer.html ($BYTES bytes)"
