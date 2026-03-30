#!/usr/bin/env bash
# Bundle separate CSS/JS files into a single portal.html for deployment
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read the HTML shell
SHELL_HTML="$SCRIPT_DIR/portal-shell.html"
if [ ! -f "$SHELL_HTML" ]; then
  echo "ERROR: portal-shell.html not found" >&2
  exit 1
fi

# Read CSS
CSS_FILE="$SCRIPT_DIR/css/portal.css"
if [ ! -f "$CSS_FILE" ]; then
  echo "ERROR: css/portal.css not found" >&2
  exit 1
fi
CSS_CONTENT=$(cat "$CSS_FILE")

# Read JS files in order
JS_CONTENT=""
for f in app.js charts.js tab-chat.js tab-topology.js tab-experiments.js tab-dreams.js tab-inoculate.js constellation.js system-card.js consciousness-music.js ceremony.js; do
  JS_FILE="$SCRIPT_DIR/js/$f"
  if [ ! -f "$JS_FILE" ]; then
    echo "ERROR: js/$f not found" >&2
    exit 1
  fi
  JS_CONTENT+="$(cat "$JS_FILE")"
  JS_CONTENT+=$'\n'
done

# Use awk for reliable placeholder substitution (handles special chars in content)
# Write CSS and JS to temp files, then use awk to splice them in
CSS_TMP=$(mktemp)
JS_TMP=$(mktemp)
trap "rm -f '$CSS_TMP' '$JS_TMP'" EXIT

echo "$CSS_CONTENT" > "$CSS_TMP"
echo "$JS_CONTENT" > "$JS_TMP"

awk -v css_file="$CSS_TMP" -v js_file="$JS_TMP" '
  /__CSS_PLACEHOLDER__/ {
    while ((getline line < css_file) > 0) print line
    close(css_file)
    next
  }
  /__JS_PLACEHOLDER__/ {
    while ((getline line < js_file) > 0) print line
    close(js_file)
    next
  }
  { print }
' "$SHELL_HTML" > "$SCRIPT_DIR/portal.html"

BYTES=$(wc -c < "$SCRIPT_DIR/portal.html")
echo "Built portal.html ($BYTES bytes)"
