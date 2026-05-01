#!/usr/bin/env bash
# Check that all .rs files in symtropy/crates/ have SPDX headers matching their Cargo.toml license.

set -e

FAILED=0

for d in symtropy/crates/*; do
  if [ -f "$d/Cargo.toml" ]; then
    pkg_license=$(grep '^license =' "$d/Cargo.toml" | cut -d'"' -f2)
    if [ -z "$pkg_license" ]; then
      echo "No license found in $d/Cargo.toml"
      continue
    fi

    # Check all .rs files in src, tests, examples, benches
    while read -r f; do
      if ! grep -q "SPDX-License-Identifier:" "$f"; then
        echo "Missing SPDX header: $f"
        FAILED=1
        continue
      fi

      spdx_header=$(grep "SPDX-License-Identifier:" "$f" | head -n 1 | cut -d':' -f2- | xargs)
      if [ "$pkg_license" != "$spdx_header" ]; then
        echo "License mismatch: $f (Cargo: $pkg_license, Header: $spdx_header)"
        FAILED=1
      fi
    done < <(find "$d" \( -path "$d/src" -o -path "$d/tests" -o -path "$d/examples" -o -path "$d/benches" \) -name "*.rs")
  fi
done

if [ $FAILED -ne 0 ]; then
  echo "License check failed."
  exit 1
fi

echo "License check passed."
