#!/usr/bin/env bash
BAD="/srv/luminous-dynamics/crates"
if [ -e "$BAD" ]; then
  echo "ERROR: unauthorized root crates directory exists: $BAD"
  exit 1
fi
