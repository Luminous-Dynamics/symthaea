#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "\${TLA2TOOLS_JAR:?Set TLA2TOOLS_JAR to a locally verified tla2tools.jar (v1.7.4)}"
[[ -f "$TLA2TOOLS_JAR" ]] || { echo "TLA2TOOLS_JAR does not exist: $TLA2TOOLS_JAR" >&2; exit 2; }
expected="bee4a54f3ee3d4afc347c3240ec2d9e93b075104"
actual="$(sha1sum "$TLA2TOOLS_JAR" | awk '{print $1}')"
[[ "$actual" == "$expected" ]] || {
  echo "wrong tla2tools.jar SHA-1: expected $expected, got $actual" >&2
  exit 2
}
java -XX:+UseParallelGC -cp "$TLA2TOOLS_JAR" tlc2.TLC \
  -workers 1 \
  -config MACEEffectFence.cfg \
  MACEEffectFence.tla
