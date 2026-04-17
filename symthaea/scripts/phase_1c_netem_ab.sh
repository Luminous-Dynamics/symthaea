#!/usr/bin/env bash
# Phase I.C WS-vs-QUIC loss A/B via user+net namespace + tc netem.
#
# Runs the built-in `holon_transport_ab` example inside an unprivileged
# user+net namespace with `tc qdisc netem loss N%` applied to loopback,
# so the localhost socket traffic for both WS and QUIC traverses a
# lossy link. Exercises:
#
#   - QUIC unreliable datagrams: drop-and-continue (the design intent).
#   - WebSocket/TCP: retransmit with HoL blocking (the baseline).
#
# No sudo needed — Linux user namespaces give CAP_NET_ADMIN inside
# the new namespace while the outer user remains unprivileged.
#
# Usage:
#   scripts/phase_1c_netem_ab.sh            # 1% loss, 30 frames, default
#   LOSS=5 FRAMES=60 scripts/phase_1c_netem_ab.sh
#
# Required:
#   - Linux with user namespaces enabled
#     (`unshare --user --net --map-root-user /usr/bin/env true`)
#   - `tc` (iproute2) available in PATH
#   - holon_transport_ab built: `cargo build --features holon-viewer`

set -euo pipefail

LOSS="${LOSS:-1}"
FRAMES="${FRAMES:-30}"
BIN="${BIN:-target/debug/examples/holon_transport_ab}"

if [[ ! -x "$BIN" ]]; then
    echo "holon_transport_ab not found at $BIN" >&2
    echo "build first:" >&2
    echo "  cargo build --no-default-features --features holon-viewer --example holon_transport_ab" >&2
    exit 2
fi

# Probe the kernel once so the error message is clear if disabled.
if ! unshare --user --net --map-root-user -- /usr/bin/env true 2>/dev/null; then
    echo "unprivileged user+net namespaces blocked by kernel" >&2
    echo "check: sysctl kernel.unprivileged_userns_clone" >&2
    exit 3
fi

# Resolve to absolute path — unshare does NOT cd to cwd on all invocations.
BIN_ABS="$(readlink -f "$BIN")"

echo "==[ Phase I.C loss A/B: netem loss=${LOSS}% frames=${FRAMES} ]=="
unshare --user --net --map-root-user -- /usr/bin/env bash -c "
    set -euo pipefail
    ip link set lo up
    tc qdisc add dev lo root netem loss ${LOSS}%
    echo '---[ loopback qdisc ]---'
    tc qdisc show dev lo
    echo '---[ A/B run under loss ]---'
    '${BIN_ABS}' --frames=${FRAMES}
"
echo "==[ done ]=="
