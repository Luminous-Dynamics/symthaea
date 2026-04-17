#!/usr/bin/env bash
# Filter miniF2F-v2 problems to those plausibly in Phase 2 Option A scope.
# OUT-OF-SCOPE patterns (any match → reject):
#   - Real.sqrt, Real.sin/cos/tan, Real.log, Real.exp, Real.pi
#   - Nat.Prime, Nat.gcd, Nat.choose, Nat.fib, Nat.factorial, Nat.digits
#   - Finset, ∑, ∏, ∃!
#   - Function, fun, List., String, Matrix
#   - abs  (too many forms)
#
# IN-SCOPE target: theorem with only polynomial / rational arithmetic.
set -euo pipefail

root="${1:-/srv/luminous-dynamics/symthaea/data/benchmarks/minif2f/MiniF2F}"

OUT_OF_SCOPE='Real\.sqrt|Real\.sin|Real\.cos|Real\.tan|Real\.log|Real\.exp|Real\.pi|Nat\.Prime|Nat\.gcd|Nat\.choose|Nat\.fib|Nat\.factorial|Nat\.digits|Finset|∑|∏|∃!|Function|\bfun |List\.|String|Matrix|\babs |maxOn|Int\.floor|Nat\.log|Real\.sqrt'

for d in Valid Test; do
  for f in "$root/$d"/mathd_algebra_*.lean "$root/$d"/mathd_numbertheory_*.lean; do
    [ -f "$f" ] || continue
    if grep -qE "$OUT_OF_SCOPE" "$f"; then
      continue
    fi
    echo "$f"
  done
done
