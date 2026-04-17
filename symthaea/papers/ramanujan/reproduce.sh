#!/usr/bin/env bash
#
# reproduce.sh — regenerate the Ramanujan Protocol showcase end-to-end.
#
# Produces:
#   - papers/ramanujan/showcase_stdout.txt  (full pipeline trace)
#   - papers/ramanujan/showcase_stderr.txt  (compiler warnings, timings)
#   - papers/ramanujan/results_table.tex   (the LaTeX table ready to paste
#                                           into main.tex)
#
# Requirements:
#   - Rust stable (tested with 1.75+)
#   - Z3 >= 4.13 on PATH (optional — absence downgrades PROVEN rows to Numeric)
#
# Determinism:
#   The showcase pins seed = 42. Same host + same Z3 version → bit-identical
#   output. Cross-host Z3 version drift can flip a row from PROVEN to Numeric
#   (never the other way — see papers/ramanujan/VERIFY.md).
#
# Usage:
#   ./reproduce.sh                    # regenerate everything
#   ./reproduce.sh --verify-proofs    # also run every committed SMT-LIB2
#                                       witness through Z3 independently

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PAPER_DIR}/../.." && pwd)"

STDOUT_FILE="${PAPER_DIR}/showcase_stdout.txt"
STDERR_FILE="${PAPER_DIR}/showcase_stderr.txt"
TABLE_FILE="${PAPER_DIR}/results_table.tex"

echo "[reproduce] repo root: ${REPO_ROOT}"
echo "[reproduce] paper dir: ${PAPER_DIR}"
echo "[reproduce] running ramanujan_showcase (release profile)..."

cd "${REPO_ROOT}"

# Primary run.
cargo run --release \
  -p symthaea-physics-bridge \
  --example ramanujan_showcase \
  > "${STDOUT_FILE}" \
  2> "${STDERR_FILE}"

echo "[reproduce] showcase complete."
echo "[reproduce]   stdout: ${STDOUT_FILE}"
echo "[reproduce]   stderr: ${STDERR_FILE}"

# Extract the LaTeX table block (between "\\begin{table}" and "\\end{table}"),
# sanitizing Unicode that pdflatex rejects (↳ → `--`). XeLaTeX/LuaLaTeX would
# accept it natively, but we target pdflatex for simpler CI.
awk '
  /^\\begin\{table\}/ { in_table = 1 }
  in_table { print }
  /^\\end\{table\}/ && in_table { in_table = 0; print ""; exit }
' "${STDOUT_FILE}" \
  | sed 's/↳ MATCHES/-- matches/g; s/↳ resembles/-- resembles/g; s/↳ weakly/-- weakly/g; s/↳ NO/-- no/g' \
  > "${TABLE_FILE}"

echo "[reproduce]   table:  ${TABLE_FILE}"

# Optional: regenerate + re-verify SMT witnesses if --verify-proofs passed.
if [[ "${1:-}" == "--verify-proofs" ]]; then
  if ! command -v z3 > /dev/null 2>&1; then
    echo "[reproduce] z3 not on PATH; skipping independent verification." >&2
    exit 0
  fi

  # Regenerate the SMT-LIB2 witnesses from source. This guarantees the
  # committed files match what our code currently says, not a stale copy.
  echo "[reproduce] regenerating SMT-LIB2 witnesses via verify_invariants_formal..."
  cargo run --release \
    -p symthaea-physics-bridge \
    --example verify_invariants_formal \
    > "${PAPER_DIR}/verify_invariants_results.csv" \
    2>> "${STDERR_FILE}"

  echo "[reproduce]   witness CSV: ${PAPER_DIR}/verify_invariants_results.csv"

  # Independent re-verification with whichever Z3 is on the reader's PATH.
  echo "[reproduce] re-verifying committed SMT-LIB2 witnesses..."
  proofs_dir="${PAPER_DIR}/proofs"
  if [[ ! -d "${proofs_dir}" ]]; then
    echo "[reproduce] no committed proofs at ${proofs_dir}." >&2
    exit 0
  fi
  all_ok=1
  for proof in "${proofs_dir}"/*.smt2; do
    [[ -f "${proof}" ]] || continue
    result="$(z3 -smt2 "${proof}" 2>&1 | tail -1 || true)"
    case "${result}" in
      unsat) echo "  OK    $(basename "${proof}")" ;;
      *)     echo "  FAIL  $(basename "${proof}"): ${result}" >&2; all_ok=0 ;;
    esac
  done
  if [[ "${all_ok}" -eq 0 ]]; then
    exit 2
  fi
  echo "[reproduce] all SMT witnesses re-verified as unsat."
fi

echo "[reproduce] done."
