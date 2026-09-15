#!/usr/bin/env bash
set -euo pipefail

QUALIFIED_002A_SHA="eb73527d05a913e79d1f05135ad6b06c1da8e2ee"
QUALIFIED_CENSUS_BYTES="1830977"
QUALIFIED_CENSUS_SHA256="49ff56a49ac730d960d7625b65ddb0139171fa3862d145a6deb3d68245c7e711"
TOOL_DIR="tools/paradox_002a_independent"

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"
subject_sha="$(git rev-parse HEAD)"
parent_sha="$(git rev-parse HEAD^)"

[[ "${parent_sha}" == "${QUALIFIED_002A_SHA}" ]]
[[ "$(git rev-list --count "${QUALIFIED_002A_SHA}..${subject_sha}")" == "1" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]

unexpected="$(git diff --name-only "${QUALIFIED_002A_SHA}" "${subject_sha}" -- | grep -Ev '^tools/paradox_002a_independent/' || true)"
if [[ -n "${unexpected}" ]]; then
  printf 'unexpected G2b diff surface:\n%s\n' "${unexpected}" >&2
  exit 1
fi

for required in PROTOCOL.md format.py theorem.py verify.py qualify.sh; do
  test -f "${TOOL_DIR}/${required}"
done

# Independence source audit applies to Python theorem/parser/driver only.
python_sources=("${TOOL_DIR}/format.py" "${TOOL_DIR}/theorem.py" "${TOOL_DIR}/verify.py")
! grep -Eqi 'qualify_fixture|subprocess|ctypes|cffi|rustimport|pyo3|maturin|cargo|rustc' "${python_sources[@]}"

# Syntax-check without producing .pyc files in the evidence worktree.
python3 - "${python_sources[@]}" <<'PY'
from pathlib import Path
import sys
for name in sys.argv[1:]:
    compile(Path(name).read_text(encoding="utf-8"), name, "exec")
PY

# The exact qualified emitter is the only Rust program permitted to construct
# the input corpus. Use a detached worktree so verifier bytes cannot affect it.
tmp="$(mktemp -d)"
qualified_wt="${tmp}/qualified-002a"
cleanup() {
  git worktree remove --force "${qualified_wt}" >/dev/null 2>&1 || true
  rm -rf "${tmp}"
}
trap cleanup EXIT

git worktree add --detach "${qualified_wt}" "${QUALIFIED_002A_SHA}" >/dev/null
manifest="${qualified_wt}/crates/research/symthaea-paradox-fixtures/Cargo.toml"
census="${tmp}/census.bin"

rustc +1.96.0 --version | grep -Fq 'rustc 1.96.0'
cargo +1.96.0 run --locked --offline --quiet --manifest-path "${manifest}" --bin qualification_census > "${census}"

bytes="$(wc -c < "${census}" | tr -d '[:space:]')"
digest="$(sha256sum "${census}" | awk '{print $1}')"
if [[ "${bytes}" != "${QUALIFIED_CENSUS_BYTES}" || "${digest}" != "${QUALIFIED_CENSUS_SHA256}" ]]; then
  printf 'QualifiedCorpusReproductionMismatch bytes=%s sha256=%s\n' "${bytes}" "${digest}" >&2
  exit 1
fi
printf 'qualified corpus reproduced bytes=%s sha256=%s\n' "${bytes}" "${digest}"

receipt_a="${tmp}/receipt-a.json"
receipt_b="${tmp}/receipt-b.json"
(
  cd "${TOOL_DIR}"
  PYTHONDONTWRITEBYTECODE=1 python3 verify.py "${census}" --verifier-sha "${subject_sha}" --receipt "${receipt_a}" >/dev/null
)
(
  cd "${TOOL_DIR}"
  PYTHONDONTWRITEBYTECODE=1 python3 verify.py "${census}" --verifier-sha "${subject_sha}" --receipt "${receipt_b}" >/dev/null
)
cmp -s "${receipt_a}" "${receipt_b}"
cat "${receipt_a}"

# Postflight: qualification must not mutate the exact verifier subject.
[[ "$(git rev-parse HEAD)" == "${subject_sha}" ]]
[[ "$(git rev-parse HEAD^)" == "${QUALIFIED_002A_SHA}" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
