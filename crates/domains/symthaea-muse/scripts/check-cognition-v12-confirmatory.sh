#!/usr/bin/env bash
set -euo pipefail
root="${1:-.}"
required=(
  src/confirmatory_collection_protocol.rs
  src/confirmatory_collection_monitor.rs
  src/confirmatory_cohort_registry.rs
  src/confirmatory_collection_close.rs
  src/confirmatory_unblinding.rs
  src/confirmatory_analysis_execution.rs
  src/confirmatory_publication.rs
  src/post_publication_audit.rs
  src/confirmatory_final_release.rs
  CONFIRMATORY_EXECUTION_AND_PUBLICATION.md
  scripts/verify_cognition_study_v12.py
)
for path in "${required[@]}"; do
  test -f "$root/$path" || { echo "missing $path" >&2; exit 1; }
done
grep -q 'symthaea-muse-study-orchestration-v3' "$root/src/study_orchestration.rs"
grep -q 'ConfirmatoryCollectionCloseReceipt' "$root/src/study_orchestration.rs"
grep -q 'ConfirmatoryPublicationRecord' "$root/src/study_orchestration.rs"
python3 "$root/scripts/verify_cognition_study_v12.py" --self-test
printf 'V12 confirmatory execution static checks passed\n'
