#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat >"$TMP/lsn-one.json" <<'JSON'
{"offsets":[{"partition":{"server":"mon025","database":"mon025"},"offset":{"lsn":101,"lsn_proc":99,"lsn_commit":101,"messageType":"INSERT","lsn_events_processed":1,"txId":42}}]}
JSON
cat >"$TMP/message.json" <<'JSON'
{"offsets":[{"partition":{"server":"mon025","database":"mon025"},"offset":{"lsn":120,"lsn_proc":119,"lsn_commit":120,"messageType":"MESSAGE","lsn_events_processed":2,"txId":43}}]}
JSON
cat >"$TMP/wrong-count.json" <<'JSON'
{"offsets":[{"partition":{"server":"mon025","database":"mon025"},"offset":{"lsn":101,"lsn_proc":99,"lsn_commit":101,"messageType":"INSERT","lsn_events_processed":0,"txId":42}}]}
JSON
cat >"$TMP/wrong-type.json" <<'JSON'
{"offsets":[{"partition":{"server":"mon025","database":"mon025"},"offset":{"lsn":101,"lsn_proc":99,"lsn_commit":101,"messageType":"MESSAGE","lsn_events_processed":1,"txId":42}}]}
JSON
bash "$ROOT/scripts/verify-offset-target.sh" "$TMP/lsn-one.json" lsn_events_processed_1 "$TMP/lsn-one-proof.json"
bash "$ROOT/scripts/verify-offset-target.sh" "$TMP/message.json" transactional_message "$TMP/message-proof.json"
set +e
bash "$ROOT/scripts/verify-offset-target.sh" "$TMP/wrong-count.json" lsn_events_processed_1 "$TMP/wrong-count-proof.json"
wrong_count_rc=$?
bash "$ROOT/scripts/verify-offset-target.sh" "$TMP/wrong-type.json" lsn_events_processed_1 "$TMP/wrong-type-proof.json"
wrong_type_rc=$?
set -e
[ "$wrong_count_rc" -eq 3 ] || { echo "wrong count was not rejected with indeterminate status" >&2; exit 1; }
[ "$wrong_type_rc" -eq 3 ] || { echo "wrong type was not rejected with indeterminate status" >&2; exit 1; }
jq -e '.target_reached == true and .offset.messageType == "INSERT" and .offset.lsn_events_processed == 1' "$TMP/lsn-one-proof.json" >/dev/null
jq -e '.target_reached == true and .offset.messageType == "MESSAGE" and .offset.lsn_events_processed == 2' "$TMP/message-proof.json" >/dev/null
echo "MON-033 source-offset target parser fixtures passed."
