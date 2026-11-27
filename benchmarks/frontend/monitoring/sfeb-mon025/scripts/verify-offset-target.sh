#!/usr/bin/env bash
set -euo pipefail
OFFSETS_FILE="$1"
TARGET="$2"
PROOF_FILE="$3"
[ -s "$OFFSETS_FILE" ] || { echo "offset JSON is unavailable" >&2; exit 66; }
case "$TARGET" in lsn_events_processed_1|transactional_message) ;; *) echo "invalid target" >&2; exit 64 ;; esac
[ ! -e "$PROOF_FILE" ] || { echo "proof output already exists" >&2; exit 73; }
jq -e --arg target "$TARGET" '
  (.offsets | type == "array" and length == 1) and
  (.offsets[0].partition | type == "object") and
  (.offsets[0].offset | type == "object") and
  (.offsets[0].offset) as $o |
  ($o.lsn != null and $o.lsn_proc != null and $o.lsn_commit != null) and
  (if $target == "lsn_events_processed_1" then
     ((try ($o.lsn_events_processed | tonumber) catch -1) == 1) and
     (["INSERT","UPDATE","DELETE","TRUNCATE"] | index($o.messageType) != null)
   else
     ($o.messageType == "MESSAGE") and ($o.txId != null)
   end)
' "$OFFSETS_FILE" >/dev/null || {
  echo "observed source offset is not the requested complete target" >&2
  exit 3
}
captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
digest="$(sha256sum "$OFFSETS_FILE" | awk '{print $1}')"
jq -n --arg target "$TARGET" --arg captured "$captured_at" --arg digest "$digest" \
  --slurpfile raw "$OFFSETS_FILE" \
  '($raw[0].offsets[0]) as $entry |
   {profile:"mon033-postgres-offset-target-v1",version:"1.0.0",
    target:$target,target_reached:true,captured_at:$captured,offsets_sha256:$digest,
    partition:$entry.partition,offset:$entry.offset,
    semantics:{message_type:$entry.offset.messageType,
      lsn_events_processed:($entry.offset.lsn_events_processed // 0),
      counted_operations:["INSERT","UPDATE","DELETE","TRUNCATE"],
      begin_commit_message_origin_noop_are_not_counted:true}}' >"$PROOF_FILE"
echo "Target state verified: $TARGET"
