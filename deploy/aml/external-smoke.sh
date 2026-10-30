#!/usr/bin/env bash
set -euo pipefail

: "$AML_BASE_URL"
: "$AML_API_KEY"

if [[ "$AML_BASE_URL" != https://* ]]; then
  echo "AML_BASE_URL must use https://" >&2
  exit 2
fi

for tool in curl jq openssl sed; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required command: $tool" >&2
    exit 2
  fi
done

BASE_URL="$(printf '%s' "$AML_BASE_URL" | sed 's:/*$::')"
RUN_ID="$(openssl rand -hex 16)"
REQUEST_ID="broca-smoke-$RUN_ID"
USER_ID="broca-smoke-user-$RUN_ID"
SESSION_ID="broca-smoke-session-$RUN_ID"
CONTENT="broca-aml-external-smoke-$RUN_ID"

echo "[1/4] checking public HTTPS health"
HEALTH="$(curl --silent --show-error --fail-with-body \
  --connect-timeout 5 --max-time 20 \
  "$BASE_URL/health")"
jq -e '.status == "ok"' <<<"$HEALTH" >/dev/null || {
  echo "Health response did not contain status=ok: $HEALTH" >&2
  exit 1
}

echo "[2/4] checking unauthenticated Add is rejected"
UNAUTH_STATUS="$(curl --silent --show-error \
  --connect-timeout 5 --max-time 20 \
  --output /dev/null --write-out '%{http_code}' \
  --header 'Content-Type: application/json' \
  --data '{"request_id":"unauthenticated-smoke","user_id":"unauthenticated-smoke","session_id":"unauthenticated-smoke","messages":[{"role":"user","content":"must be rejected"}]}' \
  "$BASE_URL/add")"
if [[ "$UNAUTH_STATUS" != "401" ]]; then
  echo "Expected unauthenticated Add to return 401, got $UNAUTH_STATUS" >&2
  exit 1
fi

echo "[3/4] checking authenticated Add and exact response echoes"
ADD_BODY="$(jq -n \
  --arg request_id "$REQUEST_ID" \
  --arg user_id "$USER_ID" \
  --arg session_id "$SESSION_ID" \
  --arg content "$CONTENT" \
  '{request_id:$request_id,user_id:$user_id,session_id:$session_id,messages:[{role:"user",content:$content}]}')"
ADD_RESPONSE="$(curl --silent --show-error --fail-with-body \
  --connect-timeout 5 --max-time 30 \
  --header "Authorization: Bearer $AML_API_KEY" \
  --header 'Content-Type: application/json' \
  --data "$ADD_BODY" \
  "$BASE_URL/add")"
jq -e \
  --arg request_id "$REQUEST_ID" \
  --arg user_id "$USER_ID" \
  --arg session_id "$SESSION_ID" \
  '.success == true and .request_id == $request_id and .user_id == $user_id and .session_id == $session_id' \
  <<<"$ADD_RESPONSE" >/dev/null || {
    echo "Add response did not satisfy the echo contract: $ADD_RESPONSE" >&2
    exit 1
  }

echo "[4/4] checking read-after-write visibility and cross-user isolation"
SEARCH_BODY="$(jq -n \
  --arg query "$CONTENT" \
  --arg user_id "$USER_ID" \
  '{query:$query,user_id:$user_id,top_k:10}')"
SEARCH_RESPONSE="$(curl --silent --show-error --fail-with-body \
  --connect-timeout 5 --max-time 30 \
  --header "Authorization: Bearer $AML_API_KEY" \
  --header 'Content-Type: application/json' \
  --data "$SEARCH_BODY" \
  "$BASE_URL/search")"
jq -e --arg expected "$CONTENT" \
  '.data | type == "array" and any(.[]; .content == $expected)' \
  <<<"$SEARCH_RESPONSE" >/dev/null || {
    echo "Add was not immediately visible in Search: $SEARCH_RESPONSE" >&2
    exit 1
  }

OTHER_USER_BODY="$(jq -n \
  --arg query "$CONTENT" \
  --arg user_id "$USER_ID-other" \
  '{query:$query,user_id:$user_id,top_k:10}')"
OTHER_USER_RESPONSE="$(curl --silent --show-error --fail-with-body \
  --connect-timeout 5 --max-time 30 \
  --header "Authorization: Bearer $AML_API_KEY" \
  --header 'Content-Type: application/json' \
  --data "$OTHER_USER_BODY" \
  "$BASE_URL/search")"
jq -e --arg expected "$CONTENT" \
  '(.data | type == "array") and (all(.data[]?; .content != $expected))' \
  <<<"$OTHER_USER_RESPONSE" >/dev/null || {
    echo "Search leaked the smoke record to a different user scope: $OTHER_USER_RESPONSE" >&2
    exit 1
  }

echo "PASS: public HTTPS health, auth rejection, Add echo, read-after-write Search, and user isolation."
echo "Smoke record user_id=$USER_ID; it remains subject to the service's configured retention policy."
echo "This is an operator smoke check, not the organizer's official Smoke or Full evaluation."
