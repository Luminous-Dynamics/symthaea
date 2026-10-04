#!/usr/bin/env bash
set -euo pipefail

# TPM plumbing smoke test only. An emulator is not hardware-backed authority.
# This validates that Spore's Nix development environment can exercise the
# primitives required by the later concrete regenerative-health adapter.

: "${TMPDIR:=/tmp}"
ROOT=$(mktemp -d "${TMPDIR%/}/symthaea-tpm-smoke.XXXXXX")
EVIDENCE_FILE="${GITHUB_WORKSPACE:-$PWD}/tpm2-spore-boundary-evidence.txt"
TPM_STATE="$ROOT/tpm-state"
CTRL_PORT=2322
TPM_PORT=2321
NV_INDEX=0x1500016
NV_AUTH=index
CHALLENGE=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
CHALLENGE_DIGEST=$(printf '%s' "$CHALLENGE" | sha256sum | cut -d' ' -f1)
printf 'schema=tpm2-spore-boundary-smoke-v2\n' > "$EVIDENCE_FILE"
printf 'initial_challenge_sha256=%s\n' "$CHALLENGE_DIGEST" >> "$EVIDENCE_FILE"
printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || echo unavailable)" >> "$EVIDENCE_FILE"
printf 'swtpm=%s\n' "$(swtpm --version 2>/dev/null | head -1 || echo unavailable)" >> "$EVIDENCE_FILE"
printf 'tpm2_tools=%s\n' "$(tpm2_getcap --version 2>/dev/null | head -1 || echo unavailable)" >> "$EVIDENCE_FILE"
printf 'nix=%s\n' "$(nix --version 2>/dev/null || echo unavailable)" >> "$EVIDENCE_FILE"
for pkg in tss2-esys tss2-tctildr tss2-mu; do printf '%s=%s\n' "$pkg" "$(pkg-config --modversion "$pkg" 2>/dev/null || echo unavailable)" >> "$EVIDENCE_FILE"; done
cleanup() {
  if [ -n "${SWTPM_PID:-}" ]; then
    kill "$SWTPM_PID" 2>/dev/null || true
    wait "$SWTPM_PID" 2>/dev/null || true
  fi
  rm -rf "$ROOT"
}
trap cleanup EXIT

for cmd in swtpm swtpm_setup tpm2_startup tpm2_shutdown tpm2_getcap tpm2_nvdefine tpm2_nvincrement tpm2_nvreadpublic tpm2_nvread tpm2_createprimary tpm2_readpublic tpm2_quote tpm2_checkquote tpm2_nvcertify tpm2_verifysignature tpm2_nvundefine tpm2_print od sha256sum tr grep sed; do
  command -v "$cmd" >/dev/null || { echo "missing required command: $cmd" >&2; exit 2; }
done

mkdir -p "$TPM_STATE"
swtpm_setup --tpm2 --tpmstate "$TPM_STATE" --overwrite >/dev/null
swtpm socket --tpm2 --tpmstate "dir=$TPM_STATE,fsync" --ctrl "type=tcp,port=$CTRL_PORT,bindaddr=127.0.0.1" --server "type=tcp,port=$TPM_PORT,bindaddr=127.0.0.1" --flags not-need-init >/dev/null 2>&1 &
SWTPM_PID=$!

for _ in $(seq 1 50); do
  if TPM2TOOLS_TCTI="swtpm:host=127.0.0.1,port=$TPM_PORT" tpm2_getcap properties-fixed >/dev/null 2>&1; then break; fi
  sleep 0.1
done
export TPM2TOOLS_TCTI="swtpm:host=127.0.0.1,port=$TPM_PORT"

tpm2_startup -c
tpm2_getcap properties-fixed > "$ROOT/tpm-properties.txt"
grep -A2 'TPM2_PT_FAMILY_INDICATOR:' "$ROOT/tpm-properties.txt" | grep -Fq 'value: "2.0"'

# Define an eight-byte counter and advance it to generation 1.
tpm2_nvdefine -Q -C o -s 8 -a 'ownerread|authread|authwrite|nt=counter' "$NV_INDEX" -p "$NV_AUTH"
tpm2_nvincrement -Q -C "$NV_INDEX" "$NV_INDEX" -P "$NV_AUTH"
tpm2_nvread -Q -C "$NV_INDEX" -s 8 -P "$NV_AUTH" -o "$ROOT/counter.bin" "$NV_INDEX"
read_hex() { od -An -tx1 -v "$1" | tr -d ' \\n'; }\n[ "$(read_hex "$ROOT/counter.bin")" = "0000000000000001" ]

# Pin the public NV object and its Name.
tpm2_nvreadpublic > "$ROOT/nv-public.txt"
grep -Fq "$NV_INDEX" "$ROOT/nv-public.txt"

# Create a restricted RSA signing key for the emulator test.
tpm2_createprimary -Q -C o -g sha256 -G rsa -a 'fixedtpm|fixedparent|sensitivedataorigin|userwithauth|restricted|sign' -c "$ROOT/ak.ctx"
tpm2_readpublic -Q -c "$ROOT/ak.ctx" -f pem -o "$ROOT/ak.pem"

# Quote PCR 7 and authenticate the exact fresh challenge.
tpm2_quote -Q -c "$ROOT/ak.ctx" -l sha256:7 -q "$CHALLENGE" -m "$ROOT/quote.attest" -s "$ROOT/quote.sig" -o "$ROOT/quote.pcrs" -g sha256
tpm2_checkquote -Q -u "$ROOT/ak.pem" -m "$ROOT/quote.attest" -s "$ROOT/quote.sig" -f "$ROOT/quote.pcrs" -g sha256 -q "$CHALLENGE" -l sha256:7
tpm2_print -Q -t TPMS_ATTEST "$ROOT/quote.attest" > "$ROOT/quote.yaml"

# A valid old Quote must not verify for a fresh challenge.
REPLAY_CHALLENGE=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
REPLAY_CHALLENGE_DIGEST=$(printf '%s' "$REPLAY_CHALLENGE" | sha256sum | cut -d' ' -f1)
printf 'replay_challenge_sha256=%s\n' "$REPLAY_CHALLENGE_DIGEST" >> "$EVIDENCE_FILE"
if tpm2_checkquote -Q -u "$ROOT/ak.pem" -m "$ROOT/quote.attest" -s "$ROOT/quote.sig" -f "$ROOT/quote.pcrs" -g sha256 -q "$REPLAY_CHALLENGE" -l sha256:7 >/dev/null 2>&1; then
  echo "ERROR: old Quote was accepted for a different challenge" >&2
  exit 1
fi

# Restart the emulator against the same persistent state and ensure the counter survives.
tpm2_shutdown -c
kill "$SWTPM_PID" 2>/dev/null || true
wait "$SWTPM_PID" 2>/dev/null || true
unset SWTPM_PID
swtpm socket --tpm2 --tpmstate "dir=$TPM_STATE,fsync" --ctrl "type=tcp,port=$CTRL_PORT,bindaddr=127.0.0.1" --server "type=tcp,port=$TPM_PORT,bindaddr=127.0.0.1" --flags not-need-init >/dev/null 2>&1 &
SWTPM_PID=$!
for _ in $(seq 1 50); do
  if TPM2TOOLS_TCTI="swtpm:host=127.0.0.1,port=$TPM_PORT" tpm2_getcap properties-fixed >/dev/null 2>&1; then break; fi
  sleep 0.1
done
export TPM2TOOLS_TCTI="swtpm:host=127.0.0.1,port=$TPM_PORT"
tpm2_startup -c
tpm2_nvread -Q -C "$NV_INDEX" -s 8 -P "$NV_AUTH" -o "$ROOT/counter-after-restart.bin" "$NV_INDEX"
[ "$(read_hex "$ROOT/counter-after-restart.bin")" = "0000000000000001" ] || {
  echo "ERROR: TPM NV counter did not survive restart" >&2
  exit 1
}

# A post-restart generation advance proves the counter remains usable after recovery.
tpm2_nvincrement -Q -C "$NV_INDEX" "$NV_INDEX" -P "$NV_AUTH"
tpm2_nvread -Q -C "$NV_INDEX" -s 8 -P "$NV_AUTH" -o "$ROOT/counter-generation-2.bin" "$NV_INDEX"
[ "$(read_hex "$ROOT/counter-generation-2.bin")" = "0000000000000002" ] || {
  echo "ERROR: TPM NV counter did not advance to generation 2" >&2
  exit 1
}

# Transient TPM key handles do not survive TPM restart; recreate the attestation key.
tpm2_createprimary -Q -C o -g sha256 -G rsa -a 'fixedtpm|fixedparent|sensitivedataorigin|userwithauth|restricted|sign' -c "$ROOT/ak-after-restart.ctx"
tpm2_readpublic -Q -c "$ROOT/ak-after-restart.ctx" -f pem -o "$ROOT/ak-after-restart.pem"

# Use a fresh challenge for post-restart evidence.
POST_RESTART_CHALLENGE=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \\n')
POST_RESTART_CHALLENGE_DIGEST=$(printf '%s' "$POST_RESTART_CHALLENGE" | sha256sum | cut -d' ' -f1)
printf 'post_restart_challenge_sha256=%s\n' "$POST_RESTART_CHALLENGE_DIGEST" >> "$EVIDENCE_FILE"
tpm2_quote -Q -c "$ROOT/ak-after-restart.ctx" -l sha256:7 -q "$POST_RESTART_CHALLENGE" -m "$ROOT/quote-after-restart.attest" -s "$ROOT/quote-after-restart.sig" -o "$ROOT/quote-after-restart.pcrs" -g sha256
tpm2_print -Q -t TPMS_ATTEST "$ROOT/quote-after-restart.attest" > "$ROOT/quote-after-restart.yaml"
tpm2_checkquote -Q -u "$ROOT/ak-after-restart.pem" -m "$ROOT/quote-after-restart.attest" -s "$ROOT/quote-after-restart.sig" -f "$ROOT/quote-after-restart.pcrs" -g sha256 -q "$POST_RESTART_CHALLENGE" -l sha256:7

QUOTE_SIGNER=$(grep -m1 '^qualifiedSigner:' "$ROOT/quote-after-restart.yaml" | sed 's/^qualifiedSigner:[[:space:]]*//')
[ -n "$QUOTE_SIGNER" ] || {
  echo "ERROR: post-restart Quote is missing a qualified signer" >&2
  exit 1
}

# Certify the complete eight-byte NV counter at generation 2 with that same fresh challenge.
tpm2_nvcertify -Q -C "$ROOT/ak-after-restart.ctx" -c "$NV_INDEX" -p "$NV_AUTH" -g sha256 -f plain -s rsassa -o "$ROOT/nv.sig" --attestation "$ROOT/nv.attest" --size 8 --offset 0 -q "$POST_RESTART_CHALLENGE" "$NV_INDEX"
test -s "$ROOT/nv.attest"
test -s "$ROOT/nv.sig"
tpm2_print -Q -t TPMS_ATTEST "$ROOT/nv.attest" > "$ROOT/nv.yaml"
NV_EXTRA_DATA=$(grep -m1 '^extraData:' "$ROOT/nv.yaml" | sed 's/^extraData:[[:space:]]*//')
[ "$NV_EXTRA_DATA" = "$POST_RESTART_CHALLENGE" ] || {
  echo "ERROR: NV_Certify attestation did not bind the fresh challenge" >&2
  exit 1
}
grep -Fq "type: 8014" "$ROOT/nv.yaml"
NV_SIGNER=$(grep -m1 '^qualifiedSigner:' "$ROOT/nv.yaml" | sed 's/^qualifiedSigner:[[:space:]]*//')
[ "$NV_SIGNER" = "$QUOTE_SIGNER" ] || {
  echo "ERROR: NV_Certify was signed by a different qualified signer than the Quote" >&2
  exit 1
}
grep -Fq "indexName:" "$ROOT/nv.yaml"
grep -Fq "offset: 0" "$ROOT/nv.yaml"
NV_CONTENTS=$(grep -m1 '^      nvContents:' "$ROOT/nv.yaml" | sed 's/^ *nvContents:[[:space:]]*//')
[ "$NV_CONTENTS" = "0000000000000002" ] || {
  echo "ERROR: NV_Certify did not certify the expected generation-2 counter contents" >&2
  exit 1
}
printf 'nv_certification_contents_binding=verified\\n' >> "$EVIDENCE_FILE"
printf 'nv_certification_challenge_binding=verified\\n' >> "$EVIDENCE_FILE"
tpm2_verifysignature -Q -c "$ROOT/ak-after-restart.ctx" -g sha256 -m "$ROOT/nv.attest" -s "$ROOT/nv.sig" -f rsassa

# The TPM will also certify a partial range. This is a negative domain fixture:
# the authoritative contract requires the complete eight-byte counter at offset 0.
tpm2_nvcertify -Q -C "$ROOT/ak-after-restart.ctx" -c "$NV_INDEX" -p "$NV_AUTH" -g sha256 -f plain -s rsassa -o "$ROOT/nv-partial.sig" --attestation "$ROOT/nv-partial.attest" --size 4 --offset 4 -q "$POST_RESTART_CHALLENGE" "$NV_INDEX"
test -s "$ROOT/nv-partial.attest"
test -s "$ROOT/nv-partial.sig"
tpm2_verifysignature -Q -c "$ROOT/ak-after-restart.ctx" -g sha256 -m "$ROOT/nv-partial.attest" -s "$ROOT/nv-partial.sig" -f rsassa
printf 'partial_nv_certification_accepted_by_tpm=true_domain_should_reject=true\n' >> "$EVIDENCE_FILE"

tpm2_nvundefine -Q -C o "$NV_INDEX"

echo 'TPM2 Spore boundary smoke test: PASS'
echo "initial_challenge_sha256=$CHALLENGE_DIGEST"
echo "nv_index=$NV_INDEX"
echo 'counter_generation=2'
echo 'counter_persisted_across_tpm_restart=true'
echo 'old_quote_replay_with_fresh_challenge=rejected'
echo 'post_restart_quote=pcr7+fresh_challenge verified'
echo 'nv_certification=full_contents offset=0 size=8 verified'
echo 'authority_claim=none (software TPM emulator)'
