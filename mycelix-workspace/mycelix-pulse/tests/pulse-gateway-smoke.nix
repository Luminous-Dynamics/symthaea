# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 5A.3.a — single-VM nixosTest for pulse-smtp-gateway.
#
# Boots one VM with the module enabled, no TLS, port 2525, stub secrets.
# Drives an in-VM smtplib client at the SMTP listener and asserts:
#   1. systemd unit reaches active state
#   2. unit is NOT in failed state (catches CAP_NET_BIND_SERVICE,
#      LoadCredential=, DynamicUser binding errors that would crash
#      the daemon at startup before it ever accepts a connection)
#   3. TCP port 2525 is bound
#   4. smtplib send to alice@mycelix.test succeeds (RCPT-resolves through
#      stub alias map, persists to StubZomeBridge)
#   5. unit STILL active after the send (defends against panic-in-handler
#      regressions)
#   6. journalctl shows the structured log line confirming pipeline ran

{ pkgs, pulseModule }:

let
  # Build the gateway with rust-overlay's recent toolchain — nixos-24.05's
  # stock cargo (1.77) is too old for several transitive deps that need
  # edition2024 (cargo 1.85+). See flake.nix for the analogous wiring.
  rustToolchainForBuild = pkgs.rust-bin.stable.latest.default;
  gatewayPkg = (pkgs.makeRustPlatform {
    cargo = rustToolchainForBuild;
    rustc = rustToolchainForBuild;
  }).buildRustPackage {
    pname = "pulse-smtp-gateway";
    version = "0.1.0-alpha.1";
    src = ../crates/pulse-smtp-gateway;
    cargoLock.lockFile = ../crates/pulse-smtp-gateway/Cargo.lock;
    nativeBuildInputs = [ pkgs.pkg-config rustToolchainForBuild ];
    buildInputs = [ ];
    doCheck = false;
  };
in
pkgs.testers.runNixOSTest {
  name = "pulse-gateway-smoke";

  # Both static checks (mypy + pyflakes) misread the embedded heredoc
  # indentation. The script runs identically without them; skip to keep
  # focus on actual runtime semantics.
  skipTypeCheck = true;
  skipLint = true;

  nodes.gateway = { config, lib, ... }: {
    imports = [ pulseModule ];

    # Bare-minimum host config. nixosTest gives us a default networking
    # stack on a private subnet; we don't need internet access.
    networking.firewall.enable = false;

    # Stub secret files — random bytes are fine for Phase 5A. Phase 5B
    # will load real DKIM RSA, Ed25519, and Dilithium keys from BWS or
    # an equivalent secret store.
    environment.etc."pulse-gateway-secrets/dkim-rsa".text =
      "fake-rsa-key-for-test-only-not-a-real-pem";
    environment.etc."pulse-gateway-secrets/dkim-ed25519".text =
      "fake-ed25519-32-byte-seed-test-only";
    environment.etc."pulse-gateway-secrets/dilithium".text =
      "fake-dilithium-key-test-only";
    # The VERP HMAC secret is the only one the gateway actually reads
    # at boot (config.rs validate path) — must be real bytes that work
    # with HMAC-SHA256.
    environment.etc."pulse-gateway-secrets/verp-hmac".text =
      "test-verp-hmac-secret-32-bytes-of-entropy-for-phase-5a-smoke";

    services.pulse-smtp-gateway = {
      enable = true;
      package = gatewayPkg;
      domain = "mycelix.test";
      hostname = "gateway.mycelix.test";
      port = 2525; # unprivileged — sidesteps CAP_NET_BIND_SERVICE for the smoke
      bind = "127.0.0.1";
      conductorUrl = "ws://localhost:8888";
      gatewayDid = "did:mycelix:gateway-mail-test";
      postmasterDid = "did:mycelix:test-postmaster";
      abuseDid = "did:mycelix:test-abuse";
      enableRspamd = false; # rspamd needs its own warm-up; defer to 5A.3.b
      openFirewall = false;

      secrets = {
        "dkim-rsa" = "/etc/pulse-gateway-secrets/dkim-rsa";
        "dkim-ed25519" = "/etc/pulse-gateway-secrets/dkim-ed25519";
        "dilithium" = "/etc/pulse-gateway-secrets/dilithium";
        "verp-hmac" = "/etc/pulse-gateway-secrets/verp-hmac";
      };

      # Phase 5A test alias — without this, the pipeline returns 5.1.1
      # "no such user" for every recipient.
      testAliases = {
        "alice@mycelix.test" = "did:mycelix:alice";
      };
    };

    # Python with smtplib for the in-VM test client. smtplib is in the
    # stdlib; no extra packages needed.
    environment.systemPackages = with pkgs; [ python3 ];
  };

  # All script lines must be flush-left because Nix's `''` indent-strip
  # uses MIN(leading whitespace across non-empty lines), and the embedded
  # smtplib heredoc has zero-indent lines that would otherwise pin the
  # strip to 0 and leave the rest of the script indented — Python then
  # raises IndentationError at line 1.
  testScript = ''
start_all()

# 1. Service must reach active state.
gateway.wait_for_unit("pulse-smtp-gateway.service")

# 2. Service must NOT be in failed state. Catches DynamicUser /
#    LoadCredential / CAP_NET_BIND_SERVICE crash-at-startup bugs:
#    wait_for_unit returns even on restart-loop, so check is-active
#    + is-failed explicitly.
gateway.succeed("systemctl is-active --quiet pulse-smtp-gateway.service")
gateway.fail("systemctl is-failed --quiet pulse-smtp-gateway.service")

# 3. SMTP listener must be bound on 2525.
gateway.wait_for_open_port(2525)

# 4. smtplib send to alice@mycelix.test must succeed.
gateway.succeed(r"""
python3 - <<'PY'
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg["From"] = "external@example.org"
msg["To"] = "alice@mycelix.test"
msg["Subject"] = "Phase 5A.3.a VM smoke"
msg.set_content("In-VM smoke test body. Pipeline must persist this.")

with smtplib.SMTP("127.0.0.1", 2525, timeout=15) as s:
    s.helo("test-driver.mycelix.test")
    refused = s.send_message(msg)
    assert refused == {}, f"recipients refused: {refused}"
print("smtplib send OK")
PY
""")

# 5. Unit STILL active after the send. Defends against
#    panic-in-handler regression that would let smtplib see 250
#    while the gateway thread crashes underneath.
gateway.succeed("systemctl is-active --quiet pulse-smtp-gateway.service")

# 6. Structured log shows the pipeline ran.
#    The stub zome's `receive_external` emits a tracing::info!
#    event with the recipient_did. JSON log format means we grep
#    on field name. Sleep briefly — log writer is async.
gateway.sleep(2)
gateway.succeed(
    "journalctl -u pulse-smtp-gateway.service --no-pager "
    "| grep -F 'stub receive_external'"
)
gateway.succeed(
    "journalctl -u pulse-smtp-gateway.service --no-pager "
    "| grep -F 'did:mycelix:alice'"
)

# ---- Negative paths ------------------------------------------------
# The happy path passes. Now verify the error paths reject cleanly
# AND don't pollute the stub zome with spurious persistence.

# 7. Unknown alias on our domain → smtplib raises SMTPRecipientsRefused.
#    `gateway.fail` asserts the shell command exits non-zero; smtplib's
#    exception on 5xx gives us exactly that.
gateway.fail(r"""
python3 - <<'PY'
import smtplib
from email.message import EmailMessage
msg = EmailMessage()
msg["From"] = "external@example.org"
msg["To"] = "bob@mycelix.test"  # NOT pre-registered in testAliases
msg["Subject"] = "negative: unknown alias"
msg.set_content("server must 5xx this")
with smtplib.SMTP("127.0.0.1", 2525, timeout=15) as s:
    refused = s.send_message(msg)
    print(f"UNEXPECTED: accepted unknown alias, refused={refused}")
PY
""")

# 8. Foreign domain (not mycelix.test) → same 5xx rejection path.
gateway.fail(r"""
python3 - <<'PY'
import smtplib
from email.message import EmailMessage
msg = EmailMessage()
msg["From"] = "external@example.org"
msg["To"] = "carol@foreign.example"
msg["Subject"] = "negative: foreign domain"
msg.set_content("server must 5xx this")
with smtplib.SMTP("127.0.0.1", 2525, timeout=15) as s:
    refused = s.send_message(msg)
    print(f"UNEXPECTED: accepted foreign domain, refused={refused}")
PY
""")

# 9. Anti-leak: stub zome should STILL show exactly 1 persisted
#    email (alice's one). The two rejected messages must NOT have
#    reached receive_external.
gateway.sleep(2)
gateway.succeed(
    "test $(journalctl -u pulse-smtp-gateway.service --no-pager "
    "| grep -c 'stub receive_external') -eq 1"
)

# 10. Unit STILL active after multiple 5xx rejections. Defends
#     against a regression where a rejection path panics.
gateway.succeed("systemctl is-active --quiet pulse-smtp-gateway.service")

print("Phase 5A.3.a smoke + negative paths: GREEN")
  '';
}
