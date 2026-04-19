// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end integration test: real TCP SMTP on loopback, real lettre
//! client, real mailin-embedded server, real pipeline, StubZomeBridge
//! as the "DHT" capture layer.
//!
//! Proves Phase 5A: the gateway accepts an SMTP session over TCP, runs
//! the full inbound pipeline (rate-limit → parse → alias resolve →
//! zome persist), and the stub captures what would have been written
//! to the DHT.
//!
//! This is the local-only equivalent of what the nixosTest harness
//! will eventually do across VMs. No VPS, no real DNS, no Let's
//! Encrypt, no conductor.

use lettre::message::Message;
use lettre::transport::smtp::client::Tls;
use lettre::{SmtpTransport, Transport};
use pulse_smtp_gateway::{
    pipeline::RealInboundPipeline, rate_limit::PerIpLimiter, receiver::SmtpReceiver,
    zome::StubZomeBridge,
};
use std::net::TcpListener;
use std::sync::Arc;
use std::time::Duration;

fn find_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn alice_external_email_reaches_stub_zome() {
    // --- Setup: pick a free port, build the stub, pre-register alice. ---
    let port = find_free_port();
    let zome = StubZomeBridge::new();
    zome.set_alias("alice@mycelix.test", "did:mycelix:alice")
        .await;

    // We want to inspect the stub after the test, so we need a second
    // handle. StubZomeBridge is Default, uses interior mutability
    // (Mutex<Vec<_>>) — clone-via-reconstruction won't work, but we can
    // wrap it in Arc and pass the Arc through.
    let zome = Arc::new(zome);
    let zome_for_pipeline = ArcZome(zome.clone());

    let limiter = PerIpLimiter::new(&pulse_smtp_gateway::config::RateLimitConfig {
        per_ip_per_minute: 30,
        per_ip_per_hour: 300,
    })
    .expect("limiter");

    let pipeline = RealInboundPipeline::new(
        limiter,
        zome_for_pipeline,
        "mycelix.test".to_string(),
        tokio::runtime::Handle::current(),
    );
    let receiver = SmtpReceiver::new(pipeline);

    // --- Spawn the server on its own thread. Lives for the whole test. ---
    let bind = format!("127.0.0.1:{}", port);
    let bind_for_server = bind.clone();
    std::thread::spawn(move || {
        let mut server = mailin_embedded::Server::new(receiver);
        server.with_name("test.mycelix.test");
        server
            .with_addr(bind_for_server.as_str())
            .expect("with_addr");
        server.serve().expect("serve"); // blocks forever
    });

    // Give the listener a moment to bind before the client fires.
    // mailin-embedded's serve() spawns listener threads internally;
    // 200ms is generous on dev hardware.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // --- Send an SMTP message via lettre. ---
    let email = Message::builder()
        .from("external@example.org".parse().unwrap())
        .to("alice@mycelix.test".parse().unwrap())
        .subject("hello from outside")
        .body("This is the plaintext body of the Phase 5A test email.".to_string())
        .expect("build email");

    let mailer = SmtpTransport::builder_dangerous("127.0.0.1")
        .port(port)
        // Disable TLS — our test server has no cert and mailin-embedded's
        // rustls builder rejects naked TCP unless we say so explicitly.
        .tls(Tls::None)
        .timeout(Some(Duration::from_secs(10)))
        .build();

    let result = tokio::task::spawn_blocking(move || mailer.send(&email))
        .await
        .expect("spawn_blocking");

    let sent = result.expect("SMTP send should succeed");
    assert!(
        sent.is_positive(),
        "expected 2xx SMTP reply, got {:?}",
        sent
    );

    // --- Assertion: the stub has exactly one capture with our fields. ---
    // Poll briefly — pipeline runs on a worker thread and writes under mutex.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let received = zome.received.lock().await;
        if received.len() == 1 {
            let r = &received[0];
            assert_eq!(r.recipient_did, "did:mycelix:alice");
            assert_eq!(r.from_external, "external@example.org");
            assert_eq!(r.subject, "hello from outside");
            assert!(
                std::str::from_utf8(&r.encrypted_body)
                    .map(|s| s.contains("plaintext body of the Phase 5A test"))
                    .unwrap_or(false),
                "body did not round-trip: {:?}",
                r.encrypted_body
            );
            return;
        }
        drop(received);
        if std::time::Instant::now() >= deadline {
            let count = zome.received.lock().await.len();
            panic!(
                "stub did not receive the email within 5s (found {} captures)",
                count
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ----------------------------------------------------------------------------
// Small adapter so we can share one StubZomeBridge between the pipeline and
// the test assertions. The pipeline takes ZomeBridge by value; without this
// adapter we'd need two separate StubZomeBridges and no shared state.
// ----------------------------------------------------------------------------

#[derive(Clone)]
struct ArcZome(Arc<StubZomeBridge>);

#[async_trait::async_trait]
impl pulse_smtp_gateway::zome::ZomeBridge for ArcZome {
    async fn receive_external(
        &self,
        recipient_did: &str,
        from_external: &str,
        subject: &str,
        encrypted_body: &[u8],
        message_id: &str,
    ) -> pulse_smtp_gateway::GatewayResult<()> {
        self.0
            .receive_external(
                recipient_did,
                from_external,
                subject,
                encrypted_body,
                message_id,
            )
            .await
    }
    async fn update_outbox_status(
        &self,
        message_id: &str,
        status: &str,
        diagnostic: Option<&str>,
    ) -> pulse_smtp_gateway::GatewayResult<()> {
        self.0
            .update_outbox_status(message_id, status, diagnostic)
            .await
    }
    async fn receive_bounce(
        &self,
        message_id: &str,
        dsn_status: &str,
        diagnostic: &str,
    ) -> pulse_smtp_gateway::GatewayResult<()> {
        self.0
            .receive_bounce(message_id, dsn_status, diagnostic)
            .await
    }
    async fn resolve_alias(
        &self,
        alias: &str,
    ) -> pulse_smtp_gateway::GatewayResult<Option<String>> {
        self.0.resolve_alias(alias).await
    }
}
