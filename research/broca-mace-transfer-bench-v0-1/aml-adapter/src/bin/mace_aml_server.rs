// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal single-track AML Add/Search server.
//!
//! Bind to loopback and place an HTTPS reverse proxy in front of this process.
//! The binary deliberately refuses Textual mode until a verified frozen AML
//! Adapter word counter is supplied; it never substitutes whitespace counting.

use std::env;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::net::SocketAddr;

use broca_aml_adapter::{
    durable_service::DurableAmlServiceV1,
    http_service::{router, AmlHttpConfigV1, ApiAuthV1, AuthSchemeV1},
    redb_store::{RedbDurableAmlStoreV1, RedbVisibilityVerifierV1},
    FrozenAdapterWordCounter, Message, Track,
};
use tokio::net::TcpListener;

struct NonTextualOnlyCounter;

impl FrozenAdapterWordCounter for NonTextualOnlyCounter {
    fn adapter_version(&self) -> &str {
        "not-used-for-coding-or-multimodal"
    }

    fn count_words(&self, _messages: &[Message]) -> usize {
        // Track::Textual is refused at startup, before a request can reach this
        // placeholder. Coding and Multimodal do not call the Textual counter.
        0
    }
}

fn required_env(name: &str) -> Result<String, IoError> {
    env::var(name).map_err(|_| {
        IoError::new(
            ErrorKind::InvalidInput,
            format!("required environment variable {name} is missing"),
        )
    })
}

fn parse_track(value: &str) -> Result<Track, IoError> {
    match value {
        "textual" => Ok(Track::Textual),
        "coding" => Ok(Track::Coding),
        "multimodal" => Ok(Track::Multimodal),
        _ => Err(IoError::new(
            ErrorKind::InvalidInput,
            "MACE_AML_TRACK must be textual, coding, or multimodal",
        )),
    }
}

fn authentication() -> Result<ApiAuthV1, IoError> {
    let mode = required_env("MACE_AML_AUTH_MODE")?;
    match mode.as_str() {
        "token" => ApiAuthV1::from_secret(
            AuthSchemeV1::Token,
            &required_env("MACE_AML_API_KEY")?,
        )
        .map_err(|_| IoError::new(ErrorKind::InvalidInput, "invalid Token authentication")),
        "bearer" => ApiAuthV1::from_secret(
            AuthSchemeV1::Bearer,
            &required_env("MACE_AML_API_KEY")?,
        )
        .map_err(|_| IoError::new(ErrorKind::InvalidInput, "invalid Bearer authentication")),
        "x-api-key" => ApiAuthV1::from_secret(
            AuthSchemeV1::XApiKey,
            &required_env("MACE_AML_API_KEY")?,
        )
        .map_err(|_| IoError::new(ErrorKind::InvalidInput, "invalid X-Api-Key authentication")),
        "public-smoke" if env::var("MACE_AML_ENABLE_PUBLIC_SMOKE").as_deref() == Ok("1") => {
            Ok(ApiAuthV1::public_smoke())
        }
        "public-smoke" => Err(IoError::new(
            ErrorKind::PermissionDenied,
            "public smoke mode requires MACE_AML_ENABLE_PUBLIC_SMOKE=1",
        )),
        _ => Err(IoError::new(
            ErrorKind::InvalidInput,
            "MACE_AML_AUTH_MODE must be token, bearer, x-api-key, or explicitly enabled public-smoke",
        )),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let track = parse_track(&required_env("MACE_AML_TRACK")?)?;
    if track == Track::Textual {
        return Err(Box::new(IoError::new(
            ErrorKind::Unsupported,
            "Textual mode is disabled until the official frozen AML Adapter word counter is integrated; no whitespace fallback is permitted",
        )));
    }

    let database_path = required_env("MACE_AML_DB_PATH")?;
    let bind_address = env::var("MACE_AML_BIND_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_owned())
        .parse::<SocketAddr>()?;
    if !bind_address.ip().is_loopback()
        && env::var("MACE_AML_TLS_TERMINATED").as_deref() != Ok("1")
    {
        return Err(Box::new(IoError::new(
            ErrorKind::PermissionDenied,
            "non-loopback bind requires an explicitly configured HTTPS-terminating proxy",
        )));
    }

    let body_limit = match env::var("MACE_AML_MAX_BODY_BYTES") {
        Ok(value) => value.parse::<usize>()?,
        Err(_) => http_service::DEFAULT_MAX_JSON_BODY_BYTES,
    };
    let auth = authentication()?;
    let config = AmlHttpConfigV1::with_body_limit(track, auth, body_limit)
        .map_err(|_| IoError::new(ErrorKind::InvalidInput, "invalid HTTP service configuration"))?;

    let store = RedbDurableAmlStoreV1::open(&database_path)
        .map_err(|_| IoError::new(ErrorKind::Other, "durable store could not be opened"))?;
    let service = DurableAmlServiceV1::new(store, RedbVisibilityVerifierV1, NonTextualOnlyCounter);
    let app = router(service, config);

    let listener = TcpListener::bind(bind_address).await?;
    eprintln!(
        "MACE AML server ready: track={:?}, bind={}",
        track, bind_address
    );
    axum::serve(listener, app).await?;
    Ok(())
}
