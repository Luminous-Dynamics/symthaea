//! Symthaea Live Demo Server
//!
//! Launches a browser-accessible visualization of the consciousness pipeline.
//!
//! ```bash
//! cargo run --features api_module --bin symthaea-demo
//! ```
//!
//! Then open http://localhost:8080 in your browser.

use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let addr = env::var("SYMTHAEA_DEMO_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    symthaea::api::serve_demo(&addr).await
}
