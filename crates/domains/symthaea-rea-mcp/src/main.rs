//! Stdio entry point for the read-only REA Evidence MCP server.
//!
//! stdout is reserved for MCP JSON-RPC frames; this binary emits no banners.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    symthaea_rea_mcp::serve_stdio().await
}
