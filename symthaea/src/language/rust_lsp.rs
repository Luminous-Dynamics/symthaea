// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal synchronous LSP client for `rust-analyzer`.
//!
//! This is intentionally small: it gives the coding loop type-aware navigation
//! facts (`goto_definition`, `find_references`, `hover`) without pulling async
//! infrastructure into the verified-generation path.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use serde_json::{json, Value};

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Zero-based LSP text position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LspPosition {
    pub line: u32,
    pub character: u32,
}

impl LspPosition {
    pub fn new(line: u32, character: u32) -> Self {
        Self { line, character }
    }

    fn to_json(self) -> Value {
        json!({
            "line": self.line,
            "character": self.character,
        })
    }
}

/// Zero-based LSP text range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspRange {
    pub start: LspPosition,
    pub end: LspPosition,
}

/// LSP location with a file URI and range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspLocation {
    pub uri: String,
    pub range: LspRange,
}

impl LspLocation {
    /// Convert a `file://` URI back to a local path.
    pub fn path(&self) -> Option<PathBuf> {
        file_uri_to_path(&self.uri)
    }
}

/// Hover text returned by rust-analyzer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspHover {
    pub contents: String,
}

/// A running rust-analyzer process spoken to via JSON-RPC over stdio.
pub struct RustAnalyzerClient {
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<io::Result<Value>>,
    next_id: u64,
    root_uri: String,
    shutdown_sent: bool,
    request_timeout: Duration,
}

impl RustAnalyzerClient {
    /// Spawn `rust-analyzer` and perform the LSP initialize handshake.
    pub fn start(root: impl AsRef<Path>) -> io::Result<Self> {
        Self::start_with_command(root, "rust-analyzer")
    }

    /// Spawn a custom command. Useful for integration tests with a fake server.
    pub fn start_with_command(root: impl AsRef<Path>, command: &str) -> io::Result<Self> {
        Self::start_with_command_and_timeout(root, command, DEFAULT_REQUEST_TIMEOUT)
    }

    /// Spawn a custom command with an explicit request timeout.
    pub fn start_with_command_and_timeout(
        root: impl AsRef<Path>,
        command: &str,
        request_timeout: Duration,
    ) -> io::Result<Self> {
        let root_uri = path_to_file_uri(root.as_ref())?;
        let mut child = Command::new(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "missing LSP stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "missing LSP stdout"))?;
        let responses = spawn_lsp_reader(stdout);

        let mut client = Self {
            child,
            stdin,
            responses,
            next_id: 1,
            root_uri,
            shutdown_sent: false,
            request_timeout,
        };
        if let Err(err) = client.initialize() {
            let _ = client.child.kill();
            let _ = client.child.wait();
            return Err(err);
        }
        Ok(client)
    }

    /// Set the maximum duration to wait for a matching response to each request.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Root URI passed during initialize.
    pub fn root_uri(&self) -> &str {
        &self.root_uri
    }

    /// Notify rust-analyzer that a document is open.
    pub fn did_open(
        &mut self,
        path: impl AsRef<Path>,
        language_id: &str,
        text: &str,
    ) -> io::Result<()> {
        let params = json!({
            "textDocument": {
                "uri": path_to_file_uri(path.as_ref())?,
                "languageId": language_id,
                "version": 1,
                "text": text,
            }
        });
        self.notify("textDocument/didOpen", params)
    }

    /// Query definition locations for a file position.
    pub fn goto_definition(
        &mut self,
        path: impl AsRef<Path>,
        position: LspPosition,
    ) -> io::Result<Vec<LspLocation>> {
        let result = self.request(
            "textDocument/definition",
            text_document_position_params(path.as_ref(), position)?,
        )?;
        Ok(parse_locations(&result))
    }

    /// Query references for a file position.
    pub fn find_references(
        &mut self,
        path: impl AsRef<Path>,
        position: LspPosition,
        include_declaration: bool,
    ) -> io::Result<Vec<LspLocation>> {
        let mut params = text_document_position_params(path.as_ref(), position)?;
        params["context"] = json!({ "includeDeclaration": include_declaration });
        let result = self.request("textDocument/references", params)?;
        Ok(parse_locations(&result))
    }

    /// Query hover text for a file position.
    pub fn hover(
        &mut self,
        path: impl AsRef<Path>,
        position: LspPosition,
    ) -> io::Result<Option<LspHover>> {
        let result = self.request(
            "textDocument/hover",
            text_document_position_params(path.as_ref(), position)?,
        )?;
        Ok(parse_hover(&result))
    }

    /// Gracefully shut down the language server.
    pub fn shutdown(&mut self) -> io::Result<()> {
        if self.shutdown_sent {
            return Ok(());
        }
        let _ = self.request("shutdown", Value::Null)?;
        self.notify("exit", Value::Null)?;
        self.shutdown_sent = true;
        Ok(())
    }

    fn initialize(&mut self) -> io::Result<()> {
        let params = json!({
            "processId": std::process::id(),
            "rootUri": self.root_uri,
            "capabilities": {
                "textDocument": {
                    "definition": { "dynamicRegistration": false },
                    "references": { "dynamicRegistration": false },
                    "hover": { "dynamicRegistration": false },
                }
            },
        });
        let _ = self.request("initialize", params)?;
        self.notify("initialized", json!({}))
    }

    fn request(&mut self, method: &str, params: Value) -> io::Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_message(&message)?;

        loop {
            let response = self.read_message(method)?;
            if response.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = response.get("error") {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("LSP request {method} failed: {error}"),
                ));
            }
            return Ok(response.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> io::Result<()> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_message(&message)
    }

    fn write_message(&mut self, message: &Value) -> io::Result<()> {
        let frame = build_lsp_frame(message)?;
        self.stdin.write_all(&frame)?;
        self.stdin.flush()
    }

    fn read_message(&mut self, method: &str) -> io::Result<Value> {
        match self.responses.recv_timeout(self.request_timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "timed out waiting {:?} for LSP response to {method}",
                    self.request_timeout
                ),
            )),
            Err(RecvTimeoutError::Disconnected) => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "LSP reader thread disconnected",
            )),
        }
    }
}

impl Drop for RustAnalyzerClient {
    fn drop(&mut self) {
        let _ = self.shutdown();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn text_document_position_params(path: &Path, position: LspPosition) -> io::Result<Value> {
    Ok(json!({
        "textDocument": { "uri": path_to_file_uri(path)? },
        "position": position.to_json(),
    }))
}

fn spawn_lsp_reader(stdout: std::process::ChildStdout) -> Receiver<io::Result<Value>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            match read_lsp_message(&mut reader) {
                Ok(message) => {
                    if tx.send(Ok(message)).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let is_eof = error.kind() == io::ErrorKind::UnexpectedEof;
                    let _ = tx.send(Err(error));
                    if is_eof {
                        break;
                    }
                }
            }
        }
    });
    rx
}

fn build_lsp_frame(message: &Value) -> io::Result<Vec<u8>> {
    let body = serde_json::to_vec(message).map_err(io::Error::other)?;
    let mut frame = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    frame.extend(body);
    Ok(frame)
}

fn read_lsp_message(reader: &mut impl BufRead) -> io::Result<Value> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "LSP server closed stdout",
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(value.trim().parse::<usize>().map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad Content-Length: {err}"),
                )
            })?);
        }
    }

    let len = content_length.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "LSP frame missing Content-Length",
        )
    })?;
    let mut body = vec![0; len];
    reader.read_exact(&mut body)?;
    serde_json::from_slice(&body).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid LSP JSON: {err}"),
        )
    })
}

fn parse_locations(value: &Value) -> Vec<LspLocation> {
    match value {
        Value::Array(items) => items.iter().filter_map(parse_location).collect(),
        Value::Object(_) => parse_location(value).into_iter().collect(),
        _ => Vec::new(),
    }
}

fn parse_location(value: &Value) -> Option<LspLocation> {
    if let (Some(uri), Some(range)) = (
        value.get("uri").and_then(Value::as_str),
        value.get("range").and_then(parse_range),
    ) {
        return Some(LspLocation {
            uri: uri.to_string(),
            range,
        });
    }

    let uri = value.get("targetUri").and_then(Value::as_str)?;
    let range = value.get("targetRange").and_then(parse_range)?;
    Some(LspLocation {
        uri: uri.to_string(),
        range,
    })
}

fn parse_range(value: &Value) -> Option<LspRange> {
    Some(LspRange {
        start: parse_position(value.get("start")?)?,
        end: parse_position(value.get("end")?)?,
    })
}

fn parse_position(value: &Value) -> Option<LspPosition> {
    Some(LspPosition {
        line: value.get("line")?.as_u64()? as u32,
        character: value.get("character")?.as_u64()? as u32,
    })
}

fn parse_hover(value: &Value) -> Option<LspHover> {
    let contents = value.get("contents")?;
    let text = hover_contents_to_text(contents)?;
    if text.trim().is_empty() {
        return None;
    }
    Some(LspHover { contents: text })
}

fn hover_contents_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => map
            .get("value")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().filter_map(hover_contents_to_text).collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n\n"))
            }
        }
        _ => None,
    }
}

/// Convert a local path to a `file://` URI.
pub fn path_to_file_uri(path: &Path) -> io::Result<String> {
    let path = match path.canonicalize() {
        Ok(path) => path,
        Err(_) if path.is_absolute() => path.to_path_buf(),
        Err(_) => std::env::current_dir()?.join(path),
    };
    Ok(format!(
        "file://{}",
        percent_encode_path(&path.to_string_lossy())
    ))
}

/// Convert a `file://` URI into a local path.
pub fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let raw = uri.strip_prefix("file://")?;
    percent_decode(raw).map(PathBuf::from)
}

fn percent_encode_path(path: &str) -> String {
    let mut encoded = String::new();
    for byte in path.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(*byte as char)
            }
            other => encoded.push_str(&format!("%{other:02X}")),
        }
    }
    encoded
}

fn percent_decode(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = *bytes.get(i + 1)?;
            let lo = *bytes.get(i + 2)?;
            decoded.push((hex_value(hi)? << 4) | hex_value(lo)?);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn lsp_frame_round_trips_json() {
        let message = json!({"jsonrpc": "2.0", "id": 7, "result": {"ok": true}});
        let frame = build_lsp_frame(&message).unwrap();
        let parsed = read_lsp_message(&mut Cursor::new(frame)).unwrap();
        assert_eq!(parsed["id"], 7);
        assert_eq!(parsed["result"]["ok"], true);
    }

    #[test]
    fn parses_location_and_location_link_results() {
        let value = json!([
            {
                "uri": "file:///tmp/lib.rs",
                "range": {
                    "start": {"line": 2, "character": 4},
                    "end": {"line": 2, "character": 12}
                }
            },
            {
                "targetUri": "file:///tmp/main.rs",
                "targetRange": {
                    "start": {"line": 5, "character": 0},
                    "end": {"line": 8, "character": 1}
                }
            }
        ]);

        let locations = parse_locations(&value);

        assert_eq!(locations.len(), 2);
        assert_eq!(locations[0].range.start, LspPosition::new(2, 4));
        assert_eq!(locations[1].uri, "file:///tmp/main.rs");
    }

    #[test]
    fn parses_marked_string_hover() {
        let hover = parse_hover(&json!({
            "contents": {
                "kind": "markdown",
                "value": "```rust\nfn normalize(value: &str) -> String\n```"
            }
        }))
        .unwrap();

        assert!(hover.contents.contains("normalize"));
    }

    #[test]
    fn file_uri_round_trips_spaces() {
        let path = Path::new("/tmp/symthaea lsp/lib.rs");
        let uri = path_to_file_uri(path).unwrap();
        assert_eq!(uri, "file:///tmp/symthaea%20lsp/lib.rs");
        assert_eq!(file_uri_to_path(&uri).unwrap(), path);
    }

    #[test]
    #[ignore = "requires a local rust-analyzer binary and starts a real LSP process"]
    fn live_rust_analyzer_smoke_hover_and_definition() {
        if std::env::var("SYMTHAEA_RUN_LIVE_LSP").as_deref() != Ok("1") {
            eprintln!("set SYMTHAEA_RUN_LIVE_LSP=1 to run the live rust-analyzer smoke test");
            return;
        }
        let Some(rust_analyzer) = rust_analyzer_command() else {
            eprintln!("rust-analyzer not found; skipping live LSP smoke test");
            return;
        };

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"symthaea_lsp_smoke\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let lib_path = src_dir.join("lib.rs");
        let source = r#"pub fn add_one(value: i32) -> i32 {
    value + 1
}

pub fn call_add() -> i32 {
    add_one(41)
}
"#;
        std::fs::write(&lib_path, source).unwrap();

        let mut client = RustAnalyzerClient::start_with_command_and_timeout(
            dir.path(),
            &rust_analyzer,
            Duration::from_secs(5),
        )
        .unwrap();
        client.did_open(&lib_path, "rust", source).unwrap();

        let hover = retry_lsp(|| client.hover(&lib_path, LspPosition::new(5, 6)))
            .unwrap()
            .expect("expected hover information for add_one call");
        assert!(
            hover.contents.contains("add_one") || hover.contents.contains("fn("),
            "unexpected hover contents: {}",
            hover.contents
        );

        let definitions =
            retry_lsp(|| client.goto_definition(&lib_path, LspPosition::new(5, 6))).unwrap();
        assert!(
            definitions
                .iter()
                .any(|location| location.range.start.line == 0),
            "expected definition on line 0, got {definitions:?}"
        );
    }

    fn retry_lsp<T>(mut query: impl FnMut() -> io::Result<T>) -> io::Result<T>
    where
        T: IsReady,
    {
        let mut last = None;
        for _ in 0..8 {
            let value = query()?;
            if value.is_ready() {
                return Ok(value);
            }
            last = Some(value);
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        Ok(last.expect("retry loop always runs at least once"))
    }

    trait IsReady {
        fn is_ready(&self) -> bool;
    }

    impl IsReady for Option<LspHover> {
        fn is_ready(&self) -> bool {
            self.is_some()
        }
    }

    impl IsReady for Vec<LspLocation> {
        fn is_ready(&self) -> bool {
            !self.is_empty()
        }
    }

    fn rust_analyzer_command() -> Option<String> {
        if let Ok(command) = std::env::var("SYMTHAEA_RUST_ANALYZER") {
            if command_available(&command) {
                return Some(command);
            }
            eprintln!("SYMTHAEA_RUST_ANALYZER is set but not executable: {command}");
        }

        for candidate in ["rust-analyzer", "/run/current-system/sw/bin/rust-analyzer"] {
            if command_available(candidate) {
                return Some(candidate.to_string());
            }
        }
        None
    }

    fn command_available(command: &str) -> bool {
        std::process::Command::new(command)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }
}
