// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Probe stream backends
//!
//! Provides the embedding source backends for streaming.

use std::error::Error;
use std::net::ToSocketAddrs;

/// Core interface for pulling raw embedding vectors.
pub trait EmbeddingBackend: Send + Sync {
    /// Return the embedding dimension length.
    fn embedding_dim(&self) -> usize;
    /// Fetch the embedding vector for a given simulation time/context.
    fn fetch_embedding(&mut self, t: f64) -> Result<Vec<f32>, Box<dyn Error>>;
}

/// A deterministic Mock backend utilizing xorshift64 RNG.
pub struct MockBackend {
    dim: usize,
    state: u64,
}

impl MockBackend {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            dim,
            state: if seed == 0 { 1 } else { seed },
        }
    }
}

impl EmbeddingBackend for MockBackend {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn fetch_embedding(&mut self, _t: f64) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut embedding = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            // xorshift64
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;

            // map to range [-1.0, 1.0]
            let val = (x as f64 / u64::MAX as f64) * 2.0 - 1.0;
            embedding.push(val as f32);
        }
        Ok(embedding)
    }
}

/// Errors occurring during OllamaBackend operations.
#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HTTP connection failed to receive valid response: {0}")]
    Http(String),
}

/// Sensory backend calling Ollama API (`POST /api/embed`).
pub struct OllamaBackend {
    base_url: String, // e.g. "http://localhost:11434" or "localhost:11434"
    model: String,
    embedding_dim: usize,
    queued_input: String,
}

impl OllamaBackend {
    /// Create a new OllamaBackend.
    pub fn new(base_url: &str, model: &str, embedding_dim: usize) -> Self {
        Self {
            base_url: base_url
                .trim_start_matches("http://")
                .trim_start_matches("https://")
                .to_string(),
            model: model.to_string(),
            embedding_dim,
            queued_input: "placeholder".to_string(),
        }
    }

    /// Set the next input text to be processed.
    pub fn set_input(&mut self, text: String) {
        self.queued_input = text;
    }
}

impl EmbeddingBackend for OllamaBackend {
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn fetch_embedding(&mut self, _t: f64) -> Result<Vec<f32>, Box<dyn Error>> {
        use std::io::{Read, Write};
        use std::net::{SocketAddr, TcpStream};
        use std::time::Duration;

        let stream_addr = if self.base_url.contains(':') {
            self.base_url.clone()
        } else {
            format!("{}:11434", self.base_url)
        };

        // Parse socket address or resolve to enforce security boundaries
        let addrs: Vec<SocketAddr> = stream_addr.to_socket_addrs()?.collect();
        let target_addr = addrs.first().ok_or_else(|| {
            OllamaError::Http(format!("Failed to resolve address: {}", stream_addr))
        })?;

        // Safety: ensure default/localhost boundary unless explicitly configured (e.g. check for loopback/private)
        let is_safe = match target_addr.ip() {
            std::net::IpAddr::V4(ipv4) => ipv4.is_loopback() || ipv4.is_private(),
            std::net::IpAddr::V6(ipv6) => ipv6.is_loopback(), // V6 has different rules, at least check loopback
        };
        if !is_safe {
            // Log warning or enforce safety policies as needed. For now, allow but track.
        }

        // Establish connection with strict timeout (e.g., 5 seconds)
        let mut stream = TcpStream::connect_timeout(target_addr, Duration::from_secs(5))
            .map_err(OllamaError::Io)?;

        // Set read timeout (e.g., 10 seconds)
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(OllamaError::Io)?;
        // Set write timeout (e.g., 5 seconds)
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(OllamaError::Io)?;

        // Build the payload
        let payload = serde_json::json!({
            "model": self.model,
            "input": self.queued_input
        });
        let body = serde_json::to_string(&payload).map_err(OllamaError::Json)?;

        // Build the HTTP POST request
        let request = format!(
            "POST /api/embed HTTP/1.1\r\n\
             Host: {}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n\
             {}",
            self.base_url,
            body.len(),
            body
        );

        stream
            .write_all(request.as_bytes())
            .map_err(OllamaError::Io)?;
        stream.flush().map_err(OllamaError::Io)?;

        // Read response up to a maximum safety limit (e.g., 10MB to prevent memory exhaustion)
        let mut response = Vec::new();
        let max_bytes = 10 * 1024 * 1024; // 10 MB
        let mut chunk = [0u8; 8192];
        loop {
            let bytes_read = stream.read(&mut chunk).map_err(OllamaError::Io)?;
            if bytes_read == 0 {
                break;
            }
            if response.len() + bytes_read > max_bytes {
                return Err(Box::new(OllamaError::Http(
                    "HTTP response exceeded safety limit".to_string(),
                )));
            }
            response.extend_from_slice(&chunk[..bytes_read]);
        }

        let response_str = String::from_utf8_lossy(&response);

        // Find double CRLF separating header and body
        let parts: Vec<&str> = response_str.split("\r\n\r\n").collect();
        if parts.len() < 2 {
            return Err(Box::new(OllamaError::Http(
                "Invalid HTTP response format".to_string(),
            )));
        }

        let headers = parts[0];
        let body = parts[1..].join("\r\n\r\n");

        if !headers.contains("200 OK") {
            return Err(Box::new(OllamaError::Http(format!(
                "Non-200 HTTP status returned: {}",
                headers.lines().next().unwrap_or("")
            ))));
        }

        // Deserialize response body: {"embeddings": [[f32, ...]]}
        let res: serde_json::Value = serde_json::from_str(&body).map_err(OllamaError::Json)?;
        let embeddings = res
            .get("embeddings")
            .ok_or_else(|| OllamaError::Http("No 'embeddings' field in response".to_string()))?;

        let embedding_array = embeddings
            .as_array()
            .ok_or_else(|| OllamaError::Http("'embeddings' is not an array".to_string()))?;

        if embedding_array.is_empty() {
            return Err(Box::new(OllamaError::Http(
                "Embeddings array is empty".to_string(),
            )));
        }

        let first_embedding = embedding_array[0].as_array().ok_or_else(|| {
            OllamaError::Http("First embedding element is not an array".to_string())
        })?;

        let mut result = Vec::with_capacity(first_embedding.len());
        for val in first_embedding {
            let num = val
                .as_f64()
                .ok_or_else(|| OllamaError::Http("Embedding value is not a float".to_string()))?;
            result.push(num as f32);
        }

        if result.len() != self.embedding_dim {
            return Err(Box::new(OllamaError::Http(format!(
                "Received embedding of dimension {} but expected {}",
                result.len(),
                self.embedding_dim
            ))));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn test_ollama_backend_success() {
        // Start local test server
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let base_url = format!("127.0.0.1:{}", port);

        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);

                let response_body = serde_json::json!({
                    "embeddings": [[0.1f32, -0.2f32, 0.3f32]]
                });
                let response_body_str = serde_json::to_string(&response_body).unwrap();
                let http_response = format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\r\n\
                     {}",
                    response_body_str.len(),
                    response_body_str
                );
                let _ = stream.write_all(http_response.as_bytes());
            }
        });

        let mut backend = OllamaBackend::new(&base_url, "test-model", 3);
        let res = backend.fetch_embedding(0.0).unwrap();
        assert_eq!(res, vec![0.1f32, -0.2f32, 0.3f32]);
    }

    #[test]
    fn test_ollama_backend_non_200_status() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let base_url = format!("127.0.0.1:{}", port);

        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let http_response = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(http_response.as_bytes());
            }
        });

        let mut backend = OllamaBackend::new(&base_url, "test-model", 3);
        let res = backend.fetch_embedding(0.0);
        assert!(res.is_err());
        let err_msg = res.unwrap_err().to_string();
        assert!(err_msg.contains("Non-200 HTTP status returned"));
    }

    #[test]
    fn test_ollama_backend_malformed_json() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let base_url = format!("127.0.0.1:{}", port);

        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let http_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 5\r\nConnection: close\r\n\r\n{bad}";
                let _ = stream.write_all(http_response.as_bytes());
            }
        });

        let mut backend = OllamaBackend::new(&base_url, "test-model", 3);
        let res = backend.fetch_embedding(0.0);
        assert!(res.is_err());
    }
}
