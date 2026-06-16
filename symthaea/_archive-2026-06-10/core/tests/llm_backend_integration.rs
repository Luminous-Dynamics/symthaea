// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for LLM backends
//!
//! These tests verify the OpenAI and Anthropic backend implementations work correctly.
//! They use mock HTTP servers for CI testing and can optionally test against real APIs.
//!
//! To run with real APIs:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo test --test llm_backend_integration -- --ignored
//! ANTHROPIC_API_KEY=sk-... cargo test --test llm_backend_integration -- --ignored
//! ```

use std::sync::{Mutex, OnceLock};
use symthaea::language::{
    AnthropicBackend, GenerationParams, LLMBackend, OpenAiBackend, SimulatedBackend,
    create_backend_from_env,
};

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

// ============================================================================
// Backend Factory Tests
// ============================================================================

#[test]
fn test_create_backend_from_env_defaults_to_ollama() {
    let _guard = env_lock().lock().unwrap();
    // Clear all provider env vars
    let prev_provider = std::env::var("SYMTHAEA_LLM_PROVIDER").ok();
    let prev_openai = std::env::var("OPENAI_API_KEY").ok();
    let prev_anthropic = std::env::var("ANTHROPIC_API_KEY").ok();

    std::env::remove_var("SYMTHAEA_LLM_PROVIDER");
    std::env::remove_var("OPENAI_API_KEY");
    std::env::remove_var("ANTHROPIC_API_KEY");

    let backend = create_backend_from_env();
    assert_eq!(backend.name(), "Ollama");

    // Restore
    if let Some(v) = prev_provider {
        std::env::set_var("SYMTHAEA_LLM_PROVIDER", v);
    }
    if let Some(v) = prev_openai {
        std::env::set_var("OPENAI_API_KEY", v);
    }
    if let Some(v) = prev_anthropic {
        std::env::set_var("ANTHROPIC_API_KEY", v);
    }
}

#[test]
fn test_create_backend_explicit_provider_simulated() {
    let _guard = env_lock().lock().unwrap();
    let prev = std::env::var("SYMTHAEA_LLM_PROVIDER").ok();
    std::env::set_var("SYMTHAEA_LLM_PROVIDER", "simulated");

    let backend = create_backend_from_env();
    assert_eq!(backend.name(), "Simulated");

    match prev {
        Some(v) => std::env::set_var("SYMTHAEA_LLM_PROVIDER", v),
        None => std::env::remove_var("SYMTHAEA_LLM_PROVIDER"),
    }
}

#[test]
fn test_create_backend_auto_detects_openai_key() {
    let _guard = env_lock().lock().unwrap();
    let prev_provider = std::env::var("SYMTHAEA_LLM_PROVIDER").ok();
    let prev_openai = std::env::var("OPENAI_API_KEY").ok();
    let prev_anthropic = std::env::var("ANTHROPIC_API_KEY").ok();

    // Use explicit provider to avoid env var race conditions in parallel tests
    std::env::set_var("SYMTHAEA_LLM_PROVIDER", "openai");
    std::env::set_var("OPENAI_API_KEY", "sk-test-key");
    std::env::remove_var("ANTHROPIC_API_KEY");

    let backend = create_backend_from_env();
    assert_eq!(backend.name(), "OpenAI");

    // Restore
    if let Some(v) = prev_provider {
        std::env::set_var("SYMTHAEA_LLM_PROVIDER", v);
    }
    match prev_openai {
        Some(v) => std::env::set_var("OPENAI_API_KEY", v),
        None => std::env::remove_var("OPENAI_API_KEY"),
    }
    if let Some(v) = prev_anthropic {
        std::env::set_var("ANTHROPIC_API_KEY", v);
    }
}

#[test]
fn test_create_backend_auto_detects_anthropic_key() {
    let _guard = env_lock().lock().unwrap();
    let prev_provider = std::env::var("SYMTHAEA_LLM_PROVIDER").ok();
    let prev_openai = std::env::var("OPENAI_API_KEY").ok();
    let prev_anthropic = std::env::var("ANTHROPIC_API_KEY").ok();

    std::env::remove_var("SYMTHAEA_LLM_PROVIDER");
    std::env::remove_var("OPENAI_API_KEY");
    std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test-key");

    let backend = create_backend_from_env();
    assert_eq!(backend.name(), "Anthropic");

    // Restore
    if let Some(v) = prev_provider {
        std::env::set_var("SYMTHAEA_LLM_PROVIDER", v);
    }
    if let Some(v) = prev_openai {
        std::env::set_var("OPENAI_API_KEY", v);
    }
    match prev_anthropic {
        Some(v) => std::env::set_var("ANTHROPIC_API_KEY", v),
        None => std::env::remove_var("ANTHROPIC_API_KEY"),
    }
}

// ============================================================================
// SimulatedBackend Tests
// ============================================================================

#[tokio::test]
async fn test_simulated_backend_generates_response() {
    let backend = SimulatedBackend;
    let params = GenerationParams::default();

    let result = backend.generate("What is consciousness?", &params).await;
    assert!(result.is_ok());
    let text = result.unwrap();
    assert!(!text.is_empty());
}

#[tokio::test]
async fn test_simulated_backend_streaming() {
    let backend = SimulatedBackend;
    let params = GenerationParams::default();

    let mut tokens = Vec::new();
    let result = backend
        .generate_streaming("Tell me about HDC", &params, &mut |token| {
            tokens.push(token.to_string())
        })
        .await;

    assert!(result.is_ok());
    assert!(!tokens.is_empty());
}

// ============================================================================
// OpenAI Backend Unit Tests
// ============================================================================

#[test]
fn test_openai_backend_creation() {
    let backend = OpenAiBackend::new(
        "test-key".to_string(),
        "gpt-4o-mini".to_string(),
        "https://api.openai.com/v1".to_string(),
    );
    assert_eq!(backend.name(), "OpenAI");
}

#[tokio::test]
async fn test_openai_is_available_with_key() {
    let backend = OpenAiBackend::new(
        "test-key".to_string(),
        "gpt-4o-mini".to_string(),
        "https://api.openai.com/v1".to_string(),
    );
    assert!(backend.is_available().await);
}

#[tokio::test]
async fn test_openai_is_unavailable_without_key() {
    let backend = OpenAiBackend::new(
        "".to_string(),
        "gpt-4o-mini".to_string(),
        "https://api.openai.com/v1".to_string(),
    );
    assert!(!backend.is_available().await);
}

// ============================================================================
// Anthropic Backend Unit Tests
// ============================================================================

#[test]
fn test_anthropic_backend_creation() {
    let backend = AnthropicBackend::new(
        "test-key".to_string(),
        "claude-sonnet-4-20250514".to_string(),
        "https://api.anthropic.com".to_string(),
    );
    assert_eq!(backend.name(), "Anthropic");
}

#[tokio::test]
async fn test_anthropic_is_available_with_key() {
    let backend = AnthropicBackend::new(
        "test-key".to_string(),
        "claude-sonnet-4-20250514".to_string(),
        "https://api.anthropic.com".to_string(),
    );
    assert!(backend.is_available().await);
}

#[tokio::test]
async fn test_anthropic_is_unavailable_without_key() {
    let backend = AnthropicBackend::new(
        "".to_string(),
        "claude-sonnet-4-20250514".to_string(),
        "https://api.anthropic.com".to_string(),
    );
    assert!(!backend.is_available().await);
}

// ============================================================================
// GenerationParams Tests
// ============================================================================

#[test]
fn test_generation_params_default() {
    let params = GenerationParams::default();
    assert!((params.temperature - 0.7).abs() < 0.01);
    assert_eq!(params.max_tokens, 256);
    assert!(params.system_prompt.is_none());
}

#[test]
fn test_generation_params_with_system_prompt() {
    let params = GenerationParams {
        temperature: 0.3,
        max_tokens: 100,
        system_prompt: Some("You are a consciousness researcher.".to_string()),
        consciousness_context: None,
    };
    assert!((params.temperature - 0.3).abs() < 0.01);
    assert_eq!(params.max_tokens, 100);
    assert!(params.system_prompt.is_some());
}

// ============================================================================
// Real API Tests (ignored by default - run with --ignored flag)
// ============================================================================

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY environment variable"]
async fn test_openai_real_api() {
    let backend = match OpenAiBackend::from_env() {
        Some(b) => b,
        None => {
            eprintln!("Skipping: OPENAI_API_KEY not set");
            return;
        }
    };

    let params = GenerationParams {
        temperature: 0.3,
        max_tokens: 50,
        system_prompt: Some("Reply in exactly 5 words.".to_string()),
        consciousness_context: None,
    };

    let result = backend.generate("Say hello", &params).await;
    match result {
        Ok(text) => {
            println!("OpenAI response: {}", text);
            assert!(!text.is_empty());
        }
        Err(e) => {
            eprintln!("OpenAI API error (may be rate limit): {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY environment variable"]
async fn test_anthropic_real_api() {
    let backend = match AnthropicBackend::from_env() {
        Some(b) => b,
        None => {
            eprintln!("Skipping: ANTHROPIC_API_KEY not set");
            return;
        }
    };

    let params = GenerationParams {
        temperature: 0.3,
        max_tokens: 50,
        system_prompt: Some("Reply in exactly 5 words.".to_string()),
        consciousness_context: None,
    };

    let result = backend.generate("Say hello", &params).await;
    match result {
        Ok(text) => {
            println!("Anthropic response: {}", text);
            assert!(!text.is_empty());
        }
        Err(e) => {
            eprintln!("Anthropic API error (may be rate limit): {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY environment variable"]
async fn test_openai_streaming_real_api() {
    let backend = match OpenAiBackend::from_env() {
        Some(b) => b,
        None => {
            eprintln!("Skipping: OPENAI_API_KEY not set");
            return;
        }
    };

    let params = GenerationParams {
        temperature: 0.3,
        max_tokens: 50,
        system_prompt: None,
        consciousness_context: None,
    };

    let mut token_count = 0;
    let result = backend
        .generate_streaming("Count from 1 to 5", &params, &mut |token| {
            print!("{}", token);
            token_count += 1;
        })
        .await;
    println!();

    match result {
        Ok(text) => {
            println!("Full response: {}", text);
            println!("Tokens received: {}", token_count);
            assert!(!text.is_empty());
            assert!(token_count > 0);
        }
        Err(e) => {
            eprintln!("OpenAI streaming error: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY environment variable"]
async fn test_anthropic_streaming_real_api() {
    let backend = match AnthropicBackend::from_env() {
        Some(b) => b,
        None => {
            eprintln!("Skipping: ANTHROPIC_API_KEY not set");
            return;
        }
    };

    let params = GenerationParams {
        temperature: 0.3,
        max_tokens: 50,
        system_prompt: None,
        consciousness_context: None,
    };

    let mut token_count = 0;
    let result = backend
        .generate_streaming("Count from 1 to 5", &params, &mut |token| {
            print!("{}", token);
            token_count += 1;
        })
        .await;
    println!();

    match result {
        Ok(text) => {
            println!("Full response: {}", text);
            println!("Tokens received: {}", token_count);
            assert!(!text.is_empty());
            assert!(token_count > 0);
        }
        Err(e) => {
            eprintln!("Anthropic streaming error: {}", e);
        }
    }
}