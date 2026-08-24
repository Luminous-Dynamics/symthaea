use std::sync::{Arc, Mutex};

use anyhow::Result;
use symthaea::language::{
    llm_backend::{GenerationParams, LLMBackend},
    llm_organ::{LLMOrgan, LLMOrganConfig, LLMQuery, LLMQueryParams, QueryType},
};

#[derive(Clone)]
struct RecordingBackend {
    response: String,
    calls: Arc<Mutex<Vec<(String, GenerationParams)>>>,
}

#[async_trait::async_trait]
impl LLMBackend for RecordingBackend {
    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        self.calls
            .lock()
            .unwrap()
            .push((prompt.to_owned(), params.clone()));
        Ok(self.response.clone())
    }

    async fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "strict-recording-backend"
    }
}

fn query() -> LLMQuery {
    LLMQuery {
        query_type: QueryType::Translation,
        content: "exact strict request content".into(),
        context: vec![],
        system_prompt: Some("exact strict system prompt".into()),
        params: Some(LLMQueryParams {
            temperature: Some(0.23),
            max_length: Some(321),
            stop_sequences: vec![],
        }),
    }
}

#[tokio::test]
async fn strict_execution_forwards_exact_backend_inputs_once() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let backend = Arc::new(RecordingBackend {
        response: "surface realization".into(),
        calls: calls.clone(),
    });
    let mut organ = LLMOrgan::with_backend(LLMOrganConfig::default(), backend);
    let query = query();

    let result = organ.execute_backend_strict(&query).await.unwrap();
    assert_eq!(result.text, "surface realization");

    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    let (prompt, params) = &calls[0];
    assert_eq!(prompt, &query.content);
    assert_eq!(params.system_prompt, query.system_prompt);
    assert_eq!(params.temperature.to_bits(), 0.23_f32.to_bits());
    assert_eq!(params.max_tokens, 321);
    assert!(params.consciousness_context.is_none());
}

#[tokio::test]
async fn strict_execution_respects_memory_disabled() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let backend = Arc::new(RecordingBackend {
        response: "surface realization".into(),
        calls,
    });
    let config = LLMOrganConfig {
        memory_enabled: false,
        ..Default::default()
    };
    let mut organ = LLMOrgan::with_backend(config, backend);

    organ.execute_backend_strict(&query()).await.unwrap();

    assert!(organ.conversation_history().is_empty());
    assert_eq!(organ.stats().queries_processed, 1);
    assert_eq!(organ.stats().errors, 0);
}
