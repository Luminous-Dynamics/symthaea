//! Lossless human-language types and a versioned JSON-lines worker protocol.

use crate::{SignalObservation, TimeSpan, valid_confidence};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

pub const HUMAN_PROVIDER_PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LanguageCandidate {
    pub language: String,
    pub script: Option<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreservedSpan {
    pub original: String,
    pub normalized: Option<String>,
    pub timing: Option<TimeSpan>,
    pub language: Option<String>,
    pub script: Option<String>,
    pub confidence: f32,
    pub alternatives: Vec<String>,
    pub named_entity: Option<String>,
    pub prosody: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreservedText {
    pub original: String,
    pub normalized: Option<String>,
    pub primary_language: Option<String>,
    pub script: Option<String>,
    pub spans: Vec<PreservedSpan>,
}

impl PreservedText {
    pub fn validate(&self) -> Result<(), String> {
        if self.original.is_empty() {
            return Err("original text must be preserved".into());
        }
        if self.spans.iter().any(|span| {
            span.original.is_empty()
                || !valid_confidence(span.confidence)
                || span.timing.is_some_and(|timing| !timing.is_valid())
        }) {
            return Err("invalid preserved span".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum HumanProviderRequest {
    Capabilities {
        request_id: String,
    },
    IdentifyLanguage {
        request_id: String,
        observation: SignalObservation,
    },
    Transcribe {
        request_id: String,
        observation: SignalObservation,
    },
    Translate {
        request_id: String,
        input: PreservedText,
        target_language: String,
    },
    TranslateSpeech {
        request_id: String,
        observation: SignalObservation,
        target_language: String,
    },
}

impl HumanProviderRequest {
    fn request_id(&self) -> &str {
        match self {
            Self::Capabilities { request_id }
            | Self::IdentifyLanguage { request_id, .. }
            | Self::Transcribe { request_id, .. }
            | Self::Translate { request_id, .. }
            | Self::TranslateSpeech { request_id, .. } => request_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanProviderEnvelope<T> {
    pub protocol_version: u32,
    pub provider_id: String,
    pub request_id: String,
    pub result: Option<T>,
    pub error: Option<String>,
}

impl<T> HumanProviderEnvelope<T> {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != HUMAN_PROVIDER_PROTOCOL_VERSION {
            return Err("unsupported human-provider protocol version".into());
        }
        if self.provider_id.is_empty() || self.request_id.is_empty() {
            return Err("provider and request ids are required".into());
        }
        if self.result.is_some() == self.error.is_some() {
            return Err("exactly one of result or error is required".into());
        }
        Ok(())
    }
}

pub trait HumanCommunicationProvider {
    fn identify_language(
        &mut self,
        observation: &SignalObservation,
    ) -> Result<Vec<LanguageCandidate>, String>;
    fn transcribe(&mut self, observation: &SignalObservation) -> Result<PreservedText, String>;
    fn translate(
        &mut self,
        input: &PreservedText,
        target_language: &str,
    ) -> Result<PreservedText, String>;
    fn translate_speech(
        &mut self,
        observation: &SignalObservation,
        target_language: &str,
    ) -> Result<PreservedText, String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum HumanProviderResult {
    Languages(Vec<LanguageCandidate>),
    Text(PreservedText),
}

/// A local JSON-lines worker. Model runtimes stay replaceable while Rust owns
/// evidence, preservation, policy, and evaluation.
pub struct LocalJsonlProvider {
    provider_id: String,
    child: Child,
    input: ChildStdin,
    responses: Receiver<Result<String, String>>,
    next_request: u64,
    timeout: Duration,
}

#[derive(Clone, Debug)]
pub struct WorkerPolicy {
    pub allowed_programs: BTreeSet<PathBuf>,
    pub response_timeout: Duration,
    pub maximum_response_bytes: usize,
}

impl WorkerPolicy {
    pub fn allow_one(program: &Path) -> Result<Self, String> {
        let canonical = program
            .canonicalize()
            .map_err(|error| format!("{}: {error}", program.display()))?;
        Ok(Self {
            allowed_programs: BTreeSet::from([canonical]),
            response_timeout: Duration::from_secs(120),
            maximum_response_bytes: 16 * 1024 * 1024,
        })
    }

    fn validate(&self) -> Result<(), String> {
        if self.allowed_programs.is_empty()
            || self.response_timeout.is_zero()
            || self.maximum_response_bytes == 0
        {
            return Err("worker policy requires an allowlist, timeout, and response limit".into());
        }
        Ok(())
    }
}

impl LocalJsonlProvider {
    pub fn spawn(
        provider_id: impl Into<String>,
        program: &Path,
        args: &[String],
        policy: WorkerPolicy,
    ) -> Result<Self, String> {
        policy.validate()?;
        let program = program
            .canonicalize()
            .map_err(|error| format!("{}: {error}", program.display()))?;
        if !policy.allowed_programs.contains(&program) {
            return Err(format!(
                "worker program is not allowlisted: {}",
                program.display()
            ));
        }
        let provider_id = provider_id.into();
        let mut child = Command::new(&program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("failed to start provider worker: {error}"))?;
        let input = child.stdin.take().ok_or("provider worker has no stdin")?;
        let stdout = child.stdout.take().ok_or("provider worker has no stdout")?;
        let (sender, responses) = mpsc::sync_channel(1);
        let maximum_response_bytes = policy.maximum_response_bytes;
        std::thread::Builder::new()
            .name(format!("communication-worker-{provider_id}"))
            .spawn(move || {
                let mut output = BufReader::new(stdout);
                loop {
                    let mut bytes = Vec::new();
                    let read = (&mut output)
                        .take(maximum_response_bytes as u64 + 1)
                        .read_until(b'\n', &mut bytes);
                    let response = match read {
                        Ok(0) => break,
                        Ok(_) if bytes.len() > maximum_response_bytes => {
                            Err("provider response exceeds configured limit".into())
                        }
                        Ok(_) => String::from_utf8(bytes)
                            .map_err(|_| "provider response is not UTF-8".into()),
                        Err(error) => Err(format!("failed reading provider response: {error}")),
                    };
                    let stop = response.is_err();
                    if sender.send(response).is_err() || stop {
                        break;
                    }
                }
            })
            .map_err(|error| format!("failed to start provider reader: {error}"))?;
        Ok(Self {
            provider_id,
            child,
            input,
            responses,
            next_request: 0,
            timeout: policy.response_timeout,
        })
    }

    fn request_id(&mut self) -> String {
        self.next_request += 1;
        format!("{}-{}", self.provider_id, self.next_request)
    }

    fn send(&mut self, request: &HumanProviderRequest) -> Result<HumanProviderResult, String> {
        if self
            .child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return Err("provider worker is not running".into());
        }
        serde_json::to_writer(&mut self.input, request).map_err(|error| error.to_string())?;
        self.input
            .write_all(b"\n")
            .map_err(|error| error.to_string())?;
        self.input.flush().map_err(|error| error.to_string())?;
        let response = self.responses.recv_timeout(self.timeout).map_err(|error| {
            let _ = self.child.kill();
            format!("provider response timeout or disconnect: {error}")
        })?;
        let line = response.map_err(|error| {
            let _ = self.child.kill();
            error
        })?;
        let envelope: HumanProviderEnvelope<HumanProviderResult> = serde_json::from_str(&line)
            .map_err(|error| format!("invalid provider response: {error}"))?;
        envelope.validate()?;
        if envelope.provider_id != self.provider_id {
            return Err("provider response identity mismatch".into());
        }
        if envelope.request_id != request.request_id() {
            return Err("provider response request id mismatch".into());
        }
        envelope
            .result
            .ok_or_else(|| envelope.error.unwrap_or_else(|| "provider failed".into()))
    }
}

impl Drop for LocalJsonlProvider {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl HumanCommunicationProvider for LocalJsonlProvider {
    fn identify_language(
        &mut self,
        observation: &SignalObservation,
    ) -> Result<Vec<LanguageCandidate>, String> {
        let request_id = self.request_id();
        match self.send(&HumanProviderRequest::IdentifyLanguage {
            request_id,
            observation: observation.clone(),
        })? {
            HumanProviderResult::Languages(value) => Ok(value),
            _ => Err("provider returned the wrong result kind".into()),
        }
    }

    fn transcribe(&mut self, observation: &SignalObservation) -> Result<PreservedText, String> {
        let request_id = self.request_id();
        match self.send(&HumanProviderRequest::Transcribe {
            request_id,
            observation: observation.clone(),
        })? {
            HumanProviderResult::Text(value) => {
                value.validate()?;
                Ok(value)
            }
            _ => Err("provider returned the wrong result kind".into()),
        }
    }

    fn translate(
        &mut self,
        input: &PreservedText,
        target_language: &str,
    ) -> Result<PreservedText, String> {
        input.validate()?;
        let request_id = self.request_id();
        match self.send(&HumanProviderRequest::Translate {
            request_id,
            input: input.clone(),
            target_language: target_language.to_owned(),
        })? {
            HumanProviderResult::Text(value) => {
                value.validate()?;
                Ok(value)
            }
            _ => Err("provider returned the wrong result kind".into()),
        }
    }

    fn translate_speech(
        &mut self,
        observation: &SignalObservation,
        target_language: &str,
    ) -> Result<PreservedText, String> {
        let request_id = self.request_id();
        match self.send(&HumanProviderRequest::TranslateSpeech {
            request_id,
            observation: observation.clone(),
            target_language: target_language.to_owned(),
        })? {
            HumanProviderResult::Text(value) => {
                value.validate()?;
                Ok(value)
            }
            _ => Err("provider returned the wrong result kind".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destructive_normalization_is_rejected() {
        let text = PreservedText {
            original: String::new(),
            normalized: Some("hello".into()),
            primary_language: Some("en".into()),
            script: Some("Latn".into()),
            spans: vec![],
        };
        assert!(text.validate().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn worker_rejects_mismatched_request_id() {
        let shell = Path::new("/bin/sh");
        let mut policy = WorkerPolicy::allow_one(shell).unwrap();
        policy.response_timeout = Duration::from_secs(1);
        let response = r#"{"protocol_version":1,"provider_id":"test","request_id":"wrong","result":{"kind":"languages","value":[]},"error":null}"#;
        let args = vec![
            "-c".into(),
            format!("read line; printf '%s\\n' '{response}'"),
        ];
        let mut worker = LocalJsonlProvider::spawn("test", shell, &args, policy).unwrap();
        let mut observation = SignalObservation {
            id: String::new(),
            modality: crate::Modality::Text {
                language: None,
                script: None,
            },
            samples: vec![],
            features: Default::default(),
            original_text: Some("hello".into()),
            normalized_text: None,
            uncertain_spans: vec![],
            timing: TimeSpan {
                start_s: 0.0,
                end_s: 0.0,
            },
            location: None,
            calibration: Default::default(),
            source_identity: None,
            environment: Default::default(),
        };
        observation.refresh_id().unwrap();
        assert!(
            worker
                .identify_language(&observation)
                .unwrap_err()
                .contains("request id mismatch")
        );
    }
}
