// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trace context propagation for distributed tracing

use opentelemetry::{
    global,
    propagation::{Extractor, Injector, TextMapPropagator},
    Context,
};
use std::collections::HashMap;

/// HTTP header extractor for incoming requests
pub struct HeaderExtractor<'a> {
    headers: &'a HashMap<String, String>,
}

impl<'a> HeaderExtractor<'a> {
    /// Create a new header extractor
    pub fn new(headers: &'a HashMap<String, String>) -> Self {
        Self { headers }
    }
}

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.headers.get(&key.to_lowercase()).map(|s| s.as_str())
    }

    fn keys(&self) -> Vec<&str> {
        self.headers.keys().map(|s| s.as_str()).collect()
    }
}

/// HTTP header injector for outgoing requests
pub struct HeaderInjector<'a> {
    headers: &'a mut HashMap<String, String>,
}

impl<'a> HeaderInjector<'a> {
    /// Create a new header injector
    pub fn new(headers: &'a mut HashMap<String, String>) -> Self {
        Self { headers }
    }
}

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        self.headers.insert(key.to_lowercase(), value);
    }
}

/// Extract trace context from HTTP headers
pub fn extract_context(headers: &HashMap<String, String>) -> Context {
    let extractor = HeaderExtractor::new(headers);
    global::get_text_map_propagator(|propagator| propagator.extract(&extractor))
}

/// Inject trace context into HTTP headers
pub fn inject_context(ctx: &Context, headers: &mut HashMap<String, String>) {
    let mut injector = HeaderInjector::new(headers);
    global::get_text_map_propagator(|propagator| propagator.inject_context(ctx, &mut injector));
}

/// Extract trace context from the current span
pub fn extract_current_context() -> Context {
    Context::current()
}

/// Set up the global propagator for W3C TraceContext + Jaeger
pub fn setup_propagator() {
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_jaeger_propagator::Propagator as JaegerPropagator;
    use opentelemetry::propagation::TextMapCompositePropagator;

    let composite = TextMapCompositePropagator::new(vec![
        Box::new(TraceContextPropagator::new()),
        Box::new(JaegerPropagator::default()),
    ]);

    global::set_text_map_propagator(composite);
}

/// Trace ID utilities
pub mod trace_id {
    use opentelemetry::trace::{SpanContext, TraceContextExt};
    use opentelemetry::Context;

    /// Get the current trace ID as a hex string
    pub fn current() -> Option<String> {
        let ctx = Context::current();
        let span = ctx.span();
        let span_context = span.span_context();

        if span_context.is_valid() {
            Some(span_context.trace_id().to_string())
        } else {
            None
        }
    }

    /// Get the current span ID as a hex string
    pub fn current_span_id() -> Option<String> {
        let ctx = Context::current();
        let span = ctx.span();
        let span_context = span.span_context();

        if span_context.is_valid() {
            Some(span_context.span_id().to_string())
        } else {
            None
        }
    }

    /// Create a link header value for trace correlation
    pub fn correlation_header() -> Option<String> {
        let trace_id = current()?;
        let span_id = current_span_id()?;
        Some(format!("00-{}-{}-01", trace_id, span_id))
    }
}

/// Holochain-specific context propagation
pub mod holochain {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Trace context for Holochain zome calls
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HolochainTraceContext {
        /// W3C traceparent header value
        pub traceparent: Option<String>,
        /// W3C tracestate header value
        pub tracestate: Option<String>,
        /// Additional baggage items
        pub baggage: HashMap<String, String>,
    }

    impl HolochainTraceContext {
        /// Create from current context
        pub fn from_current() -> Self {
            let mut headers = HashMap::new();
            inject_context(&Context::current(), &mut headers);

            Self {
                traceparent: headers.remove("traceparent"),
                tracestate: headers.remove("tracestate"),
                baggage: headers,
            }
        }

        /// Convert to HTTP headers for extraction
        pub fn to_headers(&self) -> HashMap<String, String> {
            let mut headers = self.baggage.clone();
            if let Some(ref tp) = self.traceparent {
                headers.insert("traceparent".to_string(), tp.clone());
            }
            if let Some(ref ts) = self.tracestate {
                headers.insert("tracestate".to_string(), ts.clone());
            }
            headers
        }

        /// Extract context from this struct
        pub fn extract(&self) -> Context {
            extract_context(&self.to_headers())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_extractor() {
        let mut headers = HashMap::new();
        headers.insert("traceparent".to_string(), "00-trace-span-01".to_string());

        let extractor = HeaderExtractor::new(&headers);
        assert_eq!(extractor.get("traceparent"), Some("00-trace-span-01"));
        assert_eq!(extractor.get("missing"), None);
    }

    #[test]
    fn test_header_injector() {
        let mut headers = HashMap::new();
        {
            let mut injector = HeaderInjector::new(&mut headers);
            injector.set("traceparent", "00-trace-span-01".to_string());
        }

        assert_eq!(headers.get("traceparent"), Some(&"00-trace-span-01".to_string()));
    }

    #[test]
    fn test_holochain_context() {
        let ctx = holochain::HolochainTraceContext {
            traceparent: Some("00-trace-span-01".to_string()),
            tracestate: None,
            baggage: HashMap::new(),
        };

        let headers = ctx.to_headers();
        assert!(headers.contains_key("traceparent"));
    }
}
