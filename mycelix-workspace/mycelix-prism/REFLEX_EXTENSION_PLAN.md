# Reflex Arc Extension Integration (Proposal)

## Overview
To provide extensions with visibility into content safety and threat detection, we will enable `ReflexArc` to dispatch events to the `prism-bridge`.

## Implementation Strategy

1.  **Reflex Event Definition:**
    Extensions need to know when a verdict is reached. We will define an `ExtensionEvent` structure that can be serialized and sent over `ExtensionMessage`.

    ```rust
    #[derive(Serialize, Deserialize)]
    pub struct ThreatEvent {
        pub url: String,
        pub safety_level: SafetyLevel,
        pub threats: Vec<ThreatType>,
    }
    ```

2.  **Dispatch Mechanism:**
    Update `ReflexArc` to optionally take a sender (e.g., an `mpsc::UnboundedSender<RendererToSpore>`) to dispatch `ExtensionMessage` directly when a `PostParseVerdict` is generated.

    ```rust
    // In prism-reflex/src/lib.rs
    pub fn on_post_parse_verdict(&self, verdict: &PostParseVerdict, url: &Url) {
        if !verdict.threats.is_empty() {
             // Logic to construct and send ExtensionMessage via bridge
        }
    }
    ```

## Benefit
Extensions can now build real-time monitoring dashboards or local-first reputation systems that track threats across a user's browsing history without requiring modification to the core engine.
