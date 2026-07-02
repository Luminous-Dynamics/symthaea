# Updating Prism Architecture (Phase 0.1 Integration)

The `mycelix-prism` architecture has been extended to support modular extensibility and multi-transport networking, consistent with the Spore-Mind principles of resilience and secure evolution.

## Updated Architectural Primitives

### 1. Safe IPC Messaging (Reflex & Extensions)
Prism now features a robust IPC bridge that supports isolated, side-car extension execution.
- **`RendererToSpore` (ExtensionMessage):** Allows the renderer to pipe event data to the kernel without executing arbitrary code in the browser context.
- **`SporeToRenderer` (ExtensionCommand):** Allows the kernel to push updates to extensions securely.
- **`ThreatEvent`:** The `ReflexArc` now emits standardized threat events, enabling extensions to observe and react to localized security threats in real-time.

### 2. Protocol-Agnostic Networking (Multi-Transport)
The networking stack has been decoupled into a transport-agnostic client.
- **`Transport` Trait:** The networking layer now defines an abstract `Transport` interface (`fetch(&self, url: &Url)`).
- **`HttpTransport`:** Default implementation for existing web-based retrieval.
- **Future-Proofing:** This allows the integration of `iroh://`, `ipfs://`, and local-DHT protocols without modifying the `PrismRenderer`.

## Security Rationale
- **Zero-JIT Policy:** These improvements uphold the foundational requirement of no JS/JIT engine in the renderer.
- **Sandboxed Extensibility:** By shifting extension complexity to the kernel (via IPC), the renderer process remains a small, inspectable security surface.
- **Structural Immunity:** The Reflex Arc’s integration with the IPC bridge ensures that localized security verdicts are broadcast globally (within the Spore's consciousness), enabling collective epistemic defense.
