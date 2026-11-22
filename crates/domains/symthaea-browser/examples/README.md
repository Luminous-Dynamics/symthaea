# Symthaea Browser examples

## Maintained example

- `praxis_journey.rs` is the maintained local Praxis journey. It routes navigation through `BrowserExecutor`, checks each receipt, and obtains observations through `BrowserExecutor::observe()`, which checks the observation capability/Phi and current URL before exposing the accessibility tree. It requires a local Praxis server and an explicitly configured development browser.

## Archived exploration harnesses

Older raw-CDP/JavaScript experiments are preserved in `archive/` with an archive banner. They are historical artifacts and not supported examples of the current controller's safety boundary. Do not use raw `evaluate` snippets or bypass `BrowserExecutor` in maintained browser automation.

The controller's native-CDP migration is tracked in [issue #7426](https://github.com/Luminous-Dynamics/symthaea/issues/7426) and [PR #7429](https://github.com/Luminous-Dynamics/symthaea/pull/7429). A Rust controller does not make Chromium itself a JavaScript-free or Rust-only rendering engine.
