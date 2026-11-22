# Archived browser exploration harnesses

These source files are retained as historical research artifacts, not maintained examples.

- `praxis_active.rs` — raw CDP/JavaScript exploration harness.
- `praxis_conscious.rs` — raw CDP/JavaScript exploration harness.

They intentionally preserve the original source and include an archive banner. They directly evaluate page scripts, bypass the normal `BrowserExecutor` route, and are **not safety evidence**. Do not copy these patterns into production or active examples.

Current browser-control implementation work is tracked in [issue #7426](https://github.com/Luminous-Dynamics/symthaea/issues/7426), with the native CDP implementation slice in [PR #7429](https://github.com/Luminous-Dynamics/symthaea/pull/7429). Keeping these files here preserves the research trail while moving them out of the maintained examples surface.
