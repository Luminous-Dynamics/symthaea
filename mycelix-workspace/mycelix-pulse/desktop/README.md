Mycelix Pulse desktop now has one canonical shell: Tauri v2 in [`src-tauri`](./src-tauri), wrapping the Leptos app in [`../apps/leptos`](../apps/leptos).

Notes:
- `identifier` remains `com.mycelix.mail` for compatibility until there is an explicit migration plan.
- The old Electron and Tauri v1 entry points were retired to avoid parallel desktop products.

Deterministic workflow:
- `nix develop path:/srv/luminous-dynamics/mycelix-pulse`
- `cd /srv/luminous-dynamics/mycelix-pulse/desktop`
- `npm run smoke`
- `npm run type-check`
- `npm run dev`

What these do:
- `npm run smoke` checks that the desktop shell still targets `apps/leptos` and keeps the compatibility-sensitive bundle identifier.
- `npm run type-check` runs `cargo check` for the Tauri v2 shell.
- `npm run dev` launches `cargo tauri dev` against the Leptos app via `src-tauri/tauri.conf.json`.
