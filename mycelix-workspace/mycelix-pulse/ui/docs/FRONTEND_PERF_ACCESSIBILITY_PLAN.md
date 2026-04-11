# Frontend Performance & Accessibility Improvement Plan

## Performance
- **Bundle slimming**: analyze Vite bundle with `vite --analyze`, split heavy routes, lazy-load rarely used views (Trust Center, Settings subsections).
- **Memoization & re-renders**: audit hot components (EmailList, ThreadView, EmailView) for avoidable re-renders; add memo where prop-stable; ensure hooks deps are minimal.
- **Virtualization**: swap EmailList to a virtualized list for large inboxes; keep skeletons/animations lightweight.
- **Network & caching**: tighten React Query stale/caching times per endpoint; coalesce refetches after WebSocket events; add backoff for transient failures.
- **Images & assets**: ensure avatars/attachments use responsive sizes and lazy loading; compress SVG/PNG assets.

## Accessibility
- **ARIA & roles**: verify EmailList rows, checkboxes, buttons, and modals have roles/labels; ensure keyboard focus states are visible (already present, but audit).
- **Keyboard flows**: ensure new trust controls (banners, buttons) are keyboard reachable and have focus order/tooltips.
- **Color contrast**: audit dark/light modes for contrast on badges/banners; adjust tailwind classes where needed.
- **Notifications**: ensure toasts and alerts are screen-reader friendly; add `aria-live` to alert banners.

## Testing
- **Component tests**: add targeted tests for EmailList/thread view selection and trust banner visibility with policy toggles.
- **E2E smoke**: minimal flows (login, open inbox, toggle trust settings, perform bulk actions) to guard regressions.

## Quick Wins
- Virtualize EmailList for >200 items.
- Add `aria-live="polite"` to trust banners/alerts.
- Memoize TrustBadge and heavy row renderers in EmailList/ThreadView.
