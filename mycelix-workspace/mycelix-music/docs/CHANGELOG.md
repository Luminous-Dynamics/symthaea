# Changelog (recent targeted changes)

- Added upload header validation and unified zod error formatting.
- Added health details query validation for `client_ts` to normalize error responses.
- Added EIP-712 smoke tests (song/claim/play) gated by envs.
- Added migrations 008 (non-negative checks) and 009 (FK plays→songs).
- Added `test:eip712` script in apps/api/package.json.
