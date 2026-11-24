# MON-003 SQLite transactional-outbox preflight

The exact tested source is stored here as deterministic gzip:

- `mon003-sqlite-transactional-outbox-v1.mjs.gz`
- Uncompressed source SHA-256: `919a5ee17ef02c15ec2154f8730344b223feefb33b043785e4720f3ed1356965`
- Compressed artifact SHA-256: `0a878fd26795a266c7f091acf66092fe212e39f287f1f8425c0a56e40eb1f836`

Run from the repository root (Node.js v22.16.0 recommended):

```sh
gzip -dc benchmarks/frontend/monitoring/mon003-sqlite-transactional-outbox-v1.mjs.gz | node --check -
gzip -dc benchmarks/frontend/monitoring/mon003-sqlite-transactional-outbox-v1.mjs.gz | node -
```

The current local report records 38/38 deterministic cases and five fresh-process sequential smoke runs. The SQLite `node:sqlite` API is experimental in Node 22. This is a local persistence preflight, not a production distributed-system qualification.
