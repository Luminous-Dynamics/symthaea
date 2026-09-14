# Public worker routing

This crate binds deterministic extension routing to the exact frozen public worker deployment authority.

It preserves package truth (`runtime = wasm`), binds the canonical request digest at selection, requires exactly one matching live public deployment authority, and returns only private-field routed capabilities. It does not mint admission, deployment, engineering-evidence, or release authority.

The later release layer must independently re-verify the routed decision, public deployment binding, live worker qualification, execution evidence, and final currentness before minting the existing `symthaea.simulation.release.v1` receipt.
