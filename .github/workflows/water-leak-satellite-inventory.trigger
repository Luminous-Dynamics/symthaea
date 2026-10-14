One-shot, metadata-only Copernicus catalogue reconnaissance trigger for Johannesburg bbox and date range defaults defined in water-leak-satellite-inventory.yml. No raster assets are downloaded; output is discovery only.
Re-triggered after inventory-verifier hardening; inspect the artifact with the current branch's verifier before interpreting counts.
Finalized strict-JSON evidence-boundary checks and updated offline suite; re-run metadata-only inventory on this exact head.
Refresh live metadata-only STAC and independent OData inventories after OData verifier edge-case hardening; evaluate both as catalogue discovery only.
Refreshed after fixing OData shell line continuations; require exact-head independent STAC and OData metadata-only runs.
Refresh catalogue runs after constraining OData endpoint-path evidence and adding the corresponding regression test.
Retry OData by calendar quarter with a 30-second per-request timeout, at most two bounded retries, and attempt receipts; refresh STAC plus OData on the current source head. Catalogue metadata remains reconnaissance only, not pipe-level leak evidence.
Refresh exact-head metadata-only inventory after the bounded retry implementation, verifier updates, documentation refresh, and final-head offline-test trigger updates. Keep the same reconnaissance bounding box/date scope; do not download rasters or claim pipe-level leak evidence.
