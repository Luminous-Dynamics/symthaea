# Symthaea Boot Protocol v1

`BootEvent` is an ordered presentation stream. Producers assign monotonically increasing sequence numbers within one boot observation lineage. Consumers ignore duplicate or older sequence numbers.

`BootSnapshot` is a point-in-time normalized state and may replace a consumer's reduced event state after successful validation.

`BootHealth::Unknown` is deliberately distinct from `Normal`; consumers must never map missing observation to health.

Recovery of one domain does not implicitly restore whole-boot health. Only an authoritative snapshot or `BootReady` event may resolve aggregate health after prior degradation/failure.

Optional detail strings are operator-facing hints only. They are bounded and must not contain control characters. Raw diagnostic truth belongs to the journal.

The planned v1 transport is a local Unix datagram endpoint with a 4096-byte application message ceiling. Transport failure affects presentation only and must not affect boot progress.
