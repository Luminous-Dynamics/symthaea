# Symthaea Boot Protocol v1

`BootEvent` is an ordered presentation stream. Producers assign monotonically increasing sequence numbers within one boot-observation lineage. Consumers ignore duplicate or older sequence numbers and reject elapsed-time regressions from newer events.

`elapsed_ms` is measured from the start of the same observation lineage. It is presentation timing, not wall-clock time, and must never be derived from a clock that can jump backwards.

`BootSnapshot` is a point-in-time normalized state and may replace a consumer's reduced event state only after successful validation. A replacement snapshot may refine state at the same sequence number, but it must not have an older sequence number or move elapsed time backwards. Domain timestamps contained by a snapshot must not be later than the snapshot itself.

`BootHealth::Unknown` is deliberately distinct from `Normal`; consumers must never map missing observation to health.

Recovery of one domain does not implicitly restore whole-boot health. Only an authoritative snapshot or `BootReady` event may resolve aggregate health after prior degradation/failure.

Optional detail strings are operator-facing hints only. They are bounded and must not contain control characters. Raw diagnostic truth belongs to the journal.

## Wire boundary

The planned v1 transport is a local Unix datagram endpoint with a 4096-byte application message ceiling.

Receivers MUST reject datagrams larger than `MAX_WIRE_BYTES` **before deserialization**, then validate the version and decoded event/snapshot before reducing it. The protocol crate exposes `validate_datagram_size` so this ceiling is an executable contract rather than documentation only.

Malformed, unsupported, oversized, stale, or temporally inconsistent messages affect presentation only. They must never affect boot progress, generation blessing, authentication, or recovery.

## Authority boundary

The protocol reports normalized observations. It does not define boot success. Linux/systemd/NixOS remain authoritative, and host-specific Last-Known-Good promotion remains outside the renderer and outside visual lineage state.

See `BOOT_ECOLOGY_CONVERGENCE_V1.md` for the one-way protocol → ecology → renderer integration contract.
