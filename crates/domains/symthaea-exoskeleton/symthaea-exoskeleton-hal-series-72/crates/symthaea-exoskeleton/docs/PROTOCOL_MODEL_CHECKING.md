# Actuation Protocol Model — Series 71

The protocol model exhaustively evaluates all 64 combinations of six
independent admission facts:

- canonical-header checksum validity,
- authorized producer role,
- expected boot epoch,
- validity-window freshness,
- zero reserved fields,
- exact system-binding digest.

Exactly one combination is admissible: all six facts true. The model executes
the real canonical decoder, contract-version receiver, and actuator-intent
decoder. It also verifies that replaying an otherwise valid contract is
rejected.

This is a bounded model of the admission boundary, not a proof of the entire
physical system. Its release value is detecting accidental permissive changes
to the wire and compatibility layers.
