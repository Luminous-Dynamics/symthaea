# Versioned Safety Contracts

Every cross-process safety message must carry a validated contract header. The
header binds the schema version, minimum compatible reader, producer role, boot
epoch, sequence, and payload digest.

Unknown, malformed, replayed, or incompatible messages are rejected and latch
the receiver. Version negotiation may establish compatibility before actuation;
it may never reinterpret an unknown payload on the active path.
