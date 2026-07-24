# Canonical Actuation Wire Format — Series 68

Safety messages use two fixed-width layers:

1. `SafetyWireHeaderV1` in `symthaea-hal`: exactly 88 bytes, little-endian,
   strict reserved fields, boot epoch, sequence, validity interval, payload
   length, cryptographic payload digest, and a transport corruption checksum.
2. `ActuationIntentV1` in `symthaea-exoskeleton`: exactly 64 bytes containing
   the complete system-binding digest, six signed torques in milli-newton-metres,
   stiffness and damping in permille, and final authority in permille.

No JSON, `serde` enum representation, architecture-sized integer, native
endianness, NaN, infinity, signed zero, or implicit trailing field is accepted
at the actuation process boundary.

Torque conversion rounds magnitude toward zero. Gains and authority round
strictly downward. The checksum detects malformed transport frames; it is not a
replacement for the authenticated digest and signature layers.
