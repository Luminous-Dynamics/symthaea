# Deterministic Numeric Contract

Safety-relevant authority is quantized downward to integer permille before it
scales the final command. This prevents a platform-specific floating-point
rounding decision from increasing authority. Non-finite or out-of-range values
produce zero authority, and signed zero is canonicalized before transport.

The quantizer is deliberately asymmetric: losing less than 0.001 authority is
acceptable; gaining authority through rounding is not.
