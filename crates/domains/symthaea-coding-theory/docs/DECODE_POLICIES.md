# Reed-Solomon decode policies

The algebraic correction radius is an upper bound, not always the operational
budget a protocol wants to accept. `ReedSolomonDecodePolicy` lets a caller state
a tighter envelope:

    2 * max_unknown_errors + max_known_erasures <= parity_symbols

The decoder still performs its normal post-correction syndrome verification.
Recovered data is released only when the observed unknown corrections and the
declared erasure count also fit the caller policy.

This is useful for staged degradation, safety cases, storage health thresholds,
and protocols that want to reject unusually damaged frames before accepting
reconstructed data. It is not authentication and cannot prove that a
beyond-radius word was not transformed into another valid codeword.
