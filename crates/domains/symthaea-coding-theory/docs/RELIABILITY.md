# Independent-errata reliability

For each symbol, the model has three mutually exclusive outcomes:

- clean with probability `1 - p_error - p_erasure`;
- unknown error with probability `p_error`;
- known erasure with probability `p_erasure`.

The inputs are exact rational probabilities. Analysis converts them to `f64`
only after validating their sum exactly with `u128` arithmetic.

The dynamic program tracks probability by errata weight. A clean symbol adds
zero, an erasure adds one, and an unknown error adds two. For a code with
minimum distance `d`, all mass at weights below `d` is algebraically guaranteed
recoverable:

    2 * unknown_errors + known_erasures < d

The reported outside-guarantee probability is not identical to decoder failure
probability. Beyond the unique-decoding radius, some patterns can still decode,
fail verification, or map ambiguously to another codeword. Authentication is
still required when undetected substitution is unacceptable.
