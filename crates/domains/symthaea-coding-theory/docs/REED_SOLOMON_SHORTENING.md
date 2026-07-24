# Reed-Solomon shortening contract

A shortened systematic Reed-Solomon code removes a known all-zero prefix from a
larger parent message. With `z = k_parent - k_transmitted`:

    parent_message  = zero[z] || transmitted_message
    parent_codeword = zero[z] || transmitted_codeword

The parity tail is unchanged because leading zero symbols do not change the
all-zero encoder remainder. Decoder coordinates are transmitted coordinates;
callers do not add `z` to erasure or correction positions.

`ReedSolomonShortenedFrame` records both parent and transmitted dimensions,
checks the parent remains within the 255-symbol GF(2^8) limit, and provides
explicit expansion/contraction helpers. Contraction rejects any non-zero symbol
inside the purported shortened prefix.
