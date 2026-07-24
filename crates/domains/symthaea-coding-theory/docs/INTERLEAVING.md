# Rectangular block interleaving

For `r` equal-length component frames of `c` symbols, source storage is
row-major:

    frame0[0..c], frame1[0..c], ... frame(r-1)[0..c]

Transmission is column-major:

    frame0[0], frame1[0], ... frame(r-1)[0], frame0[1], ...

The stable coordinate maps are:

    wire = (source % c) * r + (source / c)
    source = (wire % r) * c + (wire / r)

A contiguous wire burst of length at most `r` therefore touches each row at
most once. This is particularly useful when each row is an independent
Reed-Solomon codeword: the interleaver converts one concentrated burst into
several sparse symbol errors that remain inside each decoder's radius.

The permutation adds latency and requires exactly agreed dimensions. It does
not authenticate data, increase minimum distance, or make beyond-capacity
corruption reliably detectable.
