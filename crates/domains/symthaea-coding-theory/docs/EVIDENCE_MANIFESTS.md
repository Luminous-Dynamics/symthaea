# Deterministic coding evidence manifests

`ReedSolomonErrataExperiment::manifest()` emits a stable semicolon-delimited
preregistration line. Version 1 records every input needed to replay an
exact-count mixed errata campaign:

- the complete Reed-Solomon interoperability profile;
- fixed message and codeword lengths (`k` and `n`);
- frame count and SplitMix64 seed;
- exact unknown-error and known-erasure counts;
- the erasure placeholder byte.

Example:

    symthaea-coding-evidence-v1;kind=rs-fixed-errata;profile=rs-gf256-p11b-g03-fcr11-msb-systematic-nsym8;k=32;n=40;frames=250;seed=000051a7e22a5eed;errors=2;erasures=4;placeholder=ee

The manifest is a canonical description, not an authentication mechanism.
Evidence bundles that cross trust boundaries should sign or MAC the manifest
and the resulting report with an appropriate cryptographic system.
