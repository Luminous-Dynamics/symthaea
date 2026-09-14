# symthaea-evidence-current-independence

ASSURE-015F binds lifecycle-head currentness to exact-use verifier independence.

The crate consumes the opaque `ActiveIndependenceQualification` from ASSURE-015D and two opaque `CurrentActiveVerifierProfile` capabilities from ASSURE-015E. It refuses to promote them unless every ordered profile/record/graph/completeness/provenance/lifecycle/time binding agrees.

Both currentness attestations must also bind one caller-supplied composition challenge nonce. This turns two independent currentness checks into one atomic composition context rather than allowing unrelated currentness sessions to be spliced together.

The resulting `CurrentIndependenceQualification` is non-serializable and binds both lifecycle-head authority policy digests, both head statement/currentness attestation digests, both head sequences, the ASSURE-015D qualification digest, the separation-policy digest, the shared challenge and exact use-time.

It grants no physical authority and does not establish trusted time or universal independence.
