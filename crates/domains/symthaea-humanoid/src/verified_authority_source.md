# Verified authority-source boundary

The public humanoid authority path accepts externally-originating operator, physical, epistemic and cognitive authority only after an application-selected verifier has authenticated a canonical source claim.

The signed statement binds the authentication scheme ID. The runtime evidence identity is derived from the complete verification decision, including verifier policy/keyset identity, revocation epoch and bounded verification window.

The humanoid crate does not implement issuer key management, signature algorithms, revocation infrastructure or trusted clock synchronization. Those remain explicit upstream trust services (for example Xenia/Mycelix).

This is internal engineering authority evidence, not legal or product-safety certification.
