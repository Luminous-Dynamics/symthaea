# symthaea-tpm-reference-current-independence

Thin TPM/reference adapter over ASSURE-015F current-head-qualified verifier independence.

The adapter preserves the cross-stage theorem introduced by the earlier TPM verifier-diversity gate: campaign verification must precede obligation verification and obligation review must occur within a reviewed maximum lag. It no longer treats different flat fault-domain strings as the final independence proof.

Instead, `left` is role-bound to the campaign verifier and `right` to the obligation verifier. The adapter pins the exact graph, relation-completeness policy, separation policy, campaign lifecycle-head authority policy, and obligation lifecycle-head authority policy expected by the TPM program.

Only an opaque `CurrentIndependenceQualification` whose parent ASSURE-015D qualification carries the exact campaign/obligation verification times can produce the stronger TPM qualification.

The resulting report/receipt are evidence only. The runtime-only TPM qualification grants no physical authority.
