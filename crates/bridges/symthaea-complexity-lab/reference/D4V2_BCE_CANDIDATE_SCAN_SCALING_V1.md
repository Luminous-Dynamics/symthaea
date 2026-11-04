# D4v2 dynamic-BCE candidate scan scaling v1

**Claim ceiling:** instrumented independent source-derived reference model. Not a D4 binary benchmark, formal proof, or solver performance claim.

## Adversarial family

Hidden variable h and selected variables a,b1,...,bk.

- T = (h OR a)
- O_i = (NOT h OR b_i), i=1...k

For blocker h on T, all k opposing resolvents are non-tautological, so the protected candidate order is [O1,...,Ok] and the first sentinel is O1.

Assign a=false, then b1=true through bk=true. Each opposing clause is deactivated in sequence. D4-style logic scans the current candidate array from index 1 until finding the first active candidate, swaps it into the sentinel position, and eventually deletes T when no active candidate remains.

The source-derived model checks exact projected support/PWMC at the deletion leaf. Full-model count changes from 1 to 2, which is allowed by projected BCE.

## Scan count

For this family, the scan count is:

S(k) = 1 + 2 + ... + (k-1) + (k-1)
     = (k-1)(k+2)/2

The final (k-1) term is the failed search when the last candidate is deactivated. This produces quadratic candidate inspections despite only k-1 watcher relocations and one dynamic deletion.

| Candidate count k | Inspections S | Relocations | Deletions |
|---:|---:|---:|---:|
| 2 | 2 | 1 | 1 |
| 4 | 9 | 3 | 1 |
| 8 | 35 | 7 | 1 |
| 16 | 135 | 15 | 1 |
| 32 | 527 | 31 | 1 |
| 64 | 2,079 | 63 | 1 |

The final sentinel stays at Ok across postUpdate. Backtracking restores clauses and assignments but not the watch permutation. A sibling that leaves Ok active must not delete T; the fixture checks projected support/PWMC there too.

## Interpretation

This is evidence that ETH-031 must account for S = candidate entries inspected, not just relocations R, deletions D and assignment/active-state events U. It motivates ETH-032's rollback-aware successor-index research. It does not establish real D4 runtime or a production speedup.

## Reproduction

Run python3 d4v2_bce_candidate_scan_scaling.py beside verify_d4v2_dynamic_bce_trace.py. The script imports the sibling source-derived reference model and instruments the same forward candidate scan.
