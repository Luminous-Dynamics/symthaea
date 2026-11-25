# ETH reduction comprehension pilot — unique extension versus equisatisfiability

**Task:** `ETH-REDUCTION-COMPREHENSION-TOY-001`  
**Related:** [#10811](https://github.com/Luminous-Dynamics/symthaea/issues/10811), [#10835](https://github.com/Luminous-Dynamics/symthaea/issues/10835), [#10809](https://github.com/Luminous-Dynamics/symthaea/pull/10809)  
**Machine fixture:** `ETH_REDUCTION_COMPREHENSION_TOY_V1.json`

## 1. Source problem

[
F(p,q,r)=(p\land q)\lor r.
]

This is a small source problem with three original Boolean variables.

## 2. Correct Tseitin encoding

Introduce two fresh variables:

[
u\leftrightarrow(p\land q),\qquad
s\leftrightarrow(u\lor r),
]

and assert (s).

The CNF clauses are:

1. (\neg u\lor p)
2. (\neg u\lor q)
3. (u\lor\neg p\lor\neg q)
4. (s\lor\neg u)
5. (s\lor\neg r)
6. (\neg s\lor u\lor r)
7. (s) (root assertion)

For every assignment to the original variables, the gate biconditionals force exactly one pair ((u,s)). The root assertion makes the encoding satisfiable precisely when (F) is true.

## 3. Exhaustive finite result

| p | q | r | F | Correct extensions | Mutated extensions |
|---|---|---|---|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 | 1 | 1 |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 | 1 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 | 1 | 1 |
| 1 | 1 | 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 1 | 1 | **2** |
| **Total** | | | **5 source models** | **5** | **6** |

Thus, for this fixture:
- source #SAT count = 5;
- correct Tseitin #SAT count = 5;
- mutated encoding #SAT count = 6.

The correct encoding has one auxiliary extension iff the source formula is true.

## 4. Adversarial mutation

Delete the clause:

[
u\lor\neg p\lor\neg q.
]

The remaining AND-gate clauses enforce:

[
u\rightarrow(p\land q),
]

but not the converse.

For (p=q=r=1), both (u=0) and (u=1) are allowed, while (s=1) satisfies the root and OR gate. The same source assignment therefore has two target extensions.

The mutation still preserves the yes/no decision for every original assignment in this finite fixture. It does **not** preserve the number of satisfying assignments and does **not** satisfy unique extension.

This is the key distinction:

[
\text{equisatisfiable}\;\not\Rightarrow\;\text{parsimonious}.
]

And the stronger property needed here is:

[
\forall x,\quad
\#\{y:E(x,y)\}=\begin{cases}
1,&F(x)=\mathrm{true},\\
0,&F(x)=\mathrm{false}.
\end{cases}
]

## 5. What Symthaea should learn to explain

A good explanation must do more than say "Tseitin preserves satisfiability." It must identify:

- which direction of each gate biconditional constrains the auxiliary variable;
- why asserting the root preserves decision correctness;
- why the reverse gate implication is necessary for unique extension in a composed circuit;
- how a single missing clause changes a #SAT count without changing SAT/UNSAT answers;
- which reduction guarantee may safely be transferred to a downstream solver or complexity claim.

The system should be asked to classify the correct encoding as decision-preserving, unique-extension and parsimonious; classify the mutation as decision-preserving only; and explain the finite witness causing the difference.

## 6. Claim boundary

This is an exhaustive reference check for one formula over eight original assignments and four auxiliary assignments per original assignment. It is not a generic proof of the Tseitin construction, not a validation of the Rust implementation in PR #10809, and not an asymptotic result.

The next formal step is to prove the gate truth-table lemma once per connective and then compose those lemmas structurally by induction on the formula tree. Lean should prove the generic statement; the finite oracle remains a mutation-sensitive regression test.
