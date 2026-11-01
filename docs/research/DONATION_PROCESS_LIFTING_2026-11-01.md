# Lifting one-shot donations into evolving process certificates

**Date:** 2026-11-01  
**Status:** conditional composition lemma established; availability of required conditional donation seeds remains unproved.

## 1. The exact composition lemma

Let \((\mathcal F_t)_{t\ge0}\) be one global filtration and let \(\tau\) be a predictable integer-valued checkpoint. For a fixed intersection null \(H_S\), suppose:

1. The nonnegative seed \(D_\tau^S\) is \(\mathcal F_\tau\)-measurable and satisfies
   \[
   \mathbb E_P[D_\tau^S\mid\mathcal F_{\tau-1}]\le1
   \quad\text{a.s. for every }P\in H_S.
   \]
2. \(G_t^S\), \(t\ge\tau\), is nonnegative, adapted, \(G_\tau^S=1\), and is a test supermartingale after \(\tau\) in the same filtration:
   \[
   \mathbb E_P[G_t^S\mid\mathcal F_{t-1}]\le G_{t-1}^S,
   \quad t>\tau.
   \]
3. The post-checkpoint process is defined on paths after the checkpoint, with predictable checkpoint selection and the usual integrability conditions.

Define
\[
C_t^S=\begin{cases}
1,&t<\tau,\\
D_\tau^S G_t^S,&t\ge\tau.
\end{cases}
\]

Then \(C^S\) is a nonnegative test supermartingale starting at one. For a deterministic checkpoint, the only nontrivial one-step check is at \(\tau\), where the conditional seed bound gives \(\mathbb E[C_\tau^S\mid\mathcal F_{\tau-1}]\le1=C_{\tau-1}^S\). After the checkpoint, \(D_\tau^S\) is measurable with respect to the past and can be factored out of the conditional expectation. For predictable random \(\tau\), partition each one-step conditional expectation into \(\{\tau>t\}\), \(\{\tau=t+1\}\), and \(\{\tau\le t\}\); predictability makes the checkpoint event known at time \(t\), and the same two inequalities apply on their respective events.

This is a composition lemma, not a construction of the seed. It is only useful when both the conditional-seed requirement and the post-checkpoint process conditions are independently certified.

## 2. Why an ordinary e-value cannot simply be frozen

An ordinary e-value guarantees only \(\mathbb E[D]\le1\). It does not imply the conditional bound given the complete global history before the checkpoint.

Exact counterexample: let \(U\sim\mathrm{Bernoulli}(1/2)\) be revealed in \(\mathcal F_0\), and at checkpoint one set \(D_1=2U\). Then
\[
\mathbb E[D_1]=1,\qquad
\mathbb E[D_1\mid\mathcal F_0]=2U.
\]
Take the continuation \(G_t\equiv1\). The process is \(C_0=1,C_1=2U\). The statewise stopping time \(T=0\) if \(U=0\), and \(T=1\) if \(U=1\), is a stopping time because \(U\) is already in \(\mathcal F_0\). It yields
\[
\mathbb E[C_T]=\tfrac12(1)+\tfrac12(2)=1.5>1.
\]
Thus the frozen one-shot e-value fails the test-supermartingale / arbitrary-stopping guarantee in the richer filtration, even though its unconditional mean is exactly one.

Executable finite-state arithmetic for this counterexample is in `donation_process_lifting.py` and `test_donation_process_lifting.py`. The helper checks finite models; it does not estimate conditional validity from empirical data.

## 3. What the UAI 2026 donation theorem actually assumes

Xu, Fischer & Ramdas (2026), arXiv:2603.24792:

- Equation (4) defines each arriving e-value by the unconditional condition \(\mathbb E[E_t]\le1\) under its null.
- Their e-value filtration is \(\mathcal F_t=\sigma(E_1,\ldots,E_t)\).
- Definition 4 defines online compound e-values by the unconditional weighted expectation inequality \(\sum_{i\in\mathcal N}\gamma_i\mathbb E[\widetilde E_i]\le1\), and defines donations through a pathwise weighted budget.
- Proposition 6 and its proof use unconditional expectations and that weighted budget.

Therefore the published donation e-LOND result, as stated, supplies a valid online SupFDR decision procedure for its one-shot arriving e-value model. It does **not** by those statements alone supply \(\mathbb E[D_\tau^S\mid\mathcal F_{\tau-1}^{\rm global}]\le1\) for an arbitrary finer process filtration. This is not a refutation of the published SupFDR theorem; it is a boundary on reusing its compound e-values as dynamic test-supermartingale seeds.

## 4. A possible continuation and its zero edge case

If coordinate \(M_{i,t}\) is a nonnegative test supermartingale in the same global filtration and \(M_{i,\tau}>0\), then \(M_{i,t}/M_{i,\tau}\), for \(t\ge\tau\), is a conditional test supermartingale starting at one. If \(M_{i,\tau}=0\), nonnegativity and the supermartingale property imply it remains zero almost surely under the relevant null; define the continuation asset as one at \(\tau\) and zero afterwards (an immediate absorbing loss), rather than divide by zero or allow it to revive.

A projective portfolio merger may be built from these continuation assets, but it is still required to prove that its intersection-specific family is horizon-coherent and neutral under future-hypothesis padding. The seed family must satisfy the same projectivity. The product is not automatically an e-collection just because the scalar composition lemma is valid.

## 5. What would unlock this research track

1. A donation theorem with conditional validity relative to the complete global filtration, or an explicit filtration-lifting adjuster and its cost.
2. A seed family for every finite intersection, with monotonicity/projectivity under future extensions—not merely one compound e-value stream for a rejection algorithm.
3. A post-checkpoint continuation family with exact neutral padding and zero-safe normalization.
4. A proof of the composition theorem at predictable random checkpoints.
5. Full stopped-FDR and SupFDR comparisons, keeping one-shot donation, dynamic e-closure, and adjusted anytime-valid procedures in separate matched lanes.

## References

- Xu, Fischer & Ramdas (2026), [Improving online FDR procedures via online analogs of e-closure and compound e-values](https://arxiv.org/abs/2603.24792).
- Choe & Ramdas (2026), [Combining evidence across filtrations](https://academic.oup.com/jrsssb/article/88/4/1492/8627076).
- Tavyrikov, Goeman & de Heide (2026), [Carefree multiple testing with e-processes](https://doi.org/10.1214/26-EJS2546).
- de Heide (2026), [Dynamic e-closure for online hypotheses with any-time-valid evidence](https://arxiv.org/abs/2608.09927).