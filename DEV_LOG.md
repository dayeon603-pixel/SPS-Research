# Development Log — SPS

A chronological record of experiments, decisions, and observations. Entries are numbered by research day (approximate, relative to project start).

---

## Day 1–3 — Problem Scoping

Noticed that models on GLUE benchmarks maintained high accuracy after synonym substitution, but cosine similarity between original and substituted representations varied widely — sometimes dropping below 0.6.

**Question:** Is this a problem with cosine similarity as a metric, or with the representations themselves?

Started reading: Jacovi & Goldberg (2020) on faithfulness, Wallace et al. (2019) on NLP attacks, Sinha et al. (2021) on syntactic robustness.

None of these exactly addressed the question. They measure vulnerability to adversarial input, not semantic invariance of representations.

---

## Day 5 — First Experiment

Applied random Gaussian noise ($\sigma = 0.01$) to RoBERTa-base embeddings. Measured output consistency (KL divergence between original and perturbed softmax outputs).

**Result:** Outputs were highly stable under noise. Interpreted this as "model is robust."

**Retrospective:** Wrong interpretation. Output stability under noise is a function of the output layer's saturation, not representation quality. The experiment was measuring the wrong thing.

---

## Day 8 — Synonym Substitution Test

Replaced tokens with WordNet synonyms (single substitution, random). Measured accuracy drop and cosine similarity between [CLS] representations.

**Result:** Accuracy barely changed. Cosine similarity dropped to 0.55–0.65 on average.

**Observation:** The model produces the same answer via very different internal paths. This is the first clear signal that accuracy and representation stability are decoupled.

---

## Day 12 — Formal Problem Statement

Wrote the first draft of the sensitivity functional:

$$S(f, x, T) = \frac{d_\mathcal{Y}(f(Tx), f(x))}{c(T, x)}$$

Issues immediately apparent:
- What is $d_\mathcal{Y}$? Output-space metric is not uniquely defined for classification
- $c(T, x)$ normalises by transformation magnitude, but magnitude is not well-defined for discrete token substitutions
- Taking the sup over arbitrary $T$ gives a measure that is dominated by pathological transformations

---

## Day 15 — Realisation: Need a Constrained Transformation Family

Recognised that the problem with Day 12's formulation is that there are no constraints on $T$. If $T$ can do anything, the sup is dominated by transformations that are semantically destructive.

Key insight: the transformation family must be **semantically constrained**. Only perturbations that preserve semantic content should enter the stability calculation.

Started drafting Definition 2 (admissible transformation family). First version required:
- Identity in $\mathcal{T}$
- Bounded magnitude $c(T,x) \leq \varepsilon$
- Some notion of semantic preservation (informal at this stage)

---

## Day 18 — Formalising Semantic Preservation

The informal "semantic preservation" requirement in A5 needed a formal definition.

Options considered:
1. Human annotation (expensive, not scalable)
2. Entailment checking via NLI model (circular — we are trying to evaluate the model)
3. Operational definition: transformations drawn from a curated family (synonym swap, back-translation, controlled paraphrase)

Chose option 3 as the primary approach with option 2 as a secondary validation tool. This means $\mathcal{T}$ is defined operationally, not axiomatically, which limits universality but enables tractable computation.

Added A5 formally with three axioms: Identity, Semantic preservation (operational), Measurability.

---

## Day 22 — First Complete Definition of SPS

Wrote:
$$\mathrm{SPS}_\varepsilon(f_\theta) := \exp\!\left(-\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)]\right)$$

Why exponential? 
- Maps $[0, \infty) \to (0, 1]$, giving an interpretable bounded score
- Connects to information-theoretic intuitions (SPS ≈ 1 − entropy of sensitivity distribution)
- Makes SPS multiplicative under composition (if compositions were independent, which they are not in general — see Theorem 3)

---

## Day 27 — Theorem 2 First Draft

Connected the sensitivity functional to the Jacobian. Wrote:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x) = \|J_{f_\theta}(x)\|_{A_x}$$

**Problem:** This is only true in the limit $\varepsilon \to 0$. For finite $\varepsilon$, the sensitivity functional captures non-linear effects that the Jacobian does not.

**Bigger problem (found later):** The original statement set a single-direction LHS equal to a sup-over-directions RHS. These are not equal — this is a full theorem error, not just a $\varepsilon \to 0$ issue.

---

## Day 30 — Composition Instability Discovery

Observed that composing two individually stable transformations ($T_1, T_2$ each with small sensitivity) did not guarantee small sensitivity for $T_2 \circ T_1$.

Example: Back-translation ($T_1$) followed by synonym substitution ($T_2$). Each has high individual SPS. Composed, the representation can drift further than either alone, because each step pushes in a different direction in representation space.

This motivated Theorem 3: $\mathrm{SPS}(\mathcal{T}_2 \circ \mathcal{T}_1) \leq \min(\mathrm{SPS}_1, \mathrm{SPS}_2)$.

Note: composition is **not** always unstable — this is an upper bound, not a claim that composition always degrades stability.

---

## Day 34 — Spectral Gap Idea

Reading on Lipschitz networks: a model that compresses information along semantic directions relative to arbitrary directions should be more stable.

Formalised as: the **semantic spectral gap** $\bar{\gamma} = 1 - \|J_f\|_{A_x} / \sigma_{\max}(J_f)$.

If $A_x$ directions are orthogonal to the top singular vector, the gap is large. This is a geometric property that existing metrics cannot detect.

Implementation question: how to estimate $\sigma_{\max}(J_f(x))$ without materialising the full Jacobian? Answer: randomised power iteration via JVPs.

---

## Day 38 — Theorem 2 Error Found

Reviewing the proof for Theorem 2 carefully:

Original: $\delta(x)$ (single direction) $=$ $\|J_f(x)\|_{A_x}$ (sup over all $A_x$).

These are not the same thing. $\delta(x)$ is a single JVP evaluation; the restricted norm is the sup over all unit vectors in $A_x$. The equality only holds if $A_x = \{v_T\}$ (singleton), which is not the general case.

**Decision:** Split into two-part theorem.
- Part (i): Pointwise — JVP gives the directional derivative for a fixed $T$
- Part (ii): Taking sup over all $T$ in the $\varepsilon \to 0$ limit gives the restricted operator norm (requires A2 for sup to be achieved)

---

## Day 41 — Proposition 1 Hole Found

Reviewing Proposition 1 proof: claimed $\mathrm{SPS} > 0$ from $\exp(-\mathbb{E}[\mathrm{Sens}]) > 0$.

But $\exp(-\infty) = 0$. If $\mathbb{E}[\mathrm{Sens}] = +\infty$, the claim fails.

Added **Assumption A4 (Integrability):** makes the strict positivity proof valid. Without A4, can only claim $\mathrm{SPS}_\varepsilon \geq 0$.

---

## Day 45 — New Definitions 5–8

To make the framework computationally useful, added:

- **Definition 5 (Semantic Spectral Gap):** $\bar{\gamma}(f,\mathcal{T};x) := 1 - \|J_f(x)\|_{A_x} / \sigma_{\max}(J_f(x))$
- **Definition 6 (Empirical SPS):** Monte Carlo estimator using $N$ samples — what `core.py` computes
- **Definition 7 (Layer-wise SPS):** Apply SPS to the map $x \mapsto h^{(\ell)}(x)$ for each layer $\ell$
- **Definition 8 (Relative SPS):** $\mathrm{rSPS} := \mathrm{SPS}_{\mathcal{T}} / \mathrm{SPS}_{\text{arb}}$ — ratio of semantic SPS to arbitrary-direction SPS

rSPS close to 1 means semantic directions are equally destabilising as arbitrary ones — the model is not allocating its sensitivity budget away from semantics. rSPS close to 0 means semantic directions are disproportionately destabilising.

---

## Day 50 — Code Architecture

Decided on module structure:
```
src/sps/
  core.py         — SPSEstimator (main interface)
  jacobian.py     — JVP computation, restricted norm, spectral gap
  transformations.py — T_emb, T_syn families
  metrics.py      — SPSReport, rSPS, layer-wise
  utils.py        — math utilities, seed setting
```

Key decision: use `torch.autograd.functional.jvp` rather than materialising the full Jacobian. Full Jacobian for a 125M-parameter model would be $768 \times 768 \times \text{seq\_len}^2$ — infeasible.

JVP is $O(\text{forward pass})$ per direction. With $K=8$ probe directions, cost is $8\times$ inference. Acceptable.

---

## Day 54 — Test Suite Design

Wrote `tests/test_jacobian.py` with an `ExactLinearModel(W)` where $J_f = W$ analytically.

This allows exact verification:
- `directional_derivative_norm(f, x, v)` should equal `||Wv||`
- Spectral gap = 0 when semantic direction = top singular vector of $W$
- Spectral gap ≈ 1 when semantic directions are in the null space of $W$

This is the hardest class of test to write (requires analytical ground truth) but the most valuable — catches Jacobian implementation bugs that higher-level tests would miss.

---

## Day 58 — README Rendering Bug

Pushed README to GitHub. Assumptions A3–A5 rendered as broken symbol strings.

Root cause: blockquote syntax (`>`) + nested math subscripts triggered GitHub's italic pass before the math renderer.

Fix: removed blockquotes, moved math to standalone `$...$` blocks.

**Note for future:** Never put LaTeX subscripts inside GitHub blockquotes. Always preview on GitHub before finalising.

---

## Day 60 — Current State

Full theoretical framework: ✓  
Full implementation: ✓  
Test suite (43 tests): ✓  
Experiment runner (`experiments/estimate_sps.py`): ✓  
README: ✓  

**Open:** Empirical comparison across model families. Need GPU time for RoBERTa-large, DeBERTa, GPT-2.
