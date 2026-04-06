# Failures & Dead Ends — SPS

This file documents approaches that did not work, formal errors in the theory, and implementation bugs. Each failure is recorded with its diagnosis and resolution.

---

## 1. Random Perturbation Testing

**Hypothesis:**
Apply random noise to embeddings and measure output consistency. High consistency = stable model.

**What happened:**
- Noise was semantically ungrounded — it disrupted syntax, morphology, and meaning simultaneously
- Could not separate "model is robust because representations are stable" from "model is robust because the output layer ignores most of the input"
- Results were dominated by output saturation effects (softmax squashing), not representation geometry

**Insight:**
Random perturbations cannot test semantic invariance. The perturbation family must be constrained to semantically neutral transformations. This led directly to the admissible family $\mathcal{T}$.

---

## 2. Output-Only Consistency Metric

**Hypothesis:**
Measure $d_\mathcal{Y}(f(x), f(Tx))$ directly. If this is small across transformations, the model is stable.

**What happened:**
- Models appeared "stable" even when internal representations changed dramatically
- The issue: output-space metrics collapse information. A model can learn to be output-stable through many internally inconsistent paths
- Counter-example: a model that memorises training labels will have $d_\mathcal{Y}(f(x), f(Tx)) = 0$ for all paraphrases of training points, but completely fails on novel paraphrases — falsely appearing stable

**Insight:**
Performance ≠ semantic stability. The metric must operate in representation space, not just output space. This motivated the Jacobian-based reformulation in Theorem 2.

---

## 3. Theorem 2 — Original Error

**The bug:**
The original Theorem 2 stated:

$$\delta(x) \text{ (single direction)} \;\leftrightarrow\; \sup_{v \in A_x} (\cdots) \text{ (all directions)}$$

These are not equivalent. The LHS is a single directional derivative along a specific perturbation direction; the RHS is the supremum over all directions in $A_x$. Setting them equal requires the supremum to be achieved, which is not guaranteed — especially when $A_x$ is not a singleton.

**Diagnosis:**
The equality was being asserted where only an inequality holds in general ($\leq$). The statement confused a pointwise instantiation with the operator norm.

**Fix:**
Split into two separate claims:
- **Part (i):** For a fixed $T$, the pointwise sensitivity equals the JVP: $\lim_{\varepsilon \to 0} \frac{d_\mathcal{Y}(f(Tx), f(x))}{c(T,x)} = \|J_f(x) \cdot v_T\|$ where $v_T = \lim (Tx - x)/\|Tx - x\|$.
- **Part (ii):** Taking the sup over all $T \in \mathcal{T}$ on both sides gives $\lim_{\varepsilon \to 0} \mathrm{Sens}_{\mathcal{T},\varepsilon}(f;x) = \|J_f(x)\|_{A_x}$.

Part (ii) requires A2 (compactness of $A_x$) for the sup to be achieved. Added explicitly.

---

## 4. Proposition 1 — Missing Integrability Assumption

**The bug:**
Original Proposition 1 claimed $\mathrm{SPS}_\varepsilon(f_\theta) > 0$ without assuming the sensitivity expectation is finite.

**Diagnosis:**
If $\mathbb{E}[\mathrm{Sens}] = +\infty$, then $\exp(-\mathbb{E}[\mathrm{Sens}]) = 0$, making SPS = 0. The strict positivity claim fails.

**Fix:**
Added **Assumption A4 (Integrability):** $\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)] < \infty$.
Proposition 1 now explicitly invokes A4.

---

## 5. Theorem 4 — Missing Chain Constraint

**The bug:**
Theorem 4 (triangle inequality under chained transformations) stated that for $T_1, T_2 \in \mathcal{T}$:

$$\mathrm{Sens}(f; T_2 \circ T_1 x) \leq \mathrm{Sens}(f; T_1 x) + \mathrm{Sens}(f; T_2 \circ T_1 x)$$

But the constraint that $c(T_2, T_1 x) \leq \varepsilon$ was not stated. Without this, $T_2 \circ T_1$ may violate the $\varepsilon$-radius constraint and fall outside the admissible region where Sens is defined.

**Fix:**
Added explicit hypothesis: $c(T_2, T_1 x) \leq \varepsilon$ must hold for the composed transformation to be admissible. This is a non-trivial requirement — A5 guarantees closure under composition only when both single-step magnitudes are within $\varepsilon/2$.

---

## 6. Notation Inconsistency — Missing $\varepsilon$ Subscript

**The bug:**
Theorems 3 and 4 in the original draft used $\mathrm{Sens}_\mathcal{T}$ (no $\varepsilon$), while Theorem 2 and Proposition 1 used $\mathrm{Sens}_{\mathcal{T},\varepsilon}$.

**Diagnosis:**
The $\varepsilon$-dependence is essential — the whole framework is parameterised by the perturbation radius. Dropping it implies a global (non-localised) sensitivity, which is a different and stronger claim.

**Fix:**
Restored $\mathrm{Sens}_{\mathcal{T},\varepsilon}$ throughout `theory/proofs.md`. All theorems now consistently carry the $\varepsilon$ subscript.

---

## 7. README Assumption Rendering Bug

**The bug:**
Assumptions A3, A4, A5 in the README were wrapped in blockquote syntax (`>`). On GitHub, this caused the markdown renderer to interpret nested subscripts (e.g., `_{\mathcal{T},\varepsilon}`) as italic markers, producing broken output: raw symbols mixed with partial formatting.

**Diagnosis:**
GitHub's CommonMark renderer processes italic markers before passing content to the math renderer. Blockquote indentation + underscore subscripts triggered the italic pass.

**Fix:**
- Removed blockquote wrappers from A3–A5
- Moved math expressions into standalone `$...$` display blocks on their own lines
- Restructured A5 (which has three sub-axioms) as a numbered list with math on separate lines
- Verified locally with a CommonMark preview before committing

---

## 8. HMM Synonym Direction Initialisation

**The bug:**
The `EmbeddingPerturbationFamily` attempted to build synonym directions from WordNet for the full tokeniser vocabulary (50k+ tokens). This caused:
- Memory spikes (attempting to load all WordNet synsets at module import)
- Silent fallback to random orthogonal directions for most tokens
- No warning to the user that synonym directions were unavailable

**Fix:**
- Added `vocab_size` cap (default: 10,000 most common tokens)
- Fallback to random orthogonal basis is now explicit and logged at `WARNING` level
- `build_wordnet_synonym_map` now takes `max_synonyms_per_token` to bound memory

---

## Summary

| # | Issue | Type | Status |
|---|---|---|---|
| 1 | Random perturbations → ungrounded results | Design | → admissible family $\mathcal{T}$ |
| 2 | Output-only metric → masked instability | Design | → Jacobian reformulation |
| 3 | Theorem 2 LHS/RHS mismatch | Theory error | Fixed — two-part theorem |
| 4 | Proposition 1 missing integrability | Theory error | Fixed — A4 added |
| 5 | Theorem 4 missing chain constraint | Theory error | Fixed — $c(T_2, T_1 x) \leq \varepsilon$ added |
| 6 | Missing $\varepsilon$ in Sens subscript | Notation | Fixed — restored throughout |
| 7 | README blockquote rendering | Documentation | Fixed — removed blockquotes |
| 8 | WordNet vocab memory spike | Implementation | Fixed — 10k cap + explicit fallback |
