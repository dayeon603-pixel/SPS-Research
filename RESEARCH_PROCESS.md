# Research Process — Structured Perturbation Stability (SPS)

This document tracks the full thinking process behind this project, including failed approaches, design decisions, and unresolved questions. The goal is to make the development process transparent and reproducible.

**Paper:** Kang, Dayeon. "Structured Perturbation Stability: An Operator-Restricted Framework for Measuring Semantic Invariance in Transformer Architectures." *1st AI Agent Journal*, 2026.

---

## 1. Initial Motivation

The project started from a single uncomfortable observation:

> Large language models maintain high task accuracy even when their internal representations become unstable under small, semantically neutral perturbations.

The standard interpretability literature treats accuracy as a proxy for model understanding. If a model gets the right answer, it must have learned the right representation — or so the assumption goes.

I became suspicious of this after noticing that models can recover identical outputs from meaningfully different internal states, and fail on paraphrases that preserve all semantic content. Existing metrics (accuracy, F1, embedding cosine similarity) had no principled way to separate these cases.

The question became: **can we define a metric that measures semantic invariance in representation space, independent of task performance?**

---

## 2. Early Hypotheses

Starting assumptions (all later revised or rejected):

- Model accuracy reflects stable internal representations
- Small input perturbations should produce small, consistent output changes
- Existing robustness benchmarks (AdvGLUE, CheckList) measure what we care about
- Cosine similarity between representations is a sufficient stability proxy

All four turned out to be wrong in ways that directly shaped the framework.

---

## 3. Failed Approaches

See `FAILURES.md` for full detail. Summary:

### 3.1 Direct Random Perturbation Testing

Applied Gaussian noise to token embeddings, observed output consistency. Problems:
- No semantic grounding — noise disrupts meaning and form simultaneously
- Could not distinguish stability under paraphrase from stability under nonsense

### 3.2 Output-Only Consistency Metrics

Measured $d_\mathcal{Y}(f(x), f(Tx))$ without constraining $T$. Problems:
- Models were "stable" under arbitrary corruption because of output-layer saturation
- Performance ≠ semantic stability. A model can score $\mathrm{SPS} \approx 0$ and still get 90% accuracy

### 3.3 Existing Robustness Benchmarks

AdvGLUE and similar datasets test adversarial stability, not semantic invariance. They optimize for attacks, not for a neutral characterization of representation geometry.

---

## 4. Key Turning Point

The breakthrough was recognising that perturbation families need to be **semantically constrained**:

> Not all input changes are equal. Only structure-preserving transformations should count toward a stability measurement.

This led to the concept of an **admissible transformation family** $\mathcal{T}$, where every $T \in \mathcal{T}$ must:
- Preserve semantic content of the input
- Have bounded transformation magnitude $c(T, x) \leq \varepsilon$
- Include the identity (zero-perturbation baseline)

Once $\mathcal{T}$ is constrained, sensitivity becomes a meaningful geometric quantity: how much does the model output move when the input moves in a semantically neutral direction?

---

## 5. Core Theoretical Insight

The central result is a clean decoupling:

> A model can achieve high task accuracy while having low SPS. Conversely, a model can have high SPS and still make errors. **SPS and accuracy measure orthogonal properties.**

Formally, the sensitivity functional at point $x$ is:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) := \sup_{\substack{T \in \mathcal{T} \\ 0 < c(T,x) \leq \varepsilon}} \frac{d_\mathcal{Y}(f_\theta(Tx),\, f_\theta(x))}{c(T, x)}$$

And the global SPS score aggregates this over the data distribution:

$$\mathrm{SPS}_\varepsilon(f_\theta) := \exp\!\left(-\mathbb{E}_{x \sim \mathcal{D}}\bigl[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x)\bigr]\right) \in (0, 1]$$

The exponential map gives an interpretable bounded score: SPS = 1 means perfect invariance, SPS → 0 means the model is highly sensitive to structure-preserving changes.

---

## 6. Operator-Restricted Norm and Spectral Gap

A key extension connects SPS to Jacobian geometry. For a fixed input $x$, let $A_x \subseteq \mathbb{S}^{d-1}$ be the set of directions that semantic transformations generate. The **$A_x$-restricted operator norm** is:

$$\|J_{f_\theta}(x)\|_{A_x} := \sup_{v \in A_x} \|J_{f_\theta}(x)\, v\|$$

This is smaller than the full spectral norm $\sigma_{\max}(J_{f_\theta}(x))$ whenever semantic directions are not aligned with the top singular vector — which they should not be in a well-trained model.

The **semantic spectral gap** then measures how much the model compresses semantic directions relative to arbitrary directions:

$$\bar{\gamma}(f_\theta, \mathcal{T};\, x) := 1 - \frac{\|J_{f_\theta}(x)\|_{A_x}}{\sigma_{\max}(J_{f_\theta}(x))} \in [0, 1]$$

A large spectral gap means the model's most sensitive directions are geometrically orthogonal to the semantic manifold — a desirable property that existing metrics do not capture.

---

## 7. Iterative Development

### Version 1 — Sensitivity Functional Only
- Defined $\mathrm{Sens}_{\mathcal{T},\varepsilon}$ for a fixed family of synonym substitutions
- No operator norm connection
- Could not distinguish layer-wise contributions

### Version 2 — Jacobian Connection
- Introduced Theorem 2 (directional sensitivity = Jacobian-vector product)
- **Error found:** original Theorem 2 set single-direction LHS equal to multi-direction sup RHS — not equivalent
- Corrected to two-part theorem: (i) single direction via JVP, (ii) $\varepsilon \to 0$ limit gives restricted operator norm

### Version 3 — Spectral Gap and Full Definitions
- Added Definitions 5–8: spectral gap, empirical SPS, layer-wise SPS, relative SPS (rSPS)
- Added Assumptions A4 (integrability) and A5 (family axioms) — Proposition 1 was unproven without A4
- Added Theorem 4 fix: missing constraint $c(T_2, T_1 x) \leq \varepsilon$ added as explicit hypothesis
- Fixed notation: $\varepsilon$ subscript restored to $\mathrm{Sens}$ throughout

### Version 4 — Code and Experiments
- Implemented `StructuredSensitivityEstimator` using `torch.autograd.functional.jvp`
- Implemented `SpectralGapResult` with randomized power iteration probing
- Tested on RoBERTa-base with synonym substitution ($T_{\mathrm{syn}}$) and embedding perturbation ($T_{\mathrm{emb}}$) families
- Relative SPS (rSPS) defined: ratio of semantic SPS to arbitrary-direction SPS
- Layer-wise profiler reveals which transformer layers are semantically most sensitive

---

## 8. Theoretical Results Summary

| Result | Statement |
|---|---|
| Proposition 1 | $\mathrm{SPS}_\varepsilon \in (0, 1]$ under A3, A4 |
| Theorem 1 | Larger $\mathcal{T}$ → lower SPS (monotonicity) |
| Theorem 2 (corrected) | (i) Pointwise sens. = JVP norm; (ii) $\varepsilon \to 0$ limit = restricted op. norm |
| Theorem 3 | Composition of transformations: $\mathrm{SPS}(\mathcal{T}_2 \circ \mathcal{T}_1) \leq \min(\mathrm{SPS}(\mathcal{T}_1), \mathrm{SPS}(\mathcal{T}_2))$ |
| Theorem 4 | Triangle inequality under chained transformations (requires $c(T_2, T_1 x) \leq \varepsilon$) |
| Theorem 5 | SPS lower-bounds worst-case output perturbation magnitude |
| Corollary 1 | $\mathrm{rSPS} \in (0, 1]$; rSPS = 1 iff restricted norm = full spectral norm |
| Proposition 2 | Spectral gap $\bar{\gamma} \geq 1 - \mathrm{rSPS}^{-1}$ under Lipschitz regularity |

---

## 9. Open Questions

- **Scaling:** How does SPS behave as model size scales? Is there a phase transition at a certain parameter count?
- **Training signal:** Can SPS be directly optimised as a regularisation objective without collapsing representations?
- **Multimodal extension:** How to define admissible families $\mathcal{T}$ over image-text pairs?
- **Composition stability:** Theorem 3 gives an upper bound on composed SPS but the bound may be loose — tighter characterisation needed
- **Calibration:** Is rSPS a reliable diagnostic across architectures, or is it architecture-dependent?

---

## 10. What Comes Next

- Empirical SPS comparison across RoBERTa, BERT, DeBERTa, GPT-2 on the same transformation families
- Adversarial SPS: construct $\mathcal{T}$ families specifically designed to maximise sensitivity while preserving semantics (adversarial paraphrase generation)
- Geometric visualisation of $A_x$ directions vs. top singular vectors of $J_{f_\theta}(x)$
- Multimodal SPS applied to vision-language models (CLIP, LLaVA)
- SPS as a fine-tuning regulariser: add $\lambda \cdot \mathrm{Sens}_{\mathcal{T},\varepsilon}$ to the training loss
