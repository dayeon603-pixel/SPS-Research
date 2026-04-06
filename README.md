<p align="center">
  <strong>Structured Perturbation Stability (SPS)</strong>
</p>

<p align="center">
  <em>An Operator-Restricted Framework for Measuring Semantic Invariance in Transformer Architectures</em>
</p>

<p align="center">
  Dayeon Kang &nbsp;·&nbsp; MICA International Scholars &nbsp;·&nbsp; 1st AI Agent Journal (2026)
</p>

<p align="center">
  <a href="./Structured_Perturbation_Stability__An_Operator_Restricted_Framework_for_Measuring_Semantic_Invariance_in_Transformer_Architectures.pdf">Paper</a> ·
  <a href="./theory/definitions.md">Definitions</a> ·
  <a href="./theory/proofs.md">Proofs</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#theoretical-framework">Theory</a>
</p>

---

## Overview

Standard robustness metrics evaluate stability under arbitrary or adversarial perturbations — they do not distinguish between random noise and transformations that preserve semantic meaning.

**Structured Perturbation Stability (SPS)** addresses this gap. Given a neural network $f_\theta : \mathcal{X} \to \mathcal{Y}$ and an admissible family $\mathcal{T}$ of semantic-preserving transformations, SPS measures model sensitivity *restricted to* $\mathcal{T}$. A high SPS value means the model is genuinely invariant to meaning-preserving variation — not merely smooth under arbitrary noise.

The central question SPS answers: **does the model ignore what it should ignore?**

---

## Theoretical Framework

### Formal Assumptions

> **A1 (Fréchet Differentiability):** $f_\theta$ is Fréchet differentiable on $\mathrm{supp}(\mathcal{D})$.  
> **A2 (Transformation Compactness):** The normalized perturbation direction set $A_x^{(\varepsilon)}$ is compact in $\mathbb{S}^{d-1}$ for each $x, \varepsilon$.  
> **A3 (Measurability):** $x \mapsto \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)$ is $\mathcal{D}$-measurable.  
> **A4 (Integrability):** $\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)] < \infty$.  
> **A5 (Family Axioms):** $\mathcal{T}$ contains the identity, is semantically preserving, and is closed under composition.

### Definition 1 — Structured Local Sensitivity

For $\varepsilon > 0$:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) \;:=\; \sup_{\substack{T \in \mathcal{T} \\ 0 < c(T,x) \leq \varepsilon}} \frac{d_\mathcal{Y}(f_\theta(x),\, f_\theta(Tx))}{c(T,x)}$$

where $c(T, x) = \|Tx - x\|$ is the transformation magnitude. This is the operator-restricted analogue of the local Lipschitz constant.

### Definition 2 — Structured Perturbation Stability

$$\mathrm{SPS}_\varepsilon(f_\theta) \;:=\; \exp\!\Bigl(-\,\mathbb{E}_{x \sim \mathcal{D}}\bigl[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x)\bigr]\Bigr)$$

Under A3–A4: $\quad 0 < \mathrm{SPS}_\varepsilon(f_\theta) \leq 1$. Higher values = stronger semantic invariance.

---

## Main Results

### Proposition 1 — Boundedness

Under A3–A4, for any $f_\theta$: $\quad 0 < \mathrm{SPS}_\varepsilon(f_\theta) \leq 1$.

*The strict lower bound requires A4 (integrability). Without it, $\mathbb{E}[\mathrm{Sens}] = \infty$ is possible, giving $\mathrm{SPS} = 0$.*

### Theorem 1 — Family Monotonicity

If $\mathcal{T}_1 \subseteq \mathcal{T}_2$, then:

$$\mathrm{Sens}_{\mathcal{T}_1,\varepsilon}(f_\theta;\, x) \;\leq\; \mathrm{Sens}_{\mathcal{T}_2,\varepsilon}(f_\theta;\, x) \qquad\text{and}\qquad \mathrm{SPS}_\varepsilon^{(\mathcal{T}_1)}(f_\theta) \;\geq\; \mathrm{SPS}_\varepsilon^{(\mathcal{T}_2)}(f_\theta)$$

### Theorem 2 — Differential Characterization *(corrected)*

Under A1–A2, suppose $\mathcal{T}$ contains all perturbations $T_\alpha^v(x) = x + \alpha v$ for $v \in A_x$.

**(i) Directional derivative:** For each $v \in A_x$, $\|v\| = 1$:

$$\lim_{\alpha \to 0} \frac{\|f_\theta(x + \alpha v) - f_\theta(x)\|}{\alpha} = \|J_{f_\theta}(x)\, v\|$$

**(ii) Restricted operator norm:** As $\varepsilon \to 0$:

$$\lim_{\varepsilon \to 0}\, \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) \;=\; \sup_{\substack{v \in A_x \\ \|v\| = 1}} \|J_{f_\theta}(x)\, v\| \;=:\; \|J_{f_\theta}(x)\|_{A_x}$$

This is the $A_x$-**restricted operator norm** of the Jacobian — the maximum output sensitivity over admissible semantic directions.

### Theorem 3 — Lipschitz Stability Bound

If $f_\theta$ is globally $L$-Lipschitz: $\quad \mathrm{SPS}_\varepsilon(f_\theta) \geq e^{-L}$

### Theorem 4 — Sequential Transformation Stability

For $T_1, T_2 \in \mathcal{T}$ with $c(T_1, x) \leq \varepsilon$ and $c(T_2, T_1 x) \leq \varepsilon$:

$$d_\mathcal{Y}(f_\theta(x),\, f_\theta(T_2 T_1 x)) \;\leq\; \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x)\, c(T_1, x) \;+\; \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, T_1 x)\, c(T_2, T_1 x)$$

### Theorem 5 — Semantic Stability Representation Theorem

If $f_\theta = g_\theta \circ \phi_\theta$ and $\phi_\theta(Tx) = \phi_\theta(x)$ for all $T \in \mathcal{T}$, then:

$$f_\theta(Tx) = f_\theta(x), \qquad \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = 0, \qquad \mathrm{SPS}_\varepsilon(f_\theta) = 1$$

---

## New Results

### Definition 5 — Semantic Spectral Gap

$$\bar{\gamma}(f_\theta, \mathcal{T};\, x) \;:=\; 1 - \frac{\|J_{f_\theta}(x)\|_{A_x}}{\sigma_{\max}(J_{f_\theta}(x))} \;\in\; [0,\, 1]$$

- $\bar{\gamma} \to 0$: worst-case sensitivity direction is semantic — the model is maximally sensitive where it should be invariant.
- $\bar{\gamma} \to 1$: Jacobian annihilates all semantic directions — perfect invariance.

### Corollary 1 — Spectral Gap Stability Bound

If $\bar{\gamma}(f_\theta, \mathcal{T};\, x) \geq \gamma_0 > 0$ $\mathcal{D}$-a.s. and $f_\theta$ is $L$-Lipschitz:

$$\mathrm{SPS}_\varepsilon(f_\theta) \;\geq\; \exp\!\bigl(-(1 - \gamma_0)\, L\bigr) \;\geq\; e^{-L}$$

Strictly tighter than Theorem 3 whenever $\gamma_0 > 0$.

### Definition 6 — Empirical SPS Estimator

$$\widehat{\mathrm{SPS}}_\varepsilon^{(n,m)}(f_\theta) \;:=\; \exp\!\left(-\frac{1}{n}\sum_{i=1}^{n} \max_{j=1,\ldots,m} \frac{d_\mathcal{Y}(f_\theta(x_i),\, f_\theta(\tau_j(x_i)))}{c(\tau_j,\, x_i)}\right)$$

Converges a.s. to $\mathrm{SPS}_\varepsilon(f_\theta)$ as $n, m \to \infty$ (under A1–A4).

### Definition 7 — Layer-wise SPS (Transformer-specific)

$$\mathrm{SPS}_\varepsilon^{(l)}(f_\theta) \;:=\; \mathrm{SPS}_\varepsilon(f_\theta^{(l)}), \qquad l = 0, 1, \ldots, L$$

The **SPS depth profile** characterizes how semantic stability accumulates through transformer depth.

### Definition 8 — Relative SPS

$$\mathrm{rSPS}_\varepsilon(f_\theta;\, \mathcal{T}) \;:=\; \frac{\mathrm{SPS}_\varepsilon^{(\mathcal{T})}(f_\theta)}{\mathrm{SPS}_\varepsilon^{(\mathcal{T}_{\mathrm{arb}})}(f_\theta)}$$

- $\mathrm{rSPS} > 1$: model is more stable to semantic perturbations than arbitrary noise. *(Desired.)*
- $\mathrm{rSPS} < 1$: pathological — model is more sensitive to semantics than random noise.

---

## Repository Structure

```
SPS/
├── src/sps/
│   ├── core.py             # Definitions 1, 2, 6 — SPSEstimator
│   ├── transformations.py  # T_emb, T_syn families, WordNet synonym map
│   ├── jacobian.py         # Theorem 2, Definition 5, Corollary 1
│   ├── metrics.py          # Definitions 7, 8 — relative SPS, layerwise SPS
│   └── utils.py            # Divergence functions, seed management
├── experiments/
│   └── estimate_sps.py     # Full empirical evaluation on RoBERTa-base
├── tests/
│   ├── test_core.py        # Proposition 1, Theorems 1, 5
│   ├── test_jacobian.py    # Theorem 2, Definition 5, Corollary 1
│   └── test_transformations.py
├── theory/
│   ├── definitions.md      # All definitions with formal assumptions
│   └── proofs.md           # All theorem statements + proof sketches
├── configs/
│   └── default.yaml
└── pyproject.toml
```

---

## Quick Start

**Install:**
```bash
pip install -e ".[experiments]"
# For WordNet synonym maps:
python -m nltk.downloader wordnet omw-1.4
```

**Run the experiment:**
```bash
python experiments/estimate_sps.py --epsilon 0.1 --m-transforms 32
# Skip layer-wise analysis for speed:
python experiments/estimate_sps.py --skip-layerwise
# GPU:
python experiments/estimate_sps.py --device cuda
```

**Python API:**
```python
from transformers import AutoModel, AutoTokenizer
from sps import (
    SPSConfig, EmbeddingPerturbationFamily, EmbeddingPerturbationConfig,
    build_sps_estimator, spectral_gap, set_seed,
)

set_seed(42)
model = AutoModel.from_pretrained("roberta-base").eval()
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Build transformation family
t_emb = EmbeddingPerturbationFamily(
    embedding_layer=model.get_input_embeddings(),
    config=EmbeddingPerturbationConfig(n_directions=8),
)

# Estimate SPS
config = SPSConfig(epsilon=0.1, m_transforms=32, divergence="cosine")
estimator = build_sps_estimator(model, t_emb, config)
result = estimator.estimate(iter(batches))  # batches: list of {input_ids, attention_mask, embeddings}
print(f"SPS = {result['sps']:.4f}")

# Spectral gap (Definition 5)
gap = spectral_gap(lambda emb: model(inputs_embeds=emb).last_hidden_state[:, 0], emb, directions)
print(f"Mean spectral gap γ̄ = {gap.mean_gap:.4f}")
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## Geometric Interpretation

Semantic transformations define structured directions in input space. The transformation orbit of $x$ is:

$$\mathcal{O}_\mathcal{T}(x) := \{ Tx : T \in \mathcal{T} \}$$

Maximal semantic stability ($\mathrm{SPS} = 1$) arises when the representation $\phi_\theta$ collapses every orbit to a single point:

$$\phi_\theta(Tx) = \phi_\theta(x) \quad \forall\, T \in \mathcal{T}$$

In differential terms, this requires the Jacobian to annihilate all semantic directions:

$$J_{f_\theta}(x)\, v = 0 \quad \forall\, v \in A_x$$

The spectral gap $\bar{\gamma}$ quantifies how close a model is to this ideal — it measures the separation between semantic directions and the worst-case sensitivity direction.

---

## Roadmap

**Near-term**
- [ ] Empirical SPS evaluation across model scales (RoBERTa-base, large; GPT-2)
- [ ] Comparison: SPS vs. adversarial robustness metrics on NLI and SST-2
- [ ] Statistical confidence intervals for empirical SPS (bootstrap)

**Mid-term**
- [ ] T_back: back-translation paraphrase family
- [ ] SPS scaling laws: how SPS_eps^{(l)} varies with model depth and width
- [ ] Connection to information geometry: SPS and Fisher information metric

**Long-term**
- [ ] Multimodal SPS (vision-language models)
- [ ] SPS as an evaluation criterion for robust generalization benchmarks
- [ ] Formal connection between SPS and PAC-Bayes generalization bounds

---

## Citation

```bibtex
@article{kang2026sps,
  title     = {Structured Perturbation Stability: An Operator-Restricted Framework
               for Measuring Semantic Invariance in Transformer Architectures},
  author    = {Kang, Dayeon},
  journal   = {1st AI Agent Journal},
  year      = {2026}
}
```

---

## Contact

**Dayeon Kang** · MICA International Scholars · dayeon603@gmail.com
