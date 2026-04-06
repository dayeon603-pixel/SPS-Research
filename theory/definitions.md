# Formal Definitions and Notation

## Structured Perturbation Stability (SPS) — Theoretical Framework

**Reference:** Kang, Dayeon. "Structured Perturbation Stability: An Operator-Restricted Framework
for Measuring Semantic Invariance in Transformer Architectures." 1st AI Agent Journal, 2026.

---

## Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$ | Input space (e.g., $\mathbb{R}^d$ or token sequence space) |
| $\mathcal{Y}$ | Output space (e.g., $\mathbb{R}^k$) |
| $\mathcal{Z}$ | Latent representation space |
| $f_\theta : \mathcal{X} \to \mathcal{Y}$ | Trained neural network |
| $\phi_\theta : \mathcal{X} \to \mathcal{Z}$ | Representation map (encoder) |
| $g_\theta : \mathcal{Z} \to \mathcal{Y}$ | Prediction function (head) |
| $\mathcal{T}$ | Admissible transformation family |
| $T : \mathcal{X} \to \mathcal{X}$ | Semantic transformation operator |
| $c(T, x)$ | Transformation magnitude at input $x$ |
| $d_\mathcal{Y}$ | Metric on output space $\mathcal{Y}$ |
| $\mathcal{D}$ | Data distribution over $\mathcal{X}$ |
| $J_{f_\theta}(x)$ | Fréchet derivative (Jacobian) of $f_\theta$ at $x$ |
| $A_x \subseteq \mathbb{S}^{d-1}$ | Structured semantic direction set at $x$ |
| $\|\cdot\|_{A_x}$ | $A_x$-restricted operator norm |
| $\sigma_{\max}(M)$ | Largest singular value of matrix $M$ |
| $\varepsilon$ | Local sensitivity radius |

---

## Formal Assumptions

The following assumptions are required for the theoretical results to hold. Each theorem
states which assumptions it invokes.

**Assumption A1 (Fréchet Differentiability).**
$f_\theta$ is Fréchet differentiable at every $x \in \mathrm{supp}(\mathcal{D})$,
with Jacobian $J_{f_\theta}(x) : \mathcal{X} \to \mathcal{Y}$.

**Assumption A2 (Transformation Compactness).**
For each $x \in \mathrm{supp}(\mathcal{D})$ and $\varepsilon > 0$, the set of normalized
perturbation directions

$$A_x^{(\varepsilon)} := \left\{ \frac{Tx - x}{\|Tx - x\|} \;:\; T \in \mathcal{T},\; 0 < c(T, x) \leq \varepsilon \right\}$$

is compact in $\mathbb{S}^{d-1}$.

**Assumption A3 (Measurability).**
The map $x \mapsto \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)$ is
$\mathcal{D}$-measurable for every $\varepsilon > 0$.

**Assumption A4 (Integrability).**
$$\mathbb{E}_{x \sim \mathcal{D}}\bigl[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)\bigr] < \infty.$$

**Assumption A5 (Transformation Family Axioms).**
The admissible family $\mathcal{T}$ satisfies:
1. **Identity:** $\mathrm{Id} \in \mathcal{T}$, with $c(\mathrm{Id}, x) = 0$.
2. **Semantic preservation:** Every $T \in \mathcal{T}$ preserves semantic content
   (i.e., $Tx$ and $x$ have the same task-relevant label under ground truth).
3. **Closure under composition:** If $T_1, T_2 \in \mathcal{T}$, then $T_2 \circ T_1 \in \mathcal{T}$
   (required for Theorem 4).

---

## Section 1: Model and Transformation Framework

**Definition 0 (Neural Network).** Let

$$f_\theta : \mathcal{X} \to \mathcal{Y}$$

be a trained neural network parameterized by $\theta$. For transformer-based models,
$f_\theta = g_\theta \circ \phi_\theta$ where $\phi_\theta$ is the encoder and $g_\theta$ is
the task head.

**Definition 0.1 (Semantic Transformation Operator).**
A semantic transformation operator is a map $T : \mathcal{X} \to \mathcal{X}$ that modifies
the surface structure of the input while preserving semantic content. Examples include:

- **Paraphrasing:** $T_{\text{para}}(x)$ — syntactic restatement preserving meaning
- **Synonym substitution:** $T_{\text{syn}}(x)$ — token-level replacement via lexical equivalents
- **Embedding perturbation:** $T_\alpha(x) = x + \alpha \delta(x)$ — continuous directional shift
  in representation space along admissible direction $\delta(x) \in A_x$

**Definition 0.2 (Transformation Magnitude).**
For each $T \in \mathcal{T}$ and input $x$, the transformation magnitude is

$$c(T, x) := \|Tx - x\|.$$

In embedding space, this corresponds to the $\ell_2$ displacement from $x$ to $Tx$.

---

## Section 2: Core Definitions

**Definition 1 (Structured Local Sensitivity).**
For $\varepsilon > 0$, the structured local sensitivity of $f_\theta$ at input $x$ is

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) \;:=\; \sup_{\substack{T \in \mathcal{T} \\ 0 < c(T,x) \leq \varepsilon}} \frac{d_\mathcal{Y}\!\bigl(f_\theta(x),\, f_\theta(Tx)\bigr)}{c(T,x)}.$$

This is the maximum normalized output change induced by admissible semantic transformations
within an $\varepsilon$-neighborhood of $x$. It is the operator-restricted analogue of the
local Lipschitz constant.

**Remark (Finiteness).** Under A1, A2, and $f_\theta$ locally Lipschitz, the supremum
is finite and is attained by compactness of $A_x^{(\varepsilon)}$.

**Definition 2 (Structured Perturbation Stability).**
The Structured Perturbation Stability of $f_\theta$ with respect to $\mathcal{T}$ is

$$\mathrm{SPS}_\varepsilon(f_\theta) \;:=\; \exp\!\Bigl(-\,\mathbb{E}_{x \sim \mathcal{D}}\bigl[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x)\bigr]\Bigr).$$

Under A3 and A4, this is well-defined and satisfies $0 < \mathrm{SPS}_\varepsilon(f_\theta) \leq 1$.
Higher values indicate stronger semantic invariance.

---

## Section 3: Differential Characterization

**Definition 3 (Admissible Semantic Direction Set).**
For $x \in \mathcal{X}$, the admissible semantic direction set is

$$A_x \;:=\; \lim_{\varepsilon \to 0} A_x^{(\varepsilon)} \;=\; \left\{ \lim_{\alpha \to 0} \frac{T_\alpha x - x}{\|T_\alpha x - x\|} \;:\; T_\alpha \in \mathcal{T},\; c(T_\alpha, x) \to 0 \right\}.$$

This is the set of tangent directions to the transformation orbits at $x$.

**Definition 4 ($A_x$-Restricted Operator Norm).**
The $A_x$-restricted operator norm of the Jacobian at $x$ is

$$\|J_{f_\theta}(x)\|_{A_x} \;:=\; \sup_{\substack{v \in A_x \\ \|v\| = 1}} \|J_{f_\theta}(x)\, v\|.$$

This measures the maximum output sensitivity of $f_\theta$ over admissible semantic directions.

---

## Section 4: New Definitions

**Definition 5 (Semantic Spectral Gap).**
The semantic spectral gap of $f_\theta$ at $x$ with respect to $\mathcal{T}$ is

$$\gamma(f_\theta, \mathcal{T};\, x) \;:=\; \sigma_{\max}\!\bigl(J_{f_\theta}(x)\bigr) - \|J_{f_\theta}(x)\|_{A_x}.$$

The normalized spectral gap is

$$\bar{\gamma}(f_\theta, \mathcal{T};\, x) \;:=\; 1 - \frac{\|J_{f_\theta}(x)\|_{A_x}}{\sigma_{\max}(J_{f_\theta}(x))} \;\in\; [0, 1].$$

The expected spectral gap is $\Gamma(f_\theta, \mathcal{T}) := \mathbb{E}_{x \sim \mathcal{D}}[\bar{\gamma}(f_\theta, \mathcal{T};\, x)]$.

**Interpretation.**
- $\bar{\gamma} = 0$: the worst-case sensitivity direction is semantic — the model is maximally
  sensitive precisely along the directions it should be invariant to.
- $\bar{\gamma} = 1$: the Jacobian annihilates all semantic directions; the model has perfect
  semantic invariance ($\mathrm{SPS}_\varepsilon = 1$).
- $\bar{\gamma} \in (0, 1)$: partial semantic separation — semantic directions are less sensitive
  than the maximally sensitive (adversarial) directions.

**Definition 6 (Empirical SPS Estimator).**
Given $n$ data samples $x_1, \ldots, x_n \sim \mathcal{D}$ and $m$ transformations
$\tau_1, \ldots, \tau_m \sim P_\mathcal{T}$ sampled i.i.d. from a distribution over $\mathcal{T}$,
the empirical SPS estimator is

$$\widehat{\mathrm{SPS}}_\varepsilon^{(n,m)}(f_\theta) \;:=\; \exp\!\left(-\frac{1}{n}\sum_{i=1}^{n} \max_{j=1,\ldots,m} \frac{d_\mathcal{Y}(f_\theta(x_i),\, f_\theta(\tau_j(x_i)))}{c(\tau_j,\, x_i)}\right).$$

**Remark (Consistency).** Under A1–A4, as $n, m \to \infty$,
$\widehat{\mathrm{SPS}}_\varepsilon^{(n,m)}(f_\theta) \xrightarrow{\text{a.s.}} \mathrm{SPS}_\varepsilon(f_\theta)$
by the strong law of large numbers and compactness of $\mathcal{T}$.

**Definition 7 (Layer-wise SPS).** *[Transformer-specific.]*
Let $f_\theta^{(l)}$ denote the subnetwork of $f_\theta$ comprising the first $l$ transformer layers
(i.e., the representation extractor truncated at depth $l$). Define

$$\mathrm{SPS}_\varepsilon^{(l)}(f_\theta) \;:=\; \mathrm{SPS}_\varepsilon\!\bigl(f_\theta^{(l)}\bigr), \qquad l = 0, 1, \ldots, L.$$

The sequence $\bigl\{\mathrm{SPS}_\varepsilon^{(l)}(f_\theta)\bigr\}_{l=0}^{L}$ is the **SPS depth profile**.
It characterizes how semantic stability is accumulated (or destroyed) through transformer depth.

**Remark.** By family monotonicity (Theorem 1), the depth profile is not necessarily monotone —
it depends on whether each layer amplifies or suppresses sensitivity to admissible semantic
directions.

**Definition 8 (Relative SPS).**
Let $\mathcal{T}_{\mathrm{arb}}^{(\varepsilon)}$ denote the family of all $\ell_2$-ball perturbations
of radius at most $\varepsilon$:

$$\mathcal{T}_{\mathrm{arb}}^{(\varepsilon)} := \{ T : \mathcal{X} \to \mathcal{X} \mid \|Tx - x\| \leq \varepsilon \;\forall x \}.$$

The relative SPS is

$$\mathrm{rSPS}_\varepsilon(f_\theta;\, \mathcal{T}) \;:=\; \frac{\mathrm{SPS}_\varepsilon^{(\mathcal{T})}(f_\theta)}{\mathrm{SPS}_\varepsilon^{(\mathcal{T}_{\mathrm{arb}})}(f_\theta)}.$$

**Interpretation.**
- $\mathrm{rSPS} > 1$: the model is more stable to semantic perturbations than to arbitrary
  noise of the same magnitude — the desired regime.
- $\mathrm{rSPS} = 1$: semantic stability is indistinguishable from generic smoothness.
- $\mathrm{rSPS} < 1$: the model is paradoxically *more* sensitive to semantic transformations
  than to random noise — a pathological failure mode.

---

## Section 5: Geometric Definitions

**Definition 9 (Transformation Orbit).** The transformation orbit of $x$ under $\mathcal{T}$ is

$$\mathcal{O}_\mathcal{T}(x) := \{ Tx : T \in \mathcal{T} \}.$$

**Definition 10 (Invariant Representation Subspace).** A representation map
$\phi_\theta : \mathcal{X} \to \mathcal{Z}$ induces an invariant subspace for $\mathcal{T}$ if

$$\phi_\theta(Tx) = \phi_\theta(x) \qquad \forall\, T \in \mathcal{T},\; \forall\, x \in \mathcal{X}.$$

Equivalently, $\phi_\theta$ collapses every orbit $\mathcal{O}_\mathcal{T}(x)$ to a single point
in $\mathcal{Z}$.

**Definition 11 (Semantic Direction Annihilation).** The Jacobian $J_{f_\theta}(x)$ annihilates
semantic directions at $x$ if

$$J_{f_\theta}(x)\, v = 0 \qquad \forall\, v \in A_x.$$

This is the differential (infinitesimal) characterization of orbit collapse.
