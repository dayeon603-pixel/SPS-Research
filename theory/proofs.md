# Theorem Statements and Proof Sketches

## Structured Perturbation Stability (SPS)

**Reference:** Kang, Dayeon. "Structured Perturbation Stability: An Operator-Restricted Framework
for Measuring Semantic Invariance in Transformer Architectures." 1st AI Agent Journal, 2026.

All definitions and notation follow `theory/definitions.md`. Assumptions A1–A5 are stated there.

---

## Proposition 1 (Boundedness)

**Statement.** Under Assumptions A3 and A4, for any neural network $f_\theta$,

$$0 < \mathrm{SPS}_\varepsilon(f_\theta) \leq 1.$$

**Proof.**
Since $d_\mathcal{Y}(\cdot, \cdot) \geq 0$ and $c(T, x) > 0$ by definition, every term in the
supremum in Definition 1 is non-negative. Hence $\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) \geq 0$
for all $x$. Taking expectations preserves non-negativity:

$$\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x)] \geq 0.$$

Applying $\exp(-\cdot)$, which is strictly decreasing on $\mathbb{R}$ with range $(0, \infty)$:

$$\mathrm{SPS}_\varepsilon(f_\theta) = \exp\!\bigl(-\mathbb{E}[\mathrm{Sens}]\bigr) \leq \exp(0) = 1.$$

The strict lower bound $\mathrm{SPS}_\varepsilon(f_\theta) > 0$ follows from A4: since
$\mathbb{E}[\mathrm{Sens}] < \infty$, we have $\exp(-\mathbb{E}[\mathrm{Sens}]) > 0$. $\square$

**Remark.** Without A4, the expectation could be $+\infty$, giving $\mathrm{SPS}_\varepsilon = 0$.
The strict positivity claim therefore requires integrability of the sensitivity functional.

---

## Theorem 1 (Family Monotonicity)

**Statement.** Let $\mathcal{T}_1 \subseteq \mathcal{T}_2$ be two admissible transformation families.
Then for all $x \in \mathcal{X}$:

$$\mathrm{Sens}_{\mathcal{T}_1,\varepsilon}(f_\theta;\, x) \;\leq\; \mathrm{Sens}_{\mathcal{T}_2,\varepsilon}(f_\theta;\, x),$$

and consequently,

$$\mathrm{SPS}_\varepsilon^{(\mathcal{T}_1)}(f_\theta) \;\geq\; \mathrm{SPS}_\varepsilon^{(\mathcal{T}_2)}(f_\theta).$$

**Proof.**
The first inequality is immediate from the definition of supremum: taking a supremum over a larger
index set cannot decrease the value. Formally,

$$\mathrm{Sens}_{\mathcal{T}_1}(f_\theta;\, x) = \sup_{T \in \mathcal{T}_1} (\cdots) \;\leq\; \sup_{T \in \mathcal{T}_2} (\cdots) = \mathrm{Sens}_{\mathcal{T}_2}(f_\theta;\, x)$$

since every $T \in \mathcal{T}_1$ is also in $\mathcal{T}_2$.

For the SPS inequality, take expectations (order-preserving under pointwise inequality):

$$\mathbb{E}[\mathrm{Sens}_{\mathcal{T}_1}] \leq \mathbb{E}[\mathrm{Sens}_{\mathcal{T}_2}].$$

Since $\exp(-\cdot)$ is strictly decreasing:

$$\exp\!\bigl(-\mathbb{E}[\mathrm{Sens}_{\mathcal{T}_1}]\bigr) \geq \exp\!\bigl(-\mathbb{E}[\mathrm{Sens}_{\mathcal{T}_2}]\bigr). \qquad \square$$

**Corollary.** The SPS functional is anti-monotone with respect to expansion of the transformation
family: richer semantic families yield lower (harder) stability scores.

---

## Theorem 2 (Differential / Jacobian Characterization) — *Corrected*

**Statement.** Under Assumption A1 (Fréchet differentiability), suppose the admissible
transformation family $\mathcal{T}$ contains all perturbations of the form
$T_\alpha^v(x) = x + \alpha v$ for unit vectors $v \in A_x \subseteq \mathbb{S}^{d-1}$
and $\alpha > 0$ sufficiently small. Then:

**(i) Directional derivative recovery.** For each fixed $v \in A_x$ with $\|v\| = 1$,

$$\lim_{\alpha \to 0} \frac{\|f_\theta(x + \alpha v) - f_\theta(x)\|}{\alpha} = \|J_{f_\theta}(x)\, v\|.$$

**(ii) Restricted operator norm.** The local structured sensitivity satisfies

$$\lim_{\varepsilon \to 0}\, \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = \sup_{\substack{v \in A_x \\ \|v\| = 1}} \|J_{f_\theta}(x)\, v\| =: \|J_{f_\theta}(x)\|_{A_x}.$$

**Proof.**

*Part (i).* By Fréchet differentiability of $f_\theta$ at $x$, for any fixed direction $v$ with $\|v\| = 1$:

$$f_\theta(x + \alpha v) = f_\theta(x) + \alpha J_{f_\theta}(x)\, v + r(\alpha v), \quad \text{where } \frac{\|r(\alpha v)\|}{\alpha} \to 0 \text{ as } \alpha \to 0.$$

Therefore,

$$\frac{\|f_\theta(x + \alpha v) - f_\theta(x)\|}{\alpha} = \left\| J_{f_\theta}(x)\, v + \frac{r(\alpha v)}{\alpha} \right\| \xrightarrow{\alpha \to 0} \|J_{f_\theta}(x)\, v\|. \qquad \square_{\text{(i)}}$$

*Part (ii).* By definition,

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = \sup_{\substack{T \in \mathcal{T} \\ 0 < c(T,x) \leq \varepsilon}} \frac{d_\mathcal{Y}(f_\theta(x), f_\theta(Tx))}{c(T,x)}.$$

Since $\mathcal{T}$ contains all $T_\alpha^v$ for $v \in A_x$, writing $Tx = x + \alpha v$:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) \geq \sup_{\substack{v \in A_x, \|v\|=1 \\ 0 < \alpha \leq \varepsilon}} \frac{\|f_\theta(x + \alpha v) - f_\theta(x)\|}{\alpha}.$$

By Part (i), as $\varepsilon \to 0$, each inner ratio converges to $\|J_{f_\theta}(x)\, v\|$. By
compactness of $A_x$ (Assumption A2) and uniform convergence of the Fréchet remainder on compact
sets, the supremum over $v$ and the limit in $\varepsilon$ can be interchanged:

$$\lim_{\varepsilon \to 0}\, \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = \sup_{v \in A_x,\, \|v\|=1} \|J_{f_\theta}(x)\, v\| = \|J_{f_\theta}(x)\|_{A_x}. \qquad \square_{\text{(ii)}}$$

**Remark on the original statement.** The original paper wrote a single lim sup for a fixed
parameterization $T_\alpha(x) = x + \alpha\delta(x)$ equated to the full supremum over $A_x$.
This conflated a single directional derivative (Part i) with the restricted operator norm (Part ii).
The corrected two-part statement above separates these, and Part (ii) is the meaningful result.

---

## Theorem 3 (Lipschitz Stability Bound)

**Statement.** If $f_\theta$ is globally $L$-Lipschitz with respect to the metrics on $\mathcal{X}$
and $\mathcal{Y}$, then for all $x \in \mathcal{X}$:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) \leq L,$$

and therefore,

$$\mathrm{SPS}_\varepsilon(f_\theta) \geq e^{-L}.$$

**Proof.**
By the global $L$-Lipschitz condition on $f_\theta$:

$$d_\mathcal{Y}\!\bigl(f_\theta(x), f_\theta(Tx)\bigr) \leq L \cdot d_\mathcal{X}(x, Tx) = L \cdot c(T, x).$$

Dividing by $c(T, x) > 0$ and taking the supremum over all admissible $T$:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = \sup_{T \in \mathcal{T}} \frac{d_\mathcal{Y}(f_\theta(x), f_\theta(Tx))}{c(T,x)} \leq L.$$

Taking expectations (with the bound holding pointwise):

$$\mathbb{E}_{x \sim \mathcal{D}}\bigl[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x)\bigr] \leq L.$$

Applying $\exp(-\cdot)$ and monotonicity:

$$\mathrm{SPS}_\varepsilon(f_\theta) = \exp\!\bigl(-\mathbb{E}[\mathrm{Sens}]\bigr) \geq \exp(-L). \qquad \square$$

---

## Theorem 4 (Sequential Transformation Stability)

**Statement.** Under Assumption A5.3 (closure under composition), let $T_1, T_2 \in \mathcal{T}$
with $c(T_1, x) \leq \varepsilon$ and $c(T_2, T_1 x) \leq \varepsilon$. Then:

$$d_\mathcal{Y}\!\bigl(f_\theta(x),\, f_\theta(T_2 T_1 x)\bigr) \;\leq\; \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x)\, c(T_1, x) + \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, T_1 x)\, c(T_2, T_1 x).$$

**Proof.**
Apply the triangle inequality in $(\mathcal{Y}, d_\mathcal{Y})$:

$$d_\mathcal{Y}(f_\theta(x), f_\theta(T_2 T_1 x)) \leq d_\mathcal{Y}(f_\theta(x), f_\theta(T_1 x)) + d_\mathcal{Y}(f_\theta(T_1 x), f_\theta(T_2 T_1 x)).$$

For the first term: since $T_1 \in \mathcal{T}$ and $c(T_1, x) \leq \varepsilon$,

$$d_\mathcal{Y}(f_\theta(x), f_\theta(T_1 x)) \leq \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) \cdot c(T_1, x).$$

For the second term: treating $T_1 x$ as a basepoint and $T_2 \in \mathcal{T}$ with
$c(T_2, T_1 x) \leq \varepsilon$,

$$d_\mathcal{Y}(f_\theta(T_1 x), f_\theta(T_2(T_1 x))) \leq \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, T_1 x) \cdot c(T_2, T_1 x). \qquad \square$$

**Remark.** The constraint $c(T_2, T_1 x) \leq \varepsilon$ is necessary for the sensitivity at
$T_1 x$ to be within the $\varepsilon$-radius. This constraint was absent from the original
theorem statement and must be imposed for the bound to hold.

---

## Theorem 5 (Semantic Stability Representation Theorem)

**Statement.** Let $f_\theta : \mathcal{X} \to \mathcal{Y}$ admit the representation decomposition

$$f_\theta = g_\theta \circ \phi_\theta,$$

where $\phi_\theta : \mathcal{X} \to \mathcal{Z}$ is a representation map and
$g_\theta : \mathcal{Z} \to \mathcal{Y}$ is a prediction function. Suppose $\phi_\theta$ is
invariant under $\mathcal{T}$:

$$\phi_\theta(Tx) = \phi_\theta(x) \qquad \forall\, T \in \mathcal{T},\; \forall\, x \in \mathcal{X}.$$

Then:

1. **Output invariance:** $f_\theta(Tx) = f_\theta(x)$ for all $T \in \mathcal{T}$.
2. **Zero sensitivity:** $\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = 0$ for all $x$.
3. **Maximal stability:** $\mathrm{SPS}_\varepsilon(f_\theta) = 1$.

**Proof.**
*Step 1.* By the decomposition and $\phi_\theta$-invariance:

$$f_\theta(Tx) = g_\theta(\phi_\theta(Tx)) = g_\theta(\phi_\theta(x)) = f_\theta(x).$$

*Step 2.* Since $f_\theta(Tx) = f_\theta(x)$, we have $d_\mathcal{Y}(f_\theta(x), f_\theta(Tx)) = 0$
for every $T \in \mathcal{T}$. Hence every term in the supremum in Definition 1 equals zero:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = \sup_{T \in \mathcal{T}} \frac{0}{c(T,x)} = 0.$$

*Step 3.* Taking expectations: $\mathbb{E}[0] = 0$. Applying Definition 2:

$$\mathrm{SPS}_\varepsilon(f_\theta) = \exp(-0) = 1. \qquad \square$$

**Remark.** The converse also holds locally: if $\mathrm{SPS}_\varepsilon(f_\theta) = 1$, then
$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = 0$ for $\mathcal{D}$-a.e. $x$, implying
that $\phi_\theta$ collapses transformation orbits to points in $\mathcal{Z}$ almost surely.

---

## Corollary 1 (Spectral Gap Stability Bound) — *New*

**Statement.** Under Assumptions A1 and A2, suppose $f_\theta$ is globally $L$-Lipschitz
and the normalized spectral gap satisfies

$$\bar{\gamma}(f_\theta, \mathcal{T};\, x) \geq \gamma_0 > 0 \qquad \mathcal{D}\text{-almost surely.}$$

Then

$$\mathrm{SPS}_\varepsilon(f_\theta) \geq \exp\!\bigl(-(1 - \gamma_0)\, L\bigr).$$

This bound is strictly tighter than Theorem 3 whenever $\gamma_0 > 0$.

**Proof.**
By Theorem 2(ii) and the definition of $\bar{\gamma}$:

$$\lim_{\varepsilon \to 0} \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) = \|J_{f_\theta}(x)\|_{A_x} = (1 - \bar{\gamma}(f_\theta, \mathcal{T};\, x))\, \sigma_{\max}(J_{f_\theta}(x)).$$

Since $f_\theta$ is $L$-Lipschitz, the spectral norm of its Jacobian satisfies
$\sigma_{\max}(J_{f_\theta}(x)) \leq L$ for a.e. $x$. Applying the assumed lower bound
$\bar{\gamma}(f_\theta, \mathcal{T};\, x) \geq \gamma_0$:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta;\, x) \leq (1 - \gamma_0)\, L.$$

Taking expectations and applying $\exp(-\cdot)$:

$$\mathrm{SPS}_\varepsilon(f_\theta) \geq \exp\!\bigl(-(1 - \gamma_0)\, L\bigr). \qquad \square$$

**Remark.** The gap $\gamma_0 L$ represents a quantitative separation between semantic sensitivity
and worst-case adversarial sensitivity. Models with large $\gamma_0$ are robustly semantic-invariant
even in the worst case over the data distribution.

---

## Proposition 2 (Relative SPS Ordering) — *New*

**Statement.** For any admissible $\mathcal{T}$,

$$\mathrm{rSPS}_\varepsilon(f_\theta;\, \mathcal{T}) \leq 1 \iff \mathcal{T} \supseteq \mathcal{T}_{\mathrm{arb}}^{(\varepsilon)},$$

and

$$\mathrm{rSPS}_\varepsilon(f_\theta;\, \mathcal{T}) \geq 1 \iff \mathrm{SPS}_\varepsilon^{(\mathcal{T})}(f_\theta) \geq \mathrm{SPS}_\varepsilon^{(\mathcal{T}_{\mathrm{arb}})}(f_\theta).$$

**Proof.**
Immediate from Definition 8 and the fact that $\mathrm{SPS}$ is a ratio of two positive quantities. $\square$

**Interpretation.** $\mathrm{rSPS} > 1$ is the regime of genuine semantic robustness: the
semantic transformation family induces strictly less model sensitivity than arbitrary noise
at the same perturbation budget $\varepsilon$.
