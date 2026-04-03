# Structured Perturbation Stability (SPS)

## Measuring Semantic Invariance in Transformer Architectures

**Dayeon Kang**  
MICA International Scholars  

---

## Abstract

We introduce **Structured Perturbation Stability (SPS)**, a theoretical framework for quantifying semantic invariance in neural networks. Unlike traditional robustness metrics that consider arbitrary perturbations, SPS restricts analysis to families of semantic-preserving transformations. We formalize structured local sensitivity, derive a Jacobian-based characterization, and establish a representation theorem linking invariance to learned feature geometry.

---

## 1. Introduction

Modern neural networks often achieve high benchmark performance while remaining sensitive to small transformations that preserve semantic meaning. Existing robustness frameworks do not distinguish between arbitrary perturbations and semantically equivalent transformations.

---

## 2. Problem Setup

$$
f_\theta : \mathcal{X} \rightarrow \mathcal{Y}
$$

$$
\mathcal{T} = \{ T : \mathcal{X} \rightarrow \mathcal{X} \}
$$

$$
c(T, x) = \|T(x) - x\|
$$

---

## 3. Structured Local Sensitivity

$$
\mathrm{Sens}_{\mathcal{T}, \varepsilon}(f_\theta; x)
=
\sup_{\substack{T \in \mathcal{T} \\ 0 < c(T,x) \le \varepsilon}}
\frac{
d_{\mathcal{Y}}(f_\theta(x), f_\theta(Tx))
}{
c(T, x)
}
$$

---

## 4. Structured Perturbation Stability

$$
\mathrm{SPS}_\varepsilon(f_\theta)
=
\exp\left(
- \mathbb{E}_{x \sim \mathcal{D}}
\left[
\mathrm{Sens}_{\mathcal{T}, \varepsilon}(f_\theta; x)
\right]
\right)
$$

$$
0 < \mathrm{SPS}_\varepsilon(f_\theta) \le 1
$$

---

## 5. Differential Characterization

$$
T_\alpha(x) = x + \alpha \delta(x)
$$

$$
\limsup_{\alpha \to 0}
\frac{
\|f_\theta(T_\alpha x) - f_\theta(x)\|
}{
\|T_\alpha x - x\|
}
=
\sup_{\substack{\|v\| = 1 \\ v \in \mathcal{A}_x}}
\|J_{f_\theta}(x)v\|
$$

---

## 6. Representation Theorem

$$
f_\theta = g_\theta \circ \phi_\theta
$$

$$
\phi_\theta(Tx) = \phi_\theta(x)
$$

$$
f_\theta(Tx) = f_\theta(x)
$$

---

## 7. Repository Structure

SPS/
├── paper/
├── theory/
├── experiments/
├── benchmarks/
└── README.md

---

## 8. Paper

See /paper/SPS_paper.pdf
