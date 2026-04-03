Structured Perturbation Stability (SPS)

Measuring Semantic Invariance in Transformer Architectures

Dayeon Kang  
MICA International Scholars  

Abstract

We introduce Structured Perturbation Stability (SPS), a theoretical framework for quantifying semantic invariance in neural networks. Unlike traditional robustness metrics that consider arbitrary perturbations, SPS restricts analysis to families of semantic-preserving transformations. We formalize structured local sensitivity, derive a Jacobian-based characterization, and prove a representation theorem linking invariance to learned feature geometry.


1. Introduction

Modern neural networks achieve high benchmark performance but often fail under small semantic-preserving transformations (e.g., paraphrasing). Existing robustness frameworks fail to distinguish between arbitrary perturbations and meaningful semantic variation.

This repository provides:
- A formal definition of SPS
- Theoretical analysis and proofs
- Foundations for future empirical evaluation


2. Method

Model

Let:
f_θ : X → Y

Let T be a family of semantic-preserving transformations:
T : X → X



Structured Local Sensitivity

Sens_{T, ε}(f_θ; x) =
sup_{T ∈ T, 0 < c(T,x) ≤ ε}
[d_Y(f_θ(x), f_θ(Tx)) / c(T, x)]



Structured Perturbation Stability

SPS_ε(f_θ) =
exp(− E_{x ~ D}[Sens_{T, ε}(f_θ; x)])

0 < SPS_ε(f_θ) ≤ 1



Differential Form

T_α(x) = x + αδ(x)

limsup_{α→0}
||f_θ(T_α x) − f_θ(x)|| / ||T_α x − x||**
=
sup_{||v||=1, v ∈ A_x} ||J_fθ(x)v||



3. Theory

Theorem (Representation Invariance)

If:
f_θ = g_θ ∘ φ_θ

and:
φ_θ(Tx) = φ_θ(x)

then:
f_θ(Tx) = f_θ(x)

Thus:
SPS_ε(f_θ) = 1



Lipschitz Bound

If f_θ is L-Lipschitz:
SPS_ε(f_θ) ≥ e^(−L)


4. Repository Structure

.
├── paper/
│   └── SPS_paper.pdf
├── theory/
│   ├── definitions.md
│   ├── proofs.md
├── experiments/           (planned)
├── benchmarks/            (planned)
└── README.md



5. Planned Experiments

- SPS estimation on transformer models
- Comparison with adversarial robustness metrics
- Semantic perturbation datasets (NLP)



6. Citation

If you use this work:

Kang, D. (2026). Structured Perturbation Stability.



7. License

MIT License (recommended)



8. Notes

This repository focuses on theoretical ML contributions. Empirical validation is ongoing.
