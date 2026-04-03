<p align="center">
  # Structured Perturbation Stability (SPS)

<p align="center">
  <strong>An Operator-Restricted Framework for Measuring Semantic Invariance in Transformer Architectures</strong>
</p>

<p align="center">
  <em>Dayeon Kang</em><br/>
  MICA International Scholars
</p>

<p align="center">
  <a href="./paper/SPS_paper.pdf">Paper</a> ·
  <a href="#theoretical-framework">Theory</a> ·
  <a href="#repository-structure">Repository Structure</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

---

## Overview

Structured Perturbation Stability (SPS) is a theoretical framework for quantifying **semantic invariance** in neural networks. Classical robustness metrics typically evaluate stability under arbitrary or adversarial perturbations; SPS instead restricts analysis to **semantic-preserving transformations** and measures sensitivity only along those admissible directions.

The goal is to distinguish generic smoothness from genuine invariance to meaning-preserving variation.

---

## Contributions

This repository develops a formal framework for semantic stability in modern neural architectures.

- Introduces **structured local sensitivity** as an operator-restricted notion of model sensitivity
- Defines **Structured Perturbation Stability (SPS)** as a global semantic stability functional
- Derives a **Jacobian characterization** connecting semantic stability to local differential geometry
- Establishes core properties including **boundedness**, **family monotonicity**, and **Lipschitz stability bounds**
- Proves a **representation theorem** characterizing maximal semantic invariance through invariant latent spaces

---

## Theoretical Framework

### Model

<p align="center">
  <img src="assets/equations/model.svg" alt="model equation" />
</p>

### Semantic transformation family

<p align="center">
  <img src="assets/equations/transform_family.svg" alt="semantic transformation family" />
</p>

### Transformation magnitude

<p align="center">
  <img src="assets/equations/magnitude.svg" alt="transformation magnitude" />
</p>

### Structured local sensitivity

<p align="center">
  <img src="assets/equations/sensitivity.svg" alt="structured local sensitivity" />
</p>

Structured local sensitivity measures the maximum normalized output variation induced by admissible semantic transformations in an epsilon-neighborhood of the input.

### Structured Perturbation Stability

<p align="center">
  <img src="assets/equations/sps.svg" alt="structured perturbation stability" />
</p>

<p align="center">
  <img src="assets/equations/bounds.svg" alt="sps bounds" />
</p>

Higher SPS values indicate stronger invariance to semantic-preserving transformations.

---

## Main Results

### Differential characterization

Assume local structured transformations of the form

<p align="center">
  <img src="assets/equations/local_transform.svg" alt="local structured transformation" />
</p>

Then SPS admits the following Jacobian characterization:

<p align="center">
  <img src="assets/equations/jacobian.svg" alt="jacobian characterization" />
</p>

This connects semantic stability to a restricted Jacobian operator norm over admissible semantic directions.

### Family monotonicity

If

<p align="center">
  <img src="assets/equations/subset.svg" alt="transformation family subset relation" />
</p>

then

<p align="center">
  <img src="assets/equations/monotonicity1.svg" alt="sensitivity monotonicity" />
</p>

and

<p align="center">
  <img src="assets/equations/monotonicity2.svg" alt="sps monotonicity" />
</p>

### Lipschitz stability bound

If the model is globally Lipschitz, then

<p align="center">
  <img src="assets/equations/lipschitz1.svg" alt="lipschitz sensitivity bound" />
</p>

which yields

<p align="center">
  <img src="assets/equations/lipschitz2.svg" alt="lipschitz sps bound" />
</p>

### Semantic Stability Representation Theorem

Suppose the model decomposes as

<p align="center">
  <img src="assets/equations/decomposition.svg" alt="model decomposition" />
</p>

with

<p align="center">
  <img src="assets/equations/phi_map.svg" alt="representation and prediction maps" />
</p>

If the learned representation is invariant under the admissible transformation family,

<p align="center">
  <img src="assets/equations/invariance.svg" alt="representation invariance" />
</p>

then the model output is invariant,

<p align="center">
  <img src="assets/equations/output_invariance.svg" alt="output invariance" />
</p>

and consequently,

<p align="center">
  <img src="assets/equations/sens_zero.svg" alt="maximal semantic stability" />
</p>

---

## Geometric Interpretation

For an input x, define its transformation orbit by

<p align="center">
  <img src="assets/equations/orbit.svg" alt="transformation orbit" />
</p>

Maximal semantic stability arises when all elements of the orbit collapse to the same latent representation:

<p align="center">
  <img src="assets/equations/orbit_collapse.svg" alt="orbit collapse" />
</p>

Equivalently, semantic directions are annihilated by the local Jacobian:

<p align="center">
  <img src="assets/equations/direction_annihilation.svg" alt="jacobian annihilates semantic directions" />
</p>

This provides a geometric account of how semantic invariance can emerge in learned representation spaces.

---

## Repository Structure

```text
SPS/
├── assets/
│   └── equations/          # SVG-rendered equations for stable GitHub display
├── paper/
│   └── SPS_paper.pdf       # manuscript
├── theory/
│   ├── definitions.md      # formal definitions and notation
│   └── proofs.md           # theorem statements and proof sketches
├── experiments/            # planned empirical evaluation
├── benchmarks/             # planned semantic perturbation benchmarks
└── README.md
```

---

## Status

This repository is currently organized around the theoretical foundation of SPS.

Implemented now:
- formal problem setup
- mathematical definitions
- principal theorems
- GitHub-safe equation rendering

Planned next:
- empirical SPS estimation on transformer models
- semantic perturbation benchmark construction
- comparison against adversarial robustness metrics
- multilingual and multimodal extensions

---

## Roadmap

### Near-term
- Add theorem-by-theorem proof notes in `theory/proofs.md`
- Add notation sheet and assumptions summary
- Add pseudocode for empirical SPS estimation

### Mid-term
- Implement an evaluation pipeline for transformer encoders and decoder-only language models
- Construct semantic transformation families for NLP experiments
- Study scaling behavior of SPS across model size and architecture

### Long-term
- Extend SPS to multimodal representation learning
- Connect SPS to benchmark design for robust generalization
- Explore applications to trustworthy evaluation of advanced AI systems

---

## Citation

If you want to review this work, see the updated abstract/introduction in the `/Structured_Perturbation_Stability__An_Operator_Restricted_Framework_for_Measuring_Semantic_Invariance_in_Transformer_Architectures.pdf
`.

```bibtex
@article{kang2026sps,
  title={Structured Perturbation Stability: An Operator-Restricted Framework for Measuring Semantic Invariance in Transformer Architectures},
  author={Kang, Dayeon},
  journal={1st AI Agent Journal},
  year={2026}
}
```

---

## License

This repository currently does not include a license file. Add a project license before public release if redistribution or reuse is intended.

---

## Contact

**Dayeon Kang**  
MICA International Scholars  
dayeon603@gmail.com
