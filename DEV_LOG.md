# Dev Log — SPS

rough notes. not cleaned up.

---

**day 1–3**

started reading around the robustness/interpretability intersection. Jacovi & Goldberg on faithfulness, Wallace et al. on NLP attacks, Sinha et al. on syntactic probing. none of them quite address what i'm interested in — they're all testing adversarial vulnerability or probing for specific features. i want something more like: does this model's representation of "the dog is big" actually agree with its representation of "the large canine" in any meaningful geometric sense?

not sure yet how to formalize that. just reading for now.

---

**day 5**

first actual experiment. added gaussian noise (σ=0.01) to roberta-base token embeddings, measured KL divergence between original and perturbed output distributions. outputs were basically unchanged.

wrote in my notes: "model is robust to noise."

wrong. spent the next few days thinking this meant something. it doesn't. i was measuring softmax saturation, not representation geometry. moving on.

---

**day 8**

tried synonym substitution — replaced random tokens with wordnet synonyms, single substitution per sentence. looked at [CLS] cosine similarity and accuracy.

accuracy: barely moves. cosine sim: drops to 0.55–0.65 on average across the test set.

this is the observation that actually matters. same prediction, completely different internal representation. the model is right but for... what reason exactly? you can't tell from the output. this is the thing worth formalizing.

---

**day 12**

tried to write the sensitivity functional. first draft:

$$S(f, x, T) = \frac{d_Y(f(Tx), f(x))}{c(T, x)}$$

immediately ran into three problems:
- what even is $d_Y$ for classification? KL? total variation? not obvious.
- $c(T, x)$ for synonym substitution — what's the "magnitude" of a discrete token swap? no natural definition.
- if T is unconstrained, the sup is dominated by adversarial / semantically destructive transformations. that's not what i want.

wrote "needs constraints on T" at the top of the page and stopped. needs more thought.

---

**day 15**

ok i think i see it. the problem is that i've been treating all transformations equally. but replacing "dog" with "photosynthesis" is not the same kind of perturbation as replacing "dog" with "canine." the first destroys meaning. the second preserves it. a stability metric that can't tell these apart is useless.

so: the transformation family needs to be *semantically constrained*. only transformations that preserve meaning should count. call this an admissible family $\mathcal{T}$.

first draft of A5 (family axioms) written today. pretty rough — just: identity in T, bounded magnitude, semantic preservation (undefined). will formalize later.

---

**day 18**

how do you define "semantic preservation" without circularity?

options i wrote down:
1. human annotation — too expensive, not going to scale
2. NLI model checking entailment — circular, we're evaluating a model using another model
3. operational definition — just curate a family of known-semantic-preserving transformations (synonym swap, back-translation, controlled paraphrase) and call that T

going with 3. it's less elegant but it's the only tractable one. this means T is operationally defined, not axiomatically, which limits universality. that's a real limitation. acknowledged in the paper.

added A5 properly with three sub-axioms. also realized i needed A3 (measurability) separately from A5 and A4 (integrability) — the proposition 1 proof doesn't work without them. will add those formally.

---

**day 22**

wrote the full SPS definition:

$$\mathrm{SPS}_\varepsilon(f_\theta) := \exp\!\left(-\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)]\right)$$

the exponential felt right — maps to (0,1], interpretable, SPS=1 is perfect invariance. also some vague intuition about entropy that i haven't formalized.

one thing: this makes SPS "multiplicative under composition" in some loose sense — if you exponentiate sums you get products. but that only works if the composition terms are independent which they're not. need to be careful here. theorem 3 handles the actual composition behavior.

---

**day 27**

connected sensitivity to the Jacobian. in the $\varepsilon \to 0$ limit, the sensitivity functional should equal the Jacobian-vector product norm. wrote:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x) = \|J_{f_\theta}(x)\|_{A_x}$$

i.e. the $A_x$-restricted operator norm.

there's a problem here that i didn't catch until later (day 38). this equation is wrong as stated — the LHS has a sup baked in (over all T in T), but i was writing it like it equals a single JVP evaluation. that's only valid if $A_x$ is a singleton. noted this might be an issue but moved on — mistake.

---

**day 30**

tested composition: took two transformations each with high individual SPS (back-translation + synonym swap) and measured their composition.

result: composed SPS is lower than either alone. not by a huge amount but consistently below both.

makes sense geometrically — each transformation pushes the representation in a different direction. the composition compounds the drift. doesn't mean composition is always destabilizing but it motivates the upper bound in theorem 3: $\mathrm{SPS}(\mathcal{T}_2 \circ \mathcal{T}_1) \leq \min(\mathrm{SPS}_1, \mathrm{SPS}_2)$.

---

**day 34**

reading about Lipschitz networks and singular value bounds. had an idea: if the most sensitive directions of the Jacobian are geometrically far from the semantic directions $A_x$, the model is "allocating" its sensitivity budget to non-semantic features. that's a good thing.

spectral gap: $\bar{\gamma} = 1 - \|J_f\|_{A_x} / \sigma_{\max}(J_f)$

large gap = sensitive directions are orthogonal to semantic manifold. small gap = the model's most sensitive direction happens to be semantic, which is bad.

this isn't captured by any existing metric i'm aware of. feels like a real contribution.

implementation problem: how do you estimate $\sigma_{\max}(J_f(x))$ without materializing the full Jacobian? for a 125M param model that's totally infeasible. answer: randomized power iteration via JVPs. each JVP is O(one forward pass). do it k times with random vectors, take the max.

---

**day 38**

caught the theorem 2 error.

went back through the proof and realized: i set $\delta(x)$ (a single directional derivative) equal to $\|J_f(x)\|_{A_x}$ (the sup over all directions in $A_x$). these are not the same. the restricted norm is a sup. a single JVP is not a sup unless $A_x$ is a singleton.

i was asserting equality where only $\leq$ holds in general.

fix: split theorem 2 into two parts.
- part (i): single direction — JVP gives the directional derivative for a fixed T. this is fine.  
- part (ii): taking sup over T in the $\varepsilon \to 0$ limit gives the restricted norm. requires A2 (compactness of $A_x$) for the sup to be attained.

annoying to have to split it but the split version is actually cleaner.

---

**day 41**

reviewing proposition 1: SPS > 0.

proof: $\exp(-\mathbb{E}[\mathrm{Sens}]) > 0$ since exp is always positive.

wait. $\exp(-\infty) = 0$. if $\mathbb{E}[\mathrm{Sens}]$ diverges to infinity, the claim fails. i needed to assume the expectation is finite.

added A4 (integrability): $\mathbb{E}_{x \sim D}[\mathrm{Sens}] < \infty$.

without A4 you can only claim $\mathrm{SPS} \geq 0$. the strict inequality requires the expectation to be finite. that's a real gap in the original proposition, not just a technicality.

---

**day 45**

added definitions 5–8 to make the framework actually computable:

5. semantic spectral gap — the $\bar{\gamma}$ from day 34, formalized
6. empirical SPS — monte carlo estimator, this is what core.py implements
7. layer-wise SPS — apply SPS to $x \mapsto h^{(\ell)}(x)$ for each layer $\ell$ separately
8. relative SPS (rSPS) — ratio of semantic SPS to SPS under arbitrary directions. rSPS ≈ 1 means semantic directions are just as destabilizing as random ones. rSPS ≈ 0 means semantic directions are disproportionately bad.

rSPS is probably the most practically useful one. it normalizes out model-specific sensitivity scale.

---

**day 50**

code architecture. decided on:
```
src/sps/
  core.py           main interface — SPSEstimator
  jacobian.py       JVP, restricted norm, spectral gap
  transformations.py  T_emb and T_syn families
  metrics.py        SPSReport, rSPS, layerwise analyzer
  utils.py          seed, divergence functions, etc
```

most important implementation decision: use `torch.autograd.functional.jvp` instead of materializing the jacobian. JVP is O(one forward pass) per direction. with k=8 probe directions that's 8× inference — acceptable. full jacobian for a 125M param model is not feasible.

had a memory issue with WordNet — trying to load synonym directions for 50k tokens at import time. added a vocab cap (10k). should have caught this earlier, it's obvious in retrospect.

---

**day 54**

test suite. the hardest tests to write were for jacobian.py — you need analytical ground truth to verify the JVP is being computed correctly.

solution: `ExactLinearModel(W)` where $J_f = W$ everywhere. then:
- `directional_derivative_norm(f, x, v)` should equal $\|Wv\|$ exactly
- spectral gap = 0 when semantic direction = top singular vector of W
- spectral gap ≈ 1 when semantic directions are in the null space of W

this catches bugs that higher-level tests would miss entirely. worth the effort.

---

**day 58**

pushed the readme. A3–A5 came out as garbage on github — broken symbols, partial italic formatting.

root cause: i had the assumptions inside blockquote syntax (`>`). github's commonmark renderer does italic marker detection before math rendering. underscores in `_{\mathcal{T},\varepsilon}` inside blockquotes got interpreted as italic delimiters.

fix: removed blockquotes, put math on standalone lines.

note to self: never put latex subscripts inside github blockquotes. check math rendering on github before finalizing anything with nested subscripts.

---

**day 60**

theory: done  
code: done  
tests: done (~43 total)  
experiments: done  
readme: done (after the rendering fix)

what's left open: empirical comparison across model families (roberta-large, deberta, gpt-2). need actual GPU time for that. haven't started it.
