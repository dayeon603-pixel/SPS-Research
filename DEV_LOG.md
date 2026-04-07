# Dev Log — SPS

rough notes. not cleaned up.

---

**2026.02.08**

started this because of something that kept bothering me during the GLUE experiments last week. ran synonym substitution on a few SST-2 samples just to see what would happen — replaced "good" with "excellent", "bad" with "terrible", stuff like that. accuracy didn't move at all, which is expected. but when i looked at the [CLS] embeddings, cosine similarity between original and substituted dropped to like 0.58 in some cases. that's pretty low. like the model is treating "the movie was good" and "the movie was excellent" as meaningfully different things internally, but still predicting the same label.

that's weird right? either cosine similarity is a bad metric, or the model is genuinely routing these through different internal paths to the same output. i don't know which yet. going to dig into it.

no experiments today, just thinking and writing down the question.

---

**2026.02.09**

reading day. pulled up Jacovi & Goldberg (2020) on faithfulness — they talk about explanations being faithful to the model's actual reasoning, not just plausible-sounding. interesting framing but it's about post-hoc explanations, not about representation stability per se.

Wallace et al. (2019) on universal adversarial triggers. useful background on how fragile NLP models are, but they're looking for worst-case inputs. i don't want adversarial. i want neutral. completely different goal.

Sinha et al. (2021) on syntactic probing — models can do MNLI with shuffled word order, which suggests they're not really using syntax. tangentially relevant but also not quite what i'm after.

none of these address the specific thing i noticed yesterday. they all assume task performance is the right thing to measure. i want to measure something else.

---

**2026.02.10**

more reading. looked at some of the representation similarity work — CKA (centered kernel alignment), SVCCA. these compare representations across layers or across models. also not quite it — they're about similarity between model internals, not about how a single model responds to semantically equivalent inputs.

i think what i want is: given two inputs that mean the same thing, how similar are their representations? and can we formalize "mean the same thing" in a way that doesn't require human annotation for every pair?

wrote down this rough framing in my notebook:

*"stability = model outputs the same representation for semantically equivalent inputs. instability = model treats meaning-preserving changes as meaningful."*

still pretty fuzzy. need to make it mathematical.

---

**2026.02.12**

first actual experiment. wanted to get a baseline sense of how stable roberta-base is under noise.

setup: took 200 sentences from SST-2, added gaussian noise (σ=0.01) to all token embeddings, measured KL divergence between original and noisy softmax output distributions. also measured cosine similarity between [CLS] reps.

results:
- KL divergence: median ~0.003. basically nothing.
- cosine sim: 0.98+ for almost everything.

wrote "model is robust to noise" in my notebook and nearly moved on. glad i didn't.

the problem — which took me two more days to fully internalize — is that this experiment doesn't measure what i think it measures. output stability under small random noise is almost entirely a property of the softmax being saturated at high-confidence predictions. if the model is 99% confident on SST-2 samples, you'd need to move the logits by a lot before the output distribution changes meaningfully. i was testing softmax saturation, not representation quality. complete waste of a day essentially.

---

**2026.02.15**

tried synonym substitution properly this time. setup: 500 sentences from SST-2, replaced a single token per sentence with a WordNet synonym (randomly chosen from the first 3 synsets), measured accuracy drop and [CLS] cosine similarity.

results:
- accuracy: 91.2% original, 90.8% substituted. basically unchanged.
- cosine similarity between original and substituted [CLS]: mean 0.61, std 0.14. dropped as low as 0.43 in some cases.

this is the observation that actually started everything. the model is arriving at the same prediction through genuinely different internal representations. not a little different — sometimes the cosine similarity is 0.43, which is pretty far in high-dimensional space.

what does that mean? either (a) the metric is capturing something that doesn't matter, or (b) the model is "right for the wrong reasons" in some meaningful sense. i spent most of today oscillating between these two interpretations.

went for a walk and came back thinking: the right question isn't which interpretation is correct. the right question is — can we build a metric that distinguishes these cases? that's what i want to build.

---

**2026.02.19**

tried to formalize the sensitivity functional. first draft:

$$S(f, x, T) = \frac{d_Y(f(Tx), f(x))}{c(T, x)}$$

ratio of output change to transformation magnitude. normalizing by $c(T,x)$ so that bigger transformations don't just dominate.

immediately hit three problems that i couldn't resolve:

1. what is $d_Y$? for classification, the output is a probability distribution. KL divergence is asymmetric. total variation? L2 between logits? none of these feel canonical.

2. $c(T, x)$ for discrete token substitutions — what's the "magnitude" of swapping "dog" for "canine"? there's no obvious embedding-space distance between token sequences that accounts for what the substitution actually does semantically.

3. if $T$ is unconstrained, the sup over all transformations is dominated by the most destructive changes. replacing every word with a random word is a valid $T$ under this definition. that's not what i want.

wrote "needs constraints on T" at the top of the page and stopped for the day. feeling a bit stuck. the formalization is harder than i thought.

---

**2026.02.22**

i think i finally see the core issue with the 2026.02.19 setup.

the problem is i've been treating all transformations equally. but they're not equal. swapping "dog" for "canine" is a meaning-preserving perturbation. swapping "dog" for "photosynthesis" is not. a stability metric that lumps these together doesn't tell you anything about semantic invariance — it just tells you about robustness to arbitrary input changes, which is a different question.

so: $T$ needs to come from a constrained family. only transformations that preserve semantic content get to count. call this an admissible family $\mathcal{T}$.

first draft of A5 (family axioms) written today. pretty rough. three requirements:
- identity in $\mathcal{T}$ (zero perturbation baseline)
- bounded magnitude $c(T,x) \leq \varepsilon$
- semantic preservation (still informal — didn't define this formally yet)

the semantic preservation part is going to be the hard part. need to think about it more. but at least now the direction is clear.

also realized the sensitivity functional needs to be over the *sup* across $\mathcal{T}$, not just for a single $T$. worst-case sensitivity within the family.

---

**2026.02.25**

spent most of today thinking about how to formally define "semantic preservation" in A5 without making the whole thing circular.

options i considered:

1. **human annotation** — ask humans if two sentences mean the same thing. expensive, not scalable, subjective, and would need to be redone for every new dataset. not viable.

2. **NLI entailment** — use a model to check if $x$ entails $Tx$ and vice versa. problem: we're trying to evaluate a model's semantic understanding. using another model to define "semantic preservation" feels circular. also creates a dependency on whatever biases the NLI model has.

3. **operational definition** — curate a specific family of known-meaning-preserving transformations (synonym substitution from WordNet, back-translation, controlled paraphrase from a template, etc.) and just define $\mathcal{T}$ as "things drawn from these families." this is less elegant — $\mathcal{T}$ is now operationally defined, not axiomatically. but it's tractable.

going with option 3. it's a real limitation of the framework — universality is compromised — and i'm going to acknowledge that in the paper. but the alternatives are worse.

added A5 with three sub-axioms: identity, semantic preservation (operational), measurability. also realized while writing proposition 1 that i need A3 (measurability of the sensitivity map) and A4 (integrability) as separate assumptions. the proof doesn't close without them. will formalize later.

---

**2026.03.01**

wrote the full SPS definition today:

$$\mathrm{SPS}_\varepsilon(f_\theta) := \exp\!\left(-\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)]\right)$$

why the exponential? a few reasons:
- maps $[0, \infty)$ to $(0, 1]$, which is a clean interpretable range
- SPS = 1 means perfect invariance (zero expected sensitivity), which is the right ceiling
- SPS → 0 means unbounded average sensitivity, which is the right floor
- there's some information-theoretic intuition here that i haven't fully worked out — something about the sensitivity distribution and entropy. haven't pinned it down yet, might just be a coincidence.

one thing i wrote and then immediately flagged: "this makes SPS multiplicative under composition." that's not quite right. it's only multiplicative if the expectations decompose additively, which requires independence of the composition steps. they're not independent. theorem 3 will have to deal with this properly — the composition behavior is an upper bound, not a nice equality.

---

**2026.03.06**

tried to connect sensitivity to the Jacobian today. felt like the natural next step — in the $\varepsilon \to 0$ limit, the sensitivity functional should just be the directional derivative, which is a JVP.

wrote:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x) = \|J_{f_\theta}(x)\|_{A_x}$$

where $A_x$ is the set of directions that transformations in $\mathcal{T}$ induce at $x$.

felt clean. wrote up a proof sketch. noted in the margin that there might be an issue with the LHS having a sup over all $T$ while i was treating the RHS like a single JVP, but told myself it was probably fine and moved on.

it was not fine. caught this properly on 2026.03.17. flagging here so i don't forget — this theorem 2 draft is wrong.

---

**2026.03.09**

ran the composition experiment. took back-translation as $T_1$ and synonym substitution as $T_2$, each with high individual SPS scores (~0.78 and ~0.81 on a test set). then measured SPS for their composition $T_2 \circ T_1$.

composed SPS: 0.69. lower than both.

tried a few different pairs. consistently: composed SPS is below $\min(\mathrm{SPS}_1, \mathrm{SPS}_2)$.

the geometry makes sense — each transformation pushes the representation in a different direction in the embedding space. two small pushes in different directions compounds the drift. doesn't mean composition always makes things worse (the bound isn't tight in general) but the upper bound $\mathrm{SPS}(\mathcal{T}_2 \circ \mathcal{T}_1) \leq \min(\mathrm{SPS}_1, \mathrm{SPS}_2)$ seems right.

wrote up theorem 3 from this. straightforward from the definition.

---

**2026.03.13**

reading Sedghi et al. on singular values of convolutional networks, and some of the Lipschitz network literature. had an idea that i think is actually the most interesting contribution of this whole project.

the $A_x$-restricted operator norm $\|J_f(x)\|_{A_x}$ is almost always smaller than the full spectral norm $\sigma_{\max}(J_f(x))$, because $A_x$ is a strict subset of the sphere. the ratio tells you something: how much of the Jacobian's "sensitivity budget" is allocated toward semantic directions vs. all possible directions?

define the spectral gap:

$$\bar{\gamma} = 1 - \frac{\|J_f(x)\|_{A_x}}{\sigma_{\max}(J_f(x))}$$

large gap = semantic directions are nearly orthogonal to the top singular vectors. the model's most sensitive directions are non-semantic. this is good — the model is sensitive to things that don't affect meaning.

small gap (→ 0) = the most sensitive direction of the Jacobian happens to be a semantic direction. the model is maximally sensitive to exactly the kinds of changes that shouldn't matter. bad.

i don't think this geometric property is captured by any existing robustness or interpretability metric. feels genuinely new.

implementation challenge: computing $\sigma_{\max}(J_f(x))$ exactly requires materializing the full Jacobian, which is completely infeasible for a 125M param model (roughly $768 \times 768 \times T^2$ entries for a single input). solution: randomized power iteration using JVPs. each JVP is O(one forward pass). run $k$ times with random unit vectors, take the max observed norm. noisy but tractable.

---

**2026.03.17**

went back through the theorem 2 proof carefully today and found the error i half-noticed on 2026.03.06.

the original statement set:

$\delta(x)$ [a specific directional derivative for a fixed $T$] $=$ $\|J_f(x)\|_{A_x}$ [the sup over all directions in $A_x$]

these are not equal. $\delta(x)$ is a single JVP evaluation — $\|J_f(x) v_T\|$ for one specific direction $v_T$. the restricted norm is the sup over all unit vectors in $A_x$. equality would require $v_T$ to be the direction that achieves the sup, which is not guaranteed unless $A_x$ is a singleton (it's not).

i was asserting equality where only $\leq$ holds. the statement conflated a pointwise instantiation with the operator norm.

fix: split the theorem into two separate claims:
- **part (i):** for a fixed $T$, the pointwise sensitivity in the $\varepsilon \to 0$ limit equals $\|J_f(x) v_T\|$ where $v_T$ is the limiting perturbation direction. this is just the definition of a directional derivative. fine.
- **part (ii):** taking the sup over all $T \in \mathcal{T}$ recovers the restricted operator norm, again in the $\varepsilon \to 0$ limit. this requires A2 (compactness of $A_x^{(\varepsilon)}$) so that the sup is actually attained.

annoying to have to split it. but the split version is actually cleaner and more honest about what's being claimed where.

---

**2026.03.20**

reviewing proposition 1 today for the final draft. statement: $\mathrm{SPS}_\varepsilon(f_\theta) \in (0, 1]$.

the upper bound ($\leq 1$) is trivial — exp(-nonneg) ≤ 1. fine.

the lower bound (strict positivity, $> 0$) — the proof i had was: $\exp(-\mathbb{E}[\mathrm{Sens}]) > 0$ since $\exp$ is always strictly positive on $\mathbb{R}$.

wait. $\exp(-\infty) = 0$. that's a limit, not a value in $\mathbb{R}$. if $\mathbb{E}[\mathrm{Sens}] = +\infty$, then $\exp(-\mathbb{E}[\mathrm{Sens}])$ is 0, not strictly positive. the argument fails.

so the strict positivity claim requires assuming the expectation is finite. that needs to be an explicit assumption — it doesn't follow from anything else in the framework.

added **Assumption A4 (Integrability):** $\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)] < \infty$.

without A4, you can only prove $\mathrm{SPS}_\varepsilon \geq 0$. with A4, you get the strict inequality. the assumption is pretty mild in practice — if the sensitivity expectation diverges, SPS = 0 is a reasonable thing to say about the model anyway. but it needs to be stated explicitly.

this is a genuine gap in the original proposition, not just a technicality.

---

**2026.03.24**

the theoretical framework is in decent shape now but everything is still pretty abstract. need to make it computable. spent today adding four new definitions:

**def 5 (semantic spectral gap):** the $\bar{\gamma}$ from 2026.03.13, properly formalized. inputs: model $f_\theta$, transformation family $\mathcal{T}$, input $x$. output: scalar in $[0, 1]$.

**def 6 (empirical SPS):** monte carlo estimator — sample $N$ points from $\mathcal{D}$, average the sensitivity functional. this is what `core.py` actually computes. the continuous expectation in the definition is not tractable otherwise.

**def 7 (layer-wise SPS):** instead of applying SPS to the full model $f_\theta : \mathcal{X} \to \mathcal{Y}$, apply it to the map $x \mapsto h^{(\ell)}(x)$ for each transformer layer $\ell$ separately. lets you identify which layers are contributing most to semantic instability. this turned out to be genuinely interesting — in early experiments the middle layers (8–10 of 12 for roberta-base) are the most sensitive.

**def 8 (relative SPS):** $\mathrm{rSPS} := \mathrm{SPS}_\mathcal{T} / \mathrm{SPS}_{\text{arb}}$, ratio of semantic SPS to SPS under arbitrary random directions. if rSPS ≈ 1, the model is equally sensitive to semantic and non-semantic perturbations — it's not "protecting" semantics at all. if rSPS ≪ 1, semantic directions are disproportionately destabilizing. rSPS normalizes out the model's overall sensitivity scale, which makes comparisons across architectures cleaner.

of these four, rSPS (def 8) is probably the most practically useful diagnostic.

---

**2026.03.29**

started writing the code today. decided on the module structure:

```
src/sps/
  core.py           main interface — SPSEstimator, StructuredSensitivityEstimator
  jacobian.py       JVP, restricted norm, spectral gap
  transformations.py  T_emb and T_syn families
  metrics.py        SPSReport, rSPS, LayerwiseSPSAnalyzer
  utils.py          seed, divergence functions, normalize_directions
```

most important implementation decision: use `torch.autograd.functional.jvp` instead of materializing the Jacobian. for a 125M parameter model like roberta-base, the full Jacobian is completely infeasible — roughly $768 \times d_{\text{input}}$ per sample, times batch size. JVP is O(one forward pass) per direction. with k=8 probe directions, that's 8× inference cost. acceptable.

hit a memory problem with WordNet. the `EmbeddingPerturbationFamily` tried to build synonym directions for the entire tokenizer vocabulary — 50k+ tokens. this caused a huge memory spike at import time (trying to load all WordNet synsets upfront) and silently fell back to random orthogonal directions for most tokens. so it was computing "synonym directions" that were actually just random. not what i wanted.

fixed with a vocab_size cap (default 10k most common tokens) and explicit WARNING-level logging when falling back to random. also added `max_synonyms_per_token` to keep memory bounded. should have thought about this before trying to load 50k synsets. obvious in retrospect.

---

**2026.04.02**

writing the test suite. the hardest part is jacobian.py — you need analytical ground truth to verify that the JVP is computing the right thing. "the output looks reasonable" is not a sufficient test for Jacobian code.

solution: `ExactLinearModel(W)` — a linear model where $f(x) = Wx$, so $J_f = W$ everywhere, analytically. this lets me:

- verify `directional_derivative_norm(f, x, v)` = $\|Wv\|$ to machine precision
- construct $W$ with a known top singular vector $u_1$, set semantic direction $v = u_1$, and verify spectral gap ≈ 0 (semantic direction aligns with most sensitive direction)
- set semantic direction orthogonal to all singular vectors of $W$ (i.e., in the null space), verify spectral gap ≈ 1

this class of tests catches implementation bugs that would be completely invisible at the level of "run on roberta-base and check the output looks plausible." the exact linear model is simple enough to be wrong in interesting ways.

also wrote tests for the transformation families — shape checks, magnitude bounds, that synonym substitution actually changes tokens, that embedding perturbation stays within the $\varepsilon$ ball. pretty standard.

total tests: 29. took most of the day.

---

**2026.04.06**

pushed the readme. immediately noticed A3–A5 looked completely broken on github — raw symbols, partial italic formatting, some subscripts missing entirely.

root cause: i had wrapped the assumptions in blockquote syntax:
```
> **Assumption A3 (Measurability).** The map $x \mapsto \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)$ ...
```

github's CommonMark renderer runs italic marker detection before handing off to the math renderer. the underscore in `_{\mathcal{T},\varepsilon}` inside the blockquote got partially interpreted as an italic delimiter and the whole thing broke. output on github was a mess of raw latex mixed with half-formatted text.

fix: removed all blockquote wrappers from A3–A5, put the math expressions on their own standalone lines as display blocks. that's all it took. now it renders correctly.

lesson: never put latex subscripts inside github blockquotes. always check actual github rendering before finalizing a README that has nested subscripts. the local preview in vscode does not catch this.

---

**2026.04.07**

ran tests first thing — 28 pass, one fails:

```
FAILED tests/test_core.py::test_zero_sensitivity_for_constant_model
AssertionError: Expected ~0 sensitivity, got tensor([4.83e-06, 5.12e-06, ...])
```

spent a few minutes thinking i'd broken something. hadn't. this is float32 cosine between two identical vectors — `F.normalize` then dot product — picking up ~5e-6 numerical noise. the constant model broadcasts the exact same tensor to every position, so sensitivity should be zero, but floating point doesn't give you exactly zero. 1e-6 threshold was too tight for float32 cosine arithmetic.

changed threshold to 1e-5. that's the correct expectation for float32 — cosine similarity stacks normalize + dot product + 1 - sim, each step accumulating error. 1e-6 is a float64 precision requirement. 29/29 pass after.

---

then ran `estimate_sps.py`. spectral gap blew up immediately:

```
RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu is not implemented
```

JVP through roberta-base on CPU. flash attention doesn't have a CPU backward — it's a GPU-only optimization. `torch.autograd.functional.jvp` needs a backward to differentiate through. fix: `sdpa_kernel(SDPBackend.MATH)` forces the math (naive quadratic) attention kernel which has full CPU autograd support.

first attempt stored the context manager at module load time and planned to reuse it:

```python
_sdpa_ctx = sdpa_kernel(SDPBackend.MATH)

def _model_fn(inputs_embeds):
    with _sdpa_ctx:
        out = model(...)
```

second call to `_model_fn` failed with:

```
AttributeError: '_GeneratorContextManager' object has no attribute 'args'
```

`sdpa_kernel` is decorated with `@contextmanager`, so calling it returns a `_GeneratorContextManager` wrapping a generator function. the first `with _sdpa_ctx:` exhausts the generator. the second entry tries to resume a spent generator and hits internal attribute access that breaks. context manager *instances* from `@contextmanager` are single-use. not obvious.

fix — call the factory fresh inside the closure every time:

```python
def _model_fn(inputs_embeds):
    with _sdpa_kernel(SDPBackend.MATH):   # new instance each call
        out = model(inputs_embeds=inputs_embeds, attention_mask=mask0)
    return out.last_hidden_state[:, 0, :]
```

spectral gap ran. roberta-base on 16 SST-2 test sentences:

```
mean spectral gap:  0.1302
std:                ~0.04
range:              0.09 – 0.19
```

0.1302 means semantic directions are close to the Jacobian's top singular directions — not orthogonal to them. roberta's most sensitive internal directions largely overlap with semantic perturbation directions. consistent with the 2026.02.15 observation: model arrives at the same prediction via meaningfully different internal representations. the spectral gap is measuring exactly what i wanted to measure on 2026.03.13.

---

also fixed: rSPS display printing `rSPS = 1.0000 < 1`. rsps=0.99993 rounds to 1.0000 at 4 decimal places but the float comparison `rsps < 1.0` was still True. output made no sense. added ±5e-4 tolerance band — anything within 0.0005 of 1.0 prints as `≈ 1` now.

29/29 pass. spectral gap working. done for the day.

---

**2026.04.08**

wrapping up. current state as of 2026.04.08:

theory — done. all proofs reviewed, errors from 2026.03.17 and 2026.03.20 fixed, all assumptions explicit.  
code — done. core.py, jacobian.py, transformations.py, metrics.py, utils.py all written and working.  
tests — done. 29 tests, including exact analytical tests for the Jacobian.  
experiments — done. estimate_sps.py runs the full pipeline on 16 test sentences with roberta-base.  
readme — done (after the rendering fix on 2026.04.06).  

what's not done: the empirical comparison across model families. want to run roberta-base vs roberta-large vs deberta vs gpt-2 on the same transformation families to see if SPS scales with model size, varies by architecture, etc. need actual GPU time for that and i don't have it right now.

also haven't started the adversarial SPS work — constructing $\mathcal{T}$ families specifically designed to maximize sensitivity while preserving semantics. that's a next-phase thing.
