# Dev Log — Structured Perturbation Stability (SPS)

Research notes. Written as things happen, not cleaned up afterward.

---

**2026.02.08**

Started this project after noticing something unexpected during GLUE experiments last week. I ran synonym substitution on a small set of SST-2 samples — swapping "good" for "excellent", "bad" for "terrible" — and task accuracy was completely unaffected, which is expected. What I didn't expect: the cosine similarity between the original and substituted [CLS] embeddings dropped to around 0.58 in some cases. That's a pretty significant shift in representation space for what should be a meaning-preserving change.

So the model is arriving at the same prediction via what appear to be genuinely different internal paths. Either cosine similarity is capturing something that doesn't actually matter for the task, or the model is doing something strange — "right for the wrong reasons" in some meaningful sense. I'm not sure which yet. Going to dig into this more carefully.

No experiments today. Just wanted to write down the question before I lost the thread.

---

**2026.02.09**

Reading day. Went through a few papers to see if anyone has looked at this.

Jacovi & Goldberg (2020) on explanation faithfulness — their framing is about post-hoc explanations being faithful to what the model actually does, not just plausible-sounding. Useful framing but it's about interpretability, not representation stability.

Wallace et al. (2019) on universal adversarial triggers — relevant background on how fragile NLP models can be, but they're looking for worst-case perturbations. I want semantics-preserving ones. Different goal.

Sinha et al. (2021) on syntactic probing — models can handle MNLI with shuffled word order, suggesting they're not really using syntactic structure. Tangentially interesting but not quite what I'm after.

None of these directly address what I observed. They all treat task performance as the primary thing to measure. I want to measure something about the internal representation, not the output label.

---

**2026.02.10**

More reading. Looked at representation similarity work — CKA (centered kernel alignment) and SVCCA. These compare representations across layers or across different models, which isn't quite what I need either. I want to understand how a single model's representations change in response to semantically equivalent inputs.

Wrote down a rough version of what I'm trying to capture:

*"Stability = the model produces similar representations for semantically equivalent inputs. Instability = meaning-preserving changes cause disproportionate representation shifts."*

Still informal. Need to make it precise mathematically, but at least the direction is clearer now.

---

**2026.02.12**

First actual experiment. Wanted to establish a baseline for how roberta-base behaves under input noise.

Setup: 200 sentences from SST-2, added Gaussian noise (σ=0.01) to all token embeddings, measured KL divergence between original and noisy softmax distributions and cosine similarity between [CLS] representations.

Results:
- KL divergence: median ~0.003. Essentially unchanged.
- Cosine similarity: 0.98+ across almost everything.

I nearly concluded "model is robust to noise" and moved on. The problem — which took me a couple more days to fully work out — is that this experiment doesn't measure what I thought it did. Output stability under small random noise is almost entirely a function of softmax saturation at high-confidence predictions. If the model is 99% confident on SST-2 samples, you'd need a large logit shift to see any meaningful output change. I was measuring softmax saturation, not representation quality. The experiment was basically useless for my purposes.

---

**2026.02.15**

Ran synonym substitution more carefully this time. Setup: 500 SST-2 sentences, replaced one token per sentence with a WordNet synonym (first 3 synsets, randomly selected), measured accuracy drop and [CLS] cosine similarity.

Results:
- Accuracy: 91.2% original, 90.8% substituted. Essentially stable.
- Cosine similarity between original and substituted [CLS]: mean 0.61, std 0.14. Some pairs as low as 0.43.

This is the observation that actually motivated the whole project. The model reaches the same prediction via genuinely different internal representations — not slightly different, but sometimes with cosine similarity of 0.43, which is a substantial distance in high-dimensional embedding space.

The question I keep coming back to: does this matter? Either (a) the representation divergence doesn't affect anything that counts, or (b) the model is systematically encoding semantically irrelevant distinctions. I spent most of today going back and forth between these. Eventually landed on: the right question isn't which interpretation is correct — it's whether we can build a metric that makes the distinction measurable.

---

**2026.02.19**

First attempt at formalizing the sensitivity functional:

$$S(f, x, T) = \frac{d_Y(f(Tx), f(x))}{c(T, x)}$$

Ratio of output change to transformation magnitude, normalized so larger transformations don't trivially dominate.

Three problems I couldn't get past:

1. What is $d_Y$? For classification the output is a probability distribution. KL divergence is asymmetric, total variation seems arbitrary, L2 over logits has no clear interpretation. Nothing feels canonical here.

2. $c(T, x)$ for discrete substitutions — what's the "magnitude" of swapping "dog" for "canine"? There's no obvious distance measure between token sequences that captures what the substitution does semantically.

3. Without constraints on $T$, the supremum over all transformations is dominated by the most destructive changes. Replacing every token with a random word is a valid $T$ by this definition, which is clearly not what I want.

Wrote "needs constraints on T" at the top of the page and stopped. The formalization is harder than I initially expected.

---

**2026.02.22**

I think I finally understand the core problem with the 2026.02.19 formulation.

The issue is treating all transformations equally. Swapping "dog" for "canine" is meaning-preserving. Swapping "dog" for "photosynthesis" is not. A stability metric that treats both the same isn't measuring semantic invariance — it's measuring robustness to arbitrary perturbations, which is a different (and less interesting) question for my purposes.

So $T$ needs to come from a constrained family $\mathcal{T}$ of admissible transformations — only semantics-preserving ones count.

Wrote the first draft of Axiom A5 (family axioms) today. Three requirements:
- Identity must be in $\mathcal{T}$ (zero perturbation as baseline)
- Bounded magnitude: $c(T,x) \leq \varepsilon$
- Semantic preservation (still informal — haven't defined this formally yet)

The semantic preservation part is going to be the difficult one. But at least the direction is now clear. Also realized the sensitivity functional should be the sup over $\mathcal{T}$, not evaluated at a single $T$ — worst-case sensitivity within the admissible family.

---

**2026.02.25**

Spent most of today working through how to formally define "semantic preservation" in A5 without making the whole framework circular.

Options I considered:

1. **Human annotation** — have annotators judge whether two sentences mean the same thing. Expensive, doesn't scale, and would need to be redone for every new domain. Not viable.

2. **NLI entailment** — use an NLI model to check bidirectional entailment between $x$ and $Tx$. Problem: we're trying to evaluate a model's semantic understanding, so using another model to define "semantic preservation" feels circular. Also inherits the biases of whatever NLI model I use.

3. **Operational definition** — curate a specific set of transformation families with known meaning-preserving properties (WordNet synonym substitution, controlled back-translation, template-based paraphrase) and define $\mathcal{T}$ as transformations drawn from these families.

Going with option 3. It's a genuine limitation — universality of the framework is compromised because $\mathcal{T}$ is operationally defined rather than axiomatically. I'll acknowledge this explicitly in the paper. The alternatives have bigger problems.

Also realized while writing Proposition 1 that I need A3 (measurability of the sensitivity map) and A4 (integrability) as separate explicit assumptions — the proofs don't close without them.

---

**2026.03.01**

Wrote the full SPS definition today:

$$\mathrm{SPS}_\varepsilon(f_\theta) := \exp\!\left(-\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)]\right)$$

The exponential mapping has a few nice properties:
- Maps $[0, \infty)$ to $(0, 1]$, giving a clean interpretable range
- SPS = 1 iff expected sensitivity is zero — perfect invariance
- SPS → 0 as sensitivity diverges — the right floor

There's also some information-theoretic intuition I haven't fully worked out yet, something about the relationship between sensitivity distribution and entropy. Might be a coincidence — flagging it but not claiming it.

One thing I caught: I wrote "this makes SPS multiplicative under composition" and immediately realized that's wrong. Multiplicativity would require the expectations to decompose additively, which needs independence of composition steps — and they're not independent. Theorem 3 will need to handle this as an upper bound, not an equality.

---

**2026.03.06**

Tried to connect the sensitivity functional to the Jacobian. In the $\varepsilon \to 0$ limit the sensitivity should reduce to a directional derivative, which is a JVP.

Wrote:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x) = \|J_{f_\theta}(x)\|_{A_x}$$

where $A_x$ is the set of perturbation directions induced by $\mathcal{T}$ at $x$.

The proof sketch looked clean. I noticed a potential issue — the LHS has a supremum over all $T \in \mathcal{T}$ while the RHS is written like a single operator norm — but didn't stop to resolve it properly. Noted it in the margin and moved on.

That was a mistake. Caught the actual error on 2026.03.17.

---

**2026.03.09**

Ran the composition experiment. Used back-translation as $T_1$ and synonym substitution as $T_2$, both with solid individual SPS scores (~0.78 and ~0.81 on a held-out set). Measured SPS for their composition $T_2 \circ T_1$.

Composed SPS: 0.69 — below both individual scores.

Ran several more pairs. The pattern held: composed SPS consistently falls below $\min(\mathrm{SPS}_1, \mathrm{SPS}_2)$.

The geometric intuition makes sense: each transformation displaces the representation in a different direction in embedding space. Two small displacements in different directions compounds the total drift. The upper bound $\mathrm{SPS}(\mathcal{T}_2 \circ \mathcal{T}_1) \leq \min(\mathrm{SPS}_1, \mathrm{SPS}_2)$ holds empirically here, and follows cleanly from the definition. Wrote up Theorem 3 from this.

---

**2026.03.13**

Reading Sedghi et al. on singular values of convolutional networks, and some of the Lipschitz-constrained network literature. Had an idea while reading that I think might actually be the most interesting part of this project.

The $A_x$-restricted operator norm $\|J_f(x)\|_{A_x}$ is almost always strictly less than the full spectral norm $\sigma_{\max}(J_f(x))$, because $A_x$ is a strict subset of the unit sphere. The gap between these two quantities carries information: how much of the Jacobian's sensitivity budget is directed toward semantic perturbation directions versus all possible directions?

Define the semantic spectral gap:

$$\bar{\gamma} = 1 - \frac{\|J_f(x)\|_{A_x}}{\sigma_{\max}(J_f(x))}$$

Large gap (→ 1): semantic directions are nearly orthogonal to the Jacobian's top singular vectors. The model's most sensitive directions are non-semantic. Good — the model is sensitive to things that don't affect meaning.

Small gap (→ 0): the most sensitive direction of the Jacobian aligns with a semantic perturbation direction. The model is maximally sensitive to exactly the kinds of changes that shouldn't matter. Bad.

I don't think this geometric property is captured by existing robustness or interpretability metrics. It feels like a genuinely new characterization.

Implementation note: materializing the full Jacobian is completely infeasible for a 125M parameter model. Solution: randomized power iteration via JVPs. Each JVP costs one forward pass. With $k=8$ probe directions, that's 8× inference cost — expensive but tractable.

---

**2026.03.17**

Went back through the Theorem 2 proof carefully and found the error I'd glossed over on 2026.03.06.

The original statement implicitly equated two different things:

- $\delta(x)$: a single directional derivative for a fixed $T$ — i.e., $\|J_f(x) v_T\|$ for one specific direction $v_T$
- $\|J_f(x)\|_{A_x}$: the supremum of $\|J_f(x) v\|$ over all unit vectors $v \in A_x$

These are equal only if $v_T$ happens to be the direction achieving the supremum, which isn't guaranteed in general (and certainly isn't implied by anything in the axioms). I was asserting equality where only $\leq$ holds.

Fix: split into two separate claims.
- **Part (i):** For a fixed $T$, the pointwise sensitivity in the $\varepsilon \to 0$ limit equals $\|J_f(x) v_T\|$, where $v_T$ is the limiting perturbation direction. This is just the definition of a directional derivative.
- **Part (ii):** Taking the sup over $T \in \mathcal{T}$ recovers the restricted operator norm in the $\varepsilon \to 0$ limit. This requires A2 (compactness of $A_x^{(\varepsilon)}$) to ensure the sup is attained.

The split version is actually cleaner and more honest about what each part is claiming. Glad I caught this before writing it up.

---

**2026.03.20**

Reviewing Proposition 1 — the claim that $\mathrm{SPS}_\varepsilon(f_\theta) \in (0, 1]$.

Upper bound ($\leq 1$): trivial, since $\exp(-\text{nonneg}) \leq 1$. Fine.

Lower bound (strict positivity, $> 0$): the proof I had was "$\exp(-\mathbb{E}[\mathrm{Sens}]) > 0$ because $\exp$ is always strictly positive." This is wrong.

$\exp(-\infty) = 0$. If $\mathbb{E}[\mathrm{Sens}] = +\infty$, the expression evaluates to 0, not to something strictly positive. The strict inequality only holds when the expectation is finite, and that's an assumption — it doesn't follow from anything else already in the framework.

Added **Assumption A4 (Integrability):** $\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)] < \infty$.

Without A4, the proposition weakens to $\mathrm{SPS}_\varepsilon \geq 0$. With A4, the strict lower bound follows. The assumption is mild in practice — if the expectation diverges, SPS = 0 is arguably the right answer anyway — but it has to be stated explicitly.

---

**2026.03.24**

The theoretical framework is in reasonable shape now but everything is still abstract. Need to make it computable. Added four new definitions today:

**Def 5 (Semantic spectral gap):** Formalizes $\bar{\gamma}$ from 2026.03.13. Inputs: model $f_\theta$, transformation family $\mathcal{T}$, input $x$. Output: scalar in $[0, 1]$.

**Def 6 (Empirical SPS):** Monte Carlo estimator — sample $N$ inputs from $\mathcal{D}$, average the sensitivity functional. This is what `core.py` actually computes; the continuous expectation in the definition isn't tractable directly.

**Def 7 (Layer-wise SPS):** Apply SPS to $x \mapsto h^{(\ell)}(x)$ for each transformer layer $\ell$ separately, rather than to the full model. Useful for identifying which layers contribute most to semantic instability. Early experiments suggest middle layers (8–10 of 12 for roberta-base) are the most sensitive — interesting.

**Def 8 (Relative SPS):** $\mathrm{rSPS} := \mathrm{SPS}_\mathcal{T} / \mathrm{SPS}_{\text{arb}}$, ratio of semantic SPS to SPS under arbitrary random directions. rSPS ≈ 1 means the model is equally sensitive to semantic and non-semantic perturbations — it's not preferentially protecting semantic space. rSPS ≪ 1 means semantic directions are disproportionately destabilizing. Normalizes out the model's overall sensitivity scale, which makes cross-architecture comparisons cleaner.

Of these, rSPS (Def 8) seems like the most practically useful diagnostic.

---

**2026.03.29**

Started writing the code. Settled on this module structure:

```
src/sps/
  core.py             SPSEstimator, StructuredSensitivityEstimator
  jacobian.py         JVP, restricted norm, spectral gap estimation
  transformations.py  T_emb and T_syn perturbation families
  metrics.py          SPSReport, rSPS, LayerwiseSPSAnalyzer
  utils.py            seed utilities, divergence functions, normalize_directions
```

Key implementation decision: use `torch.autograd.functional.jvp` rather than materializing the Jacobian. For a 125M parameter model, the full Jacobian is completely infeasible — on the order of $768 \times d_\text{input}$ per sample. JVP is O(one forward pass) per direction. With k=8 probe directions, that's 8× inference cost — acceptable.

Ran into a memory issue with WordNet. `EmbeddingPerturbationFamily` was trying to build synonym directions for the entire tokenizer vocabulary (~50k tokens), which caused a large memory spike at import time and silently fell back to random orthogonal directions for most tokens. So I was computing what I thought were synonym-informed directions, but they were actually random for the majority of the vocabulary.

Fixed with a vocab_size cap (default 10k most frequent tokens) and explicit WARNING-level logging when falling back to random. Also added `max_synonyms_per_token` to keep memory bounded. This should have been caught earlier — building 50k synset lookups at import time was never a reasonable design.

---

**2026.04.02**

Writing the test suite. The hardest part is `jacobian.py` — "the output looks plausible" isn't a valid test for Jacobian code. You need analytical ground truth.

Solution: `ExactLinearModel(W)` — a linear model where $f(x) = Wx$, so $J_f = W$ everywhere analytically. This allows:

- Verifying `directional_derivative_norm(f, x, v)` = $\|Wv\|$ to machine precision
- Constructing $W$ with a known top singular vector $u_1$, setting the semantic direction $v = u_1$, and verifying spectral gap ≈ 0 (semantic direction aligns with the most sensitive Jacobian direction)
- Setting the semantic direction orthogonal to all singular vectors of $W$ (in the null space) and verifying spectral gap ≈ 1

This class of tests catches implementation bugs that would be completely invisible if I only ran the pipeline on roberta-base and checked that the numbers looked reasonable. Simple analytical models are underused for this kind of verification.

Also wrote tests for the transformation families: shape checks, magnitude bounds, that synonym substitution actually changes tokens, that embedding perturbation stays within the $\varepsilon$ ball.

Total tests: 29. This took most of the day.

---

**2026.04.06**

Pushed the README. Immediately noticed that Assumptions A3–A5 were rendering incorrectly on GitHub — raw LaTeX symbols, broken italic formatting, missing subscripts.

Root cause: I had wrapped the assumption statements in blockquote syntax:

```
> **Assumption A3 (Measurability).** The map $x \mapsto \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)$ ...
```

GitHub's CommonMark renderer runs italic marker detection before handing off to the math renderer. The underscore in `_{\mathcal{T},\varepsilon}` inside the blockquote was partially interpreted as an italic delimiter, breaking the LaTeX. The rendered output was a mix of raw LaTeX and half-formatted text.

Fix: removed all blockquote wrappers from A3–A5 and moved the math expressions to standalone display blocks. That was all it took.

Lesson: never put LaTeX subscripts inside GitHub blockquotes. The local VS Code preview doesn't catch this — always verify against actual GitHub rendering when the README has nested subscripts.

---

**2026.04.07**

Ran the full test suite first thing — 28 pass immediately, one fails:

```
FAILED tests/test_core.py::test_zero_sensitivity_for_constant_model
AssertionError: Expected ~0 sensitivity, got tensor([4.83e-06, 5.12e-06, ...])
```

My first thought was that I'd introduced a bug somewhere. After tracing through the constant model definition — it uses `self.output_embedding.expand(batch_size, seq_len, self.hidden_size)`, literally broadcasting the same tensor — I realized there's no way for the model to produce different outputs for different inputs. The issue is in the metric, not the model.

The sensitivity computation uses `F.normalize` followed by a dot product. Two separate calls to `F.normalize` on the same underlying tensor produce slightly different float32 results due to rounding order. Dotting those nearly-identical unit vectors produces a residual of around 5e-6. Not a logic error — just the expected precision of float32 cosine arithmetic.

Changed the threshold from 1e-6 to 1e-5. This is the appropriate tolerance for float32 cosine: normalize + dot product stacks multiple floating point operations, each accumulating a small rounding error. 1e-6 was asking for float64-level precision from a float32 computation.

29/29 pass after that.

---

Then ran `estimate_sps.py` to check the spectral gap numbers on roberta-base. It failed immediately:

```
RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu is not implemented
```

JVP through roberta-base requires a differentiable path through the attention mechanism. Flash attention doesn't have a CPU backward pass — it's a GPU optimization. The fix is `sdpa_kernel(SDPBackend.MATH)`, which forces PyTorch to use the standard quadratic attention kernel that has full CPU autograd support.

My first implementation stored the context manager at module scope:

```python
_sdpa_ctx = sdpa_kernel(SDPBackend.MATH)

def _model_fn(inputs_embeds):
    with _sdpa_ctx:
        out = model(...)
```

This failed on the second call to `_model_fn`:

```
AttributeError: '_GeneratorContextManager' object has no attribute 'args'
```

The problem: `sdpa_kernel` is implemented with `@contextmanager`, which wraps a generator function. Calling `with _sdpa_ctx:` calls `__enter__()`, which advances the generator. After the `with` block exits, `__exit__()` exhausts the generator. A generator can only be iterated once — trying to enter the same instance a second time fails when PyTorch internals try to access generator state that no longer exists.

The fix is straightforward: call the factory inside the closure so each invocation gets a fresh generator instance:

```python
def _model_fn(inputs_embeds):
    with _sdpa_kernel(SDPBackend.MATH):    # fresh context manager each call
        out = model(inputs_embeds=inputs_embeds, attention_mask=mask0)
    return out.last_hidden_state[:, 0, :]
```

Spectral gap ran correctly after that. Results on 16 SST-2 test sentences with roberta-base:

```
mean spectral gap:  0.1302
std:                ~0.04
range:              0.09 – 0.19
```

A gap of 0.1302 means semantic perturbation directions are substantially aligned with the Jacobian's top singular directions — they're not orthogonal. RoBERTa's most sensitive internal directions largely overlap with the semantic perturbation directions I defined. This is consistent with the 2026.02.15 observation: the model is routing synonym-substituted inputs through different internal representations while arriving at the same output label. The spectral gap is now quantifying exactly that property.

---

Also found and fixed a display bug in the rSPS output. The output was printing `rSPS = 1.0000 < 1`, which is contradictory. The underlying value was rsps = 0.99993, which rounds to 1.0000 at 4 decimal places — but the branch condition `if rsps < 1.0` evaluated the raw float, not the rounded string, so the display said 1.0000 while the code branched on "less than 1".

Fixed with a ±5e-4 tolerance band:

```python
_tol = 5e-4
if rsps > 1.0 + _tol:
    print("rSPS > 1 — numerical error likely")
elif rsps < 1.0 - _tol:
    print(f"  rSPS = {rsps:.4f} — semantic directions are disproportionately destabilizing")
else:
    print(f"  rSPS ≈ 1 ({rsps:.4f}) — model equally sensitive to semantic and random directions")
```

29/29 tests pass. Spectral gap working. Done for the day.

---

**2026.04.08**

Wrapping up the current phase. State as of 2026.04.08:

Theory — complete. All proofs reviewed, errors from 2026.03.17 and 2026.03.20 fixed, all assumptions stated explicitly.  
Code — complete. `core.py`, `jacobian.py`, `transformations.py`, `metrics.py`, `utils.py` all written and passing tests.  
Tests — complete. 29 tests including exact analytical ground truth for the Jacobian via `ExactLinearModel`.  
Experiments — complete. `estimate_sps.py` runs the full pipeline on 16 test sentences with roberta-base.  
README — complete (after the rendering fix on 2026.04.06).

What remains: the empirical comparison across model families. I want to run roberta-base vs roberta-large vs DeBERTa vs GPT-2 on the same transformation families to see whether SPS scales with model capacity, varies by architecture, etc. This requires actual GPU time which I don't have access to right now.

Also haven't started the adversarial SPS direction — constructing $\mathcal{T}$ families specifically designed to maximize semantic sensitivity while preserving meaning. That's a next phase.
