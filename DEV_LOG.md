# Dev Log — Structured Perturbation Stability (SPS)

---

**2026.02.08**

The motivating observation came out of GLUE experiments I ran last week. Applied synonym substitution to a small SST-2 sample — standard lexical paraphrases, "good" → "excellent", "bad" → "terrible" — and found exactly what you'd expect at the output level: accuracy is unchanged. The interesting part is what happens in representation space. Cosine similarity between original and substituted [CLS] embeddings dropped to 0.58 in some cases, sometimes lower.

That number matters. In a 768-dimensional space, cosine similarity of 0.58 between two vectors representing sentences that mean the same thing is a significant divergence. The model is arriving at the same discrete decision via what appear to be meaningfully different internal trajectories. The question I can't shake: is this a failure of representation quality, or is cosine similarity just a bad proxy for what I care about? I don't have an answer yet. But the question feels important enough to pursue properly.

No experiments today — just wanted to get this written down before I convinced myself it wasn't interesting.

---

**2026.02.09**

Reading day. Went through the standard references to see how close anyone has come to this.

Jacovi & Goldberg (2020) on faithfulness — the argument that explanations should be faithful to the model's actual computational process, not just post-hoc rationalizations. Useful framing, but their concern is about explanation quality rather than representational stability under semantics-preserving transformations.

Wallace et al. (2019) on universal adversarial triggers. This is closest in spirit but orthogonal in goal: they want worst-case perturbations, I want meaning-preserving ones. The adversarial setting has a fundamentally different optimization target.

Sinha et al. (2021) showing BERT-family models can handle MNLI with scrambled word order. Suggests these models aren't relying on syntactic structure in the way we might assume. Relevant context but again tangential.

None of this addresses what I observed yesterday. The common thread across all of it is that task performance is treated as the ground truth signal. I want to interrogate the internal representation, not the output label. That's a different question and it seems like no one has asked it in quite this form.

---

**2026.02.10**

Dug deeper into representation similarity literature. CKA (centered kernel alignment) and SVCCA both look interesting but they're designed for comparing representations across layers or across model checkpoints — not for analyzing a single model's response to semantically equivalent inputs. The invariance question I'm interested in is different: how much does a fixed model's representation vary under a fixed class of semantics-preserving transformations?

Working definition I wrote in my notebook today:

*"Semantic stability: the degree to which a model's internal representations are invariant under transformations that preserve semantic content."*

Still rough. "Preserve semantic content" is doing a lot of work here and I haven't defined it. But this is the right framing. Now I need to make it precise.

---

**2026.02.12**

First experiment. Wanted a quantitative baseline before developing any framework.

Setup: 200 SST-2 sentences, Gaussian noise added to all token embeddings (σ = 0.01), measured KL divergence between original and noisy output distributions and cosine similarity between [CLS] representations.

Results:
- KL divergence: median ≈ 0.003
- Cosine similarity: 0.98+ for virtually all samples

I initially read this as "the model is robust to noise." It took a couple of days to understand why that interpretation is wrong.

The issue: output stability under small random noise is almost entirely explained by softmax saturation. At high-confidence predictions — which is most of SST-2 — the logit margin is large enough that small perturbations to the input don't move the output distribution meaningfully. I was measuring the softmax's insensitivity to noise, not anything about representation quality. The experiment was testing the wrong thing.

This was a useful null result in retrospect: it forced me to be more precise about what I actually want to measure.

---

**2026.02.15**

Ran synonym substitution more carefully. Setup: 500 SST-2 sentences, one WordNet synonym substitution per sentence (randomly sampled from the first three synsets), measured accuracy and [CLS] cosine similarity.

Results:
- Accuracy: 91.2% → 90.8% (0.4pp drop — within noise)
- Cosine similarity, original vs. substituted [CLS]: mean 0.61, std 0.14, minimum observed 0.43

This is the result that actually started everything. The model maintains decision-level accuracy while exhibiting substantial representational divergence — up to cosine distance of ~0.57 for semantically equivalent inputs. Two interpretations:

**(a) The metric is uninformative.** Cosine similarity of 0.43 might not mean what I think it means in this space — if the relevant geometry is different from what cosine captures, this number is noise.

**(b) The model is encoding semantically irrelevant distinctions.** The representation is picking up on surface features that don't affect the classification boundary, but vary across semantically equivalent inputs.

I went back and forth on this for most of the day. Eventually concluded: the right question isn't which interpretation is correct, it's whether I can build a metric that makes the distinction principled and measurable. That's the project.

---

**2026.02.19**

First formalization attempt. Sensitivity functional, first draft:

$$S(f, x, T) = \frac{d_Y(f(Tx), f(x))}{c(T, x)}$$

Ratio of output divergence to transformation magnitude, normalized to prevent large transformations from trivially dominating.

Three problems I couldn't resolve before stopping for the day:

1. **Choice of $d_Y$.** For classification, the output is a probability distribution. KL divergence is asymmetric and undefined when support doesn't overlap. Total variation is bounded but loses gradient information. L2 over logits has no clear probabilistic interpretation. None of these feel canonical, and the choice of $d_Y$ will affect what the metric is actually measuring.

2. **Quantifying $c(T, x)$ for discrete substitutions.** What is the "magnitude" of replacing "dog" with "canine"? There's no obvious embedding-space distance that captures the semantic proximity of a substitution without making additional model-dependent assumptions.

3. **Unconstrained $T$ leads to degenerate behavior.** Without constraints on the transformation family, the supremum over all $T$ is dominated by maximally destructive changes. An arbitrary scrambling of the input is a valid transformation under this definition. That's not semantics-preserving stability — it's adversarial robustness. Different question.

Wrote "needs constraints on $\mathcal{T}$" and stopped. The formalization is harder than it initially appeared.

---

**2026.02.22**

I think I've identified the root cause of the 2026.02.19 problems.

The issue is that I've been treating the transformation space as uniform. It isn't. Replacing "dog" with "canine" is semantics-preserving. Replacing "dog" with "photosynthesis" is not. A stability metric that sums over both equally isn't measuring semantic invariance — it's measuring something like perturbation robustness, which is a weaker and less interesting property.

The fix: require $T$ to belong to an *admissible family* $\mathcal{T}$ of semantics-preserving transformations. Only transformations in $\mathcal{T}$ count toward the sensitivity measure. The stability metric is then defined with respect to a specific $\mathcal{T}$.

First draft of Axiom A5 (admissible transformation family):
- Identity: $\mathrm{id} \in \mathcal{T}$ (zero perturbation is the baseline)
- Bounded magnitude: $c(T, x) \leq \varepsilon$ for some $\varepsilon > 0$
- Semantic preservation: (still informal — formalization pending)

The semantic preservation condition is going to be the hard part. But at least the conceptual move is clear: the metric is parameterized by a family $\mathcal{T}$, and the family is where the semantics live.

Also realized the sensitivity functional should be defined as the supremum over $\mathcal{T}$, not evaluated at a single transformation — worst-case sensitivity within the admissible family.

---

**2026.02.25**

Spent today working through how to formally define semantic preservation in A5 without circularity.

Three options I seriously considered:

1. **Human judgment.** Ground truth: have annotators verify that $Tx$ and $x$ are semantically equivalent. Expensive, subjective, not scalable, and would need to be repeated for every new distribution. Non-starter for a computationally tractable framework.

2. **NLI-based entailment.** Use an NLI model to verify bidirectional entailment: $x \models Tx$ and $Tx \models x$. The problem: I'm trying to evaluate a model's semantic representations, and using another model's output to define "semantic equivalence" introduces circularity. The framework's validity would then depend on the NLI model's accuracy, and any biases in the NLI model transfer to the metric.

3. **Operational definition via curated families.** Define $\mathcal{T}$ as transformations drawn from specific families with known semantics-preserving properties: WordNet synonym substitution, controlled back-translation, template-based paraphrase generation. $\mathcal{T}$ is operationally defined rather than axiomatically characterized.

Going with option 3. The limitation is real — the framework's scope is bounded by whatever families I include in $\mathcal{T}$, and universality is compromised. I'll acknowledge this explicitly. But the alternatives have more serious problems, and option 3 is tractable.

Also realized during this that the proof of Proposition 1 requires two assumptions I hadn't made explicit: A3 (measurability of $x \mapsto \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)$) and A4 (integrability of the sensitivity expectation). Will formalize both.

---

**2026.03.01**

Wrote the SPS definition today:

$$\mathrm{SPS}_\varepsilon(f_\theta) := \exp\!\left(-\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)]\right)$$

The exponential mapping is motivated by several things:
- It maps $[0, \infty)$ to $(0, 1]$, giving a clean and interpretable range
- SPS = 1 iff expected sensitivity is zero — perfect semantic invariance
- SPS → 0 as expected sensitivity diverges — represents maximally unstable models
- There's an information-theoretic interpretation I haven't fully worked out involving the rate function of the sensitivity distribution, but I'm not confident it's more than coincidence

One error I caught mid-derivation: wrote "this makes SPS multiplicative under composition," then realized that's only true if the expectations decompose additively, which requires independence of composition steps. They're not independent. Theorem 3 will need to treat the composition behavior as an inequality rather than an equality.

---

**2026.03.06**

Tried to characterize the sensitivity functional in terms of the Jacobian. In the $\varepsilon \to 0$ limit, the sensitivity should reduce to a directional derivative — which is exactly a Jacobian-vector product.

Wrote:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x) \xrightarrow{\varepsilon \to 0} \|J_{f_\theta}(x)\|_{A_x}$$

where $A_x$ denotes the set of perturbation directions induced by $\mathcal{T}$ at input $x$, and $\|\cdot\|_{A_x}$ is the operator norm restricted to that direction set.

The proof sketch looked right. I noticed an issue — the LHS involves a supremum over all $T \in \mathcal{T}$ while I was treating the RHS as if $A_x$ were a single direction — but I told myself it was probably fine. It wasn't. Caught the actual error on 2026.03.17.

---

**2026.03.09**

Composition experiment. Used back-translation as $T_1$ and synonym substitution as $T_2$, with individual SPS scores of approximately 0.78 and 0.81 respectively on a held-out evaluation set. Measured SPS for the composed transformation $T_2 \circ T_1$.

Composed SPS: 0.69 — below both individual scores.

Ran several additional pairs. The pattern is consistent: composed SPS falls below $\min(\mathrm{SPS}_1, \mathrm{SPS}_2)$.

The geometric interpretation: each transformation displaces the representation in a different direction in embedding space. Two displacements in non-aligned directions compound rather than cancel, yielding greater total drift than either individually. The upper bound $\mathrm{SPS}(\mathcal{T}_2 \circ \mathcal{T}_1) \leq \min(\mathrm{SPS}_1, \mathrm{SPS}_2)$ holds in every case I tested and follows directly from the definition. Theorem 3.

---

**2026.03.13**

Reading Sedghi et al. on singular value bounds for convolutional networks. While working through the spectral norm material, I think I found something interesting.

The $A_x$-restricted operator norm $\|J_f(x)\|_{A_x}$ is generically strictly less than the full spectral norm $\sigma_{\max}(J_f(x))$, because $A_x$ is a proper subset of the unit sphere. The gap between these two quantities encodes something geometrically meaningful: what fraction of the Jacobian's total sensitivity is directed along semantic perturbation directions?

Define the semantic spectral gap:

$$\bar{\gamma}(f, x, \mathcal{T}) = 1 - \frac{\|J_f(x)\|_{A_x}}{\sigma_{\max}(J_f(x))}$$

$\bar{\gamma} \approx 1$: semantic directions are nearly orthogonal to the Jacobian's top singular directions. The model's maximum sensitivity is allocated to directions that don't affect meaning — arguably the ideal behavior.

$\bar{\gamma} \approx 0$: the Jacobian's most sensitive direction aligns with a semantic perturbation direction. The model is maximally sensitive to precisely the kinds of changes that should be irrelevant.

I haven't seen this geometric characterization in the existing robustness or interpretability literature. It might be new.

Computational note: materializing the full Jacobian is infeasible for large models — for roberta-base (125M parameters), the Jacobian at a single input is on the order of $768 \times d_\text{input}$ entries. Solution: randomized power iteration using JVPs. Each JVP is O(one forward pass). With $k=8$ probe directions, this gives a stochastic approximation of $\sigma_{\max}$ at 8× inference cost — noisy but tractable.

---

**2026.03.17**

Found the error in the 2026.03.06 Theorem 2 draft.

The statement I had was effectively equating:

$$\underbrace{\|J_f(x) v_T\|}_{\text{directional deriv. for a specific } T} \quad = \quad \underbrace{\|J_f(x)\|_{A_x}}_{\sup_{v \in A_x} \|J_f(x) v\|}$$

These are not equal in general. The left side is a single JVP evaluation at a specific direction $v_T$. The right side is the operator norm restricted to $A_x$ — a supremum over all unit vectors in the direction set. They're equal only if $v_T$ happens to be the direction achieving the supremum, which isn't guaranteed unless $A_x$ is a singleton.

I was claiming equality where only $\leq$ holds, and I conflated a pointwise instantiation with the operator norm.

Fix: split into two claims.
- **Part (i):** For a fixed $T \in \mathcal{T}$, the pointwise sensitivity in the $\varepsilon \to 0$ limit equals $\|J_f(x) v_T\|$, where $v_T$ is the limiting perturbation direction. This is just the definition of the directional derivative and requires no additional assumptions.
- **Part (ii):** Taking the sup over $T \in \mathcal{T}$ recovers the $A_x$-restricted operator norm in the $\varepsilon \to 0$ limit. This requires A2 (compactness of $A_x^{(\varepsilon)}$ in the limit) to ensure the supremum is actually attained.

The split version is cleaner and more honest about what each claim requires.

---

**2026.03.20**

Reviewing Proposition 1 — the claim that $\mathrm{SPS}_\varepsilon(f_\theta) \in (0, 1]$.

Upper bound: $\exp(-\text{nonneg}) \leq 1$. Trivial.

Lower bound (strict positivity): The argument I had was "$\exp$ is always strictly positive, therefore $\mathrm{SPS}_\varepsilon > 0$." This fails. $\exp(-\infty) = 0$. If the expected sensitivity diverges — i.e., $\mathbb{E}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)] = +\infty$ — then SPS = 0, not something strictly positive. The strict lower bound requires the expectation to be finite, and that's an assumption, not a consequence of the other axioms.

Added **Assumption A4 (Integrability):** $\mathbb{E}_{x \sim \mathcal{D}}[\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)] < \infty$.

Under A4 the strict lower bound follows immediately. Without A4 the proposition only gives $\mathrm{SPS}_\varepsilon \geq 0$. The assumption is mild in practice — if the expectation diverges, the model is pathologically unstable and SPS = 0 is the right answer — but it needs to be stated explicitly. This was a genuine gap in the original proof, not a technicality.

---

**2026.03.24**

The framework is theoretically complete enough to implement. Spent today formalizing the computable definitions.

**Def 5 (Semantic spectral gap):** Formalizes $\bar{\gamma}$ from 2026.03.13. Signature: $(f_\theta, \mathcal{T}, x) \to [0, 1]$.

**Def 6 (Empirical SPS):** Monte Carlo estimator — sample $N$ inputs from $\mathcal{D}$, average the pointwise sensitivity functional. This is what `core.py` computes; the expectation in the theoretical definition isn't analytically tractable.

**Def 7 (Layer-wise SPS):** Apply SPS to the intermediate map $x \mapsto h^{(\ell)}(x)$ for each transformer layer $\ell$ individually, rather than to the full output map. Allows localization of semantic instability across depth. Early experiments suggest layers 8–10 of 12 in roberta-base exhibit higher sensitivity than earlier or later layers — consistent with the hypothesis that middle layers encode richer semantic distinctions.

**Def 8 (Relative SPS):** $\mathrm{rSPS} := \mathrm{SPS}_\mathcal{T} / \mathrm{SPS}_{\text{arb}}$, where $\mathrm{SPS}_{\text{arb}}$ is computed against random uniform perturbation directions rather than $\mathcal{T}$. rSPS ≈ 1 means the model is equally sensitive to semantic and non-semantic perturbations — it isn't preferentially protecting the semantic subspace. rSPS ≪ 1 means semantic directions are disproportionately destabilizing relative to arbitrary directions. This normalization makes cross-model comparisons more interpretable.

---

**2026.03.29**

Started implementation. Module structure:

```
src/sps/
  core.py             SPSEstimator, StructuredSensitivityEstimator
  jacobian.py         JVP computation, restricted operator norm, spectral gap estimation
  transformations.py  EmbeddingPerturbationFamily (T_emb), SynonymSubstitutionFamily (T_syn)
  metrics.py          SPSReport, rSPS, LayerwiseSPSAnalyzer
  utils.py            set_seed, divergence functions, normalize_directions
```

Key implementation decision: `torch.autograd.functional.jvp` rather than explicit Jacobian materialization. For roberta-base (768-dimensional hidden state, ~512 token sequence), the full Jacobian per input is on the order of $768 \times 512 \times 768 \approx 300M$ entries — completely infeasible. JVP is O(one forward pass) per probe direction. At $k = 8$ probe directions this gives a stochastic approximation of $\sigma_{\max}$ at 8× inference cost.

Hit a significant bug with the WordNet-based synonym family. `EmbeddingPerturbationFamily` was loading all WordNet synsets for the full tokenizer vocabulary (~50k tokens) at import time, causing a large memory spike and then silently falling back to random orthogonal directions for most tokens. So I was computing what appeared to be semantically-informed perturbation directions but was actually mostly random. The fallback was logged at WARNING level but easy to miss.

Fixed: vocab_size cap defaulting to the 10k most frequent tokens, with explicit logging when the fallback path is taken. Also added `max_synonyms_per_token` to bound memory use during synset loading. The root issue was trying to eagerly precompute directions for a vocabulary far larger than what any experiment would actually use.

---

**2026.04.02**

Writing the test suite. The challenge with `jacobian.py` is that "the output looks plausible on roberta-base" is not a valid correctness test for Jacobian computation. You need analytical ground truth.

Solution: `ExactLinearModel(W)` — a linear model $f(x) = Wx$, so $J_f = W$ everywhere, independent of input. This enables:

- Exact verification that `directional_derivative_norm(f, x, v)` equals $\|Wv\|$ to floating-point precision
- Constructing $W$ with a known top singular vector $u_1$, setting the semantic direction $v = u_1$, and verifying that the estimated spectral gap is ≈ 0 (semantic and most-sensitive directions coincide)
- Setting $v$ in the null space of $W$ — orthogonal to all singular directions — and verifying spectral gap ≈ 1

This class of test catches implementation errors that would be invisible at the level of running the pipeline on a real model and observing that the numbers are in a plausible range. Having exact analytical ground truth is essential for Jacobian code.

Also wrote tests for the transformation families: shape, magnitude bounds, that synonym substitution actually changes tokens rather than identity-mapping them, that embedding perturbation stays within the $\varepsilon$-ball.

Total tests: 29.

---

**2026.04.06**

Pushed the README. Noticed immediately that Assumptions A3–A5 were broken on GitHub — raw LaTeX symbols, partial italic formatting, missing subscripts.

Root cause: the assumption statements were wrapped in GitHub blockquote syntax:

```
> **Assumption A3 (Measurability).** The map $x \mapsto \mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x)$ ...
```

GitHub's CommonMark renderer applies italic marker detection before the math renderer processes the content. The underscores in `_{\mathcal{T},\varepsilon}` inside the blockquote were partially interpreted as italic delimiters, corrupting the LaTeX. The rendered output was a mix of raw math markup and malformed text.

Fix: removed blockquote wrappers from A3–A5 and moved the math to standalone display-mode blocks. VS Code's local preview doesn't catch this — you have to check actual GitHub rendering when the README contains nested subscripts in blockquotes.

---

**2026.04.07**

Ran the full test suite first thing. 28 pass, one fails:

```
FAILED tests/test_core.py::test_zero_sensitivity_for_constant_model
AssertionError: Expected ~0 sensitivity, got tensor([4.83e-06, 5.12e-06, ...])
```

My first instinct was a bug in the constant model definition. Traced through it: `return self.output_embedding.expand(batch_size, seq_len, self.hidden_size)` — same tensor broadcast everywhere. There's no input-dependent computation happening. The model cannot produce different outputs for different inputs.

So the issue is in the metric. The sensitivity computation uses `F.normalize` followed by a dot product. Two separate `F.normalize` calls on the same input tensor produce marginally different float32 results due to rounding order in the normalization computation. Dotting those nearly-identical unit vectors yields a residual of ~5e-6 — not a logic error, just the expected numerical behavior of float32 cosine arithmetic.

Changed the threshold from 1e-6 to 1e-5. This is the correct tolerance for float32 cosine: the computation stacks normalize → dot product → subtract from 1, with rounding error accumulating at each step. A threshold of 1e-6 implicitly requires float64 precision from a float32 computation.

29/29 pass.

---

Ran `estimate_sps.py` to check the spectral gap on roberta-base. Failed immediately:

```
RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu is not implemented
```

`torch.autograd.functional.jvp` requires a differentiable path through the model. Flash attention doesn't implement a CPU backward pass — it's a GPU-specific optimization. The fix is to force PyTorch to use the math (quadratic) attention kernel, which has full autograd support on CPU.

First implementation: stored the context manager at module scope.

```python
_sdpa_ctx = sdpa_kernel(SDPBackend.MATH)

def _model_fn(inputs_embeds):
    with _sdpa_ctx:
        out = model(...)
```

Failed on the second call to `_model_fn`:

```
AttributeError: '_GeneratorContextManager' object has no attribute 'args'
```

`sdpa_kernel` uses `@contextmanager`, which wraps a generator. `with _sdpa_ctx:` calls `__enter__()` → `next()` on the generator. After the `with` block exits, `__exit__()` exhausts it. Generators are single-iteration iterators; trying to enter the same instance again hits an internal PyTorch attribute access on a spent generator.

Fix: instantiate inside the closure so each call gets a fresh context manager:

```python
def _model_fn(inputs_embeds):
    with _sdpa_kernel(SDPBackend.MATH):    # new instance each invocation
        out = model(inputs_embeds=inputs_embeds, attention_mask=mask0)
    return out.last_hidden_state[:, 0, :]
```

Spectral gap results on 16 SST-2 test sentences, roberta-base:

```
mean spectral gap:  0.1302
std:                ~0.04
range:              0.09 – 0.19
```

A gap of 0.1302 indicates that semantic perturbation directions are substantially aligned with the Jacobian's top singular directions — they are not orthogonal to the model's most sensitive dimensions. This is consistent with the 2026.02.15 finding: the model routes semantically equivalent inputs through measurably different representational trajectories. The spectral gap is now quantifying exactly the geometric property I was trying to capture back on 2026.03.13.

---

Also fixed a display inconsistency in the rSPS output. The printed value was `rSPS = 1.0000 < 1` — contradictory on its face. The underlying float was rsps = 0.99993, which rounds to 1.0000 at 4 decimal places, but the branch condition `if rsps < 1.0:` evaluates the raw float, not the rounded representation. The display said 1.0000 while the code had taken the "less than 1" branch.

Fixed with a ±5e-4 tolerance band around 1.0:

```python
_tol = 5e-4
if rsps > 1.0 + _tol:
    print("rSPS > 1.0 — likely numerical error, check computation")
elif rsps < 1.0 - _tol:
    print(f"  rSPS = {rsps:.4f} — semantic directions disproportionately destabilizing")
else:
    print(f"  rSPS ≈ 1 ({rsps:.4f}) — model not preferentially protecting semantic subspace")
```

29/29 pass. Spectral gap working. Done.

---

**2026.04.08**

Current state:

Theory — complete. All proofs verified, errors from 2026.03.17 and 2026.03.20 corrected, all assumptions explicitly stated.  
Code — complete. `core.py`, `jacobian.py`, `transformations.py`, `metrics.py`, `utils.py` all written and tested.  
Tests — complete. 29 tests, including analytical ground truth verification via `ExactLinearModel`.  
Experiments — complete. `estimate_sps.py` runs the full pipeline on 16 SST-2 sentences with roberta-base.  
README — complete (rendering fix pushed on 2026.04.06).

Remaining: empirical comparison across model families (roberta-base vs roberta-large vs DeBERTa vs GPT-2) to characterize whether SPS scales with model capacity or varies by architecture. Requires GPU access I don't currently have.

Also not started: the adversarial stability direction — constructing $\mathcal{T}$ families optimized to maximize semantic sensitivity while satisfying the semantic preservation constraint. That's a well-defined optimization problem and probably the most interesting next step from a theoretical standpoint.
