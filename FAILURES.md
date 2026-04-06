# Failures & Dead Ends

things that didn't work. writing these down because (a) i'll forget and (b) the failures actually explain a lot of the design decisions in the final framework.

---

## random perturbation testing

thought: add gaussian noise to embeddings, measure output consistency, call it robustness.

this was my first experiment and i was wrong about what i was measuring. outputs were super stable under noise but that's because softmax at high confidence is basically flat — you can push the pre-softmax logits around a fair amount before the output probabilities actually change. i interpreted this as "model is robust" and nearly moved on. in retrospect i was just measuring output-layer saturation, not anything about the representations.

the noise also had no semantic grounding. it disrupts syntax, morphology, and meaning all at once. you can't interpret the result.

→ this is what pushed me toward constrained transformation families. if the perturbation has no semantic meaning, neither does the stability measurement.

---

## output-only consistency metric

measuring $d_Y(f(x), f(Tx))$ without constraining $T$ seemed more principled. it wasn't.

the problem: a model that memorizes training data can appear perfectly stable because it just looks up the answer regardless of what $Tx$ is. it'll return the same prediction for the original and any paraphrase of a training point. but ask it about a novel paraphrase and it falls apart — which the metric never detects.

output-space metrics collapse information in a way that hides exactly the failure mode i care about. the metric needs to be in representation space, not just output space.

→ this is what pushed me toward the Jacobian reformulation. sensitivity needs to be measured at the representation level, not just at the prediction level.

---

## theorem 2 — this one took me a while to notice

original theorem 2 basically said:

$\delta(x)$ [single direction] $=$ $\|J_f(x)\|_{A_x}$ [sup over all directions in $A_x$]

these are not equal. a single JVP gives you the directional derivative for one specific $T$. the restricted operator norm is the supremum over all directions in $A_x$. equality would require the sup to be achieved at exactly that direction, which isn't guaranteed in general (only if $A_x$ is a singleton, which it's not).

i was asserting equality where only $\leq$ holds. the statement confused a single instantiation with the operator norm.

fixed by splitting into two claims:
- part (i): for a fixed $T$, pointwise sensitivity = JVP norm. this is fine.
- part (ii): taking sup over all $T$ gives the restricted operator norm, but only in the $\varepsilon \to 0$ limit, and only when $A_x$ is compact (A2). needed to add that explicitly.

i caught this late when i was going back through the proofs carefully before writing up the paper. would have been embarrassing to miss.

---

## proposition 1 — missing assumption

claimed $\mathrm{SPS}_\varepsilon > 0$ as a strict inequality. the proof was: $\exp(-\mathbb{E}[\mathrm{Sens}]) > 0$ because exp is always positive.

but $\exp(-\infty) = 0$. if the expectation diverges to $+\infty$, the claim fails. i needed to assume the sensitivity expectation is finite (Assumption A4, integrability) before the strict positivity claim is valid.

this was a gap in the axioms. added A4 explicitly. without it you can only claim $\mathrm{SPS}_\varepsilon \geq 0$.

---

## theorem 4 — missing chain constraint

theorem 4 is a triangle inequality for chained transformations $T_2 \circ T_1$. but i forgot to state that $c(T_2, T_1 x) \leq \varepsilon$ is required — i.e., the second transformation also needs to be within the admissible radius when applied to the already-transformed input.

without that constraint, $T_2 \circ T_1$ might push outside the $\varepsilon$-ball where Sens is defined, and the theorem doesn't apply. added as an explicit hypothesis. it's a non-trivial requirement — A5 only guarantees closure under composition when both single steps are within $\varepsilon/2$.

---

## notation inconsistency — the $\varepsilon$ subscript

theorems 3 and 4 in my drafts dropped the $\varepsilon$ subscript from Sens. wrote $\mathrm{Sens}_\mathcal{T}$ instead of $\mathrm{Sens}_{\mathcal{T},\varepsilon}$.

that's not just sloppy notation — it implies a global (non-local) sensitivity, which is a stronger and different claim. the whole framework is parameterized by the perturbation radius. dropped it inconsistently throughout two theorems.

went back and fixed everywhere. tedious but necessary.

---

## readme rendering — blockquotes + math don't mix on github

pushed the readme with A3–A5 assumptions inside blockquote syntax (`>`). on github the CommonMark renderer processes italic markers before handing off to the math renderer. so `_{\mathcal{T},\varepsilon}` inside a blockquote got partially treated as italic markup and the output was garbage — broken symbols mixed with partial formatting.

fixed by removing the blockquotes entirely and putting the math on standalone lines. obvious in retrospect. always preview math-heavy markdown on github before finalizing.

---

## wordnet vocab loading — memory spike

`EmbeddingPerturbationFamily` tried to build synonym directions from WordNet for the entire tokenizer vocabulary (50k+ tokens). this caused a massive memory spike at module import time, and silently fell back to random orthogonal directions for almost all tokens without telling anyone.

so the "synonym directions" were mostly random. not ideal.

fixed with a `vocab_size` cap (default 10k), explicit logging when falling back, and a `max_synonyms_per_token` parameter. should have thought about this earlier — 50k WordNet lookups at import is obviously going to be a problem.
