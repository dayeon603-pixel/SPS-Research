# Research Process — SPS

not meant to be polished. just tracking how this actually developed so i don't forget and so it's reproducible if anyone (including future me) needs to redo any of this.

---

## where this started

i kept noticing something weird when running synonym substitution experiments — accuracy stayed basically the same but the [CLS] embeddings were all over the place. like sometimes cosine similarity between original and paraphrased would drop to like 0.55 and the model would still get the answer right. at first i thought maybe cosine similarity was just a bad metric. but then i looked at the actual embedding vectors and they were genuinely different. the model was arriving at the same prediction through completely different internal paths.

that's when i got suspicious. the interpretability stuff i'd been reading (Jacovi & Goldberg, some of the probing work) all basically uses accuracy as a stand-in for understanding. but that assumption seemed really shaky based on what i was seeing.

the core question i kept coming back to: can a model be "right for the wrong reasons" in a way we can actually measure? and can we measure it without just doing more accuracy tests?

---

## early ideas that didn't work

first instinct was just to throw gaussian noise at embeddings and see what happens. seemed reasonable at the time. result: outputs were super stable. i wrote in my notes that the model was "robust" and almost moved on. thank god i didn't.

the problem — which took me embarrassingly long to see — is that output stability under random noise is almost entirely a function of the softmax being saturated at high confidence. it has nothing to do with whether the representation makes sense. i was measuring the wrong layer.

tried output-only consistency next. measure $d_Y(f(x), f(Tx))$ directly across a bunch of paraphrases. this felt more principled but had the same problem in disguise. a model that just memorized the training data would score perfectly stable even if it totally falls apart on novel paraphrases. the metric was masking exactly what i wanted to detect.

also looked at AdvGLUE and CheckList. those are testing adversarial robustness — they're optimizing for attacks. that's a different question. i want neutral characterization, not worst-case. those benchmarks weren't going to give me what i needed.

---

## the key realization

around day 15 i wrote this in my notes: *"not all perturbations are equal — only structure-preserving ones should count"*

that sounds obvious in retrospect but it took two weeks of failed experiments to get there. the issue with random noise and arbitrary transformations is that they're semantically destructive. if i swap "dog" for "photosynthesis" the model should be sensitive to that. the question is whether it's sensitive when i swap "dog" for "canine."

so the framework needed a constrained transformation family $\mathcal{T}$ — only transformations that preserve semantic content get to participate in the stability measurement. this immediately raised the hard question of how you define "semantic preservation" formally without circularity, but at least it pointed in a direction.

---

## formalizing it

the sensitivity functional came together pretty naturally once the constrained family idea was in place:

$$\mathrm{Sens}_{\mathcal{T},\varepsilon}(f_\theta; x) := \sup_{T \in \mathcal{T},\, c(T,x) \leq \varepsilon} \frac{d_\mathcal{Y}(f_\theta(Tx), f_\theta(x))}{c(T, x)}$$

aggregated over the data distribution via exp(-E[Sens]) to get a bounded score in (0,1].

i chose the exponential because: (1) maps [0,∞) to (0,1], interpretable. (2) SPS=1 is perfect invariance, makes sense. (3) i had an information-theoretic intuition that i haven't fully pinned down yet — it feels like it should connect to entropy of the sensitivity distribution but i haven't written that up properly.

for semantic preservation in $\mathcal{T}$ i went with an operational definition: transformations drawn from curated families (synonym substitution, controlled paraphrase, back-translation). this is less elegant than an axiomatic definition but it's tractable. human annotation is too expensive. using an NLI model to check entailment feels circular since we'd be evaluating the model using another model's judgments.

---

## the jacobian connection

connecting SPS to the Jacobian was the thing that made the theory feel complete to me. in the $\varepsilon \to 0$ limit, sensitivity is just the directional derivative along the perturbation direction — which is a JVP. and if you take the sup over all directions in $A_x$, you get the $A_x$-restricted operator norm of $J_f(x)$.

this led to the spectral gap idea: a model with high spectral gap has its most sensitive directions geometrically orthogonal to the semantic manifold. that's a desirable property. and it's something no existing metric captures.

(there was a significant error in my original Theorem 2 here — see FAILURES.md. the LHS and RHS weren't equivalent. fixed in the corrected version.)

---

## open questions i haven't resolved

- does SPS scale predictably with model size? i'd guess there's some phase transition but i have nothing empirical yet
- can you directly optimize SPS as a regularizer during training? adding $\lambda \cdot \mathrm{Sens}$ to the loss seems natural but i haven't tested if it causes representation collapse
- multimodal extension: what does $\mathcal{T}$ look like for image-text pairs? the operational definition is harder to nail down
- the composition bound in Theorem 3 ($\mathrm{SPS}(\mathcal{T}_2 \circ \mathcal{T}_1) \leq \min(\mathrm{SPS}_1, \mathrm{SPS}_2)$) is probably loose. tighter characterization would be good but i don't know what it looks like yet

---

## what i'd do next with more time

run the full empirical comparison — RoBERTa vs BERT vs DeBERTa vs GPT-2 on the same $\mathcal{T}$ families. adversarial SPS (construct $\mathcal{T}$ to maximize sensitivity while preserving semantics) would be interesting. geometric visualization of $A_x$ vs. top singular vectors of $J_f(x)$. SPS on CLIP/LLaVA.

none of this is started yet.
