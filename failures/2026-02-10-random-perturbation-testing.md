# Failure: Random Perturbation Testing

**Date:** 2026-02-10
**Status:** resolved

---

Thought: add Gaussian noise to embeddings, measure output consistency, call it robustness.

This was my first experiment and I was wrong about what I was measuring. Outputs were super stable under noise but that's because softmax at high confidence is basically flat — you can push the pre-softmax logits around a fair amount before the output probabilities actually change. I interpreted this as "model is robust" and nearly moved on. In retrospect I was just measuring output-layer saturation, not anything about the representations.

The noise also had no semantic grounding. It disrupts syntax, morphology, and meaning all at once. You can't interpret the result.

→ This is what pushed me toward constrained transformation families. If the perturbation has no semantic meaning, neither does the stability measurement.
